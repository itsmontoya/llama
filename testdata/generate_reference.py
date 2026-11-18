#!/usr/bin/env python3
"""Generate synthetic weights and an independent scalar binary64 oracle.

Only the Python standard library is used. This program never calls the Go
decoder or neuralnetwork. Weights/config epsilon are rounded to binary32;
subsequent reference arithmetic uses binary64, math.fsum, and complex RoPE.
"""

import cmath
import hashlib
import json
import math
from pathlib import Path
import platform
import struct


ROOT = Path(__file__).resolve().parent
GENERATOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ARCHITECTURE = "https://github.com/meta-llama/llama3/blob/main/llama/model.py"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def values(name, count, norm=False):
    result = []
    for index in range(count):
        digest = hashlib.sha256(f"{name}:{index}".encode("ascii")).digest()
        value = (int.from_bytes(digest[:2], "little") % 257 - 128) / 512
        result.append(f32(1 + value if norm else value))
    return result


def make_weights(name, c):
    d, f, v = c["dim"], c["ffn_dim"], c["vocab_size"]
    kv = c["kv_heads"] * (d // c["heads"])
    weights = {"token_embedding": values(f"{name}.embedding", v * d), "layers": []}
    for layer in range(c["layers"]):
        prefix = f"{name}.layer.{layer}"
        weights["layers"].append({
            key: values(f"{prefix}.{key}", count, norm)
            for key, count, norm in [
                ("attention_norm", d, True), ("wq", d * d, False),
                ("wk", kv * d, False), ("wv", kv * d, False),
                ("wo", d * d, False), ("ffn_norm", d, True),
                ("w_gate", f * d, False), ("w_up", f * d, False),
                ("w_down", d * f, False),
            ]
        })
    weights["final_norm"] = values(f"{name}.final_norm", d, True)
    if not c["tie_embeddings"]:
        weights["output"] = values(f"{name}.output", v * d)
    return weights


def rows(flat, width):
    return [flat[offset:offset + width] for offset in range(0, len(flat), width)]


def flatten(value):
    if not isinstance(value, list):
        return [value]
    return [element for child in value for element in flatten(child)]


def linear(x, flat):
    matrix = rows(flat, len(x[0]))
    return [[math.fsum(a * b for a, b in zip(row, weight)) for weight in matrix] for row in x]


def rms_norm(x, weight, epsilon):
    result = []
    for row in x:
        inverse = 1 / math.sqrt(math.fsum(value * value for value in row) / len(row) + epsilon)
        result.append([value * inverse * scale for value, scale in zip(row, weight)])
    return result


def residual(x, update):
    return [[a + b for a, b in zip(row, delta)] for row, delta in zip(x, update)]


def rotate(heads, theta):
    result = []
    for position, token in enumerate(heads):
        token_result = []
        for head in token:
            rotated = []
            for feature in range(0, len(head), 2):
                phase = cmath.exp(1j * position * theta ** (-feature / len(head)))
                pair = complex(head[feature], head[feature + 1]) * phase
                rotated.extend([pair.real, pair.imag])
            token_result.append(rotated)
        result.append(token_result)
    return result


def attend(q, k, v, c):
    t, h, width = len(q), c["heads"], len(q[0][0])
    probabilities = [[[0.0] * t for _ in range(t)] for _ in range(h)]
    result = []
    for position, queries in enumerate(q):
        merged = []
        for head, query in enumerate(queries):
            kv_head = head // (h // c["kv_heads"])
            # Iterate only visible keys; no triangular mask or head expansion.
            scores = [math.fsum(a * b for a, b in zip(query, k[key][kv_head])) / math.sqrt(width)
                      for key in range(position + 1)]
            maximum = max(scores)
            exponentials = [math.exp(score - maximum) for score in scores]
            total = math.fsum(exponentials)
            probs = [value / total for value in exponentials]
            probabilities[head][position][:position + 1] = probs
            merged.extend(math.fsum(prob * v[key][kv_head][feature] for key, prob in enumerate(probs))
                          for feature in range(width))
        result.append(merged)
    return result, probabilities


def reference(c, w, tokens):
    trace = {}
    t, d, h, kh = len(tokens), c["dim"], c["heads"], c["kv_heads"]
    width = d // h

    def record(name, value, shape):
        flat = flatten(value)
        assert len(flat) == math.prod(shape)
        assert all(math.isfinite(item) for item in flat)
        trace[name] = {"shape": shape, "values": flat}

    embedding = rows(w["token_embedding"], d)
    x = [embedding[token][:] for token in tokens]
    record("embedding", x, [t, d])
    for index, block in enumerate(w["layers"]):
        prefix = f"layer.{index}."
        n = rms_norm(x, block["attention_norm"], c["norm_epsilon"])
        record(prefix + "attention_norm", n, [t, d])
        q = [rows(row, width) for row in linear(n, block["wq"])]
        k = [rows(row, width) for row in linear(n, block["wk"])]
        v = [rows(row, width) for row in linear(n, block["wv"])]
        record(prefix + "q", q, [t, h, width])
        record(prefix + "k", k, [t, kh, width])
        record(prefix + "v", v, [t, kh, width])
        q, k = rotate(q, c["rope_theta"]), rotate(k, c["rope_theta"])
        record(prefix + "rope_q", q, [t, h, width])
        record(prefix + "rope_k", k, [t, kh, width])
        attended, probabilities = attend(q, k, v, c)
        record(prefix + "attention_probs", probabilities, [h, t, t])
        record(prefix + "attention", attended, [t, d])
        x = residual(x, linear(attended, block["wo"]))
        record(prefix + "attention_residual", x, [t, d])
        n = rms_norm(x, block["ffn_norm"], c["norm_epsilon"])
        record(prefix + "ffn_norm", n, [t, d])
        gate, up = linear(n, block["w_gate"]), linear(n, block["w_up"])
        record(prefix + "gate", gate, [t, c["ffn_dim"]])
        record(prefix + "up", up, [t, c["ffn_dim"]])
        gated = [[g / (1 + math.exp(-g)) * u for g, u in zip(gs, us)] for gs, us in zip(gate, up)]
        record(prefix + "swiglu", gated, [t, c["ffn_dim"]])
        x = residual(x, linear(gated, block["w_down"]))
        record(prefix + "output", x, [t, d])
    hidden = rms_norm(x, w["final_norm"], c["norm_epsilon"])
    record("hidden", hidden, [t, d])
    output = w["token_embedding"] if c["tie_embeddings"] else w["output"]
    record("logits", linear(hidden, output), [t, c["vocab_size"]])
    return trace


def write_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    specifications = [
        ("tiny_gqa", dict(dim=16, ffn_dim=19, layers=2, heads=4, kv_heads=2,
                          vocab_size=13, max_context=8, norm_epsilon=f32(1e-5),
                          rope_theta=5000.0, tie_embeddings=True), [1, 4, 2, 4, 0]),
        ("tiny_mha", dict(dim=8, ffn_dim=11, layers=1, heads=4, kv_heads=4,
                          vocab_size=9, max_context=6, norm_epsilon=f32(3e-4),
                          rope_theta=256.0, tie_embeddings=False), [8, 0, 3, 1]),
        ("tiny_mqa", dict(dim=18, ffn_dim=17, layers=1, heads=3, kv_heads=1,
                          vocab_size=11, max_context=7, norm_epsilon=f32(1e-6),
                          rope_theta=10000.0, tie_embeddings=False), [3, 7, 3]),
    ]
    for name, config, tokens in specifications:
        weights = make_weights(name, config)
        model_hash = write_json(ROOT / f"{name}.model.json", {"config": config, "weights": weights})
        cases = []
        for case_name, ids in [("prefix", tokens), ("single", [tokens[0]])]:
            token_hash = hashlib.sha256(b"".join(struct.pack("<q", token) for token in ids)).hexdigest()
            cases.append({"name": case_name, "tokens": ids, "tokens_sha256": token_hash,
                          "trace": reference(config, weights, ids)})
        fixture = {
            "metadata": {
                "schema": 1, "model_sha256": model_hash,
                "generator_sha256": GENERATOR_SHA256, "python": platform.python_version(),
                "reference": "original Python stdlib scalar oracle; binary64 math.fsum and complex RoPE",
                "input_dtype": "float32", "reference_dtype": "float64",
                "architecture_reference": ARCHITECTURE,
                "atol": 2e-5, "rtol": 2e-5,
            },
            "cases": cases,
        }
        fixture_hash = write_json(ROOT / f"{name}.reference.json", fixture)
        print(f"{name}: model={model_hash} reference={fixture_hash}")


if __name__ == "__main__":
    main()
