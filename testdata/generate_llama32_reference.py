#!/usr/bin/env python3
"""Audit the pinned Meta snapshot and freeze a CPU Transformers reference."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess


ROOT = Path(__file__).resolve().parent
MODEL = "meta-llama/Llama-3.2-1B"
REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
LOCK = ROOT / "llama32-reference-requirements.lock"
MANIFEST = "llama32_manifest.json"
REFERENCE = "llama32_reference.json"
PYTHON = "3.12.12"
WEIGHTS = "model.safetensors"
WEIGHTS_SHA256 = "68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a"
# Sizes and Git blob IDs from the official, revision-specific Hub inventory.
SNAPSHOT = {
    "config.json": (843, "83b8b2aebb1a987b3802dae75fb9470234a3aaaf"),
    "generation_config.json": (185, "2d73a6863086ff9d491c28e49df9fb697cd92c2b"),
    WEIGHTS: (2471645608, None),
    "special_tokens_map.json": (301, "cfabacc2620186cd3dd4b1dde9a37e057208636e"),
    "tokenizer.json": (9085657, "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8"),
    "tokenizer_config.json": (50500, "cb9ec25536e44d86778b10509d3e5bdca459a5cf"),
}
PROMPT = "The capital of France is"
BUDGET = 8
TOKENIZER_CASES = {
    "empty": "",
    "leading_spaces": "  leading  and   repeated spaces ",
    "whitespace": "\tfirst\n\nsecond\r\n \tend\t\n",
    "contractions_and_digits": "I'm YOU'RE she'll wouldn't 1234567890 42",
    "unicode": "Café Cafe\u0301 北京 مرحبا 👩🏽\u200d🚀",
    "added_token_text": "one<|begin_of_text|>two<|end_of_text|>three<|eot_id|>",
}
ENCODE = {
    "add_special_tokens": True,
    "truncation": False,
    "padding": False,
    "return_attention_mask": False,
    "return_token_type_ids": False,
}
DECODE = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
SEED = 0
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    print(f"{path}: sha256={hashlib.sha256(data).hexdigest()}")


def verify_snapshot(model_dir):
    files = {}
    unexpected = sorted(path.name for path in model_dir.glob("*.safetensors*") if path.name != WEIGHTS)
    if unexpected:
        raise ValueError(f"unexpected safetensors files in {model_dir}: {', '.join(unexpected)}")
    for name, (size, blob_id) in SNAPSHOT.items():
        path = model_dir / name
        actual_size = path.stat().st_size
        if actual_size != size:
            raise ValueError(f"{path}: expected {size} bytes, found {actual_size}")
        if blob_id is None:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != WEIGHTS_SHA256:
                raise ValueError(f"{path}: unexpected SHA-256 {digest}")
        else:
            data = path.read_bytes()
            git_blob = hashlib.sha1(f"blob {size}\0".encode() + data).hexdigest()
            if git_blob != blob_id:
                raise ValueError(f"{path}: unexpected Git blob ID {git_blob}")
            digest = hashlib.sha256(data).hexdigest()
        files[name] = {"size": size, "sha256": digest}
    return files


def runtime_versions():
    if platform.python_implementation() != "CPython" or platform.python_version() != PYTHON:
        raise ValueError(f"use CPython {PYTHON} with {LOCK.name}")
    versions = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line.startswith((" ", "#")) or not line:
            continue
        name, expected = line.rstrip(" \\").split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"{name}: lock requires {expected}, found {actual}")
        versions[name] = actual
    return versions


def cpu_name():
    if platform.system() == "Darwin":
        info = subprocess.check_output(
            ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"], stderr=subprocess.DEVNULL
        )
        hardware = json.loads(info)["SPHardwareDataType"][0]
        return hardware.get("chip_type") or hardware["cpu_type"]
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        label, _, value = line.partition(":")
        if label.strip() == "model name":
            return value.strip()
    raise ValueError("unable to identify the CPU model")


def tensor_inventory(path):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as archive:
        tensors = []
        for name in sorted(archive.keys()):
            view = archive.get_slice(name)
            tensors.append({"name": name, "dtype": view.get_dtype(), "shape": view.get_shape(), "file": path.name})
        return {
            "metadata": archive.metadata(),
            "tensor_count": len(tensors),
            "parameter_count": sum(math.prod(tensor["shape"]) for tensor in tensors),
            "classifier_present": "lm_head.weight" in archive.keys(),
            "tensors": tensors,
        }


def byte_level_mapping():
    from tokenizers.pre_tokenizers import ByteLevel

    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    escaped = iter(range(256, 512))
    mapping = [chr(value if value in visible else next(escaped)) for value in range(256)]
    if set(mapping) != set(ByteLevel.alphabet()):
        raise ValueError("the effective ByteLevel alphabet differs from its byte mapping")
    return mapping


def tokenizer_inventory(model_dir, tokenizer):
    def components(backend):
        model = dict(backend["model"])
        model["vocab_count"] = len(model.pop("vocab"))
        model["merge_count"] = len(model.pop("merges"))
        return dict(backend, model=model)

    def special_text(token):
        if isinstance(token, list):
            return [special_text(entry) for entry in token]
        return token["content"] if isinstance(token, dict) else token

    raw = components(read_json(model_dir / "tokenizer.json"))
    effective = components(json.loads(tokenizer.backend_tokenizer.to_str()))
    if effective["model"]["type"] != "BPE" or effective["decoder"]["type"] != "ByteLevel":
        raise ValueError("expected a BPE model with a ByteLevel decoder")
    special_map = read_json(model_dir / "special_tokens_map.json")
    for name, token in special_map.items():
        if special_text(token) != tokenizer.special_tokens_map.get(name):
            raise ValueError(f"special_tokens_map.json entry disagrees with the effective tokenizer: {name}")
    settings = read_json(model_dir / "tokenizer_config.json")
    settings.pop("added_tokens_decoder", None)  # Effective metadata is in the backend inventory.
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "length": len(tokenizer),
        "config": settings,
        "backend": effective,
        "raw_backend_fields_changed_by_transformers": {
            key: value for key, value in raw.items() if value != effective.get(key)
        },
        "byte_to_unicode": byte_level_mapping(),
        "special_tokens_map": special_map,
        "effective_special_tokens_map": tokenizer.special_tokens_map,
        "special_ids": tokenizer.all_special_ids,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def encode(tokenizer, prompt):
    return tokenizer(prompt, **ENCODE)["input_ids"]


def decoded_hex(tokenizer, ids):
    return tokenizer.decode(ids, **DECODE).encode("utf-8").hex()


def generation_ids(model, tokenizer):
    bos = model.generation_config.bos_token_id
    stops = model.generation_config.eos_token_id
    if type(stops) is int:
        stops = [stops]
    if not stops or not all(type(token) is int and 0 <= token < model.config.vocab_size for token in [bos, *stops]):
        raise ValueError("invalid generation BOS or EOS IDs")
    if bos != tokenizer.bos_token_id or tokenizer.eos_token_id not in stops:
        raise ValueError("generation BOS/EOS disagree with the effective tokenizer")
    return {
        "source": "generation_config.json",
        "bos_token_id": bos,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "stop_token_ids": stops,
    }


def observe(torch, row, position, sampled_ids):
    if row.dtype != torch.float32 or not torch.isfinite(row).all().item():
        raise ValueError(f"expected finite FP32 logits at position {position}")
    # The input order is ascending token ID. Stable sorting breaks ties by that order.
    top = torch.argsort(row, descending=True, stable=True)[:8].tolist()
    return {
        "position": position,
        "argmax_id": top[0],
        "top_candidates": [{"token_id": token, "logit": row[token].item()} for token in top],
        "top_two_margin": row[top[0]].item() - row[top[1]].item(),
        "sampled_logits": [
            {"token_id": token, "logit": row[token].item()} for token in sorted(sampled_ids.union(top))
        ],
    }


def reference_outputs(torch, model, tokenizer, policy):
    vocab = model.config.vocab_size
    sample_ids = {i * (vocab - 1) // 31 for i in range(32)}
    if len(sample_ids) != 32:
        raise ValueError("expected 32 distinct vocabulary sample IDs")
    sample_ids.update([policy["bos_token_id"], *policy["stop_token_ids"]])

    def forward(ids):
        tokens = torch.tensor([ids], dtype=torch.long, device="cpu")
        logits = model(input_ids=tokens, use_cache=False).logits
        if tuple(logits.shape) != (1, len(ids), vocab):
            raise ValueError("unexpected full-prefix logits shape")
        return logits[0]

    bos = [policy["bos_token_id"]]
    prompt_ids = encode(tokenizer, PROMPT)
    if len(prompt_ids) < 3 or prompt_ids[0] != bos[0]:
        raise ValueError("completion prompt needs BOS and an interior token position")
    prefix_rows = forward(prompt_ids)
    logit_cases = [
        {"name": "bos_only", "input_text": "", "input_ids": bos,
         "observations": [observe(torch, forward(bos)[0], 0, sample_ids)]},
        {"name": "completion_prompt", "input_text": PROMPT, "input_ids": prompt_ids,
         "observations": [observe(torch, prefix_rows[pos], pos, sample_ids)
                          for pos in (0, len(prompt_ids) // 2, len(prompt_ids) - 1)]},
    ]
    suffix = []
    steps = []
    for step in range(BUDGET):
        ids = prompt_ids + suffix
        row = (prefix_rows if step == 0 else forward(ids))[-1]
        observation = observe(torch, row, len(ids) - 1, sample_ids)
        observation["input_ids"] = ids
        steps.append(observation)
        selected = observation["argmax_id"]
        if selected in policy["stop_token_ids"]:
            break
        suffix.append(selected)
    if not suffix:
        raise ValueError("the reference selected a stop token before generating a suffix")

    rotary = model.model.rotary_emb
    positions = [0, 1, 7, 31]
    cos, sin = rotary(torch.empty(0, dtype=torch.float32, device="cpu"), torch.tensor([positions], device="cpu"))
    width = rotary.inv_freq.numel()
    tokenizer_cases = []
    for name, prompt in TOKENIZER_CASES.items():
        ids = encode(tokenizer, prompt)
        tokenizer_cases.append({
            "name": name,
            "input_text": prompt,
            "prompt_ids": ids,
            "decoded_bytes_hex": decoded_hex(tokenizer, ids),
        })
    return {
        "tokenizer_cases": tokenizer_cases,
        "logit_cases": logit_cases,
        "generation": {
            "input_text": PROMPT,
            "prompt_ids": prompt_ids,
            "max_new_tokens": BUDGET,
            "suffix_ids": suffix,
            "decoded_bytes_hex": decoded_hex(tokenizer, suffix),
            "steps": steps,
        },
        "rope": {"positions": positions, "inv_freq": rotary.inv_freq.tolist(),
                 "cos": cos[0, :, :width].tolist(), "sin": sin[0, :, :width].tolist()},
    }


def generate(model_dir, output_dir, files, versions):
    os.environ.update(THREAD_ENV)
    import numpy
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(SEED)
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    local = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, **local)
    if not tokenizer.is_fast:
        raise ValueError("the reference requires the official fast tokenizer")
    tensors = tensor_inventory(model_dir / WEIGHTS)
    tokenizer_components = tokenizer_inventory(model_dir, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True, **local
    ).to("cpu").eval()
    if any(param.device.type != "cpu" or param.dtype != torch.float32 for param in model.parameters()):
        raise ValueError("the reference weights must reside on CPU in FP32")
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("tokenizer and checkpoint vocabulary sizes differ")
    policy = generation_ids(model, tokenizer)
    config = read_json(model_dir / "config.json")
    architecture_fields = (
        "model_type", "architectures", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "rms_norm_eps",
        "hidden_act", "attention_bias", "mlp_bias", "tie_word_embeddings", "rope_theta", "rope_scaling",
        "max_position_embeddings", "bos_token_id", "eos_token_id",
    )
    manifest = {
        "schema_version": 1,
        "model_id": MODEL,
        "revision": REVISION,
        "files": files,
        "config": config,
        "architecture": {field: getattr(model.config, field) for field in architecture_fields},
        "generation_config": read_json(model_dir / "generation_config.json"),
        "generation_policy": policy,
        "safetensors": tensors,
        "tokenizer": tokenizer_components,
    }
    with torch.inference_mode():
        outputs = reference_outputs(torch, model, tokenizer, policy)
    reference = {
        "schema_version": 1,
        "model_id": MODEL,
        "revision": REVISION,
        "runtime": {
            "python": platform.python_version(),
            "dependencies": versions,
            "dependency_lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "platform": platform.platform(),
            "cpu": cpu_name(),
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": model.config._attn_implementation,
            "seed": SEED,
            "thread_environment": THREAD_ENV,
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "inference_mode": True,
            "evaluation_mode": not model.training,
            "weights_format": "safetensors",
            "use_cache": False,
        },
        "encoding_options": ENCODE,
        "decoding_options": DECODE,
        "generation_policy": policy,
        **outputs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / MANIFEST, manifest)
    write_json(output_dir / REFERENCE, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--download", action="store_true", help="use the local authenticated Hugging Face session")
    parser.add_argument("--verify-only", action="store_true", help="check file sizes and hashes without loading the model")
    args = parser.parse_args()

    try:
        if args.download:
            from huggingface_hub import snapshot_download

            snapshot_download(MODEL, revision=REVISION, local_dir=args.model_dir, allow_patterns=list(SNAPSHOT),
                              endpoint="https://huggingface.co", token=True)
        files = verify_snapshot(args.model_dir)
        if args.verify_only:
            manifest_path = args.output_dir / MANIFEST
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                if manifest["revision"] != REVISION or manifest["model_id"] != MODEL or manifest["files"] != files:
                    raise ValueError(f"snapshot file inventory does not match {manifest_path}")
            print(json.dumps(files, indent=2, sort_keys=True))
        else:
            generate(args.model_dir, args.output_dir, files, runtime_versions())
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
