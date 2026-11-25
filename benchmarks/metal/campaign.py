#!/usr/bin/env python3
"""Run and retain an alternating, serial Llama 3.2 1B CPU/Metal campaign."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CASES = ("short", "sustained", "27", "28", "29", "concurrent-prefill",
         "shared", "independent", "recovery")
MODULE = "github.com/itsmontoya/neuralnetwork"
REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
MANIFEST = "3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64"
REFERENCE = "3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361"
WEIGHTS = "68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a"


def output(*command):
    return subprocess.check_output(command, text=True).strip()


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_snapshot(model):
    manifest = Path("testdata/llama32_manifest.json")
    reference = Path("testdata/llama32_reference.json")
    require(digest(manifest) == MANIFEST, "manifest checksum mismatch")
    require(digest(reference) == REFERENCE, "reference checksum mismatch")
    fixture = json.loads(manifest.read_text())
    require(fixture["revision"] == REVISION and len(fixture["files"]) == 6, "manifest identity mismatch")
    require(fixture["files"]["model.safetensors"]["sha256"] == WEIGHTS, "checkpoint checksum mismatch")
    for name, props in fixture["files"].items():
        path = model / name
        require(path.is_file() and path.stat().st_size == props["size"], f"{path}: size or type mismatch")
        require(digest(path) == props["sha256"], f"{path}: SHA-256 mismatch")


def verify_module():
    selected = json.loads(output("go", "list", "-m", "-json", MODULE))
    require(selected["Version"] == "v0.12.0" and not selected.get("Replace"), "LL-06 module pin mismatch")
    release = json.loads(output("go", "mod", "download", "-json", f"{MODULE}@v0.12.0"))
    require(release["Sum"] == "h1:CsQyT1Cb59ZYgyTJamc5TihWmK6LfCYofQKfBXm7YWs=", "module checksum mismatch")
    require(release["GoModSum"] == "h1:SafbkuDehVErsuJAL5FHFsLhKJbeh/OI3ItvbmwDafw=", "go.mod checksum mismatch")
    require(release.get("Origin", {}).get("Hash") == "1638f4a627d3379c94860c0e59a6b2cf2c284479",
            "module tag origin mismatch")
    return release


def machine_state():
    # Standard macOS queries are advisory. They neither set state nor require root.
    state = {}
    for name, command in (
        ("thermal", ("/usr/bin/pmset", "-g", "therm")),
        ("power", ("/usr/bin/pmset", "-g", "ps")),
        ("swap", ("/usr/sbin/sysctl", "vm.swapusage")),
        ("pressure", ("/usr/bin/memory_pressure", "-Q")),
    ):
        try:
            record = subprocess.run(command, text=True, capture_output=True, check=False, timeout=5)
            state[name] = {"status": record.returncode, "stdout": record.stdout.strip(),
                           "stderr": record.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            state[name] = {"error": str(error)}
    return state


def run_sample(run, model, backend, case, label, profile=False, native=None, binary=None):
    prefix = run / "raw" / label
    require(not prefix.with_suffix(".json").exists(), f"refusing to overwrite {prefix}")
    environment = dict(os.environ, LLAMA_32_1B_MODEL_DIR=str(model),
                       LLAMA_PAIRED_CASE=case, LLAMA_PAIRED_BACKEND=backend,
                       LLAMA_PAIRED_OUTPUT=str(prefix), LLAMA_PAIRED_PROFILE=str(int(profile)))
    binary = binary or run / ("cpu" if backend == "cpu" else "metal")
    command = ["/usr/bin/time", "-l"]
    if native is not None:
        # SIP strips DYLD variables before launching the system timing binary.
        command.extend(("/usr/bin/env", f"DYLD_INSERT_LIBRARIES={native}"))
    command.extend((str(binary), "-test.run", "^TestLlama32PairedCampaign$",
                    "-test.count=1", "-test.timeout=10m"))
    # Launch time is immediately before spawning the macOS timing wrapper.
    # The parent records pipe arrival before parsing or writing each marker.
    begin = time.monotonic_ns()
    with (prefix.with_suffix(".stderr")).open("w") as errors:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, text=True, env=environment,
        ) as child, (prefix.with_suffix(".stdout")).open("w") as transcript:
            ready, first, sample = None, None, None
            for line in child.stdout:
                arrived = time.monotonic_ns() - begin
                if line.startswith("LLAMA_READY"):
                    ready = arrived
                if line.startswith("LLAMA_FIRST"):
                    first = arrived
                if line.startswith("LLAMA_RESULT "):
                    sample = json.loads(line.removeprefix("LLAMA_RESULT "))
                transcript.write(line)
            status = child.wait()
            process_wall = time.monotonic_ns() - begin
    require(status == 0 and ready is not None and first is not None and sample is not None,
            f"{label}: failed or incomplete; see {prefix}.stdout and .stderr")
    times = prefix.with_suffix(".stderr").read_text()
    rss = re.search(r"(\d+)\s+maximum resident set size", times)
    require(rss is not None, f"{label}: missing macOS peak RSS")
    sample.update(ready_pipe_ns=ready, first_pipe_ns=first, process_ns=process_wall,
                  peak_rss_bytes=int(rss[1]), label=label)
    prefix.with_suffix(".json").write_text(json.dumps(sample, indent=2) + "\n")
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="pinned six-file model snapshot")
    parser.add_argument("run", type=Path, help="new artifact directory")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--cooldown", type=float, default=3)
    parser.add_argument("--metal-backend", default="metal",
                        choices=("metal", "metal-bf16", "metal-bf16-kv16",
                                 "metal-fp16", "metal-fp16-kv16",
                                 "metal-q8_0", "metal-q8_0-kv16",
                                 "metal-q4_0", "metal-q4_0-kv16"))
    args = parser.parse_args()
    require(args.pairs > 0 and args.cooldown >= 0, "positive pairs and nonnegative cooldown required")
    require(Path("llama32_paired_test.go").is_file(), "run from the Llama repository root")
    run = args.run.resolve()
    require(run.is_relative_to((Path.cwd() / ".benchmarks").resolve()), "artifacts must be under .benchmarks")
    run.mkdir(parents=True, exist_ok=True)
    require(not (run / "plan.json").exists(), f"choose a new run directory: {run}")
    (run / "raw").mkdir()
    os.environ.update(GOWORK="off", CGO_ENABLED="1", GOMAXPROCS="8", GOARM64="v8.0", GOFLAGS="")
    for key in ("LLAMA_PAIRED_BACKEND", "GOGC", "GOMEMLIMIT", "GODEBUG", "DYLD_INSERT_LIBRARIES"):
        os.environ.pop(key, None)
    model = args.model.resolve()
    verify_snapshot(model)
    release = verify_module()
    subprocess.run(["go", "mod", "verify"], check=True)
    cases = args.case or CASES
    require(len(set(cases)) == len(cases), "workload names must be unique")
    metal_backend = args.metal_backend
    plan = {
        "cases": cases, "pairs": args.pairs, "cooldown_s": args.cooldown, "model": str(model),
        "metal_backend": metal_backend,
        "manifest_sha256": MANIFEST, "reference_sha256": REFERENCE, "checkpoint_sha256": WEIGHTS,
        "release": release, "commit": output("git", "rev-parse", "HEAD"),
        "git_status": output("git", "status", "--short"),
        "driver_sha256": digest(Path("llama32_paired_test.go")),
        "runner_sha256": digest(Path(__file__)),
        "go": output("go", "version"),
        "go_env": json.loads(output("go", "env", "-json", "GOOS", "GOARCH", "GOARM64", "CGO_ENABLED", "GOFLAGS", "GOWORK")),
        "gomaxprocs": os.environ["GOMAXPROCS"], "macos": output("sw_vers"),
        "kernel": output("uname", "-sm"), "sdk": output("xcrun", "--show-sdk-version"),
        "hardware": output("/usr/sbin/sysctl", "hw.model", "hw.memsize", "hw.logicalcpu",
                           "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"),
        "initial_state": machine_state(), "binaries": {},
    }
    for backend in ("cpu", "metal"):
        command = ["go", "test", "-c", "-trimpath", "-o", str(run / backend)]
        if backend == "metal":
            command.append("-tags=metal")
        subprocess.run([*command, "."], check=True)
        plan["binaries"][backend] = {"sha256": digest(run / backend),
                                     "build": output("go", "version", "-m", str(run / backend))}
    (run / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with (run / "order.jsonl").open("w", buffering=1) as log:
        for trial in range(1, args.pairs + 1):
            offset = (trial - 1) % len(cases)
            for case in (*cases[offset:], *cases[:offset]):
                order = ("cpu", metal_backend) if (trial + cases.index(case)) % 2 else (metal_backend, "cpu")
                log.write(json.dumps({"case": case, "pair": trial, "order": order,
                                      "state_before": machine_state()}) + "\n")
                for backend in order:
                    label = f"{case}-{trial:02}-{backend}"
                    print(label, flush=True)
                    sample = run_sample(run, model, backend, case, label)
                    require(sample["peak_rss_bytes"] < 16 * 2**30, f"{label}: peak exceeded 16 GiB campaign limit")
                    time.sleep(args.cooldown)
    (run / "final-state.json").write_text(json.dumps(machine_state(), indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
