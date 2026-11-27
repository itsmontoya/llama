#!/usr/bin/env python3
"""Collect separate Go profiles, Metal command times, and sampled compute stages."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from campaign import digest, require, run_sample, verify_module


def write_overlay(run):
    """Insert host timers at the explicit Llama API boundaries in a test build."""
    replacements = {}

    def instrument(name, replacements_for_file):
        source = Path(name).resolve()
        text = source.read_text().replace('import (', 'import (\n\t"time"', 1)
        for before, after in replacements_for_file:
            require(text.count(before) == 1, f"overlay anchor mismatch in {name}: {before}")
            text = text.replace(before, after)
        target = run / name
        target.write_text(text)
        subprocess.run(["gofmt", "-w", str(target)], check=True)
        replacements[str(source)] = str(target)

    instrument("hf_model.go", [
        ("var device *gpu.Device", "var phase = time.Now()\n\tvar device *gpu.Device"),
        ("var model = metalModel", 'fmt.Printf("LL04 HOST device_open 0 %d\\n", time.Since(phase).Nanoseconds())\n\tvar model = metalModel'),
    ])
    instrument("metal_model.go", [
        ("func (m *metalModel) upload(shape []int64, values []float32) (out *gpu.Tensor, err error) {\n\tvar weight *gpu.Tensor", "func (m *metalModel) upload(shape []int64, values []float32) (out *gpu.Tensor, err error) {\n\tvar phase = time.Now()\n\tvar weight *gpu.Tensor"),
        ("if err = weight.Upload(values)", '''fmt.Printf("LL04 HOST model_allocate %d %d\\n", len(values)*4, time.Since(phase).Nanoseconds())
	phase = time.Now()
	if err = weight.Upload(values)'''),
        ("if err = weight.Upload(values); err != nil {\n\t\treturn nil, errors.Join(err, weight.Close())\n\t}\n\tm.weights = append", '''if err = weight.Upload(values); err != nil {
		return nil, errors.Join(err, weight.Close())
	}
	fmt.Printf("LL04 HOST model_upload %d %d\\n", len(values)*4, time.Since(phase).Nanoseconds())
	m.weights = append'''),
    ])
    instrument("metal_attention.go", [
        ("if scratch, err = newMetalAttentionScratch", "var scratchPhase = time.Now()\n\tif scratch, err = newMetalAttentionScratch"),
        ("defer scratch.close()", '''var config = d.config
	var bytes = 4*(rows*(8*config.Dim + 3*config.KVHeads*(config.Dim/config.Heads) + 3*config.FFNDim) + logitRows*config.VocabSize)
	if scratch.last != nil {
		bytes += 4*config.Dim
	}
	fmt.Printf("LL04 HOST scratch_allocate %d %d\\n", bytes, time.Since(scratchPhase).Nanoseconds())
	defer scratch.close()'''),
        ("if err = scratch.logits.DownloadInto", "var phase = time.Now()\n\tif err = scratch.logits.DownloadInto"),
        ("return out, nil\n}\n\nfunc (d *Decoder) runMetalScope", '''fmt.Printf("LL04 HOST logits_download %d %d\\n", len(out)*4, time.Since(phase).Nanoseconds())
	return out, nil
}

func (d *Decoder) runMetalScope'''),
        ("var scope *gpu.Scope", "var phase = time.Now()\n\tvar scope *gpu.Scope"),
        ("defer func() {\n\t\t// Finish", '''fmt.Printf("LL04 HOST scope_create 0 %d\\n", time.Since(phase).Nanoseconds())
	phase = time.Now()
	defer func() {
		// Finish'''),
        ("if err = scope.Submit(ctx)", '''fmt.Printf("LL04 HOST scope_encode 0 %d\\n", time.Since(phase).Nanoseconds())
	phase = time.Now()
	if err = scope.Submit(ctx)'''),
        ("return scope.Wait(ctx)", '''fmt.Printf("LL04 HOST scope_submit 0 %d\\n", time.Since(phase).Nanoseconds())
	phase = time.Now()
	err = scope.Wait(ctx)
	fmt.Printf("LL04 HOST scope_wait 0 %d\\n", time.Since(phase).Nanoseconds())
	return err'''),
    ])
    instrument("metal_session.go", [
        ("if err = validateSessionLogits", "var phase = time.Now()\n\tif err = validateSessionLogits"),
        ("// A successful Wait made every append publishable.", '''fmt.Printf("LL04 HOST finite_validation 0 %d\\n", time.Since(phase).Nanoseconds())
	phase = time.Now()
	// A successful Wait made every append publishable.'''),
        ("state.position = next", 'state.position = next\n\tfmt.Printf("LL04 HOST kv_publish 0 %d\\n", time.Since(phase).Nanoseconds())'),
    ])
    overlay = run / "overlay.json"
    overlay.write_text(json.dumps({"Replace": replacements}, indent=2) + "\n")
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", type=Path, help="artifact directory of the completed paired campaign")
    parser.add_argument("output", type=Path, help="new diagnostic directory, under .benchmarks")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = args.measurement.resolve()
    run = args.output.resolve()
    require(run.is_relative_to((Path.cwd() / ".benchmarks").resolve()), "artifacts must be under .benchmarks")
    require((root / "final-state.json").is_file(), "complete the primary campaign before profiling")
    require(args.repeats > 0, "positive repeats required")
    plan = json.loads((root / "plan.json").read_text())
    model = Path(plan["model"])
    metal_backend = plan.get("metal_backend", "metal")
    cases = {"short", "sustained", "27", "28", "29", "shared", "independent"}
    require(cases.issubset(plan["cases"]), "profiling requires all targeted primary workloads")
    require(digest(Path("llama32_paired_test.go")) == plan["driver_sha256"], "benchmark driver differs")
    require(all(digest(root / backend) == data["sha256"] for backend, data in plan["binaries"].items()),
            "measurement binaries differ")
    run.mkdir(parents=True)
    (run / "raw").mkdir()
    os.environ.update(GOWORK="off", CGO_ENABLED="1", GOMAXPROCS="8", GOARM64="v8.0", GOFLAGS="")
    for key in ("LLAMA_METAL_SAMPLE", "DYLD_INSERT_LIBRARIES", "GOGC", "GOMEMLIMIT", "GODEBUG"):
        os.environ.pop(key, None)
    verify_module()
    library = run / "metal_probe.dylib"
    subprocess.run(["xcrun", "clang", "-O2", "-fobjc-arc", "-dynamiclib", "-Wall", "-Wextra", "-Werror",
                    "-framework", "Foundation", "-framework", "Metal",
                    str(Path(__file__).with_name("metal_probe.m")), "-o", str(library)], check=True)
    driver = None
    if metal_backend == "metal":
        driver = run / "host-timers"
        overlay = write_overlay(run)
        subprocess.run(["go", "test", "-c", "-trimpath", "-tags=metal", "-overlay", str(overlay), "-o", str(driver), "."], check=True)

    def measure(backend, case, label, binary=None, native=None, profile=False, counters=False):
        print(label, flush=True)
        if counters:
            os.environ["LLAMA_METAL_SAMPLE"] = "1"
        run_sample(run, model, backend, case, label, profile=profile, native=native,
                   binary=binary or root / ("cpu" if backend == "cpu" else "metal"))
        baseline = root / "raw" / f"{case}-01-{backend}.f32"
        measured = run / "raw" / f"{label}.f32"
        require(digest(measured) == digest(baseline), f"{label}: diagnostic changed output rows")
        os.environ.pop("LLAMA_METAL_SAMPLE", None)
        time.sleep(3)

    for backend in ("cpu", metal_backend):
        measure(backend, "sustained", f"pprof-{backend}", profile=True)
    for trial in range(1, args.repeats + 1):
        if driver is not None:
            measure(metal_backend, "short", f"host-{trial}", binary=driver)
        measure(metal_backend, "short", f"command-short-{trial}", native=library)
        measure(metal_backend, "sustained", f"command-{trial}", native=library)
        measure(metal_backend, "sustained", f"counters-{trial}", native=library, counters=True)
    for case in ("27", "28", "29", "shared", "independent"):
        measure(metal_backend, case, f"command-{case}", native=library)
    for case in ("29", "shared", "independent"):
        measure(metal_backend, case, f"counters-{case}", native=library, counters=True)
    for backend in ("cpu", metal_backend):
        prefix = run / "raw" / f"pprof-{backend}"
        for profile in ("cpu", "heap"):
            with (run / f"{backend}-{profile}-top.txt").open("w") as report:
                subprocess.run(["go", "tool", "pprof", "-top", str(root / ("cpu" if backend == "cpu" else "metal")),
                                str(prefix.with_suffix(f".{profile}.pprof"))], check=True, stdout=report)


if __name__ == "__main__":
    main()
