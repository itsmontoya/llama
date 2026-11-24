#!/usr/bin/env python3
"""Validate all matched rows, then print the per-workload and paired summaries."""

import argparse
import csv
import json
from pathlib import Path
import statistics as stats
import subprocess

from campaign import output, require

MS = 1e6
SEC = 1e9
GIB = 2**30
MIB = 2**20


def metric_map(sample):
    cold, warm = sample["phases"]
    result = {
        "startup_ms": sample["ready_pipe_ns"] / MS,
        "cold_ttft_s": sample["first_pipe_ns"] / SEC,
        "model_load_s": sample["model_load_ns"] / SEC,
        "first_model_s": sample["loads_ns"][0] / SEC,
        "tokenizer_ms": sample["tokenizer_ns"] / MS,
        "process_s": sample["process_ns"] / SEC,
        "peak_rss_gib": sample["peak_rss_bytes"] / GIB,
        "retained_rss_gib": sample["retained_rss_kib"] / MIB,
        "retained_heap_mib": sample["retained_heap_bytes"] / MIB,
        "closed_rss_gib": sample["closed_rss_kib"] / MIB,
        "closed_heap_mib": sample["closed_heap_bytes"] / MIB,
        "close_ms": sample["close_ns"] / MS,
    }
    if len(sample["loads_ns"]) == 2:
        result["second_model_s"] = sample["loads_ns"][1] / SEC
    for name, phase in (("first", cold), ("warm", warm)):
        lanes = phase["Lanes"]
        fields = {
            "cache_ms": phase["CacheNS"] / MS,
            "prefill_ms": max(lane["PrefillNS"] for lane in lanes) / MS,
            "ttft_ms": max(lane["FirstNS"] for lane in lanes) / MS,
            "wall_ms": phase["WallNS"] / MS,
            "alloc_mib": phase["AllocBytes"] / MIB,
            "allocs": phase["Allocations"],
            "predictions_s": SEC * sum(len(lane["Winners"]) for lane in lanes) / phase["WallNS"],
        }
        if len(lanes) == 2:
            ttft = [lane["FirstNS"] for lane in lanes]
            fields.update(lane1_ttft_ms=ttft[0] / MS, lane2_ttft_ms=ttft[1] / MS,
                          prefill_fairness=max(ttft) / min(ttft))
        if lanes[0]["DecodeNS"]:
            first = [lane["FirstNS"] for lane in lanes]
            ends = [lane["DecodeEndNS"][-1] for lane in lanes]
            counts = [len(lane["DecodeNS"]) for lane in lanes]
            lane_times = [end - start for start, end in zip(first, ends)]
            fields.update(
                decode_ms=stats.mean(step for lane in lanes for step in lane["DecodeNS"]) / MS,
                first_decode_ms=stats.mean(lane["DecodeNS"][0] for lane in lanes) / MS,
                decode_s=SEC * sum(counts) / (max(ends) - min(first)),
                lane1_s=SEC * counts[0] / lane_times[0],
            )
            if len(lanes) == 2:
                fields["lane2_s"] = SEC * counts[1] / lane_times[1]
                fields["fairness"] = max(lane_times) / min(lane_times)
            if counts[0] == 32:
                for index in range(4):
                    windows = [
                        lane["DecodeEndNS"][8 * index + 7] -
                        (lane["DecodeEndNS"][8 * index - 1] if index else lane["FirstNS"])
                        for lane in lanes
                    ]
                    fields[f"positions_{6+8*index}_{13+8*index}_ms"] = stats.mean(windows) / (8 * MS)
        if lanes[0]["RejectedNS"]:
            fields["invalid_ms"], fields["canceled_ms"] = [duration / MS for duration in lanes[0]["RejectedNS"]]
        result.update({f"{name}_{key}": value for key, value in fields.items()})
    return result


def span(data):
    return f"{stats.median(data):.5g} ({min(data):.5g}–{max(data):.5g})"


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def write_csv(path, records):
    fieldnames = list(dict.fromkeys(field for record in records for field in record))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def read_sample(work, pair, backend, run):
    label = f"{work}-{pair:02}-{backend}"
    prefix = run / "raw" / label
    data = json.loads(prefix.with_suffix(".json").read_text())
    require(data["case"] == work and data["backend"] == backend and data["label"] == label,
            f"{label}: invalid sample identity")
    return data, prefix.with_suffix(".f32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / "plan.json").read_text())
    order = {(event["case"], event["pair"]): event["order"] for event in
             map(json.loads, (run / "order.jsonl").read_text().splitlines())}
    compare = run / "compare_rows"
    source = Path(__file__).with_name("compare_rows.c")
    subprocess.run(["cc", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(compare)], check=True)
    records, paired_records = [], []
    frozen = {"cpu": [], "metal": []}
    fractions = {"cpu": [], "metal": []}
    margins = {"cpu": [], "metal": []}
    for work in plan["cases"]:
        samples = {"cpu": [], "metal": []}
        paired = []
        for pair in range(1, plan["pairs"] + 1):
            cpu, cpu_path = read_sample(work, pair, "cpu", run)
            metal, metal_path = read_sample(work, pair, "metal", run)
            lane_count = len(cpu["phases"][0]["Lanes"])
            row_count = len(cpu["phases"][0]["Lanes"][0]["Winners"])
            require(cpu["prompt_ids"] == metal["prompt_ids"] and cpu["capacity"] == metal["capacity"],
                    f"{work} pair {pair}: prompt or capacity mismatch")
            require(cpu["model_bytes"] == cpu["kv_bytes"] == 0 and metal["model_bytes"] > 0,
                    f"{work} pair {pair}: invalid CPU/Metal model residency")
            for left, right in zip(cpu["phases"], metal["phases"], strict=True):
                for a, b in zip(left["Lanes"], right["Lanes"], strict=True):
                    require(a["Winners"] == b["Winners"] and len(a["Winners"]) == row_count,
                            f"{work} pair {pair}: greedy mismatch")
            errors = json.loads(output(str(compare), str(cpu_path), str(metal_path), str(row_count), str(lane_count)))
            metrics = {}
            for sample in (cpu, metal):
                backend = sample["backend"]
                metrics[backend] = metric_map(sample)
                samples[backend].append(metrics[backend])
                records.append({"case": work, "pair": pair, "backend": backend,
                                "order": "-".join(order[work, pair]), **metrics[backend]})
                for phase in sample["phases"]:
                    frozen[backend].extend(phase["FrozenErrors"] or [])
                    fractions[backend].append(phase["MaxFrozenFrac"])
                    margins[backend].extend(phase["NearTieMargin"] or [])
            changes = {"case": work, "pair": pair, "order": "-".join(order[work, pair]), **errors}
            changes.update({name: metrics["metal"][name] / value - 1 for name, value in metrics["cpu"].items() if value})
            paired.append(changes)
            paired_records.append(changes)
        print(f"\n{work} ({len(paired)} pairs)")
        print("| metric | CPU median (range) | Metal median (range) | paired Metal/CPU − 1, % |")
        print("| --- | ---: | ---: | ---: |")
        for metric in samples["cpu"][0]:
            cpu_values = [row[metric] for row in samples["cpu"]]
            gpu_values = [row[metric] for row in samples["metal"]]
            pair_changes = [100 * row[metric] for row in paired if metric in row]
            print(f"| {metric} | {span(cpu_values)} | {span(gpu_values)} | {span(pair_changes) if pair_changes else '—'} |")
        print("FP32 unique trajectory error across pairs (median and range): " +
              ", ".join(f"{key} {span([row[key] for row in paired])}" for key in ("p50", "p95", "max", "rmse", "max_fraction")))
    print(f"\nPaired logits compared: {sum(row['values'] for row in paired_records):,}")
    for backend in ("cpu", "metal"):
        errors = frozen[backend]
        print(f"{backend} frozen n={len(errors):,}, p50={percentile(errors, .5):.9g}, p95={percentile(errors, .95):.9g}, "
              f"max={max(errors):.9g}, max fraction={max(fractions[backend]):.9g}, position 11 margin {span(margins[backend])}")
    write_csv(run / "samples.csv", records)
    write_csv(run / "pairs.csv", paired_records)
    failed = [f"{row['case']} pair {row['pair']}" for row in paired_records if row["violations"]]
    require(not failed, f"logit tolerance exceeded in {', '.join(failed)}")


if __name__ == "__main__":
    main()
