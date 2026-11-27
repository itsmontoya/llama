#!/usr/bin/env python3
"""Summarize native Metal, compute stage and Llama host-timer diagnostics."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as stats

from campaign import require
from summarize import MS, span


def category(name, rows, width, output, variant):
    if not name.startswith("nn_matmul"):
        return name
    if rows > 0 and width > 0 and output > 0:
        if output == 128256:
            return "classifier"
        if (width, output) in ((2048, 8192), (8192, 2048)):
            return "feed-forward projection"
        if (width, output) in ((2048, 2048), (2048, 512)):
            return "attention projection"
    return name


def read_trace(path):
    commands, stages, calibration, resources, startup = {}, defaultdict(list), {}, {}, defaultdict(list)
    support = None
    for line in path.read_text().splitlines():
        if not line.startswith("LL04 "):
            continue
        _, event, *fields = line.split()
        if event == "ERROR":
            raise RuntimeError(f"{path}: {line}")
        if event == "DEVICE":
            name, elapsed, stage, dispatch, counters, enabled, recommendation = fields[-7:]
            require(stage == "1" or enabled == "0", f"{path}: unsupported profiling configuration")
            support = (name, stage, dispatch, counters, enabled, recommendation)
            startup[event].append(int(elapsed))
        if event in ("LIBRARY", "PIPELINE", "QUEUE"):
            require(fields[-1] == "1", f"{path}: {event} failed")
            startup[event].append(int(fields[-2]))
        if event == "COMMAND":
            index, *data = map(int, fields)
            commands[index] = data
        if event == "RESOURCES":
            index, *data = map(int, fields)
            resources[index] = data
        if event == "CALIBRATION":
            index, cpu_start, cpu_end, gpu_start, gpu_end = map(int, fields)
            require(cpu_end > cpu_start and gpu_end > gpu_start, f"{path}: invalid counter calibration")
            calibration[index] = (cpu_end - cpu_start) / (gpu_end - gpu_start)
        if event == "ENCODER":
            index, encoder, name, rows, width, output, variant, start, end = fields
            start, end = int(start), int(end)
            require(0 < start <= end < 2**64 - 1, f"{path}: invalid stage timestamp, command {index}, encoder {encoder}")
            stages[int(index)].append((int(encoder), category(name, *map(int, (rows, width, output, variant))),
                                       end - start))
    require(commands and support, f"{path}: no native Metal events")
    return commands, stages, calibration, resources, startup, support


def summarize_command(path):
    sample = json.loads(path.with_suffix(".json").read_text())
    work = sample["case"]
    commands, stages, calibration, resources, startup, support = read_trace(path)
    baseline_cmds = sum(len(lane["Winners"]) for lane in sample["phases"][0]["Lanes"])
    require(set(commands) == set(range(2 * baseline_cmds)), f"{path}: command count mismatch")
    phase_metrics = defaultdict(list)
    sampled_stages = defaultdict(list)
    memory = []
    blits_set = {cmd[1] for cmd in commands.values()}
    max_blits = max(blits_set)
    for index, command in sorted(commands.items()):
        compute, blits, dispatches, submit, wait, start, end, status, bytes_allocated = command
        require(0 < compute <= dispatches and dispatches >= 242 and blits >= 0, f"{path}: command layout changed")
        require(end > start > 0 and status == 4, f"{path}: Metal command {index} failed or lacked timestamps")
        phase = "first" if index < baseline_cmds else "warm"
        if work in ("27", "28", "29"):
            phase += " prefill"
        elif len(blits_set) > 1:
            phase += " prefill" if blits == max_blits else " decode"
        else:
            phase += " decode"
        phase_metrics[phase].append(((end - start) / MS, wait / MS, submit / MS, start, end))
        memory.append(bytes_allocated)
        if support[4] == "1":
            require(len(stages[index]) == compute and index in calibration, f"{path}: missing stage samples")
            buckets = defaultdict(float)
            for encoder, name, delta in stages[index]:
                require(encoder < compute, f"{path}: invalid encoder index")
                buckets[name] += delta * calibration[index] / MS
            sampled_stages[phase].append(buckets)
    for phase, data in phase_metrics.items():
        print(f"  {phase}: n={len(data)}, GPU {span([row[0] for row in data])} ms, "
              f"native wait {span([row[1] for row in data])} ms, submit {span([row[2] for row in data])} ms")
    for phase, values in sampled_stages.items():
        keys = sorted(values[0])
        print(f"  {phase} stage sums (ms): " +
              ", ".join(f"{key}={span([entry[key] for entry in values])}" for key in keys))
    if calibration:
        print(f"  GPU timestamp calibration CPU/GPU ratio {span(list(calibration.values()))}")
    if resources:
        print(f"  NN requested live bytes {min(data[1] for data in resources.values())}–"
              f"{max(data[1] for data in resources.values())}, peak {max(data[3] for data in resources.values())}, "
              f"peak scopes {max(data[5] for data in resources.values())}")
    print(f"  Metal device currentAllocatedSize {min(memory)}–{max(memory)}")
    if work in ("shared", "independent"):
        starts = sorted((row[3], row[4]) for phase in phase_metrics.values() for row in phase)
        total, previous_end = 0, 0
        for start, end in starts:
            total += max(0, end - max(start, previous_end))
            previous_end = max(previous_end, end)
        summed = sum(end - start for start, end in starts)
        print(f"  Command intervals overlap by {(summed - total) / MS:.6g} ms summed across both phases.")
    print("  Startup: " + ", ".join(f"{key}: {len(values)} calls, {sum(values)/MS:.6g} ms"
                                   for key, values in startup.items()))
    print(f"  Device support {support}")


def summarize_host(path):
    stages = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.startswith("LL04 HOST "):
            _, _, name, byte_count, ns = line.split()
            stages[name].append((int(byte_count), int(ns) / MS))
    require(stages, f"{path}: no host stages")
    for name, data in stages.items():
        print(f"  {name}: {len(data)} calls, {sum(item[0] for item in data)} requested bytes, "
              f"{sum(item[1] for item in data):.6g} ms total, first {data[0][1]:.6g} ms, "
              f"warm decode {span([ms for _, ms in data[9:]]) if len(data) == 16 else '—'} ms")
    require(len(stages["scope_submit"]) == len(stages["scope_wait"]) == len(stages["logits_download"]) == 16,
            f"{path}: call counts differ")
    require(sum(bytes_ for bytes_, _ in stages["model_upload"]) == 4_943_388_672,
            f"{path}: fixture model upload accounting differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    args = parser.parse_args()
    for path in sorted((args.diagnostics / "raw").glob("*.stderr")):
        if path.stem.startswith(("command-", "counters-")):
            print(path.stem)
            summarize_command(path)
    for path in sorted((args.diagnostics / "raw").glob("host-*.stdout")):
        print(path.stem)
        summarize_host(path)


if __name__ == "__main__":
    main()
