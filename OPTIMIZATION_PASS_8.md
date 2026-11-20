# Llama Optimization Pass 8 results

`neuralnetwork v0.10.18` cut sustained decode latency by **13.7%** and increased two-session aggregate decode throughput by **13.2%** at the paired median. Each improved in all seven pairs. Cached eight-token generation improved by **8.8%** across all 12 pairs. The candidate passed the pinned correctness checks and the prefill guard limits.

## Baseline and ownership

The comparison used Llama `57ecb86` for both builds, based on the Pass 7 merge `3204dc9694c5d91111485d7597c9c11dadb06146`. The added two-session sustained benchmark is `d72b162`. The before build linked `neuralnetwork v0.10.17` at `cfda2464435c568c46b71b58fd5518514e72c72f` (`h1:cby0CGQgpK4KniedUtepX2TvGliln1R3EPmokwE0teM=`). The after build linked the published `v0.10.18` at `ce91677fa12ff521309764e4664a296f57a52427` (`h1:j1PoyjYzxYhlwaWOUp/KlSh4V2xu8Upt/9kyIcnK/AU=`). The published module was checksum verified and all five changed files matched the local tag byte for byte. Both linked `github.com/tphakala/simd v1.3.0` and `golang.org/x/sys v0.45.0`.

The [Pass 8 investigation](NEURALNETWORK_OPTIMIZATION_HANDOFF.md) found 89.4–91.9% of labeled sustained-decode CPU in the single-row projection loop. Projection calls accounted for 2.4608 s of a 2.5723 s decode loop. The FFN gate/up and down projections accounted for about 60% of the loop. The classifier accounted for about 18% and uses one much larger tied tensor. The upstream kernel lives behind Llama's `project` → `tensor.GemmInto` host path, so neuralnetwork owns the implementation.

Neuralnetwork `d8191c0ebbac80115c1257fcad3ca7480d895c04` computes eight outputs in two ARM64 NEON accumulators. The loop loads four consecutive K values from each of eight ordinary row-major weight rows, interleaves those values in registers, then applies four ascending-K fused multiply-adds to each lane. The Go adapter uses this tile for full eight-column blocks with `K >= 64` and `K % 4 == 0`. The portable Go loop handles irregular contractions, column tails, other architectures, `purego`, and race builds. The existing output-column sharding and tensor storage contracts apply, including when the same loop handles a one-row prefill remainder. Upstream reported a 15.5% paired reduction in real sustained decode latency and a 13.7% increase in concurrent aggregate throughput. [Upstream implementation and measurements](https://github.com/itsmontoya/neuralnetwork/pull/137)

## Measurement design

Both builds used Go 1.27.3 on Apple M3 with four performance cores, four efficiency cores, 24 GB RAM, AC power, and macOS 26.5.2 (25F84). The main series used Darwin ARM64, `GOARM64=v8.0`, `GOMAXPROCS=8`, `CGO_ENABLED=1`, default tags, empty `GOFLAGS`, and `GOWORK=off`. Each fresh process verified all six files from `meta-llama/Llama-3.2-1B` revision `4e20de362430cd3b72f300e6b0f18e50e7166e08` against `testdata/llama32_manifest.json` (SHA-256 `3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64`). The model file is 2,471,645,608 bytes with SHA-256 `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`. The frozen reference JSON SHA-256 is `3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361`.

Each process timed one operation with zero inference warm-ups. Build order alternated for each workload and all completed observations are included. For `The capital of France is`, the primary benchmark uses six prompt tokens including BOS, eight greedy output tokens, a context cap of 512, and a KV capacity of 13. TTFT covers prefill and first greedy selection. Short decode covers seven subsequent decode calls and selections. Sustained decode times positions 6–37 after prefill and the first selection. Both concurrent sessions share the decoder and hold independent mutable state. They start timed decode at one barrier and each performs 32 greedy steps. Per-session wall begins at that barrier, and aggregate wall ends when both sessions have completed. The fairness ratio is the faster session's rate divided by the slower session's rate.

The supplemental guards use the same 29-token prompt and its 27- and 28-token prefixes. The two-session prefill guard uses the six-token prompt. Whole-process wall and Darwin peak RSS include verification, initialization, and loading. The paired columns are medians of the differences and ratios within individual pairs. Positive changes in throughput are improvements, while lower latency and allocation measures are improvements. The timing series uses separate binaries and processes from the profiling diagnostics.

## Primary generation (12/12 pairs)

| Metric | v0.10.17 median (range) | v0.10.18 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill (s) | 0.1747 (0.1674–0.1785) | 0.1790 (0.1637–0.1973) | +0.0037 | +2.15% (-2.21% to +10.91%) | 4/12 |
| TTFT (s) | 0.1748 (0.1675–0.1786) | 0.1791 (0.1638–0.1974) | +0.0038 | +2.15% (-2.21% to +10.90%) | 4/12 |
| Decode (s/token) | 0.07944 (0.07857–0.08334) | 0.06985 (0.06889–0.07131) | -0.00996 | -12.49% (-15.96% to -10.19%) | 12/12 |
| Decode (tokens/s) | 12.590 (12.000–12.730) | 14.315 (14.020–14.520) | +1.785 | +14.26% (+11.32% to +19.00%) | 12/12 |
| Cached generation (s/op) | 0.7307 (0.7259–0.7584) | 0.6714 (0.6501–0.6952) | -0.0643 | -8.81% (-11.55% to -5.07%) | 12/12 |
| Output (tokens/s) | 10.948 (10.549–11.021) | 11.915 (11.508–12.306) | +1.060 | +9.66% (+5.34% to +13.06%) | 12/12 |
| Bytes/op | 131,324,588 (131,316,296–131,335,776) | 131,320,432 (131,310,352–131,332,648) | -1,980 | -0.00% (-0.01% to +0.01%) | 9/12 |
| Allocs/op | 39,902 (39,884–39,926) | 39,894 (39,873–39,913) | -8 | -0.02% (-0.08% to +0.03%) | 8/12 |
| Process wall (s) | 3.846 (3.803–4.063) | 3.802 (3.751–4.001) | -0.044 | -1.15% (-2.27% to +0.39%) | 10/12 |
| Peak RSS (GiB) | 6.374 (5.693–6.375) | 6.373 (6.373–6.374) | -0.001 | -0.01% (-0.03% to +11.95%) | 10/12 |
| Model load (s) | 1.854 (1.808–1.887) | 1.877 (1.807–1.930) | +0.015 | +0.84% (-1.17% to +2.28%) | 2/12 |
| Tokenizer setup (s) | 0.166 (0.158–0.170) | 0.168 (0.164–0.191) | +0.002 | +1.05% (-2.02% to +17.80%) | 3/12 |

## Sustained decode (7/7 pairs)

| Metric | v0.10.17 median (range) | v0.10.18 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elapsed (s/op) | 2.5759 (2.5453–2.7254) | 2.2236 (2.2182–2.2496) | -0.3533 | -13.72% (-18.58% to -12.26%) | 7/7 |
| Decode (s/token) | 0.08050 (0.07954–0.08517) | 0.06949 (0.06932–0.07030) | -0.01105 | -13.73% (-18.57% to -12.26%) | 7/7 |
| Decode (tokens/s) | 12.420 (11.740–12.570) | 14.390 (14.230–14.430) | +1.980 | +15.94% (+14.02% to +22.83%) | 7/7 |
| Bytes/op | 329,805,712 (329,783,016–329,827,840) | 329,804,704 (329,792,344–329,821,960) | +3,248 | +0.00% (-0.01% to +0.00%) | 3/7 |
| Allocs/op | 149,551 (149,500–149,597) | 149,549 (149,518–149,580) | +5 | +0.00% (-0.03% to +0.02%) | 3/7 |
| Process wall (s) | 5.899 (5.768–5.968) | 5.537 (5.452–5.673) | -0.310 | -5.26% (-8.34% to -3.83%) | 7/7 |
| Peak RSS (GiB) | 6.377 (6.365–6.378) | 6.376 (6.364–6.376) | -0.001 | -0.02% (-0.20% to +0.17%) | 6/7 |

## Two-session sustained decode (7/7 pairs)

| Metric | v0.10.17 median (range) | v0.10.18 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode wall (s/op) | 4.1910 (4.1770–4.3140) | 3.7030 (3.6730–3.8020) | -0.4880 | -11.64% (-12.91% to -11.19%) | 7/7 |
| Aggregate decode (tokens/s) | 15.270 (14.840–15.320) | 17.280 (16.840–17.420) | +2.010 | +13.16% (+12.64% to +14.81%) | 7/7 |
| Session 0 (tokens/s) | 7.646 (7.418–7.703) | 8.641 (8.418–8.712) | +1.001 | +13.07% (+11.93% to +14.76%) | 7/7 |
| Session 1 (tokens/s) | 7.635 (7.448–7.683) | 8.689 (8.484–8.754) | +1.057 | +13.84% (+13.09% to +14.81%) | 7/7 |
| Session 0 (s/token) | 0.13080 (0.12980–0.13480) | 0.11570 (0.11480–0.11880) | -0.01510 | -11.56% (-12.83% to -10.68%) | 7/7 |
| Session 1 (s/token) | 0.13100 (0.13020–0.13430) | 0.11510 (0.11420–0.11790) | -0.01600 | -12.21% (-12.89% to -11.60%) | 7/7 |
| Faster/slower rate ratio | 1.0040 (1.0031–1.0060) | 1.0051 (1.0009–1.0078) | +0.0003 | +0.02% (-0.31% to +0.21%) | 3/7 |
| Bytes/op | 659,514,720 (659,503,128–659,546,800) | 659,517,128 (659,509,288–659,533,432) | +4,872 | +0.00% (-0.01% to +0.00%) | 2/7 |
| Allocs/op | 298,908 (298,880–298,977) | 298,910 (298,895–298,945) | +4 | +0.00% (-0.03% to +0.02%) | 1/7 |
| Process wall (s) | 7.686 (7.627–7.989) | 7.168 (7.124–7.294) | -0.494 | -6.43% (-8.70% to -5.69%) | 7/7 |
| Peak RSS (GiB) | 6.379 (6.378–6.380) | 6.379 (6.379–6.380) | -0.000 | -0.00% (-0.01% to +0.01%) | 4/7 |

## 27-token prefill (5/5 pairs)

| Metric | v0.10.17 median (range) | v0.10.18 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elapsed (s/op) | 0.6888 (0.6709–0.6978) | 0.7066 (0.6791–0.7241) | +0.0183 | +2.65% (-0.73% to +4.16%) | 1/5 |
| Bytes/op | 261,780,712 (261,768,920–261,791,088) | 261,778,024 (261,768,800–261,787,992) | -11,688 | -0.00% (-0.01% to +0.01%) | 3/5 |
| Allocs/op | 7,077 (7,054–7,102) | 7,073 (7,058–7,094) | -19 | -0.27% (-0.44% to +0.57%) | 3/5 |
| Process wall (s) | 3.817 (3.739–3.846) | 3.818 (3.785–3.861) | +0.002 | +0.06% (-0.72% to +1.24%) | 1/5 |
| Peak RSS (GiB) | 6.373 (6.373–6.374) | 6.373 (6.372–6.375) | -0.000 | -0.00% (-0.01% to +0.04%) | 3/5 |

## 28-token prefill (5/5 pairs)

| Metric | v0.10.17 median (range) | v0.10.18 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elapsed (s/op) | 0.7391 (0.7364–0.7534) | 0.7488 (0.7209–0.7550) | -0.0022 | -0.29% (-2.46% to +2.27%) | 3/5 |
| Bytes/op | 271,356,880 (271,351,568–271,360,120) | 271,369,632 (271,353,232–271,379,008) | +12,016 | +0.00% (-0.00% to +0.01%) | 1/5 |
| Allocs/op | 7,066 (7,065–7,074) | 7,094 (7,062–7,118) | +20 | +0.28% (-0.06% to +0.66%) | 1/5 |
| Process wall (s) | 3.860 (3.805–3.869) | 3.868 (3.823–4.044) | -0.001 | -0.01% (-0.16% to +4.83%) | 3/5 |
| Peak RSS (GiB) | 6.373 (6.373–6.374) | 6.373 (6.373–6.374) | +0.000 | +0.00% (-0.01% to +0.00%) | 2/5 |

## 29-token prefill (5/5 pairs)

| Metric | v0.10.17 median (range) | v0.10.18 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elapsed (s/op) | 0.7517 (0.7425–0.7807) | 0.7670 (0.7560–0.8155) | +0.0120 | +1.60% (-0.16% to +8.50%) | 1/5 |
| Bytes/op | 282,508,968 (282,494,472–282,517,504) | 282,512,808 (282,504,328–282,513,000) | -2,960 | -0.00% (-0.00% to +0.01%) | 3/5 |
| Allocs/op | 7,063 (7,042–7,086) | 7,077 (7,066–7,082) | +3 | +0.04% (-0.07% to +0.55%) | 2/5 |
| Process wall (s) | 3.891 (3.824–3.930) | 3.893 (3.873–4.010) | +0.050 | +1.30% (+0.02% to +2.47%) | 0/5 |
| Peak RSS (GiB) | 6.372 (6.372–6.372) | 6.372 (5.689–6.373) | +0.000 | +0.00% (-10.73% to +0.01%) | 2/5 |

## Two-session six-token prefill (7/7 pairs)

| Metric | v0.10.17 median (range) | v0.10.18 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elapsed (s/op) | 0.2763 (0.2705–0.3009) | 0.2778 (0.2664–0.2944) | -0.0019 | -0.66% (-11.45% to +8.29%) | 4/7 |
| Bytes/op | 116,460,592 (116,445,480–116,470,016) | 116,462,336 (116,446,952–116,481,304) | +3,192 | +0.00% (-0.02% to +0.03%) | 3/7 |
| Allocs/op | 13,961 (13,937–13,988) | 13,964 (13,936–14,001) | +10 | +0.07% (-0.30% to +0.42%) | 3/7 |
| Process wall (s) | 3.416 (3.379–3.722) | 3.415 (3.386–3.685) | -0.010 | -0.31% (-0.99% to +0.49%) | 4/7 |
| Peak RSS (GiB) | 6.373 (6.372–6.373) | 6.373 (6.372–6.373) | -0.000 | -0.00% (-0.01% to +0.01%) | 4/7 |

## Timing interpretation

The primary benchmark's cached generation improvement came from faster decode. Primary TTFT moved from a per-build median of 174.8 ms to 179.1 ms, with a paired median increase of 2.2% and improvement in four of 12 pairs. TTFT changes ranged from −2.2% to +10.9%. The two-session prefill guard was similarly mixed. At 27 and 29 tokens, four of five pairs were slower, with paired medians of +2.7% and +1.6%. The 29-token range includes one +8.5% observation. The 28-token median was −0.3% with three improvements. All four prefill guard medians stayed within the investigation's 5% reference.

Both concurrent lanes improved in every pair. The candidate's median rates were 8.641 and 8.689 tokens/s, and the median faster/slower ratio moved from 1.0040 to 1.0051. The worst observed candidate imbalance was 0.78%. Allocation totals and peak RSS remained within ordinary run-to-run variation. Sustained whole-process wall improved by 5.3%, and concurrent whole-process wall improved by 6.4%, including fixture and model setup.

## Matched release profiles

Sampled CPU time sums across goroutines. The elapsed columns come from separate timers in the profiling harness. The concurrent profile covers the whole process, including model setup and prefill.

| Workload | Sampled CPU before → after | Projection loop CPU before → after | Condition signaling CPU before → after | Profiled work elapsed before → after |
| --- | ---: | ---: | ---: | ---: |
| Primary prefill ×6 | 6.86 → 1.64 s | 5.80 → 1.29 s, three-row | 0.11 → 0.04 s | 1.816 → 1.000 s |
| Short decode, 42 steps | 8.64 → 8.11 s | 7.67 → 7.21 s, single-row | 0.80 → 0.89 s | 3.491 → 2.945 s |
| Sustained decode, 64 steps | 13.16 → 12.84 s | 12.08 → 11.56 s, single-row | 1.02 → 1.26 s | 5.136 → 4.477 s |
| Two concurrent pairs of 32-step sessions | 51.68 → 46.18 s, whole process | 23.72 → 18.44 s, single-row | 5.42 → 4.67 s | 8.682 → 7.430 s |
| 27-token prefill ×3 | 5.38 → 3.96 s | 4.39 → 3.40 s, complete groups | 0.62 → 0.28 s | 2.306 → 2.154 s |
| 28-token prefill ×3 | 5.57 → 4.67 s | 3.86 → 3.50 s, complete groups | 0.59 → 0.40 s | 2.360 → 2.245 s |
| 29-token prefill ×3 | 5.73 → 5.79 s | 3.90 → 3.96 s, complete groups | 0.58 → 0.46 s | 2.337 → 2.342 s |

The sustained kernel samples moved to the NEON tile. The loop still accounted for 90.0% of labeled decode CPU, compared with 91.8% before. Gate/up sampled CPU moved from 6.40 to 8.04 s, down from 1.28 to 0.97 s, and the tied classifier from 4.02 to 2.38 s. Each shape's independent wall probes improved. Both profile wall measurements and aggregate CPU help interpret changes in worker activity. Greedy selection took 6.70 and 6.53 ms wall across 64 steps. The primary profile shows substantial prefill variation, so the alternating unprofiled pairs determine the TTFT result. At 29 tokens the two-row remainder took 0.90 and 0.97 sampled CPU seconds across three calls.

The primary allocation profiles attributed 9,757 and 9,759 MiB of cumulative sampled traffic to checkpoint loading, or 91.6% and 91.5% of their full processes. For 64 sustained decode steps, sampled allocations were 633.7 and 627.7 MiB, with 191.2 and 196.5 MiB attributed to SwiGLU. The 29-token prefill allocated 261.0 MiB in SwiGLU across three calls with either pin. These are sampled cumulative bytes. The timed allocation and peak RSS results measure different scopes and are recorded in the paired tables.

### Projection wall time

Separate timing probes captured every sequential `project` call at the public GEMM boundary. Each entry is the median and full range of three 32-step observations.

| Shape (M,K,N) | Calls | v0.10.17 elapsed s | v0.10.18 elapsed s |
| --- | ---: | ---: | ---: |
| Attention K/V (1,2048,512) | 1,024 | 0.1688 (0.1678–0.1855) | 0.1107 (0.1106–0.1115) |
| Attention Q/O (1,2048,2048) | 1,024 | 0.3042 (0.3025–0.3949) | 0.2555 (0.2553–0.2606) |
| FFN gate/up (1,2048,8192) | 1,024 | 1.0602 (1.0403–2.4322) | 0.9207 (0.9110–0.9351) |
| FFN down (1,8192,2048) | 512 | 0.4961 (0.4955–0.7316) | 0.4360 (0.4358–0.4467) |
| Tied classifier (1,2048,128256) | 32 | 0.4808 (0.4784–0.5411) | 0.4352 (0.4238–0.4372) |
| All projection calls | 3,616 | 2.5084 (2.4862–4.2852) | 2.1596 (2.1370–2.1890) |
| Full decode loop | 32 steps | 2.6202 (2.5975–4.5048) | 2.2715 (2.2503–2.3019) |

The before series includes one slow probe. Across the after probes, projection still took about 95% of decode wall. Block projection wall in the prefill probes moved from 0.5604 s to 0.5639 s at 27 tokens and from 0.6249 s to 0.6228 s at 29 tokens.

### Worker coordination and serial controls

One execution trace per build and workload observed 19,168 spawned helpers for single-session sustained decode and 38,336 across two sessions, with all helpers exiting. The counts agree with the existing output-column sharding. The trace measures time from creation to first running and time a caller spent blocked in `WaitGroup`. Join waits include the tail of useful computation, and summed waits across two callers can overlap in wall time.

| Trace metric | v0.10.17 | v0.10.18 |
| --- | ---: | ---: |
| Block helper creation to running, single-session median / p95 | 32 / 174 µs | 30 / 100 µs |
| Block helper creation to running, two-session median / p95 | 193 / 1,001 µs | 141 / 942 µs |
| Classifier helper creation to running, two-session median / p95 | 363 / 11,513 µs | 217 / 9,255 µs |
| Single-session caller join wait, summed | 0.854 s | 0.736 s |
| Two-session caller join waits, summed | 5.031 s | 3.901 s |

The public GEMM shape sweep used one process per pin and cap, 48 timed operations per case, and actual checkpoint tensors. The figures below are directional controls. The paired Llama benchmarks measure the release effect.

| GOMAXPROCS | Rotating gate/up ms, before → after | Rotating down ms, before → after | Repeated tied classifier ms, before → after |
| ---: | ---: | ---: | ---: |
| 1 | 4.804 → 2.772 | 4.509 → 2.385 | 71.415 → 41.712 |
| 2 | 2.471 → 1.621 | 2.302 → 1.351 | 36.679 → 23.373 |
| 4 | 1.328 → 1.144 | 1.241 → 1.041 | 19.941 → 16.085 |
| 8 | 1.036 → 0.936 | 0.926 → 0.843 | 15.110 → 12.920 |

The serial P=1 Llama controls preserved the same sustained suffix and the same sustained and 29-token prefill hashes. The single diagnostic sustained trial measured 2.972 and 4.241 tokens/s. The serial result exercises the existing one-worker route with the default build's eligible ARM64 tile.

## Correctness

Both Llama builds passed the default and `purego` fixture suites and `go vet`. The suites cover pinned Transformers logits and greedy winners, cached/full-prefix parity, frozen continuation IDs and byte-exact CLI output, TinyStories logits and tokenizer/generation, synthetic MHA/GQA/MQA traces, shared decoders, retained outputs, and failed-step retries. The largest Transformers absolute logit error was `4.95910645e-05` and the largest tolerance fraction was `0.0316354157` in all four fixture runs. Checked cached/full-prefix rows matched exactly. The primary benchmark validated its eight-token suffix and decoded bytes on each run. All 36 supplemental, sustained, and concurrent logs per pin matched their frozen checks.

The established race suite passed for both pins with TinyStories and local fixtures, including shared-decoder session tests. The full Meta fixture was disabled for those race runs. The assembly build constraint selects the portable kernel under `-race`. Targeted upstream GEMM and product tests passed in default, `purego`, and race builds. They cover tile and fallback boundaries, finite ordered reference values, NaN payloads, signed zero, dirty destinations, uneven worker partitions, serial/parallel equality, aliases, retries, and concurrent calls with shared weights. The ARM64 disassembly confirms the candidate uses `VFMLA` for the vector tile.

| Scope | SHA-256 of little-endian FP32 logits |
| --- | --- |
| 27-token prefill | `0b087b0839da36ec06a3a8f777d4c3c944ccc7439107fcad39b6ad0239ed9afc` |
| 28-token prefill | `cc93319c3c0701fd9cd3fdc2080afe43d2f82ba7cd236a698d19518836854e8c` |
| 29-token prefill | `6ffb4dc93003480aebdd51e9f1fee47ae3836eea9fb35ce2508f5d618a9871a5` |
| Sustained terminal row, both session counts | `b72d983e8b5c45f83ad858b52f1d27b87a7ad8118be6c5d315db2ea6fc6fd84a` |
| Two concurrent six-token prefills | `f2e070e41d1eac82f0159284cfdd0d070aa06ef703c833fa6fb781a5b7b487c1` |

## Ranked Pass 9 candidates

1. **Single-row projection data locality, neuralnetwork or cross-repository.** Projection still accounts for about 95% of measured decode wall. Across 32 steps, the candidate spends about 1.36 s in the two FFN shapes and 0.44 s in the tied classifier. Reducing effective weight traffic or improving locality could help every token after the arithmetic change. Measure cache misses, per-core execution stalls, and physical memory traffic where available. Compare rotating FFN tensors, the tied classifier in-model, and the footprint of any prepared storage. Preserve exact ascending-K FP32 arithmetic and tied-weight ownership.
2. **Projection worker coordination under two sessions, neuralnetwork.** The candidate still launches 38,336 helpers over 64 concurrent decode steps. Condition signaling used 4.67 sampled CPU seconds in the concurrent profile, and the traced median block helper delay was 141 µs. A lower coordination cost could improve aggregate work and tail latency. Compare bounded dispatch at the same process cap, measure useful work versus queue and join delay, and require both session rates and their fairness to hold.
3. **Complete-group prefill throughput, neuralnetwork.** Three-row groups account for 85.9% of labeled 27-token prefill CPU and 68.4% at 29 tokens. Candidate block wall is 0.564 s at 27 and 0.623 s at 29. More efficient reuse could help longer prompts. Sweep divisible and nondivisible lengths, separating complete groups, remainders, and the classifier while tracking worker utilization.
4. **Checkpoint staging, Llama or cross-repository.** Checkpoint loading contributes 91.5% of sampled primary-process allocation bytes and the paired candidate's median model load is about 1.88 s. Reducing load staging could improve startup and transient memory. Time conversion, finiteness checks, layout, and private copies separately. Measure loaded live heap and peak RSS before considering ownership changes.
5. **SwiGLU staging, Llama or cross-repository.** The candidate attributes 196.5 MiB to SwiGLU over 64 decode steps, including 94.8 MiB in sigmoid. The potential gain is reduced transient memory and GC work. Measure wall, collection frequency, retention, and peak RSS in sustained and shared-session inference before selecting it for latency work.

## Llama commits

- `d72b162` adds the two-session sustained decode benchmark.
- `57ecb86` records the Pass 8 investigation and upstream handoff.
- `ecf8948` pins the verified neuralnetwork release.
- The results commit adds this report and the README summary.

## Reproduction

Extract the release evidence archive from `/private/tmp/llama-pass8-release-evidence.LlUpSern.tar.gz`. Configure the local fixture paths and writable Go cache in its `env.sh`. Run from the Llama source root. The original investigation, including profiled worker-count and locality controls, is in `/private/tmp/llama-pass8-evidence.I9Q7GPj6.tar.gz`.

```sh
out=/path/to/extracted/llama-pass8-release.LlUpSern
. "$out/env.sh"
for variant in before after; do
    mkdir -p "$out/$variant"
    git show 57ecb86:go.mod > "$out/$variant/go.mod"
    git show 57ecb86:go.sum > "$out/$variant/go.sum"
done
go mod download -modfile="$out/after/go.mod" -json github.com/itsmontoya/neuralnetwork@v0.10.18
sh "$out/build.sh" "$out"
sh "$out/correctness.sh" "$out"
sh "$out/paired-later.sh" "$out"
sh "$out/profiles.sh" "$out"
sh "$out/analyze.sh" "$out"
```

The archive contains individual measurement and correctness logs, linked module metadata, paired summaries, raw and symbolized CPU and allocation profiles, execution traces, disassembly, and runnable diagnostic overlays. The raw timing series and the profiling diagnostics are separate. The hardware-counter limitation from the initial investigation also applies to this comparison.
