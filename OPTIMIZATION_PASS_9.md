# Llama Optimization Pass 9 investigation

## Baseline

The clean `optimization-pass-9` branch started at `b07f4ae100d0ada87a8a4485f294bdacb7fe137e`, the Pass 8 merge on local `main` and `origin/main`. The Pass 8 integration commit `ecf8948` is an ancestor of `main`. Its `go.mod` and `go.sum` match the baseline. The linked module is `github.com/itsmontoya/neuralnetwork v0.10.18` at tag commit `ce91677fa12ff521309764e4664a296f57a52427`. The module checksum is `h1:j1PoyjYzxYhlwaWOUp/KlSh4V2xu8Upt/9kyIcnK/AU=` and the module-file checksum is `h1:SafbkuDehVErsuJAL5FHFsLhKJbeh/OI3ItvbmwDafw=`. `go mod verify` passed, and the baseline binary records the exact published module.

Measurements used Go 1.27.3, default tags, `GOARM64=v8.0`, `GOMAXPROCS=8`, `CGO_ENABLED=1`, empty `GOFLAGS` and `GOWORK=off` on an AC-powered MacBook Air with Apple M3, four performance cores, four efficiency cores, 24 GB RAM and macOS 26.5.2 (25F84). The explicitly labeled processor-cap controls varied `GOMAXPROCS`. Each timed process verified the six pinned `meta-llama/Llama-3.2-1B` snapshot files before loading the checkpoint. The snapshot revision is `4e20de362430cd3b72f300e6b0f18e50e7166e08`. The fixture manifest SHA-256 is `3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64`. The frozen Transformers reference SHA-256 is `3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361`. The 2,471,645,608-byte checkpoint SHA-256 is `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`.

## Fresh timings

The Pass 8 harness used one timed operation per process, zero inference warm-ups and Darwin `wait4` for peak RSS. The primary prompt is `The capital of France is`, with six input tokens including BOS, eight greedy output tokens, context cap 512 and KV capacity 13. TTFT includes prefill and the first selection. Short decode comprises seven decode calls and selections. Sustained decode times 32 steps at positions 6–37 with setup, prefill, and first selection excluded. In the two-session workload, two independently prefilled sessions share one decoder and start 32 timed steps at a single barrier. Per-session rates include time from the common release. Fairness is the faster rate divided by the slower rate for each run. The prefill guards use the same 29-token prompt and its 27- and 28-token prefixes, plus two six-token sessions on one decoder.

All 48 trials are included. Figures are medians with full ranges. Process wall includes verification, checkpoint loading and tokenizer setup. In particular, the 2.203–4.159 s single-session sustained range and 0.165–1.434 s TTFT range make small changes difficult to interpret without tightly interleaved pairs. The series records the Pass 9 baseline only. Historical Pass 8 medians appear in the release report. [Pass 8 measurements](OPTIMIZATION_PASS_8.md)

| Primary, 12 processes | Median (min–max) |
| --- | ---: |
| Prefill (s) | 0.2592 (0.1650–1.4340) |
| TTFT (s) | 0.2594 (0.1651–1.4340) |
| Decode (s/token) | 0.0710 (0.0682–0.1471) |
| Decode (tokens/s) | 14.080 (6.799–14.660) |
| Cached generation (s/op) | 0.8327 (0.6426–2.3177) |
| Output (tokens/s) | 9.613 (3.452–12.449) |
| Bytes/op | 131,322,520 (131,313,712–131,333,176) |
| Allocs/op | 39,896 (39,882–39,913) |
| Process wall (s) | 4.0135 (3.6977–7.0989) |
| Process peak RSS (GiB) | 6.3049 (5.0553–6.3737) |
| Model load (s) | 1.8975 (1.8240–2.8310) |
| Tokenizer setup (s) | 0.1789 (0.1594–0.4350) |

| Sustained, seven processes each | Single session median (min–max) | Two sessions median (min–max) |
| --- | ---: | ---: |
| Decode wall (s) | 3.1376 (2.2027–4.1594) | 3.7680 (3.6900–4.3200) |
| Aggregate (tokens/s) | 10.200 (7.694–14.530) | 16.990 (14.820–17.340) |
| Session 0 (tokens/s) | 10.200 (7.694–14.530) | 8.525 (7.408–8.672) |
| Session 1 (tokens/s) | — | 8.493 (7.408–8.708) |
| Session 0 (s/token) | 0.0980 (0.0688–0.1300) | 0.1173 (0.1153–0.1350) |
| Session 1 (s/token) | — | 0.1178 (0.1148–0.1350) |
| Faster/slower rate ratio | — | 1.0038 (1.0000–1.0061) |
| Bytes/op | 329,821,024 (329,802,768–329,829,232) | 659,507,240 (659,498,104–659,535,304) |
| Allocs/op | 149,583 (149,546–149,601) | 298,891 (298,871–298,949) |
| Process wall (s) | 6.6481 (5.4491–9.0452) | 7.4625 (7.1192–8.5534) |
| Process peak RSS (GiB) | 6.3232 (5.5835–6.3760) | 6.3705 (5.9325–6.3795) |

| Prefill guard | Trials | Wall (s/op) | Bytes/op | Allocs/op | Process wall (s) | Peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 27 tokens | 5 | 0.8758 (0.7047–1.5712) | 261,776,576 (261,760,448–261,789,752) | 7,063 (7,043–7,095) | 4.0293 (3.7618–5.3291) | 6.3366 (6.2375–6.3749) |
| 28 tokens | 5 | 0.7295 (0.7210–1.7877) | 271,358,824 (271,350,800–271,377,920) | 7,068 (7,064–7,113) | 3.8592 (3.8513–5.0274) | 6.3732 (6.3382–6.3740) |
| 29 tokens | 5 | 0.7623 (0.7554–0.9521) | 282,504,736 (282,488,872–282,526,816) | 7,061 (7,028–7,099) | 3.8832 (3.8692–4.1651) | 6.3724 (5.6922–6.3728) |
| Two six-token sessions | 7 | 0.3263 (0.2703–1.2005) | 116,453,784 (116,441,384–116,468,600) | 13,962 (13,925–13,995) | 3.5674 (3.4090–5.3273) | 6.3697 (5.9774–6.4100) |

## Fresh CPU, allocation and phase wall probes

The separate diagnostic binaries label Llama projections without changing GEMM. CPU time sums sampled execution across threads. Percentages use labeled phase CPU, excluding samples with no phase label. The short, sustained and concurrent CPU runs also recorded 6.62, 6.90 and 16.22 sampled seconds with no phase label, respectively, including runtime samples. The timing column comes from the CPU diagnostic run. Allocation columns come from the separate 64 KiB sampled allocation profiles. Cumulative bytes represent allocation traffic. The load-only run measured retained live heap independently after a forced GC with the decoder held live.

| Profiled phase | Scope | Wall (s) | Labeled sampled CPU (s) | Projection CPU (s, % of phase) | Condition signaling CPU (s) | Allocation traffic (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Checkpoint load | One load | 1.965 | 1.70 | — | — | 9,757.8 |
| Primary prefill and first selection | 6 × 6 tokens | 1.899 | 6.58 | 6.48 (98.5%) | 0.21 | 334.0 |
| Short decode and selection | 42 steps | 3.139 | 7.74 | 7.62 (98.4%) | 1.32 | 413.0 |
| Sustained decode and selection | 64 steps | 4.575 | 12.98 | 12.90 (99.4%) | 1.84 | 633.7 |
| Two-session decode and selection | 128 steps, two 32-step pairs | 7.511 | 26.63 | 24.85 (93.3%) | 3.67 | 1,249.8 |
| 27-token prefill and selection | 3 iterations | 2.110 | 5.32 | 5.12 (96.2%) | 0.41 | 747.1 |
| 28-token prefill and selection | 3 iterations | 2.202 | 4.83 | 4.76 (98.6%) | 0.48 | 775.4 |
| 29-token prefill and selection | 3 iterations | 3.018 | 9.62 | 9.29 (96.6%) | 0.43 | 808.5 |

Projection CPU includes the kernel, dispatch and other labeled projection work. The single-row kernel itself accounted for 11.02 s or 84.9% of labeled sustained CPU, and 20.42 s or 76.7% of labeled concurrent decode CPU. Complete three-row loops accounted for 4.52 s (85.0%), 3.93 s (81.4%), and 7.97 s (82.8%) of labeled 27-, 28-, and 29-token prefill CPU. At 28 tokens the one-row remainder in the block projections accounted for 0.21 s and the classifier for 0.13 s of exclusive single-row kernel CPU. At 29 tokens the two-row block remainder accounted for 0.40 s, and the classifier for 0.34 s. The independent wall probes track prefill end to end and per block projection shape.

One primary allocation process attributed 9,758.8 MiB to checkpoint loading, or 91.5% of sampled process allocation traffic. The standalone load probe attributed 4,723.4 MiB to `ReadFloat32`, 4,714.2 MiB to `FromFloat32`, and 320.0 MiB to Q/K rotary row interleaving. Its roughly 4.60 GiB retained heap and 5.69–6.17 GiB process peak RSS measure different scopes. The load-only CPU profile attributed 1.05 s cumulatively to `ReadFloat32` including the reads, and 0.58 s to `weightTensor` including validation and ownership. Each call to `weightTensor` checks finiteness and makes a private tensor. Tied classification and embedding lookup share one tensor.

The sustained allocation profile sampled 194.3 MiB in SwiGLU, including 96.9 MiB in sigmoid, over 64 steps. Labeled SwiGLU sampled CPU was 0.05 s in sustained decode and 0.68 s in the concurrent profile. The sustained CPU probe observed one GC with 39 µs of stop-the-world time across 64 decode steps. The concurrent CPU probe recorded zero collections during its timed decode, with one during prefill. GC state depends on heap history. These runs provide limited evidence for a steady-state GC latency improvement. Greedy selection took 4.94 ms in the primary CPU probe across 48 selections, 6.79 ms in the sustained probe across 66, and 14.36 ms summed across concurrent lanes and setup across 132. No CPU sample carried the greedy label and the greedy loop allocates no output buffer.

Each projection wall value in the single-session column is the median of three independent 32-step probes with the full range. The column labeled shape CPU includes the single-row kernel and projection overhead from a separate profile of 64 single-session steps. The allocation sample uses the same 64-step scope and includes the Llama projection and GEMM, excluding the labeling wrapper. Concurrent values are from two matched diagnostic profiles covering 128 total steps. The concurrent projection wall values sum call durations across both lanes and can overlap.

| Projection (M,K,N) | Calls per 32 steps | Single wall (s) | Single shape CPU (s, % of phase) | Single allocations (MiB) | Concurrent summed wall (s) | Concurrent shape CPU (s, % of phase) | Concurrent allocations (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention K/V (1,2048,512) | 1,024 | 0.1206 (0.1194–0.1488) | 0.05 (0.4%) | 10.5 | 0.8234 | 0.52 (2.0%) | 18.3 |
| Attention Q/O (1,2048,2048) | 1,024 | 0.2879 (0.2701–0.4166) | 0.28 (2.2%) | 33.0 | 2.0702 | 2.57 (9.7%) | 65.3 |
| FFN gate/up (1,2048,8192) | 1,024 | 1.0179 (0.9904–1.6152) | 8.40 (64.7%) | 128.1 | 6.1850 | 10.37 (38.9%) | 257.7 |
| FFN down (1,8192,2048) | 512 | 0.4730 (0.4686–0.7795) | 1.31 (10.1%) | 17.3 | 2.9437 | 4.69 (17.6%) | 34.0 |
| Tied classifier (1,2048,128256) | 32 | 0.4463 (0.4264–0.5303) | 2.86 (22.0%) | 63.0 | 2.3114 | 6.70 (25.2%) | 125.6 |
| All projections | 3,616 | 2.3259 (2.2949–3.4903) | 12.90 (99.4%) | 252.0 | 14.3336 | 24.85 (93.3%) | 500.8 |

The projection probes' full 32-step median was 2.5037 s (2.4753–3.6446 s). Projection wall accounted for a median of 92.9% of elapsed time (92.7–95.8%). The FFN gate/up and down cumulative wall medians alone account for roughly 60% of the median loop. Allocation profiles attribute 126.2 MiB of 64-step projection allocations to host GEMM product staging and 3.8 MiB to one-row worker dispatch. The same categories measured 253.3 MiB and 7.1 MiB across 128 concurrent steps.

## Weight access and limiting factor

Each tensor is dense row-major `[N,K]`. The eight-output ARM64 tile points eight streams at consecutive rows. Neighboring streams start 8 KiB apart for K=2048 or 32 KiB apart for K=8192. Each contraction iteration loads 16 consecutive bytes from each row, then advances that row pointer by 16 B. This interleaves eight sequential streams. A single iteration touches distinct cache lines, and subsequent iterations consume the rest of the aligned lines. Unaligned row starts can cross line boundaries. The tile's full traversal visits all weight bytes once, with no duplicate weight loads within the projection. The K=2048 or K=8192 left vector is read again per eight outputs. The tied classifier makes an independent pass over its 1,050,673,152-byte weight matrix on every call. An embedding gather uses a row of the same matrix. Two sessions may revisit the same underlying addresses at different times. Their modeled scan rates can include shared cache traffic.

After one left-vector and eight weight-vector loads, the contraction body performs four `VDUP` and sixteen `VZIP1/2` instructions per four K values, then eight vector `VFMLA` instructions across two four-lane accumulators. Each lane computes one ascending-K output. The register permutation converts row-local values into the K-major vectors consumed by the FMAs. The two accumulator dependency chains and shuffle issue demand can limit core throughput even with hot weights. The dispatch shards contiguous output columns in units of four. All shards at P=8 for these shapes are eight-column aligned. Odd-size shards can reach the Go tail.

The logical weight payload for one token is 4,942,987,264 B (4.60 GiB). The nominal arithmetic intensity is 0.5 FLOP per weight byte when each FMA is counted as two operations and activation/output traffic is excluded. Dividing the logical bytes per 32-step trial by the median wall gives the following rates. These also account for projection dispatch time. Physical traffic and cache reuse can differ.

| Projection | Logical bytes per token | Effective logical reads (GB/s) |
| --- | ---: | ---: |
| Attention K/V | 134,217,728 | 35.6 |
| Attention Q/O | 536,870,912 | 59.7 |
| FFN gate/up | 2,147,483,648 | 67.5 |
| FFN down | 1,073,741,824 | 72.6 |
| Tied classifier | 1,050,673,152 | 75.3 |
| All projections | 4,942,987,264 | 68.0 |

The public GEMM control used actual loaded checkpoint tensors. Each case had 48 timed calls per process and three processes per cap. Repeat uses the same tensor. Rotate uses each layer's distinct weights in sequence. The classifier has a single tensor. Medians are in milliseconds per call.

| GOMAXPROCS | Gate/up repeat | Gate/up rotate | Down repeat | Down rotate | Classifier repeat |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.658 | 2.984 | 2.331 | 2.494 | 41.878 |
| 2 | 1.656 | 4.188 | 1.568 | 2.588 | 24.878 |
| 4 | 1.083 | 1.311 | 1.026 | 1.086 | 16.151 |
| 8 | 0.821 | 0.968 | 0.806 | 0.893 | 13.181 |

At P=1, rotating gate/up is 12% slower at the median, so extra helper launches cannot explain that gap. At P=4 it is 21% slower, with disjoint repeat and rotate ranges. The cold gate/up and classifier speedups from P=4 to P=8 were about 1.35× and 1.23×. The same-shape controls reduce the arithmetic and call-count confounds while leaving execution conditions such as frequency, cache residency and OS scheduling unobserved. One P=8 cold trial was especially slow. Its 9.580 ms gate/up result remains in the report's supporting data, and the volatile P=2 cold values have limited interpretive weight.

The contiguous streaming control scanned a 1 GiB byte buffer eight times using `bytes.Count`, with one to eight barrier-started workers. At P=8 it delivered 95.9 GB/s median logical source throughput (95.3–97.7). A copy control at P=8 delivered 83.1 GB/s by counting both reads and writes (82.8–83.1). These are software rates from independent controls. The 68.0 GB/s logical projection rate leaves a gap to the contiguous read control, and the rates alone cannot show that GEMM reaches the DRAM ceiling.

A separate two-caller experiment held (M,K,N)=(1,2048,8192), GEMM, dispatch, processor cap and operation count fixed. Each paired call started at a common barrier. The lanes used independent activation inputs and output tensors. Each lane rotated through the same 32 gate/up matrices. Changing only the offset made the lanes read the same or different matrices at each barrier. Each observation summed the slower completion across 32 barriers, with the case order alternated.

| GOMAXPROCS | Shared weight median (range), ms | Distinct weights median (range), ms | Median paired shared change (range) | Shared faster |
| --- | ---: | ---: | ---: | ---: |
| 4 | 51.171 (50.445–56.204) | 66.085 (66.001–68.285) | −22.55% (−23.67% to −17.69%) | 6/6 |
| 8 | 44.095 (43.162–52.962) | 52.699 (52.316–53.132) | −16.00% (−18.76% to +0.08%) | 5/6 |

This changes the weight working set without changing the arithmetic or worker policy. It supports a shared cache or memory-delivery effect. Synchronizing every projection is specific to the diagnostic. The real two-session benchmark synchronizes once at decode start.

The host had Command Line Tools and no usable `xctrace`. Direct hardware cache-refill, physical bandwidth, instruction throughput and stall counters were unavailable. The combined evidence supports a **mixed limit with a material weight-delivery component and secondary kernel and coordination costs**. Source inspection shows full utilization of the logical weight payload but an access geometry and transpose cost worth testing. The controls cannot distinguish cache level, memory latency, address-translation effects, physical bandwidth or the fraction of cycles exposed to the shuffle/FMA chains. A K-major tile addresses the access geometry and the register shuffle cost together. It still needs the same number of input weight bytes.

The P=8 traces observed 19,168 helpers in single-session sustained decode and 38,336 across two sessions. All exited. The concurrency trace ran 32 steps per lane in 3.712 s. The single-session trace took 2.607 s. The table reports Go scheduler states. The phase profile table above records sampled condition signaling.

| Worker observation | Single session | Two sessions |
| --- | ---: | ---: |
| Block helper creation to first running, median / p95 | 30.1 / 120.2 µs | 140.7 / 958.4 µs |
| Classifier helper creation to first running, median / p95 | 35.0 / 185.0 µs | 210.8 / 8,961.7 µs |
| Sum of helper time in Go Running state | 12.578 s | 22.610 s |
| Sum of helper time in Go Runnable state | 0.960 s | 12.215 s |
| Sum of blocked caller join time | 0.771 s | 3.871 s |
| Join time with runnable helpers and none in Go Running state | 0.010 ms | 1.829 ms |

The traces match spawned helpers to each caller and track their scheduler states while that caller waits. At least one corresponding helper was in the Go Running state for the rest of the observed blocked join time. Go Running includes useful arithmetic and may include synchronization and OS descheduling. Summed states across goroutines and summed concurrent joins can overlap. Creation-to-run is run-queue exposure. The total states and CPU samples do not identify the time a different dispatch implementation would actually save.

In the two-run-per-cap Llama control, single-session throughput medians at P=1/2/4/6/8 were 4.827, 8.526, 11.880, 12.212 and 12.790 tok/s. Concurrent aggregate medians were 4.147, 10.460, 11.470, 14.400 and 17.335 tok/s. The P=8 concurrent range was 17.330–17.340 tok/s. Lowering the cap changes both available Go parallelism and kernel shard count, and does not isolate the dispatch policy. P=1 exercised the serial default ARM64 kernel, with the same frozen suffix and digest.

## Candidate ranking

1. **Single-row projection layout. Cross-repository, with neuralnetwork owning the kernel and representation.** The FFN gate/up and down shapes take about 60% of the decode wall and stream 3 GiB of unique FP32 weights per token. Real checkpoint rotation, especially with one worker, isolates a locality penalty alongside substantial shuffle demand. The chosen optimization is an eight-output K-major FP32 weight tile for the shared immutable FFN weights. It addresses both access geometry and the in-register transpose in one change. A single authoritative representation is the preferred memory design. Preparation cost, multi-row compatibility, fallback and numerical exactness make this a higher-complexity experiment. The end-to-end gate applies before adoption.
2. **Projection coordination. Neuralnetwork.** Parallel calls generate substantial runnable delay and condition-signaling CPU. The trace separates these costs from active helper work. Coarse cap changes reduce concurrent throughput at P=4 and P=6 in the controls. A future bounded reuse experiment should keep process capacity fixed and test queueing, cleanup and fairness. The direct FFN locality effect extends to the serial case.
3. **Complete-group prefill. Neuralnetwork.** The three-row loop takes 81.4–85.0% of labeled long-prefill CPU. The same work matters to TTFT. An improvement needs divisible and remainder cases, while the current pass targets the projection shape active on every generated token. Complete groups remain a guard for any prepared weight representation.
4. **Checkpoint staging. Llama or cross-repository.** Loading is about 1.90 s and drives most sampled process allocation traffic. Reducing transient conversion or ownership copies could help startup and process RSS. The production finiteness check, private ownership and single tied embedding must be preserved. This candidate has limited sustained decode leverage.
5. **SwiGLU staging. Llama or cross-repository.** The allocation reduction could be substantial. Its CPU share and observed stop-the-world time offer less evidence for sustained latency headroom than projection. Test heap pressure, live retention and concurrent GC effects before selecting it in a later pass.

The upstream dependency owns the FP32 product and tensor storage contract. Llama would consume a prepared representation only after upstream validation. [Upstream implementation handoff](NEURALNETWORK_OPTIMIZATION_HANDOFF.md)

## Correctness and reproduction

Fixture-enabled default and `purego` suites passed with the pinned Meta and TinyStories fixtures. The checked cached/full-prefix rows matched bit for bit. The largest tested Transformers absolute logit error was `4.95910645e-05` and the largest tolerance fraction was `0.0316354157`. The established race suite passed with TinyStories and local fixtures, with the Meta fixture disabled. `go vet` passed. These suites exercised frozen greedy continuation and CLI bytes, MHA/GQA/MQA traces, independent concurrent sessions, retained outputs and failed-step retries. All 36 supplemental and sustained timing logs matched their frozen suffix or digest checks. Primary benchmarks checked the eight-token suffix and decoded text in all 12 processes.

Working evidence and standalone reproduction scripts are in `/private/tmp/llama-pass9-evidence.hqiIiZXs.tar.gz`. The archive holds the 48 raw baseline timing logs, the pinned build metadata, phase/profile/trace data, disassembly, controls, diagnostic source and summaries. From the Llama source root pinned to `b07f4ae`, extract the archive and set the local fixture paths in its `env.sh`. To keep the measured logs, run reproduction from a second extraction.

```sh
out=/path/to/extracted/pass9
. "$out/env.sh"
go version
go mod verify
sh "$out/baseline.sh" "$out"
sh "$out/investigate.sh" "$out"
sh "$out/analyze.sh" "$out"
sh "$out/correctness.sh" "$out"
```

The diagnostic overlays cover Llama benchmark/test code and projection instrumentation only. They compile against the released dependency without a module replacement. Archive metadata records the toolchain and source commit. The repository's only PR template is `PULL_REQUEST_TEMPLATE.md` at its root. The handoff precedes any implementation PR.
