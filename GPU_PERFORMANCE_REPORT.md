# Llama 3.2 1B FP32 CPU and Metal on Apple M3

In the 2026-11-24 campaign on this M3, Metal took 88.08 ms per warmed short decode call versus
68.01 ms on CPU. Single-session sustained throughput was 11.06 versus
14.76 tokens/s. Metal shortened warmed 27–29-token prefill by 25–29%
at the paired medians. Two sessions sharing one model favored CPU.
Two independent models showed a smaller median advantage for Metal
with overlapping paired ranges. Native timestamps put most single-row
time inside GPU command execution, with FFN projections the largest
sampled group.

## Source and fixtures

| Input | Verified identity |
| --- | --- |
| Llama inference source | LL-03 owner merge `d601bbce4afc5b3086a6aca58646856abf4c90df` on `main` |
| neuralnetwork module | `v0.11.2` at `8db36638be5b0166d676ae1322d89dd095823cb3` |
| Module checksum | `h1:4G/3ohNq5D95cX2JWdaIDeECcQOaeZn8EP9c+SgKcxE=` |
| Module file checksum | `h1:SafbkuDehVErsuJAL5FHFsLhKJbeh/OI3ItvbmwDafw=` |
| Snapshot | `meta-llama/Llama-3.2-1B` at `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| Manifest SHA-256 | `3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64` |
| Frozen reference SHA-256 | `3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361` |
| Safetensors SHA-256 | `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a` |
| TinyStories model SHA-256 | `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696` |
| TinyStories vocabulary SHA-256 | `037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312` |
| Mac | MacBook Air `Mac15,13`, M3 4P+4E CPU, 10-core GPU, Metal 4, 24 GiB unified memory |
| System and SDK | macOS 26.5.2 (25F84), Darwin arm64, Command Line Tools SDK 14.4 |
| Process configuration | Go 1.27.3, `GOARM64=v8.0`, `GOMAXPROCS=8`, `CGO_ENABLED=1`, `GOWORK=off`, empty `GOFLAGS` |

The runner verifies the released module origin, checksums and module cache.
It reads all six snapshot files and verifies each size and SHA-256 before
launching the samples. Both executables use the inference source above with
the same test driver. The CPU executable is untagged and links no Metal
framework. The resident executable uses `-tags=metal`, cgo, runtime Metal
shaders, and `LoadHFModelMetalAttention`. The prior LL-03 tested tree at
`aedb66aa2b5a20fea29ba797da204eb4b19747cc` matches the merge tree.
[LL-03 PR](https://github.com/itsmontoya/llama/pull/34)

## Measurement design

The short prompt is `The capital of France is`. Its six IDs are
`[128000 791 6864 315 9822 374]` and its eight frozen predictions are
`[12366 13 1102 374 279 1455 95551 3363]`. The model cap is 512
positions. Eight predictions use cache capacity 13, a single prefill
and seven decode calls. The sustained trajectory uses capacity 38,
32 calls at positions 6–37 and 33 predictions. Each concurrent sustained
run uses two lanes with distinct sessions and identical prompts. The
independent case loads two models serially, then starts both lanes together.
The concurrent prefill case starts two six-token prefills together on one
loaded model. The 27/28/29-token cases use prefixes of the checked-in
supplemental prompt and assert existing terminal host digests and winners.
Generation drives `Prefill` and `Decode` directly. The committed CLI
tests cover output bytes, stop IDs, and limits.

Six paired fresh processes per workload put each backend first three
times. The runner rotates the order of workloads across rounds, runs
one process at a time, and sleeps three seconds between processes.
Only the three concurrent workloads start two evaluation lanes. Each
process records the first inference after model loading, then repeats
the same workload with freshly allocated sessions and resident weights
and pipelines. The machine's system and shader caches can persist across
processes. The verification pass reads the checkpoint before timing and
can warm the filesystem. Each pair records AC, thermal advisories, swap,
and available memory before launch. The campaign applies a 16 GiB
per-process peak RSS bound with macOS `time -l`.

First-token time from launch is measured using the parent clock until it
receives the first greedy ID. It includes program startup, reference JSON
read, model load, tokenizer load and encoding, cache setup, and inference.
The startup marker is emitted at entry to the Go test driver. Inference
TTFT starts at the prefill barrier and ends at selection. In paired
concurrent summaries, prefill and TTFT use the slower lane. Model load
excludes tokenizer work. Decode call latency measures the API through
return of logits. Decode throughput includes greedy selection, from the
first prefill selection to the last decode selection. Aggregate decode
timing uses the earliest prefill selection and last decode selection
across lanes. Fairness is the slower lane's decode interval divided by
the faster lane's interval. Full process wall includes row validation,
artifact writes and memory and cleanup probes.

Short single-session timing. Each value summarizes six processes. The
change column uses the median and range of the six Metal/CPU ratios
minus one, expressed as a percentage.

| Metric | CPU median (range) | Metal median (range) | Paired change % (range) |
| --- | ---: | ---: | ---: |
| Process startup, ms | 16.49 (6.20–380.80) | 12.94 (6.41–343.75) | -26.0 (-65.1–1095.9) |
| Model load, s | 1.885 (1.853–3.061) | 1.860 (1.847–2.291) | -1.5 (-25.2–3.1) |
| Tokenizer load and encode, ms | 165.76 (158.58–177.01) | 168.24 (163.97–176.63) | 0.1 (-3.2–7.0) |
| First cache setup, ms | 0.013 (0.008–0.018) | 0.214 (0.196–0.231) | 1569.9 (973.7–2545.6) |
| First prefill, ms | 117.27 (108.63–663.36) | 132.51 (119.90–752.58) | 11.8 (-47.1–23.6) |
| First inference TTFT, ms | 117.38 (108.73–663.46) | 132.67 (120.05–752.69) | 11.8 (-47.1–23.6) |
| Launch to first token, s | 2.196 (2.136–4.290) | 2.200 (2.148–3.560) | -0.7 (-17.0–3.9) |
| First decode call, ms | 67.18 (65.64–89.90) | 87.09 (86.14–88.09) | 29.4 (-3.7–31.2) |
| First eight predictions, ms | 586.41 (573.72–1187.77) | 743.63 (740.99–1365.61) | 27.5 (5.5–29.4) |
| Warm prefill, ms | 103.72 (102.36–113.33) | 100.70 (98.66–101.40) | -4.0 (-10.7–-1.6) |
| Warm inference TTFT, ms | 103.83 (102.47–113.44) | 100.80 (98.75–101.50) | -4.0 (-10.7–-1.6) |
| Warm decode, ms/call | 68.01 (66.20–69.09) | 88.08 (86.90–90.31) | 30.4 (26.4–35.4) |
| Warm decode, tokens/s | 14.68 (14.45–15.08) | 11.34 (11.05–11.49) | -23.3 (-26.2–-20.9) |
| Warm eight predictions, ms | 580.56 (567.41–597.78) | 718.28 (707.88–734.80) | 24.5 (19.3–26.5) |
| Whole process, s | 3.300 (3.246–5.527) | 3.629 (3.553–4.998) | 8.7 (-9.6–12.1) |

Warmed inference through fresh sessions on the loaded models.

| Metric | CPU median (range) | Metal median (range) | Paired change % (range) |
| --- | ---: | ---: | ---: |
| Sustained decode, ms/call | 67.66 (67.11–69.95) | 90.23 (89.21–90.74) | 33.7 (28.5–35.1) |
| Sustained decode, tokens/s | 14.76 (14.27–14.88) | 11.06 (11.01–11.20) | -25.2 (-25.9–-22.1) |
| Sustained positions 6–13, ms/token | 67.73 (66.99–68.00) | 87.87 (86.49–89.19) | 30.0 (27.5–31.8) |
| Sustained positions 30–37, ms/token | 67.76 (67.03–70.88) | 92.38 (91.26–92.75) | 36.0 (30.1–38.4) |
| 27-token prefill, ms | 398.71 (384.44–550.88) | 299.88 (296.41–303.65) | -24.7 (-45.0–-21.2) |
| 28-token prefill, ms | 418.50 (411.52–486.07) | 297.70 (297.04–303.51) | -28.9 (-38.9–-26.3) |
| 29-token prefill, ms | 429.30 (427.12–629.40) | 316.56 (315.89–323.01) | -26.0 (-49.8–-24.7) |
| Concurrent six-token prefill, ms | 159.28 (156.03–195.04) | 152.62 (149.23–171.44) | -4.4 (-12.1–-1.6) |
| Concurrent six-token aggregate, predictions/s | 12.55 (10.25–12.81) | 13.10 (11.65–13.39) | 4.6 (1.6–13.7) |
| Concurrent prefill lane 1 TTFT, ms | 158.05 (150.02–195.15) | 151.30 (149.37–155.58) | -4.4 (-20.3–1.8) |
| Concurrent prefill lane 2 TTFT, ms | 154.32 (151.72–184.47) | 143.89 (137.94–171.69) | -7.5 (-10.5–0.0) |
| Concurrent prefill TTFT slower/faster | 1.053 (1.009–1.061) | 1.080 (1.027–1.104) | 2.7 (-1.6–7.0) |
| Shared-model decode aggregate, tokens/s | 17.54 (13.56–17.86) | 14.83 (14.78–15.01) | -14.9 (-17.2–10.3) |
| Shared-model lane 1, tokens/s | 8.80 (6.78–8.96) | 7.43 (7.39–7.53) | -15.1 (-17.5–10.6) |
| Shared-model lane 2, tokens/s | 8.80 (6.81–8.97) | 7.43 (7.41–7.51) | -15.2 (-17.3–9.9) |
| Shared-model slower/faster ratio | 1.003 (1.001–1.007) | 1.002 (1.000–1.003) | -0.1 (-0.4–0.2) |
| Independent models load, s | 6.195 (5.832–6.416) | 5.521 (4.567–5.916) | -10.6 (-28.8–-0.2) |
| Independent decode aggregate, tokens/s | 13.57 (9.77–16.55) | 14.59 (14.46–15.04) | 8.9 (-9.8–49.1) |
| Independent lane 1, tokens/s | 6.82 (4.91–8.29) | 7.30 (7.26–7.54) | 8.8 (-10.0–48.6) |
| Independent lane 2, tokens/s | 6.93 (4.91–8.33) | 7.30 (7.23–7.52) | 6.2 (-10.1–48.3) |
| Independent slower/faster ratio | 1.005 (1.001–1.041) | 1.003 (1.000–1.004) | -0.2 (-3.9–-0.0) |

Go allocations cover each warmed phase including cache setup and lane
bookkeeping. Post-GC RSS is sampled after closing the sessions and
dropping result rows. Close measurements follow model cleanup and GC.

| Metric | CPU median (range) | Metal median (range) | Paired change % (range) |
| --- | ---: | ---: | ---: |
| Short warm phase Go allocated, MiB | 110.54 (110.54–110.56) | 4.07 (4.07–4.07) | -96.3 (-96.3–-96.3) |
| Short warm phase Go allocation count | 39566 (39562–39618) | 2026 (2026–2027) | -94.9 (-94.9–-94.9) |
| Short peak RSS, GiB | 6.32 (5.69–6.37) | 6.60 (5.50–6.60) | 3.5 (-12.2–15.8) |
| Short retained RSS, GiB | 6.24 (5.69–6.37) | 6.60 (1.80–6.60) | 3.5 (-70.5–15.8) |
| Short retained Go heap, MiB | 4714.73 (4714.73–4714.74) | 0.34 (0.34–0.35) | -100.0 (-100.0–-100.0) |
| Short post-close RSS, GiB | 6.24 (5.69–6.37) | 6.04 (0.82–6.50) | 0.1 (-86.5–14.2) |
| Shared-model peak RSS, GiB | 6.39 (6.20–6.39) | 6.60 (6.01–6.60) | 3.3 (-5.9–6.5) |
| Independent models peak RSS, GiB | 11.28 (10.36–11.80) | 8.25 (5.79–11.00) | -24.6 (-48.2–-6.8) |
| Independent models post-close RSS, GiB | 9.77 (8.72–10.09) | 1.46 (0.91–2.72) | -84.7 (-90.6–-72.5) |

The warmed prefill gains at 27, 28, and 29 tokens occurred in all six
pairs per case. Short and sustained decode had higher Metal latency in
all six pairs. The first launch-to-token medians were close, and the
range includes slow process starts and first prefills on both backends.
The independent-model aggregate Metal lead was 8.9% at the paired median
with a range from a 9.8% loss to a 49.1% gain, and Metal won three of six
pairs. CPU won five of six shared-model aggregate comparisons. The
CPU sustained result is close to the earlier P8 baseline. The quoted
P6 15.81 tokens/s is a historical observation using a different
dependency. [Pass 11 report](https://github.com/itsmontoya/llama/blob/20e327a4de63412c4723f7f7b367f8ff8584961d/OPTIMIZATION_PASS_11.md)

Across all 54 pairs, AC power stayed connected. macOS recorded no
thermal or performance warning. The memory pressure query reported
51–82% free system memory and 8.39–16.13 GiB of existing or accrued
system swap in use. The per-process `time -l` swap count was zero in
all 108 runs. The maximum observed peak RSS was 11.80 GiB for CPU
and 11.00 GiB for Metal, both in the independent case. The first
prefill, launch, and independent case have wide ranges on this
workstation. No samples were removed. No physical temperature
or memory bandwidth rate was measured.

## Correctness and recovery

Each process checks greedy IDs at selection, checks the available eight
frozen observations for the short trajectory, and verifies every warm
row and concurrent lane against that backend's first output. Longer
prefills have existing terminal host digests and IDs but no frozen
Transformers rows for their exact prefixes. Each paired comparison
covers every complete returned row, all 128,256 values per row, at each
step, in both phases and all lanes. The verifier checks equality of
the repeated phases and lanes and computes distributions over one copy
per trajectory to avoid repeating identical values. The frozen bound
is `1e-3 + 1e-3*abs(reference)`. The complete paired bound is
`1e-4 + 1e-4*abs(CPU)`, the same mixed bound used by LL-03 for paired
rows. Any nonfinite value, exact winner mismatch, changed repeat or
fraction greater than one fails the corresponding check.

The recovery case rejects an out-of-range ID and then an already canceled
decode after successful prefill and one append. Both failures must
return no logits and preserve the cursor. Retrying the valid ID must
return the normal rows through the eight-prediction suffix. Dedicated
Metal tests also cancel after submission, force a classifier recording
error after layer appends are staged, check a nonfinite completed
classifier, retry with a different token and close during an active step.

All 108 processes and all paired comparisons passed. The raw comparisons
covered **286,267,392** values, including the exact repeats. The frozen
comparison covered 29,928 observations per backend across the processes,
lanes and phases.

| Comparison | p50 absolute | p95 absolute | Max absolute | Max tolerance fraction |
| --- | ---: | ---: | ---: | ---: |
| CPU vs frozen | 6.676e-6 | 2.289e-5 | 4.959e-5 | 0.03164 |
| Metal vs frozen | 7.629e-6 | 2.289e-5 | 4.768e-5 | 0.02076 |
| Full paired short rows | 7.153e-6 | 2.360e-5 | 8.202e-5 | 0.70488 |
| Full paired sustained rows | 1.132e-5 | 4.244e-5 | 1.214e-4 | 0.96079 |
| Full paired 27-token row | 1.717e-5 | 3.004e-5 | 6.008e-5 | 0.30093 |
| Full paired 28-token row | 7.629e-6 | 2.146e-5 | 5.722e-5 | 0.33227 |
| Full paired 29-token row | 2.345e-5 | 4.083e-5 | 6.914e-5 | 0.62007 |

The shared and independent sessions reproduced the same complete
sustained rows as the single session on each backend. The maximum
absolute paired error and tolerance fraction occurred in sustained
decode. At position 11, Metal selected frozen ID 95551 by 0.0433101654,
CPU selected it by 0.0433502197, and the frozen margin was 0.043340683.
In six paired recovery samples, the already-canceled warmed attempt
returned in about 0.96 µs on CPU and 0.79 µs on Metal. Both backends
returned the usual eight exact greedy IDs after retry and reproduced
all short-case rows. That cancellation check times rejection before
work. The dedicated tests cover draining a submitted command.

## Memory, synchronization and native profiling

The one-model Metal path requests 4,943,388,672 bytes of FP32 immutable
weights and RoPE coefficients through 148 uploads. Tied embeddings and
classifier use one allocation. Independent decoders each own that
allocation, for 9,886,777,344 requested bytes in two models. Metal KV
has two private K/V slots per layer. It uses 131,072 bytes per position
at the pinned 16 layers, eight KV heads and 64 values per head. KV
requests are 1,703,936 bytes for the short session, 4,980,736 for one
sustained session, 9,961,472 for either two-session sustained workload
and 1,572,864 for the concurrent prefill. Measured scratch requests
are 683,008 bytes for decode, 1,541,120 for six-token prefill and
5,450,752 for the 29-token prefill. The CPU caches have one host
K/V slot. Peak RSS, driver allocation size, and requested GPU bytes
account for different resources and can overlap on unified memory.

Each successful step encodes one command buffer, one wait and a single
513,024-byte logit download. The embedding uses a device gather and
multi-row cached prefill uses one extra final-row gather. No host
activations or weights upload during inference. Each of 16 layer blits
updates the private inactive KV slot, including the committed prefix.
For sustained positions 6–37 this copies 45,088,768 bytes of committed
prefix and 2,490,368 bytes of new history, including prefill, per lane.
These are logical bytes requested by the copy calls. They carry no
information about GPU traffic or cache behavior.
[Residency and byte derivations](METAL_FP32_HANDOFF.md#residency-and-publication)

The timestamp probe runs in separate processes. It layers instrumentation
onto public Metal calls in the released executable. Its reported buffers,
compute encoders, dispatches, and blits come from that real command stream.
The host overlay times Llama's Go API boundaries in separate builds.
All diagnostic logit artifacts match the corresponding primary-backend
artifacts byte for byte. [Probe and overlay source](benchmarks/metal/diagnose.py)

The M3 exposes Metal timestamp counters at compute stage boundaries.
Explicit dispatch-boundary sampling was unavailable. The released
runtime encodes one compute dispatch per encoder, so the stage samples
attribute time to one-dispatch encoders. A stage can include GPU execution
and adjacent scheduling costs. The probe calibrates CPU and GPU clocks
before and after each sampled command. The observed ratio was 1.000.
The SDK has no `xctrace` or offline `metal` command. Memory-controller
traffic and shader occupancy were unavailable.
[Apple stage sampling](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers),
[Metal command timestamps](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime),
[clock calibration](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time)

Three command-timestamp processes without stage sampling gave the
following medians and ranges. Native wait can include startup, queueing
and a host scheduling delay around the command interval. The warm
sustained figures cover 96 commands across three processes.

| Diagnostic command | GPU start to end, ms | Host native wait, ms |
| --- | ---: | ---: |
| First short prefill | 112.80 (112.13–114.22) | 133.02 (129.10–137.12) |
| Warm short prefill | 99.75 (99.60–99.95) | 100.48 (100.31–100.68) |
| Warm short decode | 87.88 (86.21–90.72) | 88.23 (86.55–91.06) |
| Warm sustained decode | 90.01 (87.12–93.80) | 90.33 (87.47–94.09) |
| Warm sustained decode with stage sampling | 92.70 (88.80–106.96) | 93.40 (89.54–107.55) |

In the short native diagnostics, Metal device creation took a median
35.54 ms (25.90–39.20) and command queue creation 0.558 ms.
The first library and 11 runtime pipelines took 1.106 ms
(1.041–1.158) summed at the Metal API. Three lazy libraries and eight
additional pipelines first used in prefill took 0.305 ms (0.299–0.335).
The operating system can reuse compiled assets across processes. These
API timings measure observed library and pipeline preparation on this
host. The first prefill's extra wall time cannot be assigned to
compilation alone.

| Llama host overlay, ms | First prefill or load stage | Warm short decode |
| --- | ---: | ---: |
| Open logical GPU device | 44.955 (35.060–200.499) | — |
| Allocate 148 model tensors | 2.924 (2.860–3.240) | — |
| Upload model and rotary tables | 292.623 (290.662–329.271) | — |
| Allocate scratch | 0.092 (0.087–0.120) | 0.042 (0.038–0.059) |
| Create scope | 0.076 (0.074–0.077) | 0.001 (0.001–0.004) |
| Encode scope | 1.306 (1.256–24.780) | 0.208 (0.191–0.236) |
| Submit scope | 0.030 (0.026–0.031) | 0.003 (0.002–0.008) |
| Wait on scope | 138.280 (131.281–142.339) | 88.265 (86.902–89.448) |
| Download logits | 0.046 (0.044–0.057) | 0.041 (0.035–0.044) |
| Validate finite logits | 0.056 (0.042–0.057) | 0.048 (0.040–0.054) |

Both profile modes saw 242 compute dispatches, 17 blit encoders for
decode and 18 for multi-row cached prefill per completed command.
Requested live buffer high water was 4,946,633,728 bytes in short
and 4,949,910,528 in the one-lane sustained run. These high water marks
include scratch, KV and model. The driver's `currentAllocatedSize`
read at those commands reached 4,947,460,096 and 4,950,605,824 bytes.
The native allocation snapshots are taken during command completion.
Post-close retention measurements use process RSS and Go heap.

Three sustained stage-sampling runs recorded the following per-token
stage sums across 96 warmed decode calls. Sampled commands had a
2.69 ms higher median GPU interval than command-timestamp-only probes.
Use the sampled distribution for attribution and the command-only
and paired results for latency.

| Compute group | Dispatches per token | Sampled GPU stage sum, ms |
| --- | ---: | ---: |
| FFN gate, up and down projections | 48 | 41.008 (39.826–45.865) |
| RMSNorm | 33 | 18.488 (18.013–20.046) |
| Attention Q, K, V and output projections | 64 | 13.567 (13.032–14.922) |
| Classifier projection | 1 | 12.072 (11.754–14.781) |
| Causal attention | 16 | 3.667 (1.197–6.144) |
| Residual adds | 32 | 0.215 (0.069–0.950) |
| RoPE | 32 | 0.096 (0.080–1.914) |
| SwiGLU | 16 | 0.045 (0.042–0.727) |

The three projection categories together occupied roughly 72% of the
sampled command time on the warmed sustained case. FFN gate and up
`[1,2048] × [8192,2048]ᵀ` summed to 26.135 ms (25.205–29.942)
and down `[1,8192] × [2048,8192]ᵀ` to 14.799 ms (14.352–16.085). Host submit and logit download
together took about 0.044 ms in the short warm overlay. The evidence
supports measuring single-row FFN projection efficiency first. It
does not establish memory-bandwidth saturation.

The shared-model and independent command-only diagnostics used one
fresh process each and peaked
at two simultaneous scopes on the same native queue. Their warmed decode
GPU command medians were 133.65 and 135.18 ms. The independent run
requested 9,899,821,056 live bytes at peak versus 4,956,432,384 when
sharing one model. The aggregate overlap of Metal's reported command intervals was
8.71 s for shared and 8.82 s for independent across both phases.
The reported intervals include scheduling within each command.
They leave the overlap of individual GPU kernels unresolved. The independent run's first prefill had waits of 434–929 ms,
with GPU intervals of 326–437 ms. The warmed waits were far lower.
Treat the first independent case as sensitive to workstation state.

The CPU inference profile sampled 19.08 CPU seconds over 4.73 seconds
elapsed. Flat samples were 38.5% pthread wait, 22.9% pthread signal,
and 36.6% across the two principal ARM64 right-transposed product
kernels. These samples include worker activity across cores and
describe the Go profile's attribution. The Metal inference Go
profile collected 70 ms over 6.21 seconds, including 40 ms of cgo
calls. Go profiles cannot attribute native GPU execution. The retained
CPU heap profile assigned about 4.60 GiB to tensor storage. Its
Metal counterpart sampled about 3.5 MiB in Go and excludes device
allocations. The profile sampling estimates and direct runtime heap
reads use different measurement methods.

## Validation and reproduction

Use the six verified files described in [testdata/README.md](testdata/README.md#llama-32-1b-transformers-reference).
Run the paired campaign and diagnostics from the repository root on an
Apple Silicon Mac with a Go 1.27.3 toolchain and Command Line Tools.
Use Python 3.11 or later. Allocate several GiB of temporary disk for
logits, compiled tests and profiles. Choose fresh artifact paths for
a new run.

```sh
run=.benchmarks/gpu-ll-04/reproduction
python3 -B benchmarks/metal/campaign.py "$model_dir" "$run"
python3 -B benchmarks/metal/summarize.py "$run" > "$run/summary.txt"
python3 -B benchmarks/metal/diagnose.py "$run" "$run/diagnostics"
python3 -B benchmarks/metal/summarize_profiles.py "$run/diagnostics" \
  > "$run/profile-summary.txt"
```

The campaign writes the run plan, executable hashes, build metadata,
sample order, machine state, complete timings, stdout, stderr, logit
rows, and summaries under the run directory. The original artifacts
are under `.benchmarks/gpu-ll-04/2026-11-24-m3/` in the isolated Llama
clone at `/private/tmp/gopherml-llama-ll01.SfRkX7Fy/llama`. Compact
per-run metrics and paired changes are checked in as CSV. `apple-m3-samples.csv` has one row per
process. `apple-m3-pairs.csv` gives Metal/CPU minus one as decimal
fractions for each metric, along with the full-row error statistics.
[Per-process samples](benchmarks/metal/data/apple-m3-samples.csv),
[per-pair changes and error distributions](benchmarks/metal/data/apple-m3-pairs.csv). The raw model
snapshot and compiled tests stay local.

Record the full test runs separately from the timed campaign. The CPU
and native Metal runs use both verified fixture sets. The portability
and race suites use TinyStories and the synthetic tests, with the 1B
fixture omitted to bound memory. Linux binaries are cross-built.

```sh
export GOWORK=off GOMAXPROCS=8 GOARM64=v8.0 CGO_ENABLED=1
export LLAMA_STORIES260K_MODEL="$stories_dir/stories260K.bin"
export LLAMA_STORIES260K_TOKENIZER="$stories_dir/tok512.bin"
unset LLAMA_32_1B_MODEL_DIR LLAMA_PAIRED_BACKEND

go mod verify
go fmt ./...
go vet ./...
go test -p 1 -count=1 -race -v ./...
CGO_ENABLED=0 go test -p 1 -count=1 ./...
CGO_ENABLED=0 go test -p 1 -count=1 -tags=purego ./...
CGO_ENABLED=0 go test -p 1 -count=1 -tags=metal ./...
CGO_ENABLED=0 go test -p 1 -count=1 -tags='metal purego' ./...
CGO_ENABLED=1 go test -p 1 -count=1 -tags='metal purego' ./...
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build ./...
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -tags=metal ./...
go vet -tags=metal ./...
go test -p 1 -count=1 -race -v -tags=metal ./...
go test -p 1 -race -tags=metal -run '^TestMetal(Attention|FFN)' -count=10 .
LLAMA_32_1B_MODEL_DIR="$model_dir" go test -p 1 -count=1 -timeout 20m -v ./...
LLAMA_32_1B_MODEL_DIR="$model_dir" go test -p 1 -count=1 -timeout 20m -v -tags=metal ./...
```

Every listed command passed. The native suite printed PASS for cached
and full-prefix Metal attention, paired synthetic MHA/GQA/MQA, the
27/28/29 prefill guards, 32-step sustained replay, shared and
independent decoders, and the synthetic failure and cancellation
tests. Its cached and uncached resident CLI emitted
`" Paris. It is the most populous city\n"`. The general suites skip
the opt-in campaign driver. It ran separately in 108 primary
processes and the diagnostic processes. Default and portability
builds exclude native Metal tests by design. The original test and
verification logs are in the run's `validation/` directory.

## Upstream handoff

See [Neuralnetwork optimization handoff](NEURALNETWORK_OPTIMIZATION_HANDOFF.md)
for the measured opportunity, tensor layouts, acceptance thresholds and
commands for a single follow-up test.
