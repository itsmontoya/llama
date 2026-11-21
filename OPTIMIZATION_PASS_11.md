# Optimization Pass 11 investigation

## Baseline identity

Pass 10 merged into `main` in PR #29 at `bdc22d2f6d23512788f26143144bd7d157941215`. At the start of the investigation, the clean `optimization-pass-11` checkout, `main`, and the local `origin/main` tracking ref all pointed there. The first parent was the supplied validation source `b6497336d10ffc8181fce3072489788702009751`. The second parent was the Pass 10 branch tip `4a68ebed2bffb4ef2050e3d7c28afa6c9db0089c`. The first-parent diff contains the five expected files `decoder.go`, `go.mod`, `go.sum`, `tensor_ops.go`, and `weights.go`.

| Item | Verified value |
| --- | --- |
| Module | `github.com/itsmontoya/neuralnetwork v0.10.20` |
| Tagged upstream commit | `4f2860f264e2af392c0a27ced40f6df0435466a0` |
| Module checksum | `h1:kRDd694eMvtoWMoa6IP+qmhFrG2GgrtAP05sjOZXJLM=` |
| Module manifest checksum | `h1:SafbkuDehVErsuJAL5FHFsLhKJbeh/OI3ItvbmwDafw=` |
| Hardware | Apple M3 Mac15,13, four performance cores, four efficiency cores, 24 GB unified memory |
| Toolchain | Go 1.27.3, Darwin/ARM64, macOS 26.5.2 (25F84), `GOARM64=v8.0`, `CGO_ENABLED=1`, empty `GOFLAGS`, `GOWORK=off`, default benchmark build tags |
| Model | `meta-llama/Llama-3.2-1B` at `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| Fixture manifest SHA-256 | `3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64` |
| Frozen reference SHA-256 | `3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361` |
| Checkpoint SHA-256 | `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a` |

`go mod download -json` returned the tagged commit and both pinned sums for the cached published module. `go mod verify` passed with `GOPROXY=off`. The snapshot verifier passed all six model files. TinyStories checkpoint and tokenizer hashes also matched their frozen values. Git's direct remote connection was unavailable in the sandbox. Merge verification used the local remote-tracking ref and the merge object.

## Fresh benchmark baseline

Three fresh processes per workload and cap ran one timed operation each. Sustained, shared and independent measurements used 32 greedy decode steps per session at positions 6–37 and excluded setup and prefill. Concurrent decode used the same start barrier for both kinds of decoder ownership. The bench build used the merged runtime and the benchmark-only commit `563976e`. The primary measurement uses six prompt tokens, eight greedy output tokens, a context cap of 512, and KV capacity 13. Its TTFT starts immediately before prefill and includes selection of the first token.

Each entry reports the median [minimum–maximum]. The first primary invocation before the benchmark build is preserved as a pilot. The 42 scheduled benchmark processes include all valid observations from the measured series.

| P8 primary metric | Median [range] |
| --- | --- |
| Prefill, ms | 113.7 [107.6–125.5] |
| TTFT, ms | 113.8 [107.7–125.6] |
| Decode, ms/token | 66.13 [65.03–66.93] |
| Decode, tok/s | 15.12 [14.94–15.38] |
| Cached generation, s | 0.5767 [0.5629–0.5942] |
| Output, tok/s | 13.87 [13.46–14.21] |
| Allocated, bytes/op | 116,073,152 [116,067,584–116,075,088] |
| Allocations/op | 39,901 [39,900–39,903] |
| Process wall, s | 3.649 [3.615–3.733] |
| Model load, s | 1.850 [1.848–1.901] |
| Peak RSS, GiB | 6.3735 [6.3734–6.3736] |

| Decode | Cap | Aggregate tok/s | Session 0 tok/s | Session 1 tok/s | Faster/slower ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sustained | 4 | 12.50 [12.49–12.56] | 12.50 [12.49–12.56] | — | — |
| Sustained | 6 | 15.81 [15.76–15.82] | 15.81 [15.76–15.82] | — | — |
| Sustained | 8 | 15.11 [9.51–15.29] | 15.11 [9.51–15.29] | — | — |
| Shared | 4 | 13.33 [13.26–13.38] | 6.66 [6.66–6.71] | 6.69 [6.63–6.69] | 1.0036 [1.0033–1.0036] |
| Shared | 6 | 17.55 [17.39–17.56] | 8.78 [8.69–8.81] | 8.78 [8.72–8.80] | 1.0033 [1.0030–1.0043] |
| Shared | 8 | 14.45 [11.43–17.98] | 7.26 [5.81–9.08] | 7.23 [5.71–8.99] | 1.0097 [1.0044–1.0172] |
| Independent | 4 | 13.05 [12.94–13.20] | 6.52 [6.47–6.65] | 6.59 [6.51–6.60] | 1.0071 [1.0062–1.0107] |
| Independent | 6 | 10.12 [6.45–16.72] | 5.06 [3.23–8.36] | 5.07 [3.28–8.43] | 1.0080 [1.0022–1.0158] |
| Independent | 8 | 9.46 [7.84–18.12] | 4.73 [3.92–9.10] | 6.39 [4.82–9.06] | 1.0186 [1.0040–1.6296] |

Sustained latency was 79.97 [79.62–80.09], 63.25 [63.22–63.47], and 66.16 [65.41–105.20] ms/token at caps 4, 6, and 8 respectively. In the slowest sustained P8 run, the first eight-step window took 1.707 s and the next three windows took 0.541–0.563 s each. The remaining two sustained P8 runs took 0.522–0.533 s per window. The fastest independent P8 run reached 18.12 tok/s and the fastest shared run reached 17.98 tok/s. The independent P6 and P8 slow runs establish variability and a fairness risk. More matched observations are needed to estimate a throughput penalty attributable to model ownership.

| Decode | Cap | Process wall, s | Peak RSS, GiB | Timed MiB/op |
| --- | ---: | ---: | ---: | ---: |
| Sustained | 4 | 5.80 [5.78–5.80] | 6.374 [6.374–6.374] | 277.73 [277.72–277.73] |
| Sustained | 6 | 5.22 [5.21–5.26] | 6.375 [5.692–6.375] | 278.28 [278.27–278.28] |
| Sustained | 8 | 5.83 [5.30–8.66] | 6.375 [6.182–6.376] | 278.89 [278.87–278.91] |
| Shared | 4 | 8.14 [8.13–8.14] | 6.378 [6.378–6.378] | 555.39 [555.38–555.40] |
| Shared | 6 | 7.02 [6.94–7.04] | 6.378 [6.378–6.379] | 556.52 [556.50–556.53] |
| Shared | 8 | 7.74 [6.86–9.19] | 6.378 [6.351–6.379] | 557.65 [557.64–557.65] |
| Independent | 4 | 13.82 [10.04–13.91] | 11.358 [11.262–11.424] | 555.39 [555.39–555.40] |
| Independent | 6 | 15.19 [12.51–18.59] | 11.198 [11.136–11.199] | 556.54 [556.52–556.54] |
| Independent | 8 | 16.65 [12.38–17.08] | 11.221 [9.625–11.422] | 557.67 [557.67–557.70] |

Timed allocation counts increased from about 139k to 150k per sustained run and from 278k to 299k per two-session run as the cap increased from four to eight. At P6, both two-session cases used about 0.543 GiB of cumulative timed allocations, close to the accepted Pass 10 result. Each valid run matched the frozen supplemental digest. All 27 sustained/concurrent runs also matched the exact 33-token suffix, including the prefill winner.

| P8 prefill guard | Wall, ms | Allocated, MiB/op |
| --- | ---: | ---: |
| 27 tokens | 389.9 [388.5–409.7] | 219.29 [219.28–219.29] |
| 28 tokens | 415.3 [415.2–441.4] | 227.29 [227.28–227.29] |
| 29 tokens | 425.6 [424.5–426.0] | 236.81 [236.79–236.82] |
| Two 6-token sessions | 163.4 [161.9–164.0] | 97.58 [97.57–97.58] |

## Phase and projection profiles

CPU, allocation, and execution trace each ran in their own process. A test-only Go overlay measured Llama call boundaries, applied pprof labels, and added trace regions. The published backend ran unchanged. These runs establish attribution. The independent benchmark processes above supply the uninstrumented timings and allocation counts.

The table shows P8 results from the CPU-run wall probe. Gate/up and down are the three prepared FFN calls in each transformer block. K/V, Q/O, and classifier are dense. Concurrent projection times sum both caller lanes. A simultaneous second can appear twice in those columns, so FFN share uses the sum of both lane durations. Primary prefill's block projections have six rows, while its final classifier uses one row.

| Phase | Elapsed s | KV caller s | QO caller s | Gate/up caller s | Down caller s | Classifier caller s | SwiGLU caller s | FFN projection share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary prefill | 0.112 | 0.006 | 0.019 | 0.035 | 0.018 | 0.014 | 0.007 | 47% |
| Primary short decode | 0.466 | 0.024 | 0.055 | 0.180 | 0.090 | 0.092 | 0.009 | 58% |
| Sustained decode | 2.101 | 0.110 | 0.251 | 0.820 | 0.409 | 0.402 | 0.038 | 59% |
| Shared decode | 3.525 | 0.431 | 0.977 | 2.762 | 1.354 | 1.160 | 0.127 | 59% |
| Independent decode | 6.462 | 0.913 | 2.047 | 5.078 | 2.522 | 1.540 | 0.216 | 59% |

The CPU profiles expose kernel and runtime samples separately. Of 0.40 sampled seconds tagged primary prefill, 0.19 s were in the packed tile, 0.10 s in dense three-row arithmetic and 0.09 s in the dense one-row tile. During short, sustained, shared and independent decode, tagged packed tile samples were 0.21, 0.53, 4.69, and 12.10 s, and tagged dense one-row tile samples were 0.58, 2.48, 3.88, and 6.54 s respectively. The solo sustained profile sampled more dense than packed CPU despite packed FFN dominating its boundary wall and helper running time. Sample coverage differed across runs and goroutines. The tagged totals are sampled CPU attribution and do not measure the work done per elapsed second.

Annotated ARM64 disassembly places essentially all tagged packed decode samples in the existing one-row loop. In shared decode, 4.63 of 4.69 sample-seconds in that loop were at a vector load or lane broadcast instruction address. In independent decode the result was 11.97 of 12.10 s. The pattern supports testing a loop that shares left loads and broadcasts across tiles. Sample locations do not identify which instructions retired slowly or what caused stalls.

The load-only allocation run attributed 9.53 GiB of cumulative allocation to loading and retained 4.604 GiB after forced GC. The sampled load profile attributed about 4,724 MiB to `safetensors.(*File).ReadFloat32` and 3,072 MiB to `tensor.PrepareRightTranspose`. The former decodes temporary BF16 source tensors into FP32. Llama's weight store validates finiteness and copies each input into owning storage, after which the temporary source can be reclaimed. The 1.884 s load-only wall probe spent 1.107 s reading/converting, 0.430 s validating, 0.214 s preparing 48 FFN matrices and 0.108 s copying dense tensors. Two independent decoders retained 9.208 GiB after forced GC, consistent with two separate model payloads.

| P8 allocation-profile phase | Cumulative MiB | Allocation count | GC cycles |
| --- | ---: | ---: | ---: |
| Primary prefill | 48.86 | 7,843 | 0 |
| Short decode | 61.23 | 38,113 | 1 |
| Sustained decode | 279.73 | 173,821 | 1 |
| Shared decode | 559.31 | 347,347 | 0 |
| Independent decode | 559.35 | 347,425 | 0 |

The attribution probe itself allocates labels and regions. Sampled SwiGLU allocations represented about 97 MiB of roughly 273 MiB sampled under sustained decode. SwiGLU occupied 38 ms of 2.101 s in the wall probe. The allocation-profile GC pause during sustained decode was about 34 μs. Instantaneous RSS collection via `ps` was denied by the sandbox. Process peak RSS came from Darwin `wait4`.

## Scheduler and locality

The kernel dispatches new helpers for each projection and the caller executes the final range inline. The released path bounds the helpers per call, then joins synchronously. The traces tracked helper creation, first run, runnable and running intervals, caller block and wake, and lane overlap. The wake stack came through `WaitGroup.Done`/`Add` and the runtime semaphore. The tagged CPU profiles also sampled `runtime.pthread_cond_signal` in the Go scheduler wake path, at 0.42 s for short decode, 2.43 s sustained, 1.53 s shared and 1.26 s independent. These are sample weights and support the presence of wake work.

| P8 decode trace | Helpers | Summed helper running s | Summed helper runnable s | Caller join blocked s | Join with a queued helper and none running, ms | Caller wake-to-running, summed ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 4,193 | 2.252 | 0.158 | 0.148 | 0.004 | 0.885 |
| Sustained | 19,168 | 10.297 | 0.723 | 0.686 | 0.012 | 4.274 |
| Shared | 38,336 | 21.191 | 11.571 | 3.690 | 10.723 | 14.699 |
| Independent | 38,336 | 45.888 | 23.553 | 7.092 | 3.523 | 15.704 |

P8 traces counted 224 classifier helpers in the single-session sustained run and 448 in each two-session run. The other helpers belong to blocks. Gate/up creation-to-running medians were 30 μs sustained, 218 μs shared, and 226 μs independent. Gate/up helper running totals were 4.468 s sustained, 9.009 s shared, and 21.041 s in the slow independent trace. Shared lane overlap was 3.597 s out of a 3.636 s decode and independent overlap was 7.421 s out of 7.696 s. Most call joins overlapped at least one running helper, so summed join or queue time cannot be reclaimed wholesale. Go's running state also includes time during which the OS may have descheduled the thread. The 4/6/8 traces used the same helper counts for shared versus independent workloads at each cap (17,600, 27,968, and 38,336 respectively).

Four trials of direct released GEMM calls switched the order of caps and reuse/rotate tests. Rotation used the real immutable operands across blocks and a fixed left row and reused destination. At P8 the median repeated versus rotated times per projection were 0.096/0.112 ms for KV, 0.248/0.289 ms for QO, 0.741/0.790 ms for gate/up, and 0.732/0.800 ms for down. P1 rotated gate/up and down took 2.309 and 2.156 ms each. P6 took 0.826 and 0.770 ms. The tied classifier repeatedly traversed a single 1,002 MiB dense weight, taking 42.77, 23.16, 16.03, 13.04, and 13.45 ms at caps 1, 2, 4, 6, and 8. This control isolates projection calls and weight address patterns. It retains the released scheduler, and provides no counters for physical memory traffic.

## Limiting factor and decision

The bottleneck is mixed. The single-row packed FFN projection accounts for the largest measured caller-time share and remains costly even in the serial, repeated-weight control. Parallelism helps the kernels, with diminishing or variable returns between caps six and eight. Scheduler pressure is visible in concurrent runnable delays, while almost all caller join time overlaps running helpers. Independent decoder contention adds a memory-residency and scheduling variable. In the slow independent trace it also increased helper running time at the same dispatch count. Shared versus independent runs at P4 were close, and one P8 independent observation matched the fastest shared observation. The available machine counters could not distinguish instruction throughput, cache stalls and memory-bandwidth pressure. `xctrace` required full Xcode and `powermetrics` required superuser privileges.

| Rank | Candidate | Rationale and owner |
| ---: | --- | --- |
| 1 | Packed FFN single-row execution | About 58–59% of measured decode caller time across workloads. An M=1-only kernel specialization can use the existing storage and worker ranges, with a numerical gate and no startup cost. neuralnetwork owns the kernel and dispatch. |
| 2 | Projection worker coordination | Stronger queue delays and wake samples with two callers support deeper work. Useful work overlaps most joins. Replacing dispatch first would add lifecycle complexity and has uncertain solo and fairness benefits. neuralnetwork. |
| 3 | Tied classifier | About 19% of solo decode wall. The wide dense path has measurable scaling limits, while repacking would introduce gather, ownership, residency and startup risks. neuralnetwork or cross-repository. |
| 4 | Independent decoder contention | High variability and roughly double the model heap. The slowdowns have multiple plausible causes and the fastest P8 run is competitive with shared inference. Cross-repository investigation. |
| 5 | Checkpoint staging | Reading and materializing owned FP32 weights dominates startup traffic. Its potential benefit is principally load time and peak memory. Llama or cross-repository. |
| 6 | SwiGLU staging | Meaningful temporary bytes with about 2% of sustained caller wall and little observed GC pause. Llama or cross-repository. |

Select one upstream optimization, an ARM64 `M=1` execution specialization for the existing prepared FFN tile format. Test issuing two adjacent complete eight-column tiles together per range to share left loads and interleave four independent vector accumulators while preserving each output's ascending contraction order. Use the existing tile paths for clipped, odd, dense-tail and multi-row cases. The existing range ownership and dispatch are part of the control. A 10% reduction in the measured FFN projection share would correspond to roughly 6% lower sustained wall if the rest of the workload holds. The repeatable real-workload gain is an acceptance test for the upstream experiment.

## Baseline correctness

The fixture-enabled default, purego and race suites passed, as did `go vet ./...`. The Llama tests cover frozen Transformers logits, exact cached/full-prefix parity, frozen greedy continuation, byte-exact CLI output, TinyStories, synthetic MHA/GQA/MQA traces, retained results, failure recovery, retries and shared-session concurrency. The largest observed Transformers absolute logit error was `4.95910645e-05` and the largest tolerance fraction was `0.0316354157`. All supplemental baseline logs and both lanes from each concurrent profile matched the frozen suffix or digest for their workload.

The released kernel's direct packed range check passed separately at each cap from 1 to 8. It checks aligned and clipped boundaries, irregular widths, dense tails, one-/two-/three-row arithmetic, untouched outputs outside the requested range and no allocations in the partial-tile path. The worker-cap and prepared GEMM suites also cover the full Llama FFN shapes, ownership and concurrent retry. The generic suites under purego and race passed, with the ARM64-only dispatch cases skipped by their existing build guards. The real-worker benchmark matrix showed steady decode allocation volume at P4/P6/P8 and exact outputs in the shared and independent cases.

## Reproduction

The raw evidence and scripts are in `/private/tmp/llama-pass11.NfWYJktg`. The compressed copy is `/private/tmp/llama-pass11-evidence.NfWYJktg.tar.gz` with SHA-256 `88c8bc9d67d105dea45b923a6759ff2827376a925ae38db319019ba0f50728e8`. The evidence includes all per-process logs, bootstrap and module checks, probes, CPU and allocation pprof files, Go traces, parsed scheduler and synchronization summaries, frozen digest checks and the original binaries for symbolization. Use the benchmark-only revision `563976e` or this investigation revision for the commands below. If using a different machine, set the fixture and writable Go cache paths in the evidence environment file. Use the same model hashes and run the series sequentially.

```sh
artifact=/private/tmp/llama-pass11.NfWYJktg
. "$artifact/env.sh"

git show --stat bdc22d2f6d23512788f26143144bd7d157941215
go mod download -json github.com/itsmontoya/neuralnetwork@v0.10.20
go mod verify
python3 testdata/generate_llama32_reference.py --model-dir "$LLAMA_32_1B_MODEL_DIR" --verify-only
shasum -a 256 testdata/llama32_manifest.json testdata/llama32_reference.json "$LLAMA_32_1B_MODEL_DIR/model.safetensors"

sh "$artifact/benchmark.sh" "$artifact"
sh "$artifact/profile.sh" "$artifact"
sh "$artifact/prep-analysis.sh" "$artifact"
python3 "$artifact/verify-profiles.py" "$artifact"
sh "$artifact/correctness.sh" "$artifact"
sh "$artifact/range-matrix.sh" "$artifact"
```

`benchmark.sh` builds the root test binary and launches fixed `-test.benchtime 1x` processes. It alternates scenario order and verifies the same prompt snapshot in every process. `summarize-benchmarks.py` validates every supplemental digest and decode suffix and reports medians and ranges. `profile.sh` builds a Llama-only overlay and collects separately labeled CPU, alloc and trace runs. `prep-analysis.sh` builds the text profile summaries, binary disassembly and per-site trace reports. `verify-profiles.py` checks all 21 profiled process logs, both lane digests where applicable, and traced helper totals.
