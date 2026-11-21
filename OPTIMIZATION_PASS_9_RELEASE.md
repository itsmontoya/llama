# Llama Optimization Pass 9 release evaluation

Llama tested the released `neuralnetwork v0.10.19` integration in an isolated checkout. The production pin stays at `v0.10.18`. The prepared path passed numerical checks and improved primary TTFT, but the paired two-session result and 27-token prefill guard did not establish release readiness. At `GOMAXPROCS=6`, unaligned packed worker shards caused whole-shard unpack allocations and a large end-to-end regression. The prepared FFN optimization remains open upstream.

## Baseline and candidate

The Llama baseline and candidate start from `45926f1041276bb7d0e6996997135b5caa4e3b9a`. Its production tree matches the Pass 8 merge `b07f4ae100d0ada87a8a4485f294bdacb7fe137e`. The baseline links `neuralnetwork v0.10.18` at `ce91677fa12ff521309764e4664a296f57a52427` with checksum `h1:j1PoyjYzxYhlwaWOUp/KlSh4V2xu8Upt/9kyIcnK/AU=`. The candidate links the published `v0.10.19` at `7335da1afc2a17f5aee831bb623c85a917a8e707`, checksum `h1:l5s0mkRq7AQmwKBt4YvFibycnccokl02pe7t5nU3xBI=`. Both module files have checksum `h1:SafbkuDehVErsuJAL5FHFsLhKJbeh/OI3ItvbmwDafw=`. The module proxy and Go checksum database verified the release, and its nine changed files matched the tag byte for byte. Both builds link `github.com/tphakala/simd v1.3.0` and `golang.org/x/sys v0.45.0`.

Upstream commit `0cc0185d6bbbbbcdceb199c6ff39ab061731155c` adds the prepared representation and kernel. Commit `90fde52656acb0b78408e7e7cd2d39ca20d74e5d` adds coverage. The prototype used one owning prepared tensor for each of the 48 FFN gate, up, and down matrices. It validates finiteness before preparation, passes row-major source payloads to `tensor.PrepareRightTranspose`, and calls the prepared `GemmInto` method at those three projection sites. The candidate keeps the pinned tied classifier and the attention weights on the dense path. It preserves the FP32 arithmetic order, worker count policy, default capacity and session behavior. [Upstream change](https://github.com/itsmontoya/neuralnetwork/pull/138)

The new kernel stores eight adjacent output weights for each K and loads them directly in one vector-pair load. It reuses them across one, two or three input rows. Assembly and packed storage are active on the eligible host build. Other architectures, `purego`, race builds and ineligible shapes use the upstream dense fallback. The probes check assembly activation and the stable mathematical output in the full Llama path.

## Measurement design

Both builds used Go 1.27.3 on Apple M3 with four performance and four efficiency cores, 24 GB unified memory, AC power and macOS 26.5.2 (25F84). Default measurements used Darwin ARM64, `GOARM64=v8.0`, `GOMAXPROCS=8`, `CGO_ENABLED=1`, empty `GOFLAGS`, `GOWORK=off` and default build tags. The pinned `meta-llama/Llama-3.2-1B` revision is `4e20de362430cd3b72f300e6b0f18e50e7166e08`. The six files were verified in every process. The manifest and frozen reference SHA-256 hashes are `3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64` and `3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361`. The 2,471,645,608-byte checkpoint hash is `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`.

Each fresh process timed one operation with zero inference warm-ups. Before and after order alternated among pairs. The 12 primary generation pairs use `The capital of France is`, six prompt tokens including BOS, eight greedy outputs, context cap 512, and KV capacity 13. TTFT covers prefill and first greedy selection. Seven sustained pairs time 32 steps at positions 6–37 after setup and prefill. Seven concurrent pairs release two already-prefilled independent sessions on one shared decoder at a common barrier. Each performs 32 steps. Fairness is the faster rate divided by the slower rate. There are five pairs each for 27-, 28-, and 29-token prefill and seven for concurrent six-token prefill. Process wall and Darwin peak RSS include setup and checkpoint loading.

The tables include all 96 completed benchmark processes. Paired columns use medians of the per-pair differences or ratios. Process observations include wide tails. The diagnostic builds and profiles run separately from the timing series. Hardware traffic and stall counters were unavailable on this host.

## Primary generation (12/12 pairs)

| Metric | v0.10.18 median (range) | v0.10.19 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill (s) | 0.1751 (0.1677–0.6477) | 0.1254 (0.1076–0.7780) | -0.0512 | -28.76% (-40.72% to +340.79%) | 11/12 |
| TTFT (s) | 0.1752 (0.1678–0.6478) | 0.1255 (0.1077–0.7781) | -0.0513 | -28.74% (-40.69% to +340.60%) | 11/12 |
| Decode (s/token) | 0.06895 (0.06780–0.07386) | 0.06873 (0.06644–0.10997) | -0.00082 | -1.17% (-7.89% to +62.20%) | 7/12 |
| Decode (tokens/s) | 14.505 (13.540–14.750) | 14.550 (9.094–15.050) | +0.165 | +1.16% (-38.35% to +8.57%) | 7/12 |
| Cached generation (s/op) | 0.6592 (0.6462–1.1325) | 0.6078 (0.5797–1.5480) | -0.0643 | -9.37% (-13.03% to +137.70%) | 9/12 |
| Output (tokens/s) | 12.137 (7.064–12.379) | 13.163 (5.168–13.801) | +1.214 | +10.34% (-57.93% to +14.98%) | 9/12 |
| Bytes/op | 131,323,444 (131,311,992–131,331,080) | 131,400,700 (131,389,808–131,407,896) | +77,716 | +0.06% (+0.04% to +0.07%) | 0/12 |
| Allocs/op | 39,900 (39,876–39,912) | 40,274 (40,252–40,289) | +374 | +0.94% (+0.85% to +1.00%) | 0/12 |
| Process wall (s) | 3.756 (3.729–5.788) | 4.067 (4.015–5.116) | +0.298 | +7.96% (-17.06% to +35.97%) | 1/12 |
| Peak RSS (GiB) | 6.373 (5.689–6.374) | 6.486 (6.294–6.486) | +0.113 | +1.77% (-0.73% to +11.79%) | 1/12 |
| Model load (s) | 1.853 (1.820–2.856) | 2.207 (2.160–2.255) | +0.356 | +19.32% (-24.37% to +23.90%) | 1/12 |
| Tokenizer setup (s) | 0.166 (0.162–0.185) | 0.173 (0.165–0.228) | +0.004 | +2.17% (-8.30% to +41.11%) | 2/12 |

## Sustained decode (7/7 pairs)

| Metric | v0.10.18 median (range) | v0.10.19 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elapsed (s/op) | 2.2455 (2.1808–3.9252) | 2.1564 (2.1303–3.9351) | -0.0938 | -4.18% (-42.50% to +70.56%) | 5/7 |
| Decode (s/token) | 0.07017 (0.06815–0.12270) | 0.06739 (0.06657–0.12300) | -0.00293 | -4.18% (-42.50% to +70.60%) | 5/7 |
| Decode (tokens/s) | 14.250 (8.153–14.670) | 14.840 (8.132–15.020) | +0.620 | +4.35% (-41.37% to +73.91%) | 5/7 |
| Bytes/op | 329,804,624 (329,788,208–329,820,400) | 330,179,800 (330,168,304–330,188,240) | +373,408 | +0.11% (+0.11% to +0.12%) | 0/7 |
| Allocs/op | 149,549 (149,514–149,583) | 151,095 (151,075–151,116) | +1,545 | +1.03% (+1.02% to +1.06%) | 0/7 |
| Process wall (s) | 5.529 (5.381–7.140) | 5.805 (5.703–7.646) | +0.256 | +4.63% (-19.81% to +35.90%) | 2/7 |
| Peak RSS (GiB) | 6.375 (5.696–6.376) | 6.500 (5.692–6.501) | +0.124 | +1.95% (-10.70% to +14.12%) | 1/7 |

## Two-session sustained decode (7/7 pairs)

| Metric | v0.10.18 median (range) | v0.10.19 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode wall (s/op) | 3.6920 (3.6550–5.4110) | 3.7530 (3.5370–4.9250) | +0.0610 | +1.65% (-34.63% to +34.01%) | 2/7 |
| Aggregate decode (tokens/s) | 17.340 (11.830–17.510) | 17.050 (13.000–18.100) | -0.290 | -1.67% (-25.33% to +53.00%) | 2/7 |
| Session 0 (tokens/s) | 8.668 (5.914–8.755) | 8.559 (6.498–9.048) | -0.109 | -1.26% (-25.37% to +52.99%) | 3/7 |
| Session 1 (tokens/s) | 8.715 (5.931–8.823) | 8.526 (6.514–9.108) | -0.189 | -2.17% (-25.64% to +53.57%) | 2/7 |
| Session 0 (s/token) | 0.11540 (0.11420–0.16910) | 0.11680 (0.11050–0.15390) | +0.00140 | +1.21% (-34.65% to +34.06%) | 3/7 |
| Session 1 (s/token) | 0.11470 (0.11330–0.16860) | 0.11730 (0.10980–0.15350) | +0.00260 | +2.27% (-34.88% to +34.41%) | 2/7 |
| Faster/slower rate ratio | 1.0041 (1.0029–1.0078) | 1.0045 (1.0025–1.0066) | -0.0008 | -0.07% (-0.36% to +0.37%) | 4/7 |
| Bytes/op | 659,506,056 (659,498,824–659,538,728) | 660,243,064 (660,221,864–660,277,240) | +724,096 | +0.11% (+0.11% to +0.11%) | 0/7 |
| Allocs/op | 298,891 (298,872–298,955) | 301,961 (301,926–302,032) | +3,062 | +1.02% (+1.01% to +1.04%) | 0/7 |
| Process wall (s) | 7.194 (6.962–8.925) | 7.952 (7.229–8.880) | +0.759 | +10.55% (-19.00% to +23.44%) | 2/7 |
| Peak RSS (GiB) | 6.379 (6.372–6.380) | 6.498 (6.246–6.505) | +0.119 | +1.86% (-2.09% to +2.09%) | 2/7 |

## 27-token prefill (5/5 pairs)

| Metric | v0.10.18 median (range) | v0.10.19 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elapsed (s/op) | 0.7108 (0.6802–1.5894) | 0.7278 (0.5637–1.0315) | +0.0476 | +6.99% (-63.92% to +45.96%) | 2/5 |
| Bytes/op | 261,775,056 (261,754,592–261,803,064) | 261,779,824 (261,772,664–261,786,600) | +1,616 | +0.00% (-0.01% to +0.01%) | 2/5 |
| Allocs/op | 7,066 (7,025–7,120) | 7,122 (7,108–7,134) | +46 | +0.65% (+0.11% to +1.55%) | 0/5 |
| Process wall (s) | 3.825 (3.798–5.578) | 4.317 (4.022–4.842) | +0.519 | +13.66% (-27.89% to +20.76%) | 1/5 |
| Peak RSS (GiB) | 6.372 (6.060–6.375) | 6.399 (5.692–6.475) | +0.027 | +0.42% (-6.07% to +1.56%) | 2/5 |

## 28-token prefill (5/5 pairs)

| Metric | v0.10.18 median (range) | v0.10.19 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elapsed (s/op) | 0.7222 (0.7062–1.9591) | 0.4511 (0.4347–0.4692) | -0.2715 | -37.95% (-76.05% to -36.42%) | 5/5 |
| Bytes/op | 271,360,936 (271,343,456–271,384,256) | 271,361,448 (271,354,928–271,375,808) | +512 | +0.00% (-0.01% to +0.01%) | 2/5 |
| Allocs/op | 7,076 (7,037–7,117) | 7,127 (7,109–7,160) | +51 | +0.72% (-0.11% to +1.75%) | 1/5 |
| Process wall (s) | 3.826 (3.677–5.152) | 3.900 (3.886–4.122) | +0.139 | +3.68% (-24.33% to +7.21%) | 1/5 |
| Peak RSS (GiB) | 6.374 (5.692–6.374) | 6.499 (6.499–6.499) | +0.125 | +1.96% (+1.96% to +14.16%) | 0/5 |

## 29-token prefill (5/5 pairs)

| Metric | v0.10.18 median (range) | v0.10.19 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elapsed (s/op) | 1.3542 (0.7385–1.5811) | 0.4684 (0.4394–1.2699) | -0.8858 | -65.41% (-72.21% to +71.95%) | 4/5 |
| Bytes/op | 282,504,064 (282,492,624–282,522,544) | 282,512,432 (282,500,392–282,525,400) | +9,712 | +0.00% (-0.01% to +0.01%) | 1/5 |
| Allocs/op | 7,056 (7,033–7,097) | 7,120 (7,106–7,153) | +65 | +0.92% (+0.13% to +1.45%) | 0/5 |
| Process wall (s) | 4.621 (3.762–5.744) | 3.931 (3.909–4.830) | -0.696 | -15.06% (-31.95% to +28.38%) | 3/5 |
| Peak RSS (GiB) | 6.035 (5.689–6.373) | 6.497 (5.693–6.498) | +0.190 | +3.01% (-4.10% to +11.01%) | 1/5 |

## Two-session six-token prefill (7/7 pairs)

| Metric | v0.10.18 median (range) | v0.10.19 median (range) | Median paired Δ | Paired Δ% median (range) | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elapsed (s/op) | 0.2793 (0.2675–0.9139) | 0.1915 (0.1635–0.5490) | -0.0778 | -29.08% (-79.04% to +84.36%) | 4/7 |
| Bytes/op | 116,454,088 (116,445,792–116,470,952) | 116,461,464 (116,449,792–116,472,880) | +4,000 | +0.00% (-0.00% to +0.02%) | 1/7 |
| Allocs/op | 13,957 (13,932–13,997) | 14,058 (14,038–14,084) | +100 | +0.72% (+0.56% to +1.03%) | 0/7 |
| Process wall (s) | 3.445 (3.363–5.036) | 3.648 (3.619–4.359) | +0.284 | +8.46% (-27.81% to +20.34%) | 2/7 |
| Peak RSS (GiB) | 6.373 (6.033–6.373) | 6.437 (5.689–6.473) | +0.065 | +1.01% (-10.73% to +7.29%) | 1/7 |

## Timing interpretation

The candidate showed its clearest effect in prefill. Primary TTFT fell 28.74% at the paired median, with gains in 11 of 12 pairs, and the 28-token guard improved in all five pairs. The 29-token guard improved in four of five pairs, with a wide range. The 27-token guard rose 6.99% at the paired median, above the prior 5% investigation threshold, and improved in two of five pairs. The 27-token and concurrent prefill ranges span substantial gains and regressions, so their effect sizes are inconclusive.

Primary cached generation improved 9.37% at the paired median in nine of 12 pairs. Its seven subsequent decode steps improved only 1.17% by median latency and had broad tails. Sustained decode improved 4.18% at the paired median in five of seven pairs, with changes from a 42.50% reduction to a 70.56% increase. This supports a possible single-session gain while leaving the repeatable decode effect size unresolved.

Two-session aggregate throughput improved in two of seven pairs and was 1.67% lower at the paired median. The lane medians were 8.559 and 8.526 tokens/s after preparation, down 1.26% and 2.17% by the corresponding paired medians. Fairness remained close to one (candidate 1.0045, range 1.0025–1.0066). The concurrent series does not establish a precise regression magnitude. Its direction agrees with upstream's independent paired trial, which reported a 4.19% median aggregate decline and slower lanes. Together with the unconfirmed 27-token guard and startup cost, the candidate falls short of the release criteria. [Upstream Llama results](https://github.com/itsmontoya/neuralnetwork/pull/138)

Median model load rose from 1.853 s to 2.207 s, a paired increase of 19.32% in 11 of 12 pairs. That increase outweighed cached generation gains in the primary benchmark's full-process wall, which rose 7.96% at the paired median. Full-process wall also rose 4.63% for sustained and 10.55% for concurrent decode. Peak RSS rose 0.113 GiB, 0.124 GiB and 0.119 GiB at the paired median in the primary, sustained and concurrent workloads, respectively. At P=8, decode allocations grew roughly 1% by count and 0.11% by bytes.

## Matched CPU, allocation and wall profiles

Each arrow reads from dense baseline to prepared candidate. Sampled CPU sums across threads and the percentages use labeled phase CPU. The wall columns use separate timers in the CPU diagnostic. The allocation columns use cumulative sampled bytes from separate 64 KiB-rate profiles. Labeled projection CPU includes kernel and dispatch work. Concurrent projection wall sums durations across both lanes and can exceed elapsed wall. Shape and preparation probes use diagnostic builds independently of the release series.

| Workload | Profiled wall (s) | Labeled sampled CPU (s) | Projection CPU (s, % of phase) | Condition signal CPU (s) | Sampled alloc traffic (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Checkpoint load | 2.706 → 2.232 | 1.72 → 1.96 | — | 0.00 → 0.00 | 9758.4 → 9757.9 |
| Primary prefill, 6 × 6 tokens | 1.018 → 0.655 | 2.09 → 1.42 | 2.09 (100.0%) → 1.40 (98.6%) | 0.19 → 0.09 | 330.7 → 334.8 |
| Short decode, 42 steps | 2.908 → 2.795 | 8.48 → 7.52 | 8.46 (99.8%) → 7.48 (99.5%) | 1.12 → 2.68 | 416.5 → 414.2 |
| Sustained decode, 64 steps | 4.547 → 4.316 | 12.74 → 11.10 | 12.73 (99.9%) → 11.03 (99.4%) | 1.44 → 3.88 | 625.4 → 617.3 |
| Concurrent decode, 128 steps | 7.454 → 8.246 | 25.76 → 28.09 | 24.37 (94.6%) → 26.30 (93.6%) | 3.68 → 2.64 | 1257.9 → 1272.2 |
| 27-token prefill ×3 | 2.123 → 1.208 | 5.51 → 2.70 | 5.27 (95.6%) → 2.63 (97.4%) | 0.31 → 0.45 | 750.9 → 751.8 |
| 28-token prefill ×3 | 2.206 → 2.247 | 5.36 → 7.02 | 5.28 (98.5%) → 6.69 (95.3%) | 0.44 → 0.20 | 775.8 → 775.6 |
| 29-token prefill ×3 | 2.302 → 2.197 | 6.45 → 7.82 | 6.17 (95.7%) → 7.41 (94.8%) | 0.37 → 0.19 | 809.6 → 808.8 |

### Sustained shape samples

| Shape (M,K,N) | Summed probe wall (s) | Shape CPU (s, % of phase) | Sampled allocation traffic (MiB) |
| --- | ---: | ---: | ---: |
| Attention K/V (1,2048,512) | 0.2251 → 0.2234 | 0.07 (0.5%) → 0.20 (1.8%) | 8.9 → 6.8 |
| Attention Q/O (1,2048,2048) | 0.5198 → 0.5130 | 0.49 (3.8%) → 1.92 (17.3%) | 33.2 → 33.1 |
| FFN gate/up (1,2048,8192) | 1.8504 → 1.6813 | 7.93 (62.2%) → 5.46 (49.2%) | 125.3 → 125.2 |
| FFN down (1,8192,2048) | 0.8742 → 0.8356 | 1.20 (9.4%) → 0.52 (4.7%) | 18.4 → 16.4 |
| Tied classifier (1,2048,128256) | 0.8511 → 0.8394 | 3.04 (23.9%) → 2.93 (26.4%) | 63.1 → 63.1 |
| All projections | 4.3207 → 4.0928 | 12.73 (99.9%) → 11.03 (99.4%) | 249.0 → 244.6 |

### Two-session decode shape samples

| Shape (M,K,N) | Summed probe wall (s) | Shape CPU (s, % of phase) | Sampled allocation traffic (MiB) |
| --- | ---: | ---: | ---: |
| Attention K/V (1,2048,512) | 0.8196 → 0.9440 | 0.39 (1.5%) → 0.43 (1.5%) | 18.9 → 18.6 |
| Attention Q/O (1,2048,2048) | 2.0197 → 2.2932 | 2.64 (10.2%) → 2.94 (10.5%) | 69.4 → 68.1 |
| FFN gate/up (1,2048,8192) | 6.2233 → 6.4149 | 10.54 (40.9%) → 11.06 (39.4%) | 255.1 → 263.6 |
| FFN down (1,8192,2048) | 2.8473 → 3.2352 | 4.25 (16.5%) → 4.79 (17.1%) | 34.7 → 34.8 |
| Tied classifier (1,2048,128256) | 2.3162 → 2.6703 | 6.55 (25.4%) → 7.08 (25.2%) | 126.2 → 126.4 |
| All projections | 14.2261 → 15.5576 | 24.37 (94.6%) → 26.30 (93.6%) | 504.3 → 511.4 |

### Labeled prefill kernel samples

| Prefill | Complete groups (s, % of phase) | One-row FFN remainder CPU (s) | Two-row block remainder CPU (s) |
| --- | ---: | ---: | ---: |
| 27 tokens | 4.75 (86.2%) → 1.98 (73.3%) | 0.00 → 0.00 | 0.00 → 0.00 |
| 28 tokens | 4.44 (82.8%) → 5.79 (82.5%) | 0.27 → 0.13 | 0.00 → 0.00 |
| 29 tokens | 4.95 (76.7%) → 6.48 (82.9%) | 0.00 → 0.00 | 0.61 → 0.10 |

## Worker caps and access controls

The shape sweep used one process and 48 calls per shape, mode and cap. The table shows time in ms per call. Rotate cycles through distinct checkpoint matrices. The classifier has one shared dense tensor. These observations are diagnostic and have broad variation, including a substantial change in the unchanged classifier at P=8.

| GOMAXPROCS | Gate/up rotate (ms) | Down rotate (ms) | Classifier repeat (ms) |
| --- | ---: | ---: | ---: |
| 1 | 3.017 → 3.061 | 2.530 → 2.653 | 42.343 → 42.216 |
| 2 | 1.703 → 3.630 | 1.492 → 1.960 | 23.342 → 23.997 |
| 4 | 1.212 → 2.480 | 1.057 → 1.627 | 16.063 → 16.974 |
| 6 | 0.982 → 7.447 | 0.948 → 5.190 | 13.298 → 13.869 |
| 8 | 1.398 → 2.069 | 1.230 → 1.409 | 13.476 → 15.741 |

P=6 makes the four-column dispatch boundaries diverge from some eight-column packed tile starts. In `rightTransposePackedRange`, a shard starting four columns into a tile allocates and unpacks its entire range before using the dense kernel. The rotating gate/up shape grew from 33,542 to 22,447,132 B/op and down grew from 8,784 to 22,553,468 B/op. That confirms the allocation mechanism independently of wall-time noise. P=8 cuts all measured FFN shapes on pack boundaries.

The paired shared/distinct operand control used the same FFN gate/up shape, worker cap, operation count and two-lane barrier within each comparison. A negative change favors both callers reading the same rotating operand. Every observation sums the slower completion time across 32 barriers.

| Build and cap | Shared median (ms) | Distinct median (ms) | Median paired shared change (range) | Shared faster |
| --- | ---: | ---: | ---: | ---: |
| v0.10.18, P=4 | 51.997 | 66.337 | −22.60% (−27.78% to −12.84%) | 6/6 |
| v0.10.19, P=4 | 56.288 | 66.223 | −16.12% (−34.25% to +317.38%) | 5/6 |
| v0.10.18, P=8 | 44.099 | 53.011 | −17.25% (−27.75% to +1.47%) | 5/6 |
| v0.10.19, P=8 | 38.555 | 49.139 | −21.59% (−28.74% to −9.05%) | 6/6 |

The prepared path remains sensitive to whether two callers reuse a weight operand. Both builds' results support a role for shared caches or delivered bandwidth. The artificial barriers show headroom for weight reuse and have limited predictive power for real sessions that start together only once.

In one additional real Llama comparison at P=6, sustained decode rose from 2.484 s to 10.895 s and from 0.307 to 32.432 GiB in cumulative allocation per 32 steps. The two-session run moved from 16.30 to 4.408 aggregate tok/s and from 0.613 to 64.864 GiB in cumulative allocation. Concurrent process peak RSS moved from 6.38 to 9.34 GiB. Both runs retained the frozen suffix and logit hash. The P=6 comparisons are cap diagnostics outside the P=8 release series. The measured allocation increase matches the extra per-shard unpack traffic across all 48 FFN matrices and 32 steps.

The Go execution traces observed the same 19,168 spawned helpers in each single-session build and 38,336 in each two-session build. All helpers exited.

| Scheduler observation | Single session | Two sessions |
| --- | ---: | ---: |
| Block helper creation to running, median / p95 (µs) | 32.2 / 368.6 → 34.9 / 388.3 | 140.8 / 951.4 → 145.3 / 936.8 |
| Classifier helper creation to running, median / p95 (µs) | 38.3 / 8,152.9 → 41.8 / 2,481.3 | 49.4 / 8,869.7 → 246.5 / 9,606.3 |
| Summed helper Running time (s) | 11.122 → 10.124 | 22.630 → 21.511 |
| Summed helper Runnable time (s) | 1.333 → 1.360 | 12.069 → 11.890 |
| Summed blocked caller join time (s) | 0.830 → 0.823 | 3.854 → 3.832 |
| Blocked join with runnable helpers and no helper Running (ms) | 0.005 → 0.017 | 2.867 → 3.037 |

At least one matching helper was in Go Running state during almost all of the observed caller join. That state can include useful work, synchronization and OS descheduling. Summed concurrent waits can overlap. The traces identify runnable-delay pressure but expose little time in which the caller waits without a running helper. A change to dispatch would need to preserve useful execution as well as reduce delays.

## Preparation and memory

The instrumented load-only runs prepared 48 FFN matrices in 0.525 and 0.540 s, including allocation and packing. That work replaces the dense allocation and copy for those matrices. The paired primary model-load delta of 0.356 s measures the actual added startup cost. The shape and payload count give 3 GiB of FFN weight storage in either layout. No secondary dense FFN tensors are retained.

With the decoder held live after a forced GC, retained heap was about 4.605 GiB on both pins (4,944,751,464 B versus 4,944,752,520 B in the CPU probes, with similarly close values in the allocation probes). Dropping the decoder and forcing another GC reduced heap to approximately 1.4 MiB in both CPU probes. RSS had a wide range across standalone load probes and stayed elevated immediately after that GC in the candidate processes. The captured runtime snapshot at the end of the sustained CPU workload was 6.370 GiB baseline and 6.507 GiB candidate. Runtime RSS includes pages held by the process after heap objects become reclaimable. The unprofiled process high-water pairs above give the more repeatable candidate memory comparison.

## Limiting factor after preparation

The candidate disassembly confirms that the packed kernel consumes adjacent K-major weight vectors directly, without the 16 ZIP instructions of the dense eight-output loop. The logical weight payload stays at 4.60 GiB per token, including 3 GiB for the FFN. In the labeled 64-step diagnostic, FFN gate/up and down summed wall fell from 2.725 s to 2.517 s. The shape timers imply about 75.7 → 81.9 GB/s of effective logical FFN weight payload. The independent three 32-step wall comparisons had dense loop times of 2.192–2.290 s and prepared loop times of 2.136–3.425 s. The packed layout reduced some single-session projection work and the more variable controls leave the gain size uncertain.

In the 128-step concurrent CPU diagnostic, summed FFN projection wall instead rose from 9.071 to 9.650 s. Other dense shapes also slowed in that probe, including the classifier. Sampled condition signaling moved in opposite directions across the single-session and concurrent profiles. Taken with the fixed-dispatch locality controls, the best-supported P=8 classification still combines weight delivery and scheduler contention. The data leaves the concurrency change unresolved. The Apple host lacked direct bandwidth and stall counters. P=6 has a separate, confirmed allocation limit from whole-shard unpacking. These two regimes call for separate measurements when a revised implementation returns.

## Correctness

Both variants passed fixture-enabled default and `purego` Llama suites, the standard Llama race suite and `go vet`. The race suite covered TinyStories, local traces and shared-session tests, with the full Meta fixture disabled as in Pass 8. Default and `purego` preserved the exact checked cached/full-prefix rows, greedy continuation IDs and byte-identical CLI output. The largest Transformers logit absolute error remained `4.95910645e-05` and the largest tolerance fraction remained `0.0316354157`. All 36 supplemental benchmark logs per variant matched their frozen digests. The 12 primary runs per variant verified their frozen eight-token suffix and decoded bytes.

The published upstream package passed targeted `RightTranspose|EmptyContraction|Gemm_Bias` tests in default, `purego` and race builds. Its affected-package tests and `go vet` also passed. The prepared-weight cases cover bitwise parity, misaligned shards, immutable ownership, shared readers, destination alias rejection, and concurrent retries. Numerical and ownership gates passed for the tested integration. The release decision follows the performance and startup gates.

## Remaining work and candidate ranking

The next revision of the selected Pass 9 optimization belongs in neuralnetwork. Keep the ordered FP32 arithmetic and single-copy weight storage. Validate packed tile alignment and bounded handling of partial shards at scheduler caps that create four-column cuts. It needs to clear real Llama two-session throughput, per-lane fairness, prefill, startup and memory guards with the default and nonaligned caps. The production pin remains v0.10.18.

Future candidates ranked from these measurements are below.

1. **Projection worker coordination, neuralnetwork.** The default concurrent trace observed 38,336 helpers, 12.07 s of summed runnable time on the baseline, and a 140.8 µs median first-run delay for block helpers. A future evaluation should keep the process cap fixed, isolate dispatch from kernel throughput, and measure queueing and both session completion times.
2. **Complete-group prefill, neuralnetwork.** Dense complete groups account for 77–86% of the labeled 27–29-token phase CPU. The prepared path's consistent 28-token win demonstrates useful prefill headroom. Profile divisible and nondivisible prompt lengths and guard sustained decode, which shares the same weights.
3. **Tied classifier access, neuralnetwork or cross-repository.** A 1,002 MiB read per token took about 0.85 s over 64 profiled single-session steps and 2.32 s of summed time over 128 concurrent steps on the dense baseline. Use controlled access experiments at the same shape and preserve shared gather/classifier ownership and row layout semantics.
4. **Checkpoint staging, Llama or cross-repository.** The dense model-load median was 1.853 s. The standalone allocation profile still sampled about 9,758 MiB of load traffic with about 4.60 GiB of retained heap. Separate reads, conversion, validation and private copies before changing ownership.
5. **SwiGLU staging, Llama or cross-repository.** Allocation traces isolate its internal staging, including sigmoid. Use measured GC time and retained memory to test whether reducing those allocations helps latency under longer and concurrent sessions.

## Reproduction

The release evidence archive contains the exact candidate patch for the isolated Llama copy, module metadata, raw timing and correctness logs, profiles, traces, disassembly, and the diagnostic overlays. The original investigation is also available. [Pass 9 investigation](OPTIMIZATION_PASS_9.md)

Extract `/private/tmp/llama-pass9-release-evidence.qZcjRjna.tar.gz`, update the local fixture paths and writable cache settings in `env.sh`, and run the following from the Llama Git checkout. Set `PATH` to the installed Go 1.27.3 toolchain. Use a second extraction to preserve the measured logs.

```sh
out=/path/to/extracted/llama-pass9-release
. "$out/env.sh"
GOPROXY=https://proxy.golang.org go mod download all
GOPROXY=https://proxy.golang.org go mod download github.com/itsmontoya/neuralnetwork@v0.10.19
sh "$out/build.sh" "$out"
sh "$out/paired.sh" "$out"
sh "$out/profiles.sh" "$out"
sh "$out/correctness.sh" "$out"
sh "$out/analyze.sh" "$out"
```
