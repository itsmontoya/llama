# NN-08 handoff from Llama LL-04

Use one upstream hypothesis for the next owner-authorized task. The
frozen LL-04 baseline and every paired sample are in the
[Llama GPU performance report](GPU_PERFORMANCE_REPORT.md). The Llama
source is `d601bbce4afc5b3086a6aca58646856abf4c90df` with neuralnetwork
`v0.11.2` at `8db36638be5b0166d676ae1322d89dd095823cb3`. Preserve the
baseline as published.

## Ranked opportunities

1. **Single-row FP32 right-transposed projection.** The warmed
   sustained GPU command median was 90.01 ms without stage samples.
   With stage samples, the 48 FFN matmuls accounted for 41.01 ms
   and the three projection groups together for about 72% of the
   command median. Gate and up summed to 26.135 ms, and down to
   14.799 ms. A 9 ms reduction would meet the 10% whole-decode
   gate. On the FFN group alone, that requires roughly 22% lower
   stage time at the same number of dispatches. Keep device-resident
   weights and the existing schedule contract. Owner is neuralnetwork.
2. **Investigate the repeated RMSNorm stage cost.** The 33 RMSNorm
   stages summed to 18.49 ms (18.01–20.05) per warmed decode in the
   counter probe. Use an isolated resident measurement to establish how
   much of that interval is reducible. Look at `[1,2048]` normalized by
   the existing epsilon, plus the 6/27/28/29-row guards. A different
   reduction order needs the same accuracy checks and an independent
   before/after series. Owner is neuralnetwork.

## Exact first candidate

Measure a single right-transposed FP32 projection specialization for
single-row decode. Preserve the public `Scope.MatMulRightTranspose`
interface and row-major tensors. Compare it first on the three FFN
projections that repeat in each block, then measure the whole decoder
using the same driver. A regression in NN-08 can be a report with the
FP32 baseline retained.

The pinned configuration uses 16 layers, width 2048, FFN width 8192,
32 query heads, eight KV heads, head width 64 and vocabulary 128256.
A projection reads a contiguous `A[M,K]`, reads immutable
`B[N,K]` with K contiguous, and writes contiguous `C[M,N]`. Metal
currently issues right-transposed variant 2. Its generic dispatch
uses a 16×16 threadgroup for a 2D grid and computes `ceil(M/16)`
threadgroups in the row dimension. Only one of the 16 output rows
is in range at M=1. Threadgroup geometry is a hypothesis to test.

| Projection | Calls per step | K | N | FP32 weight bytes per step |
| --- | ---: | ---: | ---: | ---: |
| FFN gate and up | 32 | 2048 | 8192 | 2,147,483,648 |
| FFN down | 16 | 8192 | 2048 | 1,073,741,824 |
| Attention Q and O | 32 | 2048 | 2048 | 536,870,912 |
| Attention K and V | 32 | 2048 | 512 | 134,217,728 |
| Tied classifier | 1 | 2048 | 128256 | 1,050,673,152 |

The 113 projections cover 4,942,987,264 logical bytes of resident FP32
weights per token on the pinned graph. The table describes tensor
sizes and call counts. Runtime hardware traffic depends on reuse and
the implementation. Cached prefill feeds M=6/27/28/29 to the blocks and
M=1 to the final classifier. `Forward` projects all requested rows.
Keep the existing general path and correctness for each supported M.

## Experiment and stop criteria

Use the released seed and pinned workload above. Run at least six
alternating pairs following [the full-model driver](GPU_PERFORMANCE_REPORT.md#validation-and-reproduction).
Capture the 32-step warmed sustained decode through positions 6–37,
cold launch and short TTFT, long prefill tails, and shared and independent
decode. Run the same workload shape matrix on isolated resident
projections with warm and first-use costs identified. Baseline and
candidate inputs, seeding, storage, device, command counts and readback
counts should match. If unpublished upstream work prevents a full
consumer run under the repository rules, report the limitation and show
the isolated before/after comparison and numerical gates. Leave Llama's
module pin untouched.

- Aim for at least 10% lower paired median sustained decode latency
  and a lower latency in at least five of six pairs. Use the
  sampled command intervals and primary wall together to verify that
  any reduction appears end to end.
- Recheck the full returned FP32 rows with the existing
  `1e-4 + 1e-4*abs(CPU)` bound and frozen samples with
  `1e-3 + 1e-3*abs(reference)`. All greedy IDs must agree, including
  ID 95551 at position 11. The current worst full-row fraction is
  0.96079 on sustained decode, so check every row. Include FP64 projection checks for both
  physical orientations, asymmetric dimensions, tail sizes, negative and
  nonfinite inputs under the existing upstream contract, and real Metal
  cached/full-prefix, concurrent, canceled and failed steps.
- Preserve one command, one wait and one logit download per token
  step. Keep the single tied allocation and 4,943,388,672 model bytes
  per model. Target zero added persistent device bytes. If a candidate
  needs temporary space, measure it and keep the added live bytes
  under 1 MiB per lane. The present model and KV allocation already
  doubles with two independent models. Require measured peak RSS and
  post-close retention to remain within 5% of matched baseline medians.
- Keep paired median cold TTFT and 27/28/29-row prefill within 5% of
  baseline and preserve shared and independent aggregate throughput
  within 5%. Repeat a noisy series before deciding against a candidate.
  Reject the optimization if it loses the performance gate or fails
  any correctness, memory, or compatibility gate.

The optimizer belongs in neuralnetwork's shared Metal runtime and GPU
matmul path. Preserve CPU, pure-Go, unsupported-platform, and existing
ANN behavior. The Llama integration can evaluate a separately published
decision later. [NN-07 output boundary](https://github.com/itsmontoya/neuralnetwork/pull/147)
