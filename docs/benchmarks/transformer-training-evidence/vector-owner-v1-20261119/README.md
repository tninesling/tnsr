# Vector BF16-first Adam owner experiment: no adoption

The W1024/S128/B2 complete-update experiment completed six fresh workers,
384 measured updates and four comparisons passing all six unchanged numerical
gates. Two reversed trials use the same sealed binary and source, with only
`--vector-bf16-first-adam` added to the candidate.

| Trial | Baseline ms | Candidate ms | Candidate reduction | Sampled NVML delta MiB |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6.9659345 | 7.1845775 | -3.1387% | 0 |
| 1 | 7.2988950 | 6.9652180 | +4.5716% | 0 |

The timing direction reverses, so this experiment does not justify adoption.
Aggregate native medians are 7.132415 ms baseline and 7.074898 ms candidate;
fresh compiled PyTorch CUDA-graph median is 6.835768 ms. The stronger historical
qualified target remains 6.796195 ms, with its original raw evidence missing.
This case does not meet the parity target. Experimental code stays default off;
the accepted policy and 2/8 parity count are unchanged.

All 642 candidate qualification/warmup/measured rows submit and commit exactly
20 vector parameters and 50,593,792 elements; baseline rows select zero. Candidate
status retains 38 batched F32 parameters and two old-source F32 fallbacks.
All three distinct vector lengths report 40 registers, 32 shared bytes, no local
bytes, maximum 1,024 threads and API capacity six at 256 threads. API capacity
is not an achieved-occupancy measurement. Replay remains active.

All four timed native gradient/weight snapshot pairs match SHA256 exactly.
All 642 scalar triples (1,926 F64 loss/norm/scale values) match bitwise, as do
model counters, launch counts and transfer byte/count records. All six allocator
stages and owned payload bytes match. Native sampled whole-process peaks are
2,008 MiB in both modes and both trials; these are lower bounds including setup,
separate from allocator accounting, and do not establish capacity savings.

A separate untimed first-update screen completed three workers and two fixed
comparisons. Its two native snapshot pairs and qualification/sample scalars
match exactly. Its single-sample timings are excluded from performance claims.
The source-bound build/state/leaf prerequisites remain preserved, including
NaN rejection/recovery, stale-source F32 fallback, staged-health rejection with
zero committed vector parameters and full finite state equality.

The read-only audit verifies Python/helper/binary/source-archive hashes and
compares raw monitor records with 972 normalized worker NVML rows and 66 idle
desktop pmon rows across screen and timing, with no reported activity, unknown
contexts or monitor errors. Sampling does not prove idleness between samples;
the existing terminal-tail assertion cannot be independently reconstructed from
untimestamped raw NVML rows. `audit.json` retains raw monitor/log hashes and
snapshot fingerprints. No source, accepted policy or published bundle changed.
