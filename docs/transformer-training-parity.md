# Progress toward PyTorch training parity

The goal remains complete-update parity across the larger-model suite, with the
existing numerical gates unchanged and the fastest qualified measured PyTorch
control per case. **Parity is not yet achieved.** Benchmark results
below are unprofiled synchronized forward/backward, clipping and Adam updates;
separately labeled diagnostic profiles are excluded from benchmark totals.

The repository includes a [compact machine-readable scorecard](benchmarks/transformer-training-scorecard.json).
Large local captures and duplicate source archives are excluded from Git; see
[benchmark artifact availability](benchmarks/README.md) before following raw-evidence paths.

<a id="current-scorecard-sparse-shadows-v2-20261119"></a>

<a id="current-scorecard-b32-split128-20261119"></a>

## Current scorecard (B32 register statistics 20261119)

Two of eight cases meet the throughput target. The fresh packed-attention
backward policy is accepted opt-in for B8/B16/B32 after positive paired trials
and unchanged numerical gates. B32 additionally uses the qualified 128×64 Split-K
option and register-resident forward statistics after positive complete-update
trials. W1024/S128/B2 accepts vector BF16-first Adam v2 plus sparse BF16 master
shadows after positive paired trials. Shadows increase used pool memory by
188.96875 MiB and sampled whole-process peaks by 192 MiB in both trials. The other four case policies retain their
published V30 or V27 options. Small-parameter accumulation/norm batching remains active in
seven cases and borrowed-BF16 Adam in six. Library defaults remain unchanged.
The retained W1024/S128 compiled control from the earlier vector-v2 campaign
remains the strongest qualified aggregate at 6.7812690 ms. The current shadow
campaign's fresh 7.0643060 ms compiled control is slower. The native candidate
still trails the retained strongest control by about 1.34%.

| Case | Native, ms | Strongest qualified PyTorch, ms | Native / target | Target met |
|---|---:|---:|---:|---|
| w2048_s128 | 20.3518 | 21.8892 | 0.930x | yes |
| w1024_s512 | 9.3434 | 8.0786 | 1.157x | no |
| w512_s512_b8 | 13.8079 | 8.6930 | 1.588x | no |
| w512_s512_b16 | 25.4668 | 15.8961 | 1.602x | no |
| w512_s512_b32 | 50.8735 | 29.4994 | 1.725x | no |
| w512_s128 | 2.6882 | 2.8901 | 0.930x | yes |
| w512_s512 | 5.2097 | 3.7393 | 1.393x | no |
| w1024_s128 | 6.8722 | 6.7813 | 1.013x | no |

Targets are qualified aggregate observations from equivalent configurations and
fixtures, including faster historical controls. They are not confidence bounds.
Component experiments and profiles do not establish complete-update parity.

## B32 register statistics: qualified complete-update opt-in (November 19)

B32 now adds only `--online-attention-register-statistics` to its accepted
Rectangle128x64/packed policy. The library and CLI default remain false;
other seven policies and every strongest historical target are unchanged.
Four selected online forwards retain their running statistics in registers
and export the same F32 slots 0/1 for the original backward; backward delta
slot 2 is unchanged. The original 6 MiB statistics allocation, ownership and
accounting remain present. Unsupported scope, transform, compile or resource
checks trigger a retry of the whole accepted Global forward/backward pair.
If that retry also rejects, the original graph remains available.

Both native arms use the same sealed 402-file source and release executable.
Two fresh reversed trials use 256 warmups and 64 measured synchronized complete
updates per worker:

| B32 mode | Aggregate ms | Trial 0 ms | Trial 1 ms |
| --- | ---: | ---: | ---: |
| Accepted rectangle + Global statistics | 51.487083 | 51.402957 | 51.571210 |
| Register statistics | 50.873489 | 50.823164 | 50.923815 |
| Fresh compiled PyTorch CUDA graph | 30.112859 | 29.804473 | 30.421246 |

Paired reductions are 1.127936% and 1.255343%, with 1.191744% aggregate reduction.
This complete-update observation is not a confidence or causal claim. The
compatible historical 29.499400989 ms control remains the target, with its
original reference and unavailable historical raw-snapshot status preserved.
Native remains about 1.725× that target; parity remains **2/8**. The preceding
51.26829125 ms rectangle observation and older native observations stay in
history; latest accepted observations are not selected as historical minima.

All six timing workers and four runner first-update comparisons pass all six
unchanged numerical gates. Native gradient and weight snapshots match both
arms and the earlier accepted native byte objects exactly; 1,926 compared F64
loss/norm/scale values match exactly. Every native row retains 17 rectangle
programs, four original Split-K fallthroughs, four packed backward regions and
the 6 MiB statistics workspace. The candidate selects four Register forwards;
the baseline selects four Global forwards.

Separate untimed paired math/resource tests compare 408 complete arrays and
96 read-only inputs. The live three-arm Global/Register/forced-resource-Global
V16384 unique-token decoder proof covers `[1,2,1,0]`, full state, NaN rollback
and finite recovery. That synthetic scope selects 16 rectangles and is separate
from measured V4096 fixed PyTorch gates and 17 rectangles. Forced rejection is
an explicit resource-test hook, not a claim that a real driver failure occurred.
Screens and all manual proofs remain excluded from performance totals.

All six paired allocator stages and owned tensor-payload byte deltas are zero.
Candidate-minus-baseline sampled process peaks are **+66 and 0 MiB**. Each sampled process peak is a lower bound on its true high-water mark;
these differences are observations, not allocator deltas or a memory-saving/
capacity claim. Strict existing function resource gates remain in force; register storage
alone does not establish an end-to-end speed explanation. No new profile was
captured, and the retained Square64 profile predates both later B32 options.

The [independent audit](benchmarks/transformer-training-evidence/attention-register-20261119/independent-audit.json)
and [durable evidence](benchmarks/transformer-training-evidence/attention-register-20261119/README.md)
retain full reports, raw guards, helpers, source/build bindings and snapshot
aliases. The corrected default proof verifies the original Global Rust prefix
and current default/explicit Scalar IR/PTX identity; **external historical PTX
goldens were not bound**. A legacy test name and earlier builder wording do not
establish such a comparison. Raw guard data lack query start/end timestamps and
per-query generation-success receipts; normalized reader times and source
checks support sampled closure, not continuous idleness or a guarantee against
sub-sample activity or PID reuse.

This adds one campaign, six timing workers, 384 timed complete updates and four
runner comparisons: **70 campaigns, 1,246 attempts, 1,242 successful workers,
53,312 timed updates and 828 runner comparisons**, plus six historical extra
controls. Two historical dependency failures, two capacity failures, all 15
diagnostic comparisons and two failed NCU attempts retain their prior accounting.
All 69 earlier campaign objects remain unchanged.

## B32 Split-K rectangle: qualified complete-update opt-in (November 19)

The B32 policy changes only
`--split-k-bf16-tile-policy=square64` to
`--split-k-bf16-tile-policy=rectangle128x64`. Library defaults remain Off;
the other seven case policies and every strongest historical target stay
unchanged. Seventeen qualified whole physical-column/row Split-K programs
use a 128×64×32 tile, and four existing fallthroughs retain their original
kernels. Every program keeps its four F32 partial groups, increasing K16
accumulation order and left-associated four-part sum with one final BF16 RN
store. A rejected rectangle retains a complete validated Square64 function
and schedule together.

Both native arms use the same sealed 400-file source and release executable.
Two fresh reversed trials use 256 warmups and 64 measured synchronized complete
updates per worker:

| B32 mode | Aggregate ms | Trial 0 ms | Trial 1 ms |
| --- | ---: | ---: | ---: |
| Square64 baseline | 51.754815 | 51.855210 | 51.654420 |
| Rectangle128x64 | 51.268291 | 51.215329 | 51.321254 |
| Fresh compiled PyTorch CUDA graph | 30.190971 | 30.318527 | 30.063415 |

Paired reductions are 1.23398% and 0.64499%, with 0.940055% aggregate reduction.
This is a complete-update observation, with no confidence or causal claim.
The compatible historical 29.499400989 ms control remains the target; its
original reference paths and unavailable historical raw-snapshot status are
preserved. Native latency remains about 1.738× that target, and parity remains
**2/8**. The old native observation stays in history.

All six timing workers and four runner first-update comparisons pass all six
unchanged numerical gates. Native gradient/weight snapshots match both arms
and the earlier accepted native byte objects exactly; 1,926 compared F64
loss/norm/scale values also match exactly. Separate untimed proofs cover the
three representative tile geometries, actual V4096 census and forced resource
fallback, and V16384 unique-token decoder state through `[1,2,1,0]` accumulation,
NaN rollback and finite recovery. The V16384 proof selects 16 rectangles and
avoids embedding scatter-order ambiguity; it is distinct from the measured
V4096 configuration and its 17 rectangles. Screens and manual proofs add no
performance campaign, worker or update counts.

Actual rectangle functions use 96/98 registers versus 80, 32,768 versus 18,432
shared bytes, no local bytes, and CUDA API block capacity two versus three at
256 threads. Capacity is a resource bound, not achieved occupancy or an
explanation of the measured improvement. All six paired allocator stages and
owned tensor-payload byte deltas are zero. Sampled candidate-minus-baseline
process peaks are **−64 and +66 MiB**, a mixed result that supports no
memory-saving or capacity claim.

The [independent audit](benchmarks/transformer-training-evidence/split128-20261119/independent-audit.json)
and [durable evidence](benchmarks/transformer-training-evidence/split128-20261119/README.md)
retain raw monitoring, reports, source/build bindings, manual proofs and snapshot
aliases. Raw guard files do not persist query start/end or generation-check
success receipts; normalized-reader timestamps and source checks are retained.
Sampled closure cannot establish continuous idleness. CUDA function-handle IDs
were not recorded; fallback proof combines exact label/grid/pointer/resources,
sealed whole-tuple selection source and canonical PTX byte equality.

This adds exactly one campaign, six timing workers, 384 timed complete updates
and four runner comparisons: **69 campaigns, 1,240 attempts, 1,236 successful
workers, 52,928 timed updates and 824 runner comparisons**, plus six historical
extra controls. Two historical dependency failures and two capacity failures
remain; all 15 diagnostic comparisons and two failed NCU attempts stay excluded.
All 68 prior campaign objects are preserved.

## Fresh packed-attention recovery measurements

The target directory was deleted, including unpublished old V31 raw runs.
These results come from a new source-qualified recovery run, rather than a
reconstruction of those lost timings. Both native arms use the same freshly
rebuilt executable and Rust source archive; historical accepted options define
the baseline, and the candidate adds only packed attention backward. Three
campaigns use two fresh processes per mode, reversed order, 256 warmups and
64 timed synchronized complete updates per worker.

| Case | Fresh native baseline, ms | Packed candidate, ms | Paired reduction | Trial reductions |
| --- | ---: | ---: | ---: | ---: |
| w512_s512_b8 | 14.0113 | 13.8079 | 1.45% | 1.64%, 1.26% |
| w512_s512_b16 | 26.3412 | 25.4668 | 3.32% | 4.61%, 2.01% |
| w512_s512_b32 | 52.5717 | 51.5715 | 1.90% | 1.25%, 2.54% |

All 18 workers pass, covering 1,152 measured complete updates and 12 fixed
first-update comparisons. Paired gradient/weight snapshots and optimizer
scalars match exactly. Fresh semantic tests preserve replay, accumulation
intervals `[1,2,1]`, and invalid-update rollback/recovery. A separate forced
resource rejection retains all four old online pairs and four QKV pack kernels.
The packed policy removes four pack launches, keeps its original BF16 rounding
boundaries, and retains the old allocations. B32 semantic composition uses a
V16384 unique-token fixture; the measured V4096 B32 policy keeps borrowed Adam
disabled. These are distinct scopes.

All six paired allocator stages and owned payload bytes are unchanged.
Candidate-minus-baseline sampled process peaks are B8 `[0,-32]`, B16
`[-32,-32]`, and B32 `[+64,+64]` MiB. Sampled peaks are lower bounds; no capacity
or memory-saving claim follows. The explicit desktop-monitoring protocol
retains two known idle 6 MiB desktop contexts and rejects observed activity,
identity drift, unknown CUDA processes, or monitoring errors. It cannot exclude
activity between samples.

The scorecard uses the latest accepted native observations, rather than
historical minima: fresh B8 and B32 absolute times exceed their prior published
observations despite paired improvements. Fresh CUDA-graph controls measured
9.4575, 16.5742 and 30.2836 ms; the stronger compatible historical aggregate
controls remain the targets. No confidence interval or causal explanation of
process variability is claimed.

The packed recovery publication preserved all 62 older campaign objects and added three,
for 65 campaigns, 1,216 attempts, 1,212 successful workers, two historical
pre-CUDA dependency failures and two historical capacity failures. Totals are
51,392 timed complete updates and 808 runner first-update comparisons plus six
historical extra controls; 13 diagnostic comparisons and two failed NCU
attempts remain excluded. The goal remains active at **2/8**.

The [local publication bundle](benchmarks/README.md) embeds
fresh reports and qualification records. The [Rust source archive](benchmarks/transformer-training-sources/parity-candidate-recovery-20261109.tar.gz),
[Python protocol and tests](benchmarks/transformer-training-sources/transformer-python-idle-desktop-guard-v2-recovery20261109.tar.gz),
and [prior published bundle](benchmarks/transformer-training-sources/published-v30-controls-before-recovery20261109.json.gz)
are retained outside the build directory. The subsequent vector-owner publication
preserved all 65 campaign objects and added both vector versions. The current
shadow publication preserves those 67 objects and adds one campaign, for
68 campaigns, 1,234 attempts, 1,230 successful workers, two historical pre-CUDA
dependency failures, two historical capacity failures, 52,544 timed complete
updates and 820 runner comparisons plus six historical extra controls.
The 15 diagnostic comparisons (12 native semantic and three reference replay)
and two failed NCU attempts remain excluded. The preceding B32 profile added two diagnostic comparisons only. The subsequent
B32 rectangle campaign above adds six workers, 384 timed updates and four runner
comparisons; diagnostics remain excluded.

Fresh accepted binaries, fixtures and raw snapshots currently exist for the
three packed cases and the closest shadow-v2 case. Four historical accepted
native binaries/raw snapshot sets remain missing. The new closest strongest
control has raw snapshots; strongest historical control raw sets for the other
seven cases remain missing. Embedded qualified metrics survive. This does not
claim fresh reproduction of the four historical policies or restore deleted
raw evidence. The experimental NN matrix kernel remains unadopted.

## Current accepted B32 diagnostic (November 19)

A separate five-update Nsight capture of the unchanged accepted packed B32
policy passes both fixed first-update comparisons, exact native gradient/weight
snapshots against both trials, and 66 F64 optimizer scalars over the corresponding
first eleven updates. Its ordered model-node/resource mapping passes independent
review. GEMMs total 23.507646 ms and online attention 11.088338 ms per update in
summed diagnostic GPU activity. These are profiler observations, not latency
comparisons or speedup claims, and do not change the scorecard or **2/8** result.

The [profile report](transformer-training-profile.md#current-accepted-b32-diagnostic-november-19)
and [durable raw evidence](benchmarks/transformer-training-evidence/packed-b32-profile-20261119/README.md)
retain trace/SQLite, source/helper provenance, raw F32 gradients/weights and
fixture data with deduplicated aliases. Kernel resources and legacy timestamp
limitations are disclosed. This adds two excluded diagnostic comparisons
(13 to 15); 68 benchmark campaigns, 1,234 attempts, 52,544 timed complete updates
and 820 runner comparisons remain unchanged.

## NN128x64 follow-up: not adopted (November 19)

A fresh B32 complete-update comparison passed all four fixed first-update
comparisons and exact native snapshots/scalars. Candidate reductions were
+0.82% and −0.67% in reversed trials, so the matrix candidate remains disabled.
Allocator stages were identical; sampled process peaks changed by 0/+66 MiB.
The fresh compiled CUDA-graph control measured 29.6868 ms versus 50.6966 ms
for the candidate. The accepted scorecard and 2/8 count remain unchanged.

The [separate experiment report](benchmarks/transformer-training-evidence/nn128x64-20261119/README.md)
and checksummed reports retain this follow-up outside the current 68-campaign bundle
count above. A preceding failed attempt is retained separately: two controls
could not find a C compiler, and one native trial encountered Files GPU
activity. The successful rerun used Python 3.12.16, the repaired compiler and
NVIDIA driver 595.99.02. Raw snapshots and monitor logs remain under the
reported `target/recovery-nn-performance-*` paths.

## Vector Adam owner v1: retained but not adopted (November 19)

The experimental vector optimizer passed the actual W1024/S128/B2 state test,
including replay, accumulation, stale-source fallback and staged-overflow
rollback. The archived screen and two reversed complete-update trials passed
all fixed numerical comparisons with exactly matching native snapshots and
optimizer scalars. Every candidate update selected 20 vector parameters.

Complete-update effects reversed: −3.14% and +4.57% latency reductions. The
aggregate candidate was 7.0749 ms versus 6.8358 ms for the fresh compiled
CUDA-graph control. Allocator stages and payloads matched; sampled process peak
deltas were 0/0 MiB. This first version remains unadopted and the scorecard remains 2/8. The [separate audit](benchmarks/transformer-training-evidence/vector-owner-v1-20261119/README.md)
and reports retain this six-worker, 384-update follow-up. It is now also
embedded in the bundle counts above as an unadopted experiment.

## Vector Adam owner v2: closest-case opt-in (November 19)

A bounded revision caches code-owned CUDA function handles while retaining the
same vector kernel IR and arithmetic. Data pointers remain freshly validated;
no tensor bindings are cached. Fresh state tests preserve replay, `[1,2,1]`
accumulation, full finite state bits, stale-source fallback and invalid-update
rollback/recovery. The unchanged all-pattern leaf proof is hash-bound separately.

Both native arms use the same v2 executable and options; the candidate adds only
`--vector-bf16-first-adam` with borrowed BF16 enabled. Two reversed fresh trials,
256 warmups and 64 synchronized complete updates per worker give:

| W1024/S128/B2 | Aggregate, ms | Trial 0, ms | Trial 1, ms |
| --- | ---: | ---: | ---: |
| Native baseline | 6.9832915 | 6.9799675 | 6.9866155 |
| Vector v2 | 6.9648263 | 6.9574145 | 6.9722380 |
| Compiled PyTorch CUDA-graph SDPA | 6.7812690 | 6.7953685 | 6.7671695 |

Paired reductions are 0.3231% and 0.2058% (aggregate 0.2644%). These small
observations support the closest-case experimental opt-in, with no confidence
or causal claim. The new qualified aggregate PyTorch result replaces the older
6.7961948 ms target; the fastest single trial is never used as the target.
The candidate remains about 2.71% slower, so parity remains **2/8**.

All four fixed first-update comparisons pass; native gradient/weight snapshots
and all qualification, warmup and measured loss/norm/scale values match exactly.
Every measured candidate selects 20 vector parameters. Six allocator stages
and payload bytes match; native sampled process peaks are 2,008 MiB in both
arms (0/0 MiB deltas), versus 1,340 MiB for the reference. These are sampled
whole-process lower bounds, separate from allocator peaks; no memory-saving
or capacity claim follows. Other seven policy argument vectors remain unchanged,
and the library default remains false.

These runs use Python 3.12.16 and driver 595.99.02, versus driver 580.196.07 in
older campaigns. Comparisons within each fresh campaign share their environment;
changes across environments are not attributed to the optimization. The monitor
retains one known idle 6 MiB desktop context and reports no observed violations;
sub-sample activity and the existing terminal-tail evidence limitation remain.

Rust sources are retained as [v1](benchmarks/transformer-training-sources/parity-vector-owner-v1.tar.gz)
and [v2](benchmarks/transformer-training-sources/parity-vector-owner-v2.tar.gz).
The [Python and environment archive](benchmarks/transformer-training-sources/vector-owner-python-environment-20261119.tar.gz)
retains controllers, helpers, fixed validator, build/state audits, kernel proofs
and compiler/runtime provenance. The [prior packed publication](benchmarks/transformer-training-sources/published-packed-recovery-before-vector-owner-20261119.json.gz)
is preserved separately. Both complete timing reports are embedded in the
[local publication bundle](benchmarks/README.md); first-update
screens and state tests remain excluded from performance campaign totals.

## Sparse BF16 shadows v2: closest-case opt-in with memory regression (November 19)

The closest case accepts `--bf16-master-shadows` in addition to its existing
vector/BF16-borrowed options. The library default remains false and the other
seven policies are unchanged. **This improves measured latency while increasing
memory use:** used allocator memory rises by 198,148,096 bytes (188.96875 MiB),
and sampled process peaks rise from 2,008 to 2,200 MiB in both fresh trials.
No memory-saving or capacity claim follows.

Twenty-one whole-matrix casts, covering 54,788,096 elements, become aliases of
resident paired BF16 versions. F32 masters and their consumers remain present.
Every accepted Adam publication stages the exact F32 master and matching BF16
rounding together, preserving revision checks and transactional health rejection.
Tied token/head and multi-microbatch updates retain paired F32 Adam; admitted
single-microbatch updates can use paired scalar or vector BF16 Adam. Unsupported
casts remain ordinary casts. Active typed allocation leases preserve old versions;
inactive captured graphs hold no strong leases. Graph keys use physical address
pairs, with revision and typed identity checked separately.

Both native arms use the same sealed v2 binary and source archive. The candidate
adds only the shadow flag. Two reversed fresh trials, 256 warmups and 64 timed
synchronized complete updates per worker give:

| W1024/S128/B2 | Aggregate, ms | Trial 0, ms | Trial 1, ms |
| --- | ---: | ---: | ---: |
| Vector baseline | 7.0450170 | 7.0152530 | 7.0747810 |
| Sparse-shadow candidate | 6.8722195 | 6.7925060 | 6.9519330 |
| Fresh compiled PyTorch CUDA-graph SDPA | 7.0643060 | 7.0807675 | 7.0478445 |

Paired reductions are 3.1752% and 1.7364%; the aggregate reduction is 2.452762%.
The fresh PyTorch aggregate is slower than the existing strongest qualified
6.7812690 ms vector-v2 control, which remains the target with its original
campaign and raw reference paths. The new native result is 1.3412% slower than
that target. The score remains **2/8**, with no confidence or causal claim.

All six workers and four unchanged fixed first-update comparisons pass. Native
qualification snapshots and all finite qualification/warmup/measured optimizer
scalars match exactly. The captured model drops from 347 to 326 kernels by
removing 21 casts. Replay, `[1,2,1]` accumulation, host mutation, stale-source
fallback, optional-vector fallback, invalid-update rollback/recovery and retained
old readers pass separate untimed state proofs. Current addresses can return
under fresh Arc identities without putting revision numbers into capture keys.

All six allocator stages are retained. The initial stage is empty in both arms;
used current/high memory increases by 198,148,096 bytes in every subsequent
stage. Reserved memory increases by 160 MiB after setup and 192 MiB after
qualification and in steady state. Sampled whole-process peaks are separate
lower bounds, not allocator peaks.

The measured used-pool increase is explained by the recorded storage counts:

```
219,152,384 bytes in two owned BF16 versions
-21,004,288 bytes of physical private BF16 storage removed
=198,148,096 bytes of additional used pool memory
```

Removing the 109,576,192-byte *unpooled logical cast payload* does not remove
that much reusable physical storage. The earlier 104.5 MiB logical net estimate
was not a prediction of allocator or process peaks. Normal timed updates retain
one current and one retired BF16 version (109,576,192 bytes each); the independent
old-reader/rejection state proof reaches three versions. Retained F32 counters
are bookkeeping of existing paired versions, not another measured F32 allocation
increase.

The first source-qualified integration and its instrumented reproduction failed
an overstrict source-node-label check during construction. The real F32 region
used a local graph namespace while its current master pointer and extent matched.
The fix compares the runtime dtype, source/logical shapes, exact view chain and
normalized map to the graph-declared cast, retaining storage/revision/Weak identity,
context and full-allocation checks. A same-shape square alternate bijection is
explicitly rejected. These two semantic failures, the passing component/census/
state proofs and the one-sample screen are excluded from timed campaign counts.

The [v2 Rust archive](benchmarks/transformer-training-sources/shadow-owner-v2-20261119.tar.gz),
[Python/QA/environment archive](benchmarks/transformer-training-sources/shadow-owner-python-environment-20261119.tar.gz)
and [prior 67-campaign publication](benchmarks/transformer-training-sources/published-vector-owner-before-shadow-owner-20261119.json.gz)
are retained outside `target/`. The original failed integration source archive
and [instrumented failed source](benchmarks/transformer-training-sources/shadow-owner-diagnostic-v2-source-20261119.tar.gz)
remain identified in the bundle. Timing uses the November 19 environment and
one explicitly monitored idle 6 MiB desktop context. No observed monitor violations
occurred; sub-sample activity and existing terminal-tail evidence limitations remain.

## Compact optimizer health checks

The first accepted implementation replaces one health value per parameter
element with one zero/NaN partial per logical 256-element group. Current/staged
weights and moments remain separate; nonfinite updates are rejected before
publication. Standalone 1024-thread and batched 256-thread launches retain their
original geometry. Partial blocks execute neutral lanes through all collectives.

Two fresh processes per mode, eight warmups and 32 timed updates per process:

| Case | Retained native baseline, ms | Compact health, ms | Throughput gain | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: | ---: |
| 211M / S128 / B1 | 40.234 | 37.540 | 1.072x | 22.476 |
| 55M / S512 / B1 | 18.904 | 18.114 | 1.044x | 12.176 |

Mode order reverses in trial 1. The 16 workers recorded 512 measured updates.
All native updates retain replay. All 12 first-step comparisons against the
retained native baseline, matched eager PyTorch and compiled SDPA pass the fixed
gates. Native before/after full gradient and updated-weight snapshots are
SHA256-identical in both trials of both configurations.

For 211M, exact steady and lifecycle pool USED_HIGH decrease by **842,111,300
bytes (803.10 MiB)**. Pool reserved highs decrease by 832 MiB. Sampled whole-process
NVML peaks decrease from 8,886 to 8,054 MiB; those sampled peaks remain lower
bounds, and differ in scope from exact pool counters. The 55M steady pool saving
is 220,842,400 bytes; its lifecycle saving is 220,626,944 bytes.

This is a useful improvement, but native still takes 1.67x and 1.49x the
corresponding compiled PyTorch latency in this campaign.

## Cast and softmax experiments

Each experiment uses the health-only binary as its native control. Two processes
per mode, eight warmups, 32 timed updates, reversed process order:

| Experiment / case | Native control, ms | Candidate, ms | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: |
| Physical-order casts / 211M S128 B1 | 37.595 | 34.199 | 22.564 |
| Physical-order casts / 55M S512 B1 | 18.165 | 17.498 | 10.768 |
| Compact softmax / 55M S512 B1 | 18.031 | 17.824 | 11.977 |
| Compact softmax / 15M S512 B8 | 40.584 | 39.802 | 10.461 |

Physical-order casts narrow complete physical storage contiguously, then retain
proven bijective view metadata. The experiment rejects broadcasts, masks, OOB
maps and unproven access chains. All 21 large weight casts use it. It is default
off; both tested configurations pass all fixed gates against compiled PyTorch
and the health-only control. First-step native gradient/weight hashes also match
the control in both trials. The sampled process peaks are unchanged at 8,054 MiB
(211M) and 3,536 MiB (55M).

Compact softmax backward operates only in locally f32 regions; surrounding BF16
boundaries remain intact. Both configurations pass all fixed gates. This is an
algebraic rounding change: versus the native control, first-step clipped-gradient
relative L2 is about 0.0043–0.0046 and update-vector relative L2 about 0.041–0.051.
It provides only 1–2% throughput gains in this screen and remains default off.
It is not memory-linear attention. At B8 the sampled process peak drops from
7,216 to 6,960 MiB, while compiled PyTorch peaks at 1,172 MiB.

Across these three campaigns, all 40 workers succeed, recording 1,280 timed
updates and 28 passing first-step comparisons. Compiled 55M controls vary
substantially: individual process medians range from 9.66 to 12.57 ms. One run
settles during measurement, while others fluctuate without a consistent trend.
Software/configuration match and no foreign compute was observed; the cause is
unresolved. Treat these as descriptive screens, not tight confidence bounds.
Future confirmation uses longer fixed warmups and more independent processes.

## Further work in progress

First-accumulation overwrite avoids clearing gradients and reading zeros before
the first microbatch. Default off; it passes 26 real-device resident tests
including microbatch sums, empty steps, dirty-range failure recovery, groups,
padding and replay, plus strict Clippy. Two fresh processes per mode, 32 warmups
and 64 timed updates per process compare against the retained physical-cast
control:

| Case | Physical-cast control, ms | With first-overwrite, ms | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: |
| 211M S128 B1 | 34.242 | 32.269 | 22.549 |
| 55M S512 B1 | 17.530 | 17.128 | 11.676 |

All eight first-step comparisons pass fixed gates; native gradient and weight
hashes match the control in both trials. Timed native loss/norm/scale observations
also match. Exact pool measurements are unchanged: this saves traffic, not
allocation. The 55M sampled NVML peak is lower, but unchanged pool counters do
not establish a true memory saving. Recorded kernel-launch counters also remain
unchanged. These four campaigns total 52 successful workers and 2,048 timed
updates; they still establish no PyTorch parity.

The batch-8 diagnostic trace after casts, first-overwrite and compact softmax
attributes 39.4% of summed kernel time to matmul, 12.5% to reductions, 6.7% to
materialization and 38.7% to other kernels. Adam plus accumulation are only 2.6%.
These prefix-based families are not yet a semantic attention/FFN attribution.
The diagnostic first-step snapshot passes the same gates against compiled
PyTorch; its profiled timings are excluded from throughput comparisons.

The existing opt-in GEMM shared-memory reuse is integrated into mixed training.
Its independent campaign uses the retained cast/first-overwrite binary as the
control, with 32 warmups and 64 measured updates, two processes per mode:

| Case | Control, ms | Shared reuse, ms | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: |
| 211M S128 B1 | 32.279 | 32.056 | 22.613 |
| 55M S512 B1 | 17.117 | 17.034 | 10.498 |
| 15M S512 B8 | 40.223 | 39.813 | 10.724 |

All 12 first-step comparisons pass the fixed gates. Gains are only about
0.5–1.0% in complete updates, despite larger isolated GEMM improvements. The
option remains off by default. This campaign adds 18 workers and 1,152 measured
updates; the five campaigns total 70 workers and 3,200 timed updates.

Incremental reuse during compilation is now qualified. Previously the entire
unpooled payload existed before final compaction, causing setup OOM where
steady storage could fit. The opt-in prefix policy retains future aliases and
pinned sources/sinks while reusing dead allocations at completed emission
boundaries. Two processes per mode, 32 warmups and 32 measured updates:

| Batch, W512/S512 | Old native, ms | Prefix reuse, ms | Compiled PyTorch, ms | Native process peak, old → new MiB |
| --- | ---: | ---: | ---: | ---: |
| 8 | 39.861 | 39.864 | 10.287 | 7,344 → 3,216 |
| 16 | 82.290 | 82.102 | 17.572 | 14,096 → 5,678 |
| 32 | CUDA OOM twice | 174.146 | 33.526 | unavailable → 10,896 |

Batch 32 completes both attempts and passes all fixed gates against compiled
PyTorch. Its all-requested-reference qualification field remains false because
the old native baseline OOMs and provides no snapshot to compare; the compiled
reference's two validations both pass. Batch 8/16 also qualify against the
retained native baseline. No thresholds changed. This establishes a tested
batch-32 native fit, not maximum capacity or throughput parity.

The first trial's pool lifecycle USED_HIGH drops from 7,447,878,516 to
3,141,316,980 bytes at B8, and from 14,560,009,588 to 5,800,653,172 bytes at B16.
These allocator peaks differ in scope from sampled process peaks. Steady
storage changes modestly; the principal saving removes construction overlap.
Measured mixed compilation time is 1.02 → 1.16 s at B8 and 2.14 → 1.44 s at B16
in the first trial; these are cold observations, not isolated compiler speedups.

This adds 18 attempts, including two baseline OOMs and 512 measured updates.
The six campaigns total 88 attempts, 86 successful workers, 3,712 timed updates
and 58 passing first-step benchmark comparisons. Prefix reuse also passes 53
mixed/resident CUDA tests and four pure planner tests, with strict Clippy.

## Isolated narrow GEMM and attention-normalization fusion

V5 adds two independent campaigns. Both use two fresh processes per mode,
32 warmups and 32 measured complete updates, reversing process order in trial 1.
The narrow-tile candidate adds only `--narrow-bf16-tile` to the V4 control's
physical-order casts, first-overwrite, shared-memory reuse and incremental
allocation reuse. Compact softmax and attention fusion are off in this campaign.

| Narrow BF16 tile / case | Native control, ms | Candidate, ms | Throughput gain | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: | ---: |
| 211M S128 B1 | 32.086 | 31.994 | 1.003x | 22.361 |
| 55M S512 B1 | 17.044 | 16.692 | 1.021x | 11.838 |
| 15M S512 B8 | 39.826 | 37.859 | 1.052x | 10.817 |

The eligible narrow N64 BF16 GEMM schedule provides small gains on the larger
models and a larger 5.2% gain on B8. All 18 processes succeed; all 12 first-step
comparisons pass. The candidate/control global clipped-gradient, postweight and
update relative L2 errors are exactly zero in both trials of every case.
Model-kernel counts, complete-step launch counts and exact retained pool payloads
are unchanged. The sampled B8 process peak changes from 3,216 to 3,190 MiB,
but identical exact pool counters do not establish an allocation saving.

The fusion campaign adds only `--attention-softmax-fusion` to a V5 control with
all four V4 options **and compact softmax backward enabled**. Narrow tiles are
off in both arms. Thus it isolates forward normalization/probability-cast fusion
from the already-selected backward rewrite; it is not a combined narrow/fusion
measurement.

| Fused attention normalization / case | Native control, ms | Candidate, ms | Throughput gain | Compiled PyTorch, ms | Sampled native process peak, old → new MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 55M S512 B1 | 16.797 | 16.129 | 1.041x | 11.260 | 2,614 → 2,580 |
| 15M S512 B8 | 38.976 | 34.308 | 1.136x | 11.183 | 2,966 → 2,800 |
| 15M S512 B16 | 80.220 | 70.169 | 1.143x | 18.019 | 5,200 → 4,816 |
| 15M S512 B32 | 170.643 | 149.598 | 1.141x | 33.626 | 9,614 → 9,104 |

Every candidate records four fused attention regions; every control records
zero. Model kernels decrease by 28 per update: 582→554 at W1024, 630→602 at
B8, 646→618 at B16 and 647→619 at B32. Complete-step launch counts likewise
fall by 28, to 645, 693, 709 and 710, respectively. All measured native steps
retain graph replay. These are closed normalization regions that retain BF16
probabilities and f32 values needed by backward, not memory-linear attention.

The fixed limits remain normalized loss error ≤0.005, clipped-gradient relative
L2 ≤0.02, postweight relative L2 ≤0.001 and update relative L2 ≤0.15, with finite
values and native-own-gradient Adam tolerances of 2e-7 absolute and 2e-6 relative.
All 24 fusion workers succeed and all 16 first-step comparisons pass unchanged
gates. Candidate/control gradient, postweight and update relative L2 errors are
exactly zero in all eight comparisons. Against compiled SDPA, gradient relative
L2 spans 0.0163941969–0.0168099249, postweight relative L2 spans
0.0003657753–0.0004740724, and update relative L2 spans
0.1375231377–0.1439700219. The update gate remains 0.15; these results do not
justify relaxing it. Both B32 candidates now qualify against both native and
compiled controls, unlike the earlier prefix-reuse campaign whose old control
OOMed.

At B32, exact lifecycle pool USED_HIGH decreases from 9,957,437,812 to
9,319,903,604 bytes, and steady USED_HIGH from 9,471,802,084 to 8,834,267,876
bytes: a 637,534,208-byte (608 MiB) reduction in each scope. The sampled
whole-process peaks above are lower bounds and include memory outside this
pool. The candidate retains 5,843,956,044 f32 bytes and 2,563,295,232 BF16 bytes;
its 22,701,418,828-byte unpooled counter is cumulative conceptual payload,
not simultaneous allocated memory. Setup, qualification snapshots and profiling
are excluded from throughput timing. Only two process repetitions were used;
compiler/reference variation is visible, so cross-campaign timing differences
are not isolated effects.

The two V5 campaigns add 42 successful workers, 1,344 measured updates and
28 passing first-step comparisons. All eight campaigns total **130 attempts,
128 successful workers, two historical baseline OOMs, 5,056 timed complete
updates and 86 passing benchmark comparisons**. One separate diagnostic-profile
qualification is excluded from those comparison totals. V5 also passes five
targeted real-device CUDA tests (signature identity, narrow tiles, fused B2/H2,
fused compact training and nonfinite probabilities), strict Clippy and formatting;
this is not a full-suite test claim. All new options remain off by default.

## Combined-policy control and transposed-output no-op

The V6 campaign directly compares the combined V5 policy with compiled PyTorch
on four configurations. Both native arms enable physical-order casts,
first-overwrite, shared-memory reuse, incremental allocation reuse, compact
softmax backward, narrow BF16 tiles and attention-normalization fusion. Each
mode uses two fresh processes, 32 warmups and 32 measured updates; trial 1
reverses process order. The table uses the retained **V5 baseline arm**, so it
attributes no benefit to the proposed output transpose.

| Combined policy / case | V5 native, ms | V5 tokens/s | Compiled PyTorch, ms | Native / compiled latency |
| --- | ---: | ---: | ---: | ---: |
| 211M S128 B1 | 31.956 | 4,005 | 22.429 | 1.425x |
| 55M S512 B1 | 15.852 | 32,299 | 10.847 | 1.461x |
| 15M S512 B8 | 32.455 | 126,207 | 10.268 | 3.161x |
| 15M S512 B32 | 140.947 | 116,243 | 33.737 | 4.178x |

The V6 candidate adds `--transposed-bf16-outputs`, but its actual transformed
program counter is **zero in all eight qualification steps and all 256 candidate
measured updates**. Autograd inserts a same-dtype BF16 cast between the weight
GEMM and its f32 promotion; V6 required an immediate f32 promotion and missed
that real graph pattern. This is retained as a **matcher miss and no-op
experiment**, not a successful transpose optimization. Candidate medians are
32.034, 15.847, 32.495 and 141.026 ms, respectively; these small differences
provide no evidence of a transpose gain.

All 24 workers pass, all 16 first-step comparisons against the retained native
and compiled controls pass the unchanged gates, and all measured native updates
retain replay. Candidate/control loss, gradient, postweight and update errors
are exactly zero. Exact setup, lifecycle and steady pool counters match between
native arms in both trials; differing sampled NVML peaks establish no memory
saving. The combined policy records four fused attention regions for S512 and
zero for S128, where the existing shape guard excludes fusion. This campaign
qualifies the measured combined policy on these cases, not an active output
transformation. The revised matcher requires its own structural, numerical and
complete-update qualification.

The ninth campaign adds 768 measured updates and 16 passing comparisons. The
bundle now contains **154 worker attempts: 152 passed, two historical baseline
OOMs, 5,824 timed complete updates and 102 passing first-step benchmark
comparisons**. Two separate diagnostic-profile comparisons remain excluded.
Two process repetitions provide descriptive evidence; they do not establish
parity or tight confidence bounds.

## Current 211M optimizer profile

A separate V6 diagnostic records five profiled W2048/S128/B1 complete updates
after 32 warmups. Its first step passes every unchanged gate against compiled
SDPA: clipped-gradient relative L2 is 0.0161417, postweight relative L2 is
0.000633211 and update relative L2 is 0.133097. These five samples are excluded
from benchmark worker/timed-update totals; this adds the second diagnostic-only
comparison; these diagnostic samples add no benchmark comparisons.

| Name-prefix kernel family | Summed activity per update, ms | Share of summed kernel activity |
| --- | ---: | ---: |
| Adam | 9.874 | 33.77% |
| GEMM | 8.448 | 28.89% |
| Gradient accumulation | 5.179 | 17.71% |

The trace contains 3,420 kernels: 684 per update and 29.240 ms of summed kernel
activity per update. Reported resident counters record 683; the extra cached
loss-readback materialization is outside that counter and costs 0.001044 ms per
update. It remains included in complete-update wall timing. The profiled wall
mean is 32.416 ms, which is diagnostic rather than an unprofiled benchmark.

Standalone Adam accounts for 9.863 ms across 34 launches per update, versus
0.0103 ms for four batched launches. Its observed configuration is 1,024
threads, 26 registers/thread, 128 static shared bytes and zero local bytes.
Device-limit arithmetic permits at most one such block per SM, or 32 of 48
warps. A hypothetical 256-thread launch at the same register count permits a
six-block/48-warp ceiling. These are resource ceilings, **not measured achieved
occupancy**, and do not establish a 256-thread speedup. They motivate a separate
scheduling experiment with unchanged update arithmetic, health and rollback.

Device-to-device master refresh adds 844,591,104 copied bytes per update
(211,147,776 f32 parameters), with 2.794 ms of summed copy activity. Copy and
kernel activity sums are separate; they are not a critical-path decomposition
or additive wall-time prediction. Gradient accumulation remains a substantial
independent target. Prefix-family classifications are syntactic, not a semantic
attention/FFN mapping, and this profile had no accompanying NVML contention
sampler.

The bundle embeds bounded SQL queries, derived resource/activity tables,
complete result and first-step validation, plus trace hashes and V6 source/binary
association. The raw SQLite/Nsight traces remain under
`target/transformer-training/parity-profile-v6-211m`; their bytes are not embedded.

## Adam launch geometry: neutral performance result

V7 tests 256 threads instead of 1,024 for standalone Adam launches. Generated
arithmetic, 256-element health groups, staged moments/weights, transactional
publication and existing batched 256-thread launches remain unchanged. The
candidate adds only `--adam-256-threads` to the V6 control's combined casts,
first-overwrite, shared/incremental reuse, compact backward, forward attention
fusion and narrow-tile options. Transposed-output and attention-backward fusion
options are off in both arms. All new options remain off by default.

Two processes per mode, 32 warmups and 32 measured updates, reversed trial order:

| Case | Native control, ms | Adam256, ms | Compiled PyTorch, ms | Native sampled peak, both arms MiB |
| --- | ---: | ---: | ---: | ---: |
| 211M S128 B1 | 31.926 | 31.839 | 22.492 | 7,446 |
| 55M S512 B1 | 15.794 | 15.822 | 11.312 | 2,580 |

This is a neutral performance experiment: the small changes do not establish a
useful complete-update improvement. The resource ceiling in the earlier profile
is not an achieved-occupancy measurement, and a higher theoretical ceiling did
not establish a throughput gain. The scheduling option remains experimental.

All 12 workers pass and all eight first-step comparisons pass the unchanged
fixed gates. Against the native control, gradient, postweight and update relative
L2 errors are exactly zero in all four comparisons. Model counts remain 581/554
kernels and reported complete-step launch counts remain 683/645 for 211M/55M;
retained tensor-payload counters match. All measured native steps retain replay.

V7 qualification records 172 CPU library passes, 370 real-device CUDA library
passes with five existing ignored tests, 16 focused passes, strict Clippy and
formatting. The initial broader CUDA attempt stopped after 113 passes and one
stale test failure: its assertion expected retry after failed accumulation.
The test was corrected to assert interval invalidation and recovery through
`begin_step`; the subsequent broad run passed 370 tests. Production failure
semantics were preserved. The CPU-library and focused counts are scoped counts,
not an all-target test-suite claim.

The tenth campaign adds 384 measured updates and eight passing comparisons.
The bundle totals **166 worker attempts: 164 passed, two historical baseline
OOMs, 6,208 timed complete updates and 110 passing benchmark comparisons**.
Two diagnostic-profile comparisons remain excluded from these totals. PyTorch
complete-update parity remains unachieved.

## Active transposed BF16 outputs

V7 corrects the V6 matcher miss by recognizing the same-dtype BF16 identity
casts inserted by autograd. Closed rank-two BF16 GEMM outputs are computed in
transposed physical storage while preserving their logical views and BF16
rounding. The candidate adds only `--transposed-bf16-outputs` to the retained
V6 combined policy. Adam256 and attention-backward row fusion are off in both
arms. Two processes per mode use 32 warmups and 32 measured complete updates;
trial 1 reverses process order.

| Case | Native control, ms | Active transpose, ms | Throughput gain | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: | ---: |
| 211M S128 B1 | 31.941 | 29.086 | 1.098x | 22.630 |
| 55M S512 B1 | 15.797 | 14.856 | 1.063x | 11.411 |
| 15M S512 B8 | 32.386 | 32.088 | 1.009x | 10.304 |

The transformed-program counts are **21, 21 and 17**, respectively, in both
qualification steps and every candidate measured update. Controls record zero.
This is an active transformation, unlike the preserved V6 no-op campaign.
All 18 workers succeed and all 12 first-step comparisons pass unchanged global
gates. Against the native control, normalized loss error and gradient, postweight
and update relative L2 errors are exactly zero in all six comparisons. Against
compiled SDPA, gradient relative L2 spans 0.0161417313–0.0167991530,
postweight relative L2 0.0003663690–0.0006332113, and update relative L2
0.1330974331–0.1439499122. The update limit remains 0.15.

All native measured steps retain replay. Model kernels remain 581/554/602 and
reported complete-step launches remain 683/645/693. Retained f32/BF16 payload,
conceptual unpooled payload, and lifecycle/steady pool counters match between
arms in both trials. Sampled native process peaks are unchanged at 7,446 and
2,580 MiB for the larger cases; B8 changes from 2,774 to 2,800 MiB, which is not
an exact allocation regression given unchanged pool counters. The main observed
gain is at 211M/55M; the B8 gain is small. The option remains default off.

## Compact attention backward row fusion

A separate V7 campaign adds only `--attention-softmax-backward-fusion` to the
same retained V6 combined policy. Active output transpose and Adam256 are off
in both arms. The closed S512 compact VJP keeps the existing f32 probabilities
and promoted BF16 incoming gradient, reduces `sum(P*dP)` with the existing
128-thread tree, then evaluates `(P*(dP-dot))*0.125` before the BF16 dScore store.
It emits no full-square f32 dScore payload. Unsupported shapes, dtype boundaries,
views and escaping intermediates retain the existing path; nonfinite arithmetic
is not replaced by zero/mask shortcuts.

| Case | Native control, ms | Backward row fusion, ms | Throughput gain | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: | ---: |
| 55M S512 B1 | 15.756 | 15.703 | 1.003x | 11.202 |
| 15M S512 B8 | 32.349 | 30.913 | 1.046x | 10.741 |
| 15M S512 B32 | 140.745 | 134.558 | 1.046x | 33.302 |

W1024 is a near-neutral result, with overlapping process medians. B8 and B32
show about 4.6% throughput gains. Each candidate records **four backward fused
regions** in qualification and every measured update; controls record zero.
Model kernels decrease by eight, 554→546, 602→594 and 619→611. Reported
complete-step launches decrease to 637/685/702. All measured native steps retain
replay. This still retains quadratic probability storage and is not fused
memory-linear attention.

All 18 workers succeed and all 12 first-step comparisons pass unchanged global
gates. Native-control normalized loss, clipped-gradient, postweight and update
errors are exactly zero in all six comparisons. Compiled-SDPA gradient relative
L2 spans 0.0164133061–0.0168202392, postweight relative L2
0.0003657237–0.0004741147, and update relative L2 0.1375641319–0.1439312495.
The fixed 0.15 update gate has not changed. Separate focused tests document that
surrounding automatic f32 fusion can contract a stored product/add into FFMA;
forward probabilities remain exact and numerical qualification is still required.

At W1024, retained payload and lifecycle/steady pool USED_HIGH fall by exactly
16 MiB; pool reservation and sampled process peak remain unchanged. At B8/B32,
retained payload and exact pool counters remain unchanged despite removing
intermediates, because other live ranges still determine the retained slots.
Conceptual unpooled payload falls by 64.125/256.5/1,026 MiB across the three
cases; these are cumulative construction counters, not simultaneous allocation
or measured peak savings. Sampled process peaks are 2,580→2,580, 2,800→2,774
and 9,134→9,104 MiB; each sample maximum remains a lower bound.

These two independent campaigns add 36 successful workers, 1,152 timed updates
and 24 passing comparisons. Recomputed from all raw records, the twelve-campaign
bundle contains **202 worker attempts: 200 passed, two historical baseline
OOMs, 7,360 timed complete updates and 134 passing first-step benchmark
comparisons**. Two diagnostic-profile comparisons remain excluded. Both options
remain default off. These V7 campaigns did not measure the new transpose and
backward-row options together; their gains must not be multiplied. The V8
control below supplies a separate combined-policy measurement. Complete-update
PyTorch parity remains unachieved.

## Eight computing warps: neutral complete-update result

V8 repartitions an already-selected BF16 64×64×32 tile from four computing
warps with four accumulator fragments each to eight computing warps with two
fragments each. Block size, tile dimensions, operand layouts, selected pipeline
and resource guards remain unchanged. The retained V7 control enables the
corrected transpose and attention-backward options together with the preceding
policy; V8 adds only `--bf16-eight-compute-warps`. Two processes per mode each
exclude 32 warmups and record 32 complete updates, reversing order in trial 1.

| Case | V7 control, ms | Eight-warps candidate, ms | Control / candidate throughput ratio | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: | ---: |
| 211M S128 B1 | 29.035 | 29.005 | 1.001x | 22.439 |
| 55M S512 B1 | 14.573 | 14.660 | 0.994x | 11.669 |
| 15M S512 B8 | 30.559 | 30.423 | 1.004x | 10.962 |
| 15M S512 B32 | 134.599 | 133.741 | 1.006x | 33.849 |

The full-update changes are neutral or small: the 55M case regresses by 0.60%
in latency, while other latency reductions are about 0.10–0.64%. This does not
justify enabling the option by default. The actual compiled-program counter is
21, 13, 43 and 42 for the four cases, respectively, in both qualification trials
and every measured candidate step; this is an active scheduling experiment.
Computing-warp ownership is compiler metadata, not measured occupancy or
hardware utilization.

All 24 workers and all 16 first-step comparisons pass unchanged fixed gates.
Candidate/control clipped-gradient, postweight and update relative L2 errors are
exactly zero in every comparison. Exact setup, lifecycle and steady pool counters
match in both trials for every case. Lower sampled NVML peaks on some cases do
not establish an allocation saving. Five focused tests pass: three offline
schedule/codegen/signature tests and two real-device CUDA tests covering exact
finite outputs and gradients, replay, activation, fallback and option toggles.
All 16 flag combinations have distinct compilation identities under signature
version 40; strict Clippy and formatting pass. An initial test expected one
Linear program where the graph retained two; its expected count was corrected
from graph operations without a production change.

This campaign adds 768 updates and 16 comparisons. The thirteen-campaign total
is **226 attempts, 224 successful workers, two historical baseline OOMs, 8,128
timed complete updates and 150 passing first-step benchmark comparisons**,
with two diagnostic-profile comparisons excluded. The V7 control here directly
qualifies corrected transpose and attention backward together on these four
cases; their separate campaign gains still must not be multiplied. Native
remains slower than compiled PyTorch, and eight-warps stays default off.

## Direct resident masters: eight-case campaign

V9 adds default-off `--direct-master-bindings`. It removes the mixed executor's
private f32 master copies, reading immutable resident allocations directly.
Active leases preserve the last uploaded source snapshot, including reads after
Adam publishes a newer version. On the next complete refresh, retired leases
return to the optimizer for exclusive staging reuse. Existing readers or another
executor can prevent reclamation; the safe allocation fallback remains. Inactive
capture variants own no masters. Validation and new capture construction precede
refresh commit; failed refresh or Adam health checks preserve committed state.

The isolated candidate adds only direct bindings to the retained V8 control's
combined policy. Both arms use physical-order casts, first-overwrite,
shared/incremental reuse, compact softmax backward, forward/backward attention
fusion, narrow tiles and eligible transposed outputs. Adam256 and eight computing
warps are off. Two fresh processes per mode use 32 warmups and 32 measured
complete updates, reversing order in trial 1:

| Case | Copied-master control, ms | Direct masters, ms | Compiled SDPA, ms | Sampled native process peak, old → new MiB |
| --- | ---: | ---: | ---: | ---: |
| 211M S128 B1 | 29.086 | 26.436 | 22.230 | 7,446 → 6,650 |
| 55M S512 B1 | 14.639 | 14.038 | 11.752 | 2,580 → 2,360 |
| 15M S512 B8 | 30.595 | 30.601 | 11.092 | 2,800 → 2,716 |
| 15M S512 B16 | 62.740 | 62.925 | 17.895 | 4,912 → 4,880 |
| 15M S512 B32 | 134.580 | 134.881 | 33.836 | 9,006 → 9,038 |
| W512 S128 B2 | 4.178 | 3.937 | 11.416 | 886 → 828 |
| W512 S512 B2 | 9.327 | 9.198 | 11.033 | 1,270 → 1,212 |
| W1024 S128 B2 | 9.847 | 9.211 | 11.804 | 2,294 → 2,076 |

The 211M and 55M/S512 cases improve by about 10.0% and 4.3% in throughput.
B8/B16/B32 latency is neutral; this campaign establishes no improvement there.
Three smaller native configurations beat compiled SDPA in this screen. That is
not final parity: the larger cases remain slower, required older eager/matched
controls still need fresh confirmation, and only two process repetitions were
used. The B32 native/compiled latency ratio remains about 3.99x.

All 48 workers and all 32 unchanged fixed-gate comparisons pass. Both trials of
all eight configurations have byte-identical first-step native-control clipped
gradient and postweight files (32 SHA256 pair comparisons). All 65-step
qualification/warmup/sample loss, norm and scale trajectories are also bitwise
identical in each of the 16 native trial pairs. The audit preserves the hashes
and exact per-trial counters; compiled-reference comparisons pass numerical
tolerances rather than byte identity.

Every candidate trial records 60 unique directly bound masters. Every measured
step records zero parameter refresh copies/bytes, two capture misses, two cached
variants and active replay; hits increase from 32 to 63. The miss counter includes
the initial preparation capture. There is no measured fallback, miss growth or
partial binding in any of the 16 candidate processes.

Owned f32 payload and exact lifecycle/steady pool USED_HIGH decrease by precisely
one master array in both trials of every case: **844,591,104 bytes (805.46 MiB)**
for 211M, 221,493,248 bytes for 55M/S512, 60,940,288 bytes for the S512/W512
batches, 60,153,856 bytes for W512/S128 and 219,920,384 bytes for W1024/S128.
Borrowed master bytes are reported separately from owned f32 payload; the
resident owner still owns those masters. Sampled whole-process peaks above are
lower bounds and include context/modules/capture resources outside the queried
pool. In particular, B32's sampled peak rises despite its exact pool saving;
there is no B32 whole-process memory-improvement claim.

Prequalification covers 379 tests across the CUDA-feature library and three
selected integration files, with five existing tests skipped. Five focused
cases also pass serially, including captured/uncaptured execution, six updates
with two executors, stale source snapshots across publication, retained old
readers, bounded capture rebuilding, delayed staging reclamation, failed-capture
rollback and nonfinite-update recovery. Strict Clippy and formatting pass.
The bundle also retains the separate W64/S16/B2/L1 debug smoke: its 18 masters,
254,208-byte pool/payload saving and zero-copy/two-variant replay are correctness
evidence, not performance samples or an exact release-binary attestation.

The V9 source archive SHA256 is
`d5bfed9342002b96a349df418b8e7a54dbc7e9d770e6219c0d9588902a002f0f`,
and retained release binary SHA256 is
`a9e4bad2f92aaaa4d7cebeca67a65cd60f63707d8e300be335436fd77ccd6b5a`.
The complete raw report is embedded as `direct_masters`, with baseline source V8
and per-trial audit hashes/counters. Later source changes are not measurement
evidence for this snapshot. The campaign adds 1,536 timed updates and 32 passing
comparisons. Completed benchmark totals are **14 campaigns, 274 worker attempts,
272 passed, two historical baseline OOMs, 9,664 timed complete updates and
182 passing first-step comparisons**, excluding three separately bundled
diagnostic-profile comparisons. All new options remain off by default and
complete-update parity remains unachieved.

## Current B32 profile: model work dominates

A separate V9 diagnostic records five profiled W512/S512/B32 updates after
32 warmups. It passes the unchanged compiled-reference first-step gates. Its
five updates are excluded from campaign throughput and worker totals; this is
the third separately bundled diagnostic qualification.

The trace contains 3,515 kernel instances and 134.847 ms of summed kernel
activity per update; profiled complete-update wall mean is 136.408 ms. The
captured model contributes 133.728 ms. Its 611 labels align across all five
repetitions, with identical kernel names, grids, blocks and static resources
across the two direct-master address variants. The audit also validates 387
label-family prefix matches. No model kernel reports local-memory bytes.

| Disjoint shape-based model domain | Summed activity per update, ms | Share of captured model activity |
| --- | ---: | ---: |
| FFN | 36.465 | 27.27% |
| Attention core | 34.207 | 25.58% |
| F32 width-sized normalization/residual spine | 25.476 | 19.05% |
| QKV projection and gradient packing | 20.398 | 15.25% |
| Head, embedding and loss | 9.697 | 7.25% |
| Unambiguous attention-output projection | 1.159 | 0.87% |
| Explicitly ambiguous ownership | 6.327 | 4.73% |

These assignments use exact shapes and the decoder contract; width-only
contribution additions are grouped with normalization/residual and must not be
read as a pure LayerNorm measurement. The explicit ambiguous category is retained.
The detailed mapping, grouping rules, query/source hashes and per-kernel resources
are preserved in `profile_v9_b32` in the evidence bundle.

Across the complete update, ordinary GEMMs cost 38.355 ms, split-K partials
11.690 ms and their reductions 0.183 ms. The largest grouped attention GEMMs
cost 8.530 ms across eight N64-output products, using 54 registers/thread and
256-thread blocks. Other listed GEMMs use 46–92 registers/thread with zero
reported local memory. These resource counts alone establish neither achieved
occupancy nor a scheduling improvement.

The rest is substantial: region pointwise work costs 20.809 ms, unfused
pointwise kernels 14.409 ms, ordinary reductions 10.619 ms, layout/contiguous
copies 9.383 ms, explicit cast materializations 7.644 ms, broadcast materialization
4.765 ms and concat materialization 3.843 ms. Fused attention normalization costs
3.368 ms forward and 3.612 ms backward. Head indexed-loss forward/backward cost
4.969 ms, including 3.466 ms for its backward kernel. These are disjoint activity
families, not independent achievable savings.

Adam costs only 0.651 ms and owner gradient accumulation 0.368 ms here, together
about 0.76% of kernel activity. The trace records no device-to-device copies.
Consequently the remaining B32 native/compiled gap points to model computation
and avoidable memory traffic, including QKV gradient padding/concat and attention
layouts, rather than further owner-launch capture work. A closed QKV backward
fusion still needs an actual graph/closure and rounding proof before implementation.

Profiled activity sums are not unprofiled wall-time shares or a prediction that
removing one family yields the same latency saving. The compiled comparison is
an independent campaign measurement, not a jointly profiled attribution. The
provided profile artifacts contain no NVML contention record; no absence of
foreign compute is claimed. Raw SQLite/Nsight bytes remain under
`target/transformer-training/parity-profile-v9-b32`; bounded derived evidence,
results and validation are embedded in the bundle.

## Gradient norm partials: prequalification only

V10 adds a default-off owner option, `--gradient-norm-partials`, requiring
first-accumulation overwrite. It squares the **actual stored f32 result of
`+0.0 + incoming`**, including promoted BF16 inputs. Complete globally aligned
1,024-element rows are reduced by 128 lanes visiting eight ascending elements
per lane, then the existing descending 64→1 shared tree. Rows split across
parameter ranges and the final padded row are repaired from the completed
resident gradient buffer. The second stage directly reduces `[R,1]` along axis
zero; regrouping arbitrary per-parameter sums would change the norm tree.

The fast path is valid only after one successful first-overwrite accumulation.
A second attempt invalidates it before work and retains the full-gradient norm
fallback. Empty intervals retain gradient clearing and full norm; failed
intervals still forbid update/retry until `begin_step`. Partial rows have exactly
one writer, source maps use the existing virtual-gradient indexing, and all lanes
remain present through block barriers. The workspace belongs to the resident
owner; no external MixedExecutor gradient-output ownership is introduced.

Five distinct focused tests pass: three CPU IR/coverage tests and two serial
real-device owner tests with CUDA required. These cover irregular positive f32
mantissas, subnormals, high dynamic range, BF16 gradients, signed zero, dirty
buffers/padding, nonfinite/overflow rejection, invalid-feed recovery, rectangular
BF16 view indexing, and 1→2→0→1 microbatch intervals. Norm f64 bits, clip-scale
f32 bits and full clipped-gradient bits match the existing path; postweights and
Adam moments also match. Row counts 5, 33 and 257 exercise serial, subgroup and
block second stages. Homogeneous replay asserts graph launches and zero
fallbacks; mixed tests verify replay and active direct-master bindings with zero
parameter refresh copies. These are focused tests, not a new full-suite claim.

The contraction audit uses generated PTX, actual compiled baseline norm modules,
and offline SM89 `ptxas`/`nvdisasm`. All twelve actual baseline modules use FFMA
for square accumulation. Isolated baseline stage 1 and repair each have eight
static FFMA instructions, zero FMUL and seven tree FADD; fused accumulation also
has eight FFMA and zero FMUL, with eight additional FADD for `+0+incoming`.
Static instruction counts are not dynamic operation counts. Exact GPU norm/clip
comparisons additionally qualify the emitted arithmetic; interpreting both
rounding policies on CPU alone would not establish this match.

CPU/CUDA checks, strict all-target/all-feature Clippy and formatting pass. The
bundle preserves command results, bounded logs, source hashes and all fifteen
PTX/SASS module pairs under `gradient_norm_partials_prequalification`, mapped to
[candidate-v10](benchmarks/transformer-training-sources/parity-candidate-v10.tar.gz)
(SHA256 `96610be73b799a7b1a0bb577f51771b719db22891d60aeceedb16db235e0418d`),
with frozen binary SHA256
`937b59e739bcb2d0ac133b7223c5b05c031cee39092c0d6af15ae05f7b130f28`.
The five recorded qualification source hashes match that archive. This
prequalification section adds no benchmark attempts, timed updates or comparisons.
The completed V10 campaign below is recorded separately. Prequalification alone
establishes no speedup; existing fixed gates are unchanged and PyTorch parity
remains unachieved.

## Gradient norm partials: complete-update campaign

The completed V10 campaign adds only `--gradient-norm-partials` to the retained
V9 combined policy, including direct resident masters. The option remains
default off. Each mode uses two fresh processes, 32 warmups and 32 measured
complete updates; trial 1 reverses process order.

| Case | V9 native control, ms | Norm partials, ms | Throughput gain | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: | ---: |
| 211M S128 B1 | 26.469 | 25.034 | 1.057x | 22.505 |
| 55M S512 B1 | 13.952 | 13.709 | 1.018x | 10.787 |
| 55M S128 B2 | 9.187 | 8.883 | 1.034x | 10.480 |
| 15M S512 B8 | 30.595 | 30.533 | near-neutral | 10.969 |
| 15M S512 B32 | 134.754 | 134.806 | neutral | 33.283 |

The larger-model cases improve in this screen; B8/B32 do not establish a useful
norm-partial gain. These are case-dependent complete-update observations, not a
promise of a general gain. The change also replaces the accumulation geometry
with 128 threads visiting eight values, so the saving cannot be attributed only
to eliminating a gradient-buffer read. Compiled reference variability is visible,
including 10.073–11.501 ms process medians for 55M/S512; two process repetitions
do not provide tight confidence bounds.

All ten candidates record `enabled=true` and `last_step_used=true` in their
qualification, 32 warmup and 32 measured samples. Across all **650 paired rows**,
gradient norm f64 bits, clip-scale f32 bits and loss f32 bits match the V9 control.
All first-step clipped-gradient and postweight binary snapshots are independently
SHA256-identical to their controls: twenty matching file pairs. All 30 workers
succeed and all 20 first-step comparisons pass unchanged global gates.
Candidate/control loss, gradient, postweight and update errors are zero. Against
compiled SDPA, gradient relative L2 spans 0.0161417313–0.0168246030,
postweight relative L2 0.0003656008–0.0006332113, and update relative L2
0.1330974331–0.1438772105; the existing 0.15 update limit is unchanged.

| Case | Global norm rows R | Repaired rows | Requested norm workspace, bytes |
| --- | ---: | ---: | ---: |
| 211M S128 B1 | 206,199 | 48 | 824,988 |
| 55M S512 B1 | 54,076 | 52 | 216,512 |
| 55M S128 B2 | 53,692 | 52 | 214,976 |
| 15M S512 B8/B32 | 14,878 | 42 | 59,680 |

Workspace payload is `4R + 4*repaired_rows`: f32 partials plus u32 row IDs.
Mixed-executor f32/BF16 payload and conceptual unpooled counters are unchanged;
they exclude this resident workspace. Exact pool USED_CURRENT increases by the
workspace size immediately after setup. Once the old norm row-buffer allocation
is replaced, steady USED_HIGH increases only by the row IDs: 192, 208 or 168
bytes, respectively. Lifecycle USED_HIGH increases by those same metadata bytes
in the larger cases, and by 59,680 bytes at B8/B32 where construction determines
the peak. Pool reservation counters are unchanged in both trials of every case.

Sampled whole-process peaks are unchanged at 6,650/2,360/2,076/2,736 MiB for the
first four cases; B32 changes from 9,038 to 8,880 MiB. That sampled reduction does
not establish an allocation saving: exact pool counters increase slightly, and
NVML sample maxima are lower bounds with a different scope. All native rows
retain replay. Model-kernel counts remain 581/546/627/594/611 and reported
complete-step launch counts remain 683/637/718/685/702. Boundary repair plus the
second-stage reduction replace the former two norm kernels, preserving launch
counts. Direct-master refresh copies remain zero.

The complete report is embedded as `gradient_norm_partials`, with a dedicated
audit preserving all active statuses, bit checks, snapshot hashes, replay and
memory scopes. All 136 recorded source hashes match the immutable V10 archive
identified above; the baseline is V9. This campaign adds 30 successful workers,
960 measured updates and 20 passing comparisons. Recomputed raw totals are
**15 campaigns, 304 attempts, 302 successful workers, two historical baseline
OOMs, 10,624 timed complete updates and 202 passing benchmark comparisons**.
Three diagnostic-profile comparisons remain excluded. PyTorch parity remains
unachieved; results from independent campaigns must not be multiplied.

## Medium BF16 reuse tiles: prequalification only

V11 adds an off-by-default `MediumBf16TilePolicy` with `SquareOnly` and
`SquareThenTall` choices. This revisits the large-M heuristic using checked grid
coverage. Rank-two ordinary BF16 contractions need M≥128, N≥256, K≥64 divisible
by 32, full 64-row tiles and at least 128 candidate CTAs. Square mode requires
full 64-column tiles and selects 64×64×32; the second mode can fall back to
64×32×32 with full 32-column tiles. The threshold is a scheduling bound, not a
measurement of occupancy. Already-large 64×64 tiles, batched outputs, specialized
split-K and non-BF16/strict scalar paths retain their schedules.

Selection precedes the existing shared-layout, operand-orientation and pipeline
proofs. Those conservative resource checks rerun unchanged, before shared-memory
reuse; the option does not relax them. Actual compiled square and tall program
counters are separate and exclude unchanged legacy large tiles. Compilation
signature version 41 distinguishes all 48 combinations of the four existing
kernel booleans and three policy values.

Seven focused tests pass: four offline schedule/codegen/signature tests and
three real-device CUDA tests, serialized under the GPU lock. They cover checked
resource/grid overflow, batch/tail fallback, actual square/tall activation,
layout/pipeline IR, cache invalidation, exact finite forward and all gradients,
changed feeds, views and replay. An independent eligible bias fixture verifies
that `1.0078125² - 1.015625` yields `2^-14`; prematurely rounding the matmul to
BF16 would yield zero. CPU and CUDA checks, strict all-target/all-feature Clippy
and formatting pass. This is focused prequalification, not a full-suite claim.

The frozen source is
[candidate-v11](benchmarks/transformer-training-sources/parity-candidate-v11.tar.gz)
(SHA256 `d0e3d33b447cd134de266fa0efd9d1f622a6237bb0eb173c333c287d99e2df1f`);
all 136 recorded source hashes match. Its binary SHA256 is
`2c364a8a17a750b65f75714446041c992eb512fb88d2253bf9e7aa6083697cfc`.
The square comparison uses retained V10 as control. The isolated tall comparison
uses V11 `SquareOnly` as control and V11 `SquareThenTall` as candidate.
This subsection records prequalification; tests add no benchmark workers,
timed updates or comparisons. The separate completed campaigns below provide
the full-model numerical and performance evidence.

## Medium tiles: completed square and tall screens

Both V11 campaigns use two fresh processes per mode, 32 warmups and 32 measured
complete updates, reversing mode order in trial 1. Latencies are medians of
process medians; reported tokens/s is batch×sequence divided by that latency.
The square comparison adds only `SquareOnly` to retained V10's selected policy.

| Square policy / case | V10 control, ms | Square, ms | Throughput gain | Square tokens/s | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 211M S128 B1 | 25.151 | 23.875 | 1.053x | 5,361 | 22.357 |
| 55M S512 B1 | 13.676 | 12.462 | 1.097x | 41,086 | 11.716 |
| 15M S512 B8 | 30.662 | 30.007 | 1.022x | 136,502 | 10.395 |

Actual selected-square counters are 17, 49 and 16, respectively, in both
qualification trials and every timed candidate update. Tall counts are zero.
This is an active square-tile comparison in all three cases.

The second campaign isolates the additional tall fallback: **both arms use the
same V11 binary and source**, with `SquareOnly` as control and `SquareThenTall`
as candidate. All other options match.

| Tall extension / case | Square control, ms | Square-then-tall, ms | Candidate tokens/s | Compiled PyTorch, ms | Additional tall programs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 211M S128 B1 | 23.903 | 22.3555 | 5,726 | 22.3580 | 25 |
| 55M S512 B1 — no-op control | 12.439 | 12.498 | 40,967 | 10.453 | 0 |
| 15M S512 B8 — no-op control | 29.900 | 29.980 | 136,625 | 10.315 | 0 |

Only 211M activates the tall extension: its square/tall counts are 17/25 in
both qualification steps and every measured update, giving a 1.069x throughput
ratio against the separately measured square control. The 55M/B8 counts remain
49/0 and 16/0. They are explicit no-op controls; timing differences there do not
establish a tall-tile benefit or regression. Compiled controls also vary between
campaigns, particularly at 55M, so cross-campaign deltas are not isolated effects.

At 211M the two-process medians are effectively tied: **22.3555 ms native versus
22.3580 ms compiled PyTorch**. Native process medians are 22.3721 and 22.3389 ms;
compiled medians are 22.3569 and 22.3592 ms. This is a screen-level tie requiring
more independent confirmation, not confirmed parity, a universal ranking or
completion of the larger-suite goal. The 55M and B8 gaps remain substantial.

All 36 workers and all 24 first-step comparisons pass unchanged fixed gates.
Across both campaigns, all 24 native-control full gradient/updated-weight file
pairs are SHA256-identical. Timed loss, gradient norm, clip scale and optimizer
step observations match the native controls. Every measured native update uses
replay; direct capture variants remain at two and parameter-copy counts at zero.
Model-kernel and complete-step launch counts are unchanged. Exact setup,
lifecycle and steady pool counters match between native arms in both trials;
sampled NVML variation does not establish an allocation saving. No foreign
compute was observed by the campaign samplers. These are first-step numerical
and repeated-update timing checks, not a training-convergence result.

The two screens add 1,152 measured updates and 24 comparisons. Totals recomputed
through the medium-tile screens are **17 campaigns, 340 worker attempts, 338
successful workers, two historical baseline OOMs, 11,776 timed complete updates
and 226 passing first-step benchmark comparisons**, plus three excluded
diagnostic-profile comparisons. Both complete reports, per-process snapshot
hashes, activation/lifetime audits and source-policy mappings are preserved as
`medium_bf16_square` and `medium_bf16_tall` in the raw bundle. Their 136 source
hashes match the frozen V11 archive above; square's baseline is V10, while
tall's baseline is the same V11 source/binary with the square policy selected.

## QKV gradient packing: completed prequalification

V12 adds `MixedOptions.qkv_grad_pack_fusion`, disabled by default. It replaces
closed Q/K/V Narrow-gradient padding and their two BF16 Add joins with one
packing kernel per layer. The primal Narrow views remain unchanged. The matcher
requires exact positive-zero pads, supported bijective BHSD/BHDS views and no
escaping interiors. The kernel loads only the selected branch and retains the
original ordered BF16 casts and joins through per-thread typed shared scratch.
Unsupported graph patterns retain the ordinary lowering.

The actual four-layer decoder graph produces four closed matches in both the
tiny B2/H4 fixture and the W512/S512/B32/H8 topology; all 12 primal Narrow nodes
remain. On the tiny GPU decoder, kernels decrease from 601 to 509. Three complete
updates preserve identical loss, gradient, weight and first/second Adam-moment
bits with replay, direct masters and incremental reuse enabled, with transposed
output and attention-fusion options also enabled. Changed-feed tests cover
captured and uncaptured execution. Raw BF16 tests compare against the original
GPU kernels for signed zeros, subnormals, infinities and quiet/signaling NaNs:
finite/Inf/signed-zero bits and NaN classes match, with zero observed NaN-payload
bit differences. That payload result is a device diagnostic, not a portable
cross-device promise. Closure, nonzero/negative-zero pads, dtype changes,
arithmetic, rank-two raw leaves, predicate/OOB metadata and rounding-IR guards
also have regression coverage.

All six focused tests pass. The serial real-device CUDA-feature library run
passes **373 tests, with four existing ignored tests**; all 174 CPU library tests,
strict all-target/all-feature Clippy and formatting pass. These tests are
correctness qualification; the completed campaign below supplies the separate
complete-update timing evidence.

The bundle records this evidence as `qkv_grad_pack_prequalification`, including
commands, the complete CUDA test log hash and the debug-qualification scope.
All 143 recorded source hashes match
[candidate-v12](benchmarks/transformer-training-sources/parity-candidate-v12.tar.gz)
(SHA256 `1b19b7d74033d6887a99ba181f777f7d7695f8afe2d72e40f004f1cce9ec97c0`).
The frozen candidate binary SHA256 is
`9031d7b22a5bb63ddbaa462a0c81f547d571ec2c6fb0f4cfc361a9c29d1df357`;
the campaign uses V11 as its native control. No performance conclusion follows
from these qualification tests.

## QKV gradient packing: completed eight-case campaign

Two fresh processes per mode use 32 warmups and 32 measured complete updates,
with process order reversed in the second trial. The V12 candidate adds only
`--qkv-grad-pack-fusion` to the V11 control's options; both use
`SquareThenTall` as their medium-tile policy. The complete raw report is embedded
as `qkv_grad_pack`, with its independent counter/scalar audit alongside it.

| Case | Native control, ms | QKV packing, ms | Compiled PyTorch, ms | Native throughput gain |
| --- | ---: | ---: | ---: | ---: |
| 211M / S128 / B1 | 22.344 | 21.946 | 22.685 | 1.018x |
| 55M / S512 / B1 | 12.511 | 11.525 | 10.353 | 1.086x |
| W512 / S512 / B8 | 29.867 | 25.885 | 11.169 | 1.154x |
| W512 / S512 / B16 | 62.633 | 54.001 | 17.866 | 1.160x |
| W512 / S512 / B32 | 134.756 | 116.243 | 33.543 | 1.159x |
| W512 / S128 / B2 | 3.879 | 3.429 | 10.994 | 1.131x |
| W512 / S512 / B2 | 8.472 | 7.537 | 10.633 | 1.124x |
| W1024 / S128 / B2 | 8.693 | 8.248 | 11.293 | 1.054x |

All 48 workers pass, contributing 1,536 timed updates. All 32 unchanged fixed
first-step comparisons pass: 16 against the native control and 16 against
compiled SDPA. Native gradient, postweight and update relative L2 errors are
zero. Validator-recorded SHA256s match for all 32 gradient/postweight snapshot
comparisons. Across all 16 native trial pairs, all **1,040 paired scalar rows**
(first update, warmups and measurements) match exactly for loss, pre-clip norm,
clip scale and optimizer step. Compiled-control relative L2 ranges are
0.016142–0.016904 for clipped gradients, 0.000348–0.000633 for postweights and
0.133097–0.143895 for updates, within the existing gates of 0.02, 0.001 and 0.15.
This qualifies these updates, not long-term training convergence.

Every candidate has four active packing regions. Model kernels decrease by 76
for 211M/S128 and W1024/S512, and by 92 for the other six shapes; existing view
materialization elision makes the baseline count shape-dependent. Every measured
candidate retains two capture variants and two capture misses, with capture hits
progressing from 32 to 63. Replay remains active and parameter-copy counts and
bytes remain zero across all 65 updates per native run. Unique borrowed-master
bytes equal the model master-parameter bytes.

Allocation counts need separate interpretation. At 211M, retained F32 decreases
by 3,145,728 bytes and BF16 by 1,572,864 bytes; exact steady and lifecycle pool
USED_HIGH both decrease by 4,718,592 bytes (4.5 MiB). At B32, cumulative unpooled
allocation falls by 3,356,493,824 bytes, but retained F32/BF16 and both exact pool
highs are unchanged: steady USED_HIGH is 8,773,327,756 bytes and lifecycle
USED_HIGH is 9,259,022,996 bytes. The planner already reuses those eliminated
temporaries, so the B32 unpooled reduction is **not a VRAM saving**. Sampled
whole-process peaks at B32 are 8,942 MiB for the native control, 9,040 MiB for
packing and 3,106 MiB for compiled PyTorch. These NVML samples are lower bounds
in a different scope from exact current/default-pool accounting; neither metric
includes a claim that all process memory has been measured exactly.

The largest case is 21.9462 ms versus the compiled control's 22.6848 ms in this
two-process screen. That observed advantage needs confirmation with more fresh
processes. It does not establish suite parity: B32 remains 116.243 ms versus
33.543 ms, and several other established cases also remain slower. All new
options remain disabled by default.

Totals through the 18 complete QKV-era campaigns are **388 worker attempts,
386 successful workers, two historical baseline OOMs, 13,312 timed complete
updates and 258 passing first-step benchmark comparisons**, plus three excluded
diagnostic-profile comparisons. V12 maps to the immutable archive/binary above,
with V11 as its native baseline; all 143 source hashes match the V12 archive.
Later working-tree source is not provenance for these measurements.

## SwiGLU forward/backward pair: qualification and complete-update campaign

V13 adds `MixedOptions.swiglu_pair_fusion`, disabled by default. It matches a
closed mixed-storage SwiGLU forward/VJP pair and recomputes saved scalar locals
from the same BF16 gate/up operands. It preserves the primitive expression and
gradient-Add order, all BF16 rounding boundaries and six selective saved-F32
checkpoints; it does not substitute a compact derivative. Only proven contiguous
physical BF16 inputs and full-volume reshape/flatten aliases qualify. Escaping
saved values, unsupported views and inference-only patterns retain the ordinary
lowering. Hidden output is published at its original node; both backward outputs
are published at the earliest ready output, with source lifetimes extended.

The actual four-layer B32 decoder graph has four pairs. Seven focused tests,
380 CUDA-feature library tests (four existing ignored), all 174 CPU library
tests, CPU/CUDA checks, strict all-target/all-feature Clippy and formatting pass.
Tests cover all 65,536 BF16 gate patterns, nonfinite operands, signed zero and
270-element tails. All non-NaN bits match; NaN categories match without a payload
promise. Three evolving tiny-decoder updates retain exact loss, clipped
gradients, weights and both Adam moments with replay, incremental reuse, direct
masters and active norm partials. Offline PTX/SM89 SASS audits of seven modules
confirm the original ordered contractions and explicit rounding. The two new
kernels use 15/22 registers and 1,536 shared bytes, with zero barriers/spills in
that offline compilation; this is not a driver-JIT disassembly or occupancy claim.

The campaign compares V13 with the retained V12 native control, adding only
`--swiglu-pair-fusion`. Both use the established combined options including QKV
packing. Two fresh processes per mode use 32 warmups and 32 measured complete
updates, with process order reversed in the second trial.

| Case | Native control, ms | SwiGLU pair, ms | Compiled PyTorch, ms | Native throughput gain |
| --- | ---: | ---: | ---: | ---: |
| 211M / S128 / B1 | 21.938 | 21.615 | 22.431 | 1.015x |
| 55M / S512 / B1 | 11.557 | 10.728 | 9.927 | 1.077x |
| W512 / S512 / B8 | 25.930 | 22.304 | 10.600 | 1.163x |
| W512 / S512 / B16 | 54.141 | 47.080 | 17.798 | 1.150x |
| W512 / S512 / B32 | 116.080 | 102.465 | 33.561 | 1.133x |
| W512 / S128 / B2 | 3.640 | 3.431 | 11.053 | 1.061x |
| W512 / S512 / B2 | 7.557 | 6.761 | 8.890 | 1.118x |
| W1024 / S128 / B2 | 8.148 | 7.735 | 11.775 | 1.053x |

All 48 workers pass, adding 1,536 measured complete updates. All 32 unchanged
fixed first-step comparisons pass: 16 against native controls and 16 against
compiled SDPA. Fresh stream hashes verify all 104 campaign F32 files, including
96 snapshot files and eight input archives, against the stored provenance. All
32 native gradient/postweight snapshot comparisons are hash-identical, with
zero native gradient, postweight and update relative L2 error. Across the 16
native trial pairs, all 1,040 paired loss/norm/scale/optimizer-step records are
identical. Compiled-control relative L2 ranges are 0.016142–0.016904 for clipped
gradients, 0.000348–0.000633 for postweights and 0.133097–0.143876 for updates,
inside the original fixed gates. These checks do not establish convergence.

Every candidate emits four forward and four backward pair regions, preserves
four QKV packing regions and removes exactly 20 model kernels. Every measured
candidate retains two capture variants and two capture misses; hits advance
from 32 to 63. Replay remains active and master-copy counts/bytes remain zero
through all 65 updates in each native run. Unique borrowed-master bytes still
equal model master-parameter bytes.

Unlike the earlier QKV unpooled-count reduction, this change lowers retained
payload and exact pool usage. In both B32 trials, owned F32 falls by
3,187,671,040 bytes and BF16 by 138,412,032 bytes. Steady pool USED_HIGH drops
from 8,773,327,756 to 5,447,244,684 bytes; lifecycle USED_HIGH drops from
9,259,022,996 to 5,932,939,924 bytes. Both save **3,326,083,072 bytes
(3,172 MiB)**. Exact steady/lifecycle savings are 793 MiB at B8 and 76.625 MiB
at 211M. Sampled whole-process B32 peaks are 8,974 MiB for the native control,
5,776 MiB for the candidate and 3,106 MiB for compiled PyTorch. Pool counters
cover the queried current/default pool and exclude context/modules/other pools;
NVML samples are whole-process lower bounds. Their different scopes must not be
interchanged.

The current 211M screen is 21.6149 ms versus 22.4311 ms for compiled PyTorch,
but still has only two processes per mode. Small-case controls vary: the
W512/S128 V12 candidate measured 3.429 ms in the prior campaign, while the same
V12 control is 3.640 ms here; the new candidate is 3.431 ms. Thus that small
screen improvement needs confirmation rather than a stronger attribution from
cross-campaign values. Compiled controls also move between campaigns. B32
remains 102.465 ms versus 33.561 ms, and the full-suite parity goal remains unmet.
All new options remain disabled by default.

The complete report, audit, prequalification manifest and seven-module offline
contraction audit are retained as `swiglu_pair`, `swiglu_pair_campaign_audit`
and `swiglu_pair_prequalification`. All 144 campaign source hashes, all three
prequalification source hashes and all 102 qualification artifacts verify.
The source is
[candidate-v13](benchmarks/transformer-training-sources/parity-candidate-v13.tar.gz)
(SHA256 `c2f210a48d9b171db07aebd6992afcb6552aa2ea269990f7147e96e8f876cbf2`);
the frozen candidate binary SHA256 is
`7c9154966eac8e1498c7a453570414ffcade46bcc0b54a857213e8066699b582`,
with V12 as the native baseline. Later working-tree edits are not provenance for
these measurements.

Totals through V13's **19 complete campaigns** are **436 worker attempts,
434 successful workers, two historical baseline OOMs, 14,848 timed complete
updates and 290 passing first-step benchmark comparisons**, plus three excluded
diagnostic-profile comparisons. The separately recorded V13 profile below is diagnostic and is excluded from
these benchmark totals.

## Current B32 profile after QKV and SwiGLU fusion

One retained-V13 Nsight profile records five complete updates after 32 warmups.
All five repetitions of 499 model labels match their kernel names, launch
geometry and resources; 251 label-family checks pass. Two cached capture graphs
supply two and three repetitions. The trace has 591 kernel activities per update,
versus the reported complete-step count of 590: the additional kernel is the
scalar-loss readback's `materialize_view`, outside the captured model. There is
no local-memory allocation among the mapped model kernels.

Profiled complete-update wall time averages 104.061 ms. Summed CUDA activity is
102.360 ms per update, of which 101.364 ms belongs to the captured model. The
shape/label mapping assigns that model activity as follows; these durations are
not unprofiled wall-time shares or achievable speedups.

| Domain | Summed model kernel activity, ms/update |
| --- | ---: |
| Attention core | 28.315 |
| Normalization and residual spine | 24.167 |
| FFN | 23.668 |
| QKV projections | 10.156 |
| Head and loss | 10.015 |
| Explicitly ambiguous | 3.885 |
| Attention output projection | 1.158 |

The normalization/residual bin includes branch-gradient and residual additions;
it does not prove that all 24.167 ms is LayerNorm or can be fused. Across the
complete step, GEMM kernels total 38.571 ms and split-K partials 11.814 ms.
Attention QK/PV and their four backward contractions remain material work;
removing all attention-core activity ideally would reduce summed model activity
by only about 1.38x, insufficient to explain away the roughly 3x unprofiled
B32 gap to compiled PyTorch. Neither this ideal bound nor summed activity is a
critical-path speedup prediction.

Profile first-step gradients, weights and scalar values match both V13 native
trial controls exactly. The compiled-reference fixed gates also pass: gradient
relative L2 0.016706, postweight 0.000366, update 0.143644 and normalized loss
error 0.00000908. Replay is active, direct captures remain at two variants/two
misses, and no master-copy traffic returns. A descendant-aware sampler requested
at 50 ms intervals observes a 5,852 MiB whole-worker peak and no foreign compute;
this sampled peak is a lower bound, while exact pool counters remain in the raw
result with their narrower scope.

The raw bundle's `profile_v13_b32` includes all per-kernel labels/resources and
five durations, complete results, fixed/native validations, commands, source
provenance and trace hashes. Its source and executable are the immutable V13
archive/binary above, not the subsequently edited column-policy source. This
adds the **fourth excluded diagnostic profile** and changes no benchmark worker,
measured-update or first-step benchmark-comparison total.

## Adjacent-column reduction policy: qualification

V14 adds the opt-in `ColumnReductionPolicy::{Four,Eight}`; the default is `Off`.
The policy assigns four or eight independent adjacent columns to each block,
while preserving the original 128 contributors and arithmetic tree per column.
Contributor t still visits rows t+128*k in order, followed by shared-memory
pair reductions at offsets 64,32,16,8,4,2,1. Producer and epilogue SSA, typed
loads/stores, BF16 rounding and preceding copies remain unchanged; no arithmetic
padding terms are added to valid columns. Invalid tail columns contribute the
neutral identity and reach every barrier.

The bounded eligibility includes F32 [B,S,C] axis-1 sums and BF16 [R,C] axis-0
sums with canonical physical layouts and the existing Block128 schedule.
Unsupported layouts, other reductions and device/function limits retain the
legacy mapping. Four uses 512 threads and 2,048 shared bytes; Eight uses 1,024
threads and 4,096 shared bytes. Actual JIT function limits are checked separately
from device maxima. This geometry is not an occupancy or bandwidth claim.

The full campaign compares the frozen V14 `Four` candidate with retained V13,
adding only the column policy to the established combined options. The Eight
policy's qualification is separate from Four's complete-update measurements.

Completed V14 prequalification passes 178 CPU library tests and 392 CUDA library
tests (four existing ignored), plus CPU/CUDA checks, strict all-target/all-feature
Clippy, formatting and diff checks. Thirteen focused checks cover Four and Eight
separately, exact outputs/gradients and BF16 rounding, signed zero/nonfinite
classes, changed feeds, replay, cache Off/Four/Eight/Off transitions and the
forced-128-thread fallback in the same compiled function. The frozen V14 cache
identity matrix has 144 combinations; later expanded-source matrices are not
evidence for this frozen executable. The actual B32 graph has 18 F32 and 20 BF16
boundary-eligible reductions; that CPU count is not a compiled resource claim.

The actual JIT function's maximum thread count and nonzero-active-block check
can select the uniform legacy-128 body, without removing copies or conversions.
Qualified sample driver functions report 27 registers/512 shared bytes for the
old mapping and 30 registers/2 KiB or 4 KiB for the new mappings, with no local
memory. Independent SM89 SASS assembled from the exact PTX with CUDA 12.9 is not
driver-JIT disassembly; its per-body contractions retain the old arithmetic tree.
Neither these resource checks nor the prequalification establishes a speedup.

Evidence: target/paper-kernels/column-qualification/audit.json and sibling PTX,
SASS and resource manifests. CPU test run aad7651b-3dd6-4fca-9bc7-4c2f7f9b08e2;
CUDA run 7e4f42ba-f0e8-4a6f-81ad-ce2e0e0bad45, serialized under the GPU lock
with TNSR_REQUIRE_CUDA=1. Source: parity-candidate-v14.tar.gz SHA256
90cbf92f167f7ff84eb5c18bde012139aae2d318770a40b750064c8fdea7d7fe;
retained binary SHA256
884ccb51c5d7c1d7b02677290862731b44d2ed51d8e7e68d4ff60081da87fbd6.
The completed Four and Eight campaigns below supply the separate timing evidence.

## Column Four: completed eight-case campaign

Two fresh processes per mode run 32 warmups and 32 measured complete updates;
the second trial reverses process order. V14 adds only
`--column-reduction-policy=four` to the retained V13 combined policy. Kernel
counts remain unchanged: the experiment changes the physical assignment of
independent reduction columns, not the graph or per-column arithmetic tree.

| Case | Native control, ms | Four, ms | Compiled PyTorch, ms | Active Four regions |
| --- | ---: | ---: | ---: | ---: |
| 211M / S128 / B1 | 21.574 | 21.554 | 22.455 | 0 |
| 55M / S512 / B1 | 10.774 | 10.457 | 11.070 | 38 |
| W512 / S512 / B8 | 22.341 | 20.993 | 10.238 | 38 |
| W512 / S512 / B16 | 47.214 | 44.621 | 18.169 | 38 |
| W512 / S512 / B32 | 102.517 | 97.372 | 33.849 | 38 |
| W512 / S128 / B2 | 3.430 | 3.330 | 9.619 | 20 |
| W512 / S512 / B2 | 6.747 | 6.494 | 10.447 | 38 |
| W1024 / S128 / B2 | 7.738 | 7.739 | 11.560 | 20 |

The 211M B1/S128 case is an intentional zero-activation control. Every S512
case has 38 active regions: 18 F32 and 20 BF16. The smaller B2/S128 cases have
20 BF16 regions, because their flattened row extent is 256; their F32 S128
axis is excluded. Thus S128 is not universally a no-op. The earlier QKV/SwiGLU
tables' three smaller cases are corrected to B2, matching the raw configurations.
W1024/S128 remains neutral despite 20 active regions.

All 48 workers and all 32 fixed first-step comparisons pass. All 1,040 native
paired scalar rows are exact, and all 32 fresh gradient/postweight snapshot
hash comparisons are identical. Candidate/native gradient, weight and update
relative L2 errors are zero. Model kernel counts, two retained capture variants,
two steady capture misses and hits 32–63 remain stable; all 65 native updates
retain replay and zero master-copy counts/bytes.

The two-process screen improves B8/B16/B32 over the native control, but small
cases vary: W512/S128 candidate process medians are 3.423 and 3.237 ms, so its
aggregate needs confirmation. The 55M compiled control is 11.070 ms here versus
9.927 ms in the V13 campaign; this does not establish stable 55M parity. The
largest case has zero column activation, and its small native/compiled screen
advantage cannot be attributed to this change. B32 remains 97.372 ms versus
33.849 ms for compiled PyTorch; the full-suite goal remains unmet.

## Column Eight: neutral comparison against Four

Both modes use the **same V14 binary and source**. The only policy difference
is Four versus Eight; both retain all established options. Two fresh processes
per mode again use 32 warmups and 32 measurements.

| Case | Four control, ms | Eight, ms | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: |
| 55M / S512 / B1 | 10.421 | 10.485 | 11.424 |
| W512 / S512 / B8 | 20.949 | 21.117 | 10.253 |
| W512 / S512 / B16 | 44.561 | 44.421 | 18.084 |
| W512 / S512 / B32 | 97.384 | 97.310 | 33.751 |

All 24 workers and all 16 comparisons pass. All 520 native paired scalar rows
and 16 fresh snapshot-hash comparisons are exact. The 38 active Four regions
become 38 active Eight regions, with unchanged kernel counts, replay, capture
stability and zero master copies. These small timing differences demonstrate no
Eight benefit; retain Four in the experimental combined policy. Both remain off
by default. No occupancy, bandwidth or throughput-mechanism claim follows.

Across both campaigns, all owned F32/BF16 payloads and exact steady/lifecycle
pool highs are unchanged in each native trial pair. There is no tensor-payload
or exact-pool memory-saving claim. Sampled B32 whole-process peaks are
5,872/5,840 MiB for the Four campaign's control/candidate and 5,776/5,840 MiB for
the Eight campaign's Four/Eight controls; those sampled lower bounds have a
different scope from exact pool accounting. No foreign compute was observed by
the campaign samplers. Compiled-reference errors remain within the original
fixed gates; no gate changed.

All 156 new F32 files (144 snapshots and 12 archives) were freshly stream-hashed
before verified content-preserving hardlink deduplication. Their historical stat
fingerprints and content hashes are retained in the audits and shared verified
cache; deduplication changes file metadata, not snapshot contents. The raw bundle
retains both complete reports as `column_reduction_four` and
`column_reduction_eight`, their counter/scalar/allocator audits and the completed
prequalification manifest. Both reports' 146 source hashes match immutable
[candidate-v14](benchmarks/transformer-training-sources/parity-candidate-v14.tar.gz)
(SHA256 `90cbf92f167f7ff84eb5c18bde012139aae2d318770a40b750064c8fdea7d7fe`);
the candidate binary SHA256 is
`884ccb51c5d7c1d7b02677290862731b44d2ed51d8e7e68d4ff60081da87fbd6`.
Four's control is V13; Eight's control is V14 with Four selected.

Totals through the V14 column screens are **21 complete campaigns, 508 worker attempts, 506 successful
workers, two historical baseline OOMs, 17,152 timed complete updates and 338
passing first-step benchmark comparisons**, plus four excluded diagnostic
profiles. These reports do not complete the parity goal.

## Large ordinary BF16 rectangles: qualified, no consistent benefit

V15 adds off-by-default `LargeBf16TilePolicy::{Tall,Wide,TallThenWide}` for a
bounded ordinary rank-two BF16 domain. Tall uses 128x64x32 tiles with a 4x2
computing-warp topology; Wide uses 64x128x32 and 2x4. Both use 256 threads and
2x2 fragments per computing warp. Candidates require the original ordinary
64x64x32 configuration, explicit shared-memory reuse, M>=2048, N>=256,
K>=64 divisible by 32, full candidate M/N tiles and a checked grid of at least
128 blocks. Split-K and batched policies are unchanged.

Admission applies operand-copy/layout and pipeline proofs, two stages and the
existing conservative two-block resource reserve. A rejected candidate retains
the complete old configured schedule; short-K Auto startup rules are not silently
overridden, and no synchronous candidate is substituted. Per-output increasing
K16 order and Linear's bias-before-final-BF16-round remain unchanged, including
the explicit 2^-14 bias fixture. The larger CTA handles twice the output area;
its doubled static final-conversion count is per CTA, not per output.

The 13 focused checks, all 181 CPU library tests and 404 real-device CUDA library
tests (four existing ignored) pass, along with CPU/CUDA checks, strict Clippy and
formatting/diff checks. Frozen V15 compilation identity covers 576 combinations.
The qualification audit retains 27 PTX/SASS modules and 12 paired real-device
fixtures. Qualified candidate driver functions report 96/99/101 registers,
32 KiB shared and zero local memory, with API active-block capacity two. Controls
report 78 registers/20 KiB/capacity three or 92 registers/19 KiB/capacity two.
These API capacities are not achieved occupancy. Offline CUDA12.9 SASS matches
each PTX's recorded SM80 or SM89 target; it is not driver-JIT disassembly and
does not establish a hardware traffic or bandwidth benefit.

The two campaigns below are **independent comparisons against accepted V14
Column Four with LargeOff**. Both use two fresh processes per mode, 32 warmups
and 32 measured complete updates, reversing mode order in trial 1. Wide is not
measured as a follow-on improvement over TallThenWide.

| TallThenWide case | V14 control, ms | TallThenWide, ms | Compiled PyTorch, ms | Actual Tall/Wide programs |
| --- | ---: | ---: | ---: | ---: |
| 211M / S128 / B1 | 21.698 | 21.696 | 22.705 | 17/0 |
| 55M / S512 / B1 | 10.458 | 10.508 | 11.531 | 5/8 |
| W512 / S512 / B8 | 21.033 | 21.293 | 10.295 | 43/0 |
| W512 / S512 / B16 | 44.516 | 44.757 | 17.861 | 43/0 |
| W512 / S512 / B32 | 97.353 | 97.530 | 33.301 | 42/0 |
| W512 / S128 / B2 | 3.264 | 3.329 | 10.787 | 1/0 |
| W512 / S512 / B2 | 6.436 | 6.450 | 11.169 | 1/0 |
| W1024 / S128 / B2 | 7.774 | 7.752 | 11.612 | 5/8 |

| Wide-only case | V14 control, ms | Wide, ms | Compiled PyTorch, ms | Actual Wide programs |
| --- | ---: | ---: | ---: | ---: |
| 211M / S128 / B1 | 21.561 | 21.619 | 22.350 | 17 |
| 55M / S512 / B1 | 10.414 | 10.575 | 11.711 | 13 |
| W512 / S512 / B8 | 21.010 | 21.387 | 10.971 | 43 |
| W512 / S512 / B32 | 97.360 | 97.528 | 33.530 | 42 |

All 48 TallThenWide workers and 24 Wide workers pass, adding 2,304 measured
updates. All 48 fixed first-step comparisons pass. All 1,560 native paired
scalar records match, and all 48 fresh gradient/postweight snapshot-hash
comparisons are identical. Native gradient, weight and update relative L2 errors
are zero; compiled controls pass the unchanged fixed gates. Every candidate
keeps the original model kernel count, active replay, two captured variants and
two steady capture misses, with hits 32–63 and zero master-copy counts/bytes
through all 65 updates. Actual counters match both trials of every case.

The 211M structural boundary fixture had 21 candidates per orientation, while
the complete campaigns activate 17. The four final nonactivations are
**unattributed**; structural eligibility is not proof of final copy/resource
activation, and no QKV or other fallback cause is asserted. Actual 55M
TallThenWide activation is five Tall plus eight Wide, while Wide-only has 13 Wide.
The B8/B16 cases have 43 Tall programs and B32 has 42; Wide-only has 43 and 42 for
its measured B8/B32 cases. These counters demonstrate actual selection, not
performance benefit.

Global owned F32/BF16 payloads, cumulative unpooled counts and exact steady and
lifecycle pool highs are unchanged in all native trial pairs. Sampled B32
whole-process peaks are 5,680/5,840 MiB for TallThenWide control/candidate and
5,616/5,840 MiB for Wide-only control/candidate. Those sampled lower bounds
include process memory outside the queried pool and are not exact VRAM totals;
there is no allocator or VRAM-saving claim. No foreign compute was observed by
the samplers.

Neither orientation demonstrates consistent complete-update benefit. Keep
**LargeOff with Column Four** as the accepted experimental combination. The small
native/compiled 211M screen advantage remains a two-process observation and
cannot establish suite parity; these rectangles do not close B32's roughly
97.5 ms versus 33.5 ms gap. All policies remain off by default.

The complete raw reports and audits are embedded as `large_bf16_tall_then_wide`
and `large_bf16_wide`, with `large_bf16_prequalification`. Both reports' 156 source
hashes match immutable
[candidate-v15](benchmarks/transformer-training-sources/parity-candidate-v15.tar.gz)
(SHA256 `19ec5faab431b2766a015143168a4eb52e030c87ad00bb6fa998b4bc12f5e4e5`);
the frozen binary SHA256 is
`3dde06ed1e4526148e9006fc7c77ee9c1a514a4124953645ea53dc79a2ed6be6`.
Both native baselines map to V14. All 156 new F32 files were freshly stream-hashed
before verified content-preserving hardlink deduplication; historical stat
fingerprints and unchanged content hashes are preserved in the audits/cache.

Totals through the V15 ordinary-rectangle screens are **23 complete campaigns, 580 worker attempts, 578 successful
workers, two historical baseline OOMs, 19,456 timed complete updates and 386
passing first-step benchmark comparisons**, plus four excluded diagnostic
profiles. The parity goal remains unmet.

## Batched attention rectangles: complete-update improvement at larger batches

V16 adds default-off `BatchedAttentionBf16TilePolicy::{TallScores,WideScores}`.
The initial domain is BF16 S512/D64 attention with B*H between 64 and 65,535.
Candidates require whole-volume bijective maps, aligned async copies, complete
tiles, at least 128 batched CTAs, physical shared-memory reuse and a final
resource/layout/pipeline proof. Admission configures a clone and installs it
only after all guards pass; unsupported cases retain their complete ordinary
schedule. The policy alone explicitly requests two pipeline stages; the common
Auto startup rule is unchanged. Increasing K16 accumulation order and final
BF16 rounding remain unchanged.

Four-layer B8 and B32 decoder fixtures prove 24 eligible GEMMs: eight QK/dP score
contractions, twelve PV/dV/dQ query-width contractions and four dK width-query
contractions. Query-width always uses 128x64 tiles; width-query uses 64x128.
TallScores uses 128x64 for score outputs, selecting twenty tall/four wide
programs. WideScores selects twelve tall/twelve wide. Actual B8/B16/B32 campaign
counters confirm all 24; S128 and insufficient-BH cases have zero activation.

All 184 CPU library tests and 413 real-device CUDA library tests pass, with four
existing ignored tests, along with CPU/CUDA checks, strict Clippy and formatting.
Nine new focused tests and three existing copy-proof tests cover exact whole-map
proofs, guard fallback, all role output bits/nonfinite classes, cache invalidation
and changed feeds. Actual B8 model training checks both policies over three
evolving intervals with one, two and one microbatches. Loss, norm/scale bits,
all clipped gradients, masters and Adam moments match the ordinary policy exactly
with replay, incremental allocation, direct masters and norm partials enabled.

Driver queries on the 24 actual B8 programs per policy report 94–100 registers,
32 KiB static shared memory, zero local bytes and an API two-block capacity.
This capacity is not achieved occupancy. The offline audit covers 52 PTX modules,
including 50 selected rectangles: eight PTX WMMA instructions map to sixteen
HMMA instructions in each offline cubin, and every selected module drains async
copies before its shared epilogue. Offline CUDA12.9 `ptxas`/`cuobjdump` output is
not driver-JIT SASS and does not establish traffic or bandwidth gains.

The first campaign adds only TallScores to accepted V14 Column Four with
LargeOff. It uses two fresh processes per mode, 32 warmups and 32 measured
complete updates, with reversed mode order in trial 1:

| Case | V14 control, ms | TallScores, ms | Compiled PyTorch, ms | Active attention programs |
| --- | ---: | ---: | ---: | ---: |
| 211M / S128 / B1 | 21.540 | 21.541 | 22.446 | 0 |
| 55M / S512 / B1 | 10.415 | 10.482 | 10.262 | 0 |
| W512 / S512 / B8 | 21.008 | 17.456 | 11.094 | 24 |
| W512 / S512 / B16 | 44.589 | 38.186 | 18.120 | 24 |
| W512 / S512 / B32 | 97.327 | 84.141 | 33.545 | 24 |
| W512 / S128 / B2 | 3.344 | 3.417 | 9.203 | 0 |
| W512 / S512 / B2 | 6.442 | 6.442 | 9.789 | 0 |
| W1024 / S128 / B2 | 7.719 | 7.743 | 10.253 | 0 |

TallScores improves the eligible B8/B16/B32 screens by about 1.20x/1.17x/1.16x
against their fresh native controls. Ineligible cases are no-op controls;
small timing differences there are not attention-policy effects. The 211M
screen remains a two-process observation, and 55M/S512 remains near the compiled
reference. Neither establishes suite parity. B32 still takes 84.141 ms versus
33.545 ms compiled PyTorch, about 2.51x its latency.

The second campaign uses the same retained V16 binary in both native arms,
changing TallScores to WideScores only. It measures the three active cases:

| Case | TallScores control, ms | WideScores, ms | Compiled PyTorch, ms |
| --- | ---: | ---: | ---: |
| W512 / S512 / B8 | 17.511 | 17.590 | 10.184 |
| W512 / S512 / B16 | 38.152 | 38.107 | 17.751 |
| W512 / S512 / B32 | 84.047 | 84.003 | 33.336 |

WideScores is neutral. Retain **Column Four + LargeOff + BatchedTallScores** as
the accepted experimental combination; all policy defaults remain off. Fresh
compiled controls vary between the two campaigns, so these tables must not be
chained into a predicted speedup.

All 66 workers pass and all 44 fixed first-step comparisons pass. Native paired
loss/norm/scale/optimizer-step trajectories match across all 1,430 update pairs.
All 44 paired full gradient/postweight snapshot comparisons are SHA256-identical;
native gradient, postweight and update relative L2 errors are zero. Every native
candidate retains its control's model kernel count, replay, two capture variants
and two steady capture misses, and zero master-copy counts/bytes through all
65 updates. Compiled-reference gates use their original thresholds, including
2% gradient relative L2 and 15% update relative L2; none were relaxed.

Owned F32/BF16 payloads, cumulative unpooled counts, and exact steady/lifecycle
pool USED_HIGH values are unchanged in every native pair. B32 sampled whole-worker
peaks span 5,648–5,712 MiB for V14 controls and 5,808–5,904 MiB for TallScores;
the separate Tall/Wide campaign observes 5,712–5,838 / 5,744–5,838 MiB. These
requested-50-ms NVML peaks are lower bounds with a broader scope than exact pool
counters. They do not establish VRAM savings or a precise VRAM regression.
No foreign compute was observed.

The complete reports are embedded as `batched_attention_tall_scores` and
`batched_attention_wide_scores`, with dedicated audits and prequalification.
All 167 recorded source hashes in each report match immutable
[candidate-v16](benchmarks/transformer-training-sources/parity-candidate-v16.tar.gz)
(SHA256 `d152cc7cb2db4bb314e43ec608465391d4ebf89c3d96efc19fbba08f21232070`);
the retained binary SHA256 is
`f57be01588155ccd2e081164294d3ae807b00a4496b4f54e4dfcd51d9dd6ad6c`.
Tall's baseline maps to V14; Wide's baseline maps to V16 with TallScores.
All 143 campaign F32 files were freshly stream-hashed before content-preserving
hardlink deduplication. Historical prelink stat fingerprints remain recorded;
content SHA256 values are unchanged and the root's postlink cache is updated.

Totals through the V16 batched-rectangle screens are **25 complete campaigns, 646 worker attempts, 644 successful workers,
two historical baseline OOMs, 21,568 timed complete updates and 430 passing
first-step benchmark comparisons**. The diagnostic profiles and failed counter
attempts below are excluded. The parity goal remains unmet.

## Current B32 diagnostic after batched attention tuning

One retained-V16 Nsight Systems profile captures five complete updates after
32 warmups with the accepted combined flags. All five repetitions match 499
model labels, launch geometry/resources and 251 family checks. Two cached
graphs supply two and three repetitions. The trace contains 591 CUDA kernel
activities per update versus the reported complete-step count of 590: scalar
loss readback adds one `materialize_view` outside model capture and owner
counters. No mapped model kernel has local-memory allocation.

Profiled complete-update wall time averages 85.829 ms. Summed CUDA kernel
activity is 84.159 ms/update, including 83.111 ms in the captured model:

| Domain | Summed model kernel activity, ms/update |
| --- | ---: |
| Normalization and residual spine | 22.077 |
| FFN | 21.952 |
| Attention core | 15.355 |
| Head and loss | 9.781 |
| QKV projections | 9.466 |
| Explicitly ambiguous | 3.322 |
| Attention output projection | 1.157 |

The attention contraction total is 8.313 ms: QK 1.661, PV 1.508, dV 1.231,
dP 1.612, dK 1.129 and dQ 1.172 ms. Forward/backward attention normalization
still contributes 3.422/3.463 ms. The largest individual kernel is indexed-loss
backward at 3.468 ms; loss forward costs 1.496 ms. Complete-step label-informed
families include GEMMs 25.634 ms, split-K partials 11.636 ms, unfused pointwise
12.372 ms, other pointwise regions 5.992 ms and reductions 5.241 ms.

Compared descriptively with the separate V13 profile, attention-core activity
falls from 28.315 to 15.355 ms, norm/residual from 24.167 to 22.077, and FFN from
23.668 to 21.952. V16 includes both Column Four and BatchedTallScores; these
separate profiles do not isolate either effect. Forward/backward normalization
is slightly higher than V13's 3.354/3.359 ms, and the indexed-loss cost persists.
Ambiguous activity remains explicit at 3.322 ms. Norm/residual includes branch
contribution and residual adds, so its whole duration cannot be attributed to
LayerNorm or claimed as fusible. Summed durations are neither unprofiled
wall-time shares nor achievable speedups.

Profile first-step gradients, postweights and scalar bits match both native
TallScores trial controls exactly, as do loss/norm/scale bits for all five
measured updates. The unchanged compiled-reference gates pass: gradient relative
L2 0.0167094, postweight 0.000365773, update 0.143637 and normalized loss error
0.00000908. All five steps retain active 24-region counters, replay, direct
bindings and norm partials. The descendant/executable-aware sampler observes one
worker, no foreign compute and a 5,818 MiB whole-worker sampled peak, a lower
bound across setup, qualification, warmups and profiling. Exact pool counters
remain separately scoped in the raw record.

The bundle's `profile_v16_b32` preserves the full mapping, trace/source hashes,
commands, results and numerical checks. This adds the **fifth excluded qualified
diagnostic profile**, with no benchmark worker/update/comparison increments.

## Counter observability attempts

Two retained-V14 B32 Nsight Compute attempts targeted the proposed head split-K
kernel by name. The first requested hardware-counter sections; the second
requested only `LaunchStats`. Both terminated with `ERR_NVGPUCTRPERM` and exit
code 1. Commands, logs and worker artifacts are retained as
`failed_ncu_observability_attempts`. No driver settings changed. Neither attempt
verified the intended kernel's grid/role or produced accepted counters or
driver-JIT SASS. They add no qualified diagnostic-profile, benchmark-worker,
measured-update or first-step benchmark-comparison count. Offline SASS remains
explicitly distinct from the unavailable driver-JIT disassembly.

## Stronger PyTorch control: compiled model and loss with CUDA graph replay

The latest screen keeps native V16 and its accepted Column Four + LargeOff +
BatchedTallScores combination unchanged. It adds a stronger PyTorch reference:
`torch.compile(model_loss, mode="reduce-overhead", dynamic=False, fullgraph=False)`
includes both the model and cross entropy, with automatic backward compilation.
Global clipping and stock fused Adam remain outside `torch.compile`. The older
control uses default compilation options on the model only; cross entropy is
outside that compiled callable. Both retain SDPA, F32 master parameters, explicit
BF16 boundaries, disabled TF32/reduced-precision reduction flags, the same fixed
synthetic batch and complete synchronized update timing.

This is a change in **both compilation boundary and mode**. Its gain cannot be
attributed to CUDA graph replay alone. Historical reports and tables above remain
unchanged and describe their original model-only/default controls. Future parity
claims must use the fastest qualified measured control per case, rather than
selecting a weaker control. Here that means the median of two fresh-process
medians, not the fastest single process.

Two fresh processes per mode, reversed order in trial 1, each perform 32 warmups
and 32 measured complete updates:

| Case | Native V16, ms | Default model-only, ms | Model+loss / reduce-overhead, ms | Fastest qualified control | Native/control latency |
| --- | ---: | ---: | ---: | --- | ---: |
| 55M / S512 / B1 | 10.602 | 10.473 | 8.258 | Model+loss / reduce-overhead | 1.284x |
| W512 / S512 / B32 | 84.182 | 33.863 | 29.991 | Model+loss / reduce-overhead | 2.807x |
| 211M / S128 / B1 | 21.587 | 22.158 | 22.397 | Default model-only | 0.974x |

The stronger control is consistently faster in the 55M and B32 processes:
55M takes 8.285/8.232 ms, and B32 takes 29.860/30.121 ms. The older 55M default
control varies substantially at 9.042/11.905 ms; its aggregate 10.473 ms must not
be used to claim near-parity against native's 10.602 ms. At 211M, default model-only
is the faster measured control (22.158 ms versus 22.397 ms). Native's 21.587 ms
remains a two-process screen advantage, not full-suite parity. B32's current gap
is about **2.81x latency**, rather than the approximately 2.51x gap against the
historical weaker control.

All 18 workers pass, all 576 timed updates have finite metrics, and all 12
first-step comparisons pass the unchanged fixed gates, including full clipped
gradient, postweight, update and own-gradient Adam checks. Qualification,
compilation, warmups and snapshot dumps are excluded from steady timing. This
adds one campaign; the three configurations do not establish a new all-shape
PyTorch baseline or equivalent later training trajectories.

### Direct replay verification at actual sizes

The small control smoke profile observes six `cudaGraphLaunch` calls and 62
captured nodes repeated three times. Two additional Nsight Systems profiles use
the actual 55M/S512 and B32 fixtures, 32 warmups and five measured updates.
Each observes ten graph launches: forward/loss and backward replay separately.
The 55M trace has 243 captured nodes repeated exactly five times (1,215 captured
kernel instances); B32 has 249 repeated five times (1,245 instances). Kernel
names, geometry and resources are stable for each repeated node. The timed traces record no begin/end
capture, graph instantiation, exec-update or graph-upload APIs inside
either ROI. FlashAttention forward and backward kernels occur in both
compiled traces. These observations prove replay for those two cases; the 211M
control was not directly profiled here. Raw result flags that say replay was
not directly verified are left intact; external audits supply that evidence.

All six supplementary first-step comparisons pass: native versus each profiled
reference, and each profiled reference versus both unprofiled runs of the same
model+loss policy. These reference executions are **not bit-identical**. Profile
versus unprofiled gradient relative L2 is 0.00353–0.00379 at 55M and
0.00169–0.00204 at B32; update relative L2 is 0.0336–0.0359 and 0.0231–0.0274.
The first-step loss values match between reference runs. Five measured B32
updates differ by up to 0.00877 in loss, 0.85508 in norm and 0.09861 in clip
scale; the two unprofiled runs themselves differ by up to
0.00516/0.31096/0.02592 over the same five steps. All metrics remain finite.
The audit preserves these differences without a new fitted threshold or a
claim about their exact cause. Fixed qualification concerns the first update;
it does not establish later trajectory identity or learning quality. This is
distinct from the native V16 profile above, whose snapshots and five scalar
records match its native controls exactly.

### Memory and provenance scope

B32's unprofiled process sampler observes maxima of 5,806 MiB native,
3,106 MiB default model-only and 2,032 MiB model+loss/reduce-overhead. These
requested-50-ms whole-process observations are lower bounds. Torch's reset
steady peak-allocated counter for the graph control is 186,835,968 bytes, while
its lifecycle peak allocated is 1,655,988,736 bytes and lifecycle peak reserved
reaches 1,784,676,352 bytes. The steady allocated counter does **not** mean the
whole control uses 186 MB of VRAM: private graph-pool reservations, allocator
scope and context/module memory must remain distinct. Native exact pool counters
and Torch allocator counters are retained in the raw data with their own scopes.
No allocation counter is substituted for whole-process memory or a capacity
limit. No foreign compute was observed by the unprofiled campaign samplers.

The B32 reference profile's exact interpreter/reference/archive/output argument
matching verifies its sampled worker and observes 1,968 MiB, again a lower bound.
The optional 55M profile sampler failed to identify Nsight's reparented target;
its original record is retained, and its samples are explicitly unattributed
and excluded from worker-peak and foreign-compute conclusions. The GPU workload
and trace completed successfully under the shared lock, with idle before/after
snapshots; no GPU rerun or driver setting change was needed for that telemetry
limitation. Profiled timings are excluded from performance tables.

The bundle embeds `cudagraph_controls`, its audit, and
`reference_cuda_graph_replay_verifications`. Native code maps to immutable V16.
The 167 source checks comprise 164 archived compilation inputs and three actual
Python inputs saved separately as
[reference control v1](benchmarks/transformer-training-sources/reference-cudagraph-control-v1.py),
[runner control v1](benchmarks/transformer-training-sources/run-cudagraph-control-v1.py)
and [validator control v1](benchmarks/transformer-training-sources/validate-cudagraph-control-v1.py).
Their hashes match the recorded live campaign/profile sources; the older Python
files inside the V16 Rust archive were not used for these reference runs. All
47 new F32 files (43 campaign/smoke and four actual-profile snapshots) have
persisted content hashes; postlink cache verification preserves those contents.

`TallScoresByGrid` initially had targeted prequalification only. The following
V17 campaign supplies its first paired measurements with the stronger reference.
That signature-45 variant was absent from V16; its source is now retained in V17.

Totals through the stronger three-case control screen are **26 complete benchmark campaigns, 664 attempts, 662 successful
workers, two historical baseline OOMs, 22,144 timed complete updates and 442
passing first-step benchmark comparisons**. Eight qualified diagnostic captures
are excluded: five native profiles, the small reference smoke profile and these
two actual-size reference profiles. Two failed NCU counter attempts remain
separately excluded and do not count as qualified captures. The parity goal
remains unmet.

## V17 LayerNorm and grid policy: faster batches with a memory cost

The accepted **opt-in benchmark candidate** at this stage combines V17 LayerNorm-pair
fusion with `TallScoresByGrid`, retaining Column Four and LargeOff. Both new
options remain off by default. This complete sweep compares the frozen V17
binary against V16 TallScores and fresh PyTorch model+loss/reduce-overhead SDPA
controls at all eight shapes. Each mode uses two fresh processes, reversed order
in trial 1, with 32 warmups and 32 measured complete updates.

The two changes have disjoint runtime eligibility. At W512/S512/B8–32,
LayerNorm fusion accepts nine forward/backward pairs, while high-BH attention
geometry remains twenty tall/four wide programs in both native arms. At
55M/S512/B1 and W512/S512/B2, LayerNorm is ineligible and ByGrid activates only
eight score GEMMs; the 64-CTA width roles retain their ordinary schedules.
S128 cases have neither change. All candidate counters remain stable through
qualification, warmups and timing; no LayerNorm runtime fallback is recorded.

| Case | V16 control, ms | V17 candidate, ms | Fresh model+loss CG control, ms | Active change | Native/CG latency |
| --- | ---: | ---: | ---: | --- | ---: |
| 211M / S128 / B1 | 21.478 | 21.670 | 22.045 | No-op | 0.983x |
| 55M / S512 / B1 | 10.528 | 10.386 | 8.214 | 8 score GEMMs | 1.264x |
| W512 / S512 / B8 | 17.483 | 15.707 | 9.114 | 9 LN pairs | 1.723x |
| W512 / S512 / B16 | 38.195 | 32.907 | 15.900 | 9 LN pairs | 2.070x |
| W512 / S512 / B32 | 84.129 | 68.188 | 29.622 | 9 LN pairs | 2.302x |
| W512 / S128 / B2 | 3.318 | 3.323 | 3.127 | No-op | 1.062x |
| W512 / S512 / B2 | 6.587 | 6.241 | 3.932 | 8 score GEMMs | 1.587x |
| W1024 / S128 / B2 | 7.766 | 7.725 | 6.936 | No-op | 1.114x |

The active LayerNorm cases improve by about 1.11x/1.16x/1.23x at B8/B16/B32.
ByGrid improves the measured low-BH screens by about 1.014x at 55M and 1.055x
at W512/B2/S512. These are paired full-update observations, not gains to multiply
with prior campaigns. The approximately 0.2 ms change at the 211M no-op and
other unsupported-shape differences are unattributed process variation, not
causal speedups or regressions from the new kernels.

A scan of completed reports with matching model geometry finds these fresh CG
aggregate controls are the fastest recorded qualified controls for all eight
cases. That is a record-level comparison, not proof of a universally optimal
controller. V17 does not remeasure default model-only compilation; the earlier
three-case control screen remains intact. Native remains slower on seven cases,
including **68.188 versus 29.622 ms at B32 (2.30x latency)** and **10.386 versus
8.214 ms at 55M (1.26x)**. The 211M candidate's small 21.670 versus 22.045 ms
screen advantage does not establish suite parity.

### Numerical and source contract

LayerNorm fusion replaces kernels only after inspecting their original selected
plans and all 18 relevant stored/same-kernel boundaries. It preserves the
primitive mean, centered variance, exp/log inverse standard deviation and VJP
operation order. Explicit F32 round checkpoints apply at original stored edges;
original same-kernel FMA opportunities remain. It does not substitute a compact
LayerNorm derivative or disable FMA globally. Original y/n/m/a/r identities and
typed BF16 incoming-gradient widening remain valid for ordinary consumers.
Unsupported or escaping structures retain ordinary execution.

Qualification passes 185 CPU library tests, 429 real-device CUDA library tests
(four existing ignored), 312 integration tests (one existing ignored), strict
Clippy, CPU/CUDA checks and formatting. The focused B8 four-layer test uses a
smaller FFN hidden extent 64 and checks three evolving [1,2,1]-microbatch intervals
with exact loss/norm/scale, gradients, masters and both Adam moments. It also
checks eight surviving pairs after one saved-stat escape, unsupported-partition
fallback, nonfinite rollback/recovery and all 65,536 BF16 incoming bit patterns.
NaN-category checks do not promise NaN payload identity.

The offline audit assembles 111 retained modules and verifies 43 original
same-kernel witnesses. Its static FFMA counts are per thread: a fused row thread
processes four elements, versus one in the original pointwise thread. Actual
row-kernel driver queries report 38 forward/40 backward registers, 1,536 static
shared bytes and zero local bytes. Offline SASS is not driver-JIT disassembly,
and no achieved-occupancy or V17 profiling claim follows from these checks.

The full hidden 1408 benchmark independently confirms all nine pairs and passes
all 32 fixed first-step comparisons. Native-control gradient/postweight/update/
loss errors are zero. All 32 paired full gradient/postweight content hashes match,
and all 1,040 native update pairs retain bit-identical loss/norm/scale and matching
optimizer steps across qualification, warmups and timing. Every native step
retains replay, zero master copies and active norm partials; steady captures
retain two variants/two misses and hits 32–63. Model kernel counts fall 157 on each
active LayerNorm case (482→325,498→341,499→342); low-BH and no-op counts remain
unchanged. Full later gradient/moment identity is established by the focused
three-interval fixture, not by scalar-only benchmark rows.

### Memory regression is part of the result

Fewer launches do not imply lower memory. The late rewrite preserves original
compiled storage and extends newly required reads through backward before
incremental reuse planning. It does not automatically free every allocation
whose producer kernel is removed. The source allows these longer lifetimes to
change retained buffers and transient pool peaks; this audit does not attribute
every allocation block or the entire NVML delta to an individual tensor.

| Case | Owned F32 and steady pool USED_HIGH increase, bytes | Lifecycle pool USED_HIGH increase, bytes | Sampled whole-process peak, control→candidate MiB |
| --- | ---: | ---: | --- |
| W512 / S512 / B8 | 8,519,680 | 92,422,144 | 1914–1936→2000–2064 |
| W512 / S512 / B16 | 17,039,360 | 184,844,288 | 3152–3248→3376–3472 |
| W512 / S512 / B32 | 34,078,720 | 369,688,576 | 5616–5712→6288 |

At B32 this is **34,078,720 additional owned F32 bytes and steady USED_HIGH
bytes**, plus **369,688,576 bytes in lifecycle USED_HIGH**. Steady pool reserved
high increases301,989,888 bytes and lifecycle reserved high 402,653,184 bytes.
The sampled process increases 576/672 MiB in the two native trial pairs.
BF16 payload and cumulative unpooled allocation counts remain unchanged.
Other cases retain their native payload/pool counts; ByGrid-only processes
observe a small 2 MiB sampled change without a corresponding pool change.

Pool USED_HIGH/RESERVED_HIGH are exact for the queried CUDA pool, not whole-process
VRAM, and current/default views of the same pool must not be added together.
NVML peaks are requested-50-ms sampled lower bounds including broader process
memory. Extra pool reservation accounts for only part of the larger process
observation; the remainder is unattributed. The two scopes establish a memory
regression here, not an exact physical allocation decomposition. This candidate
still completes B32; no larger batch/model capacity limit was measured. The
largest parameter-count configuration remains the tested 211,147,776-parameter
model, not a maximum-fit claim. Torch graph-pool counters retain the separate
scope described above.

The complete report, audit and prequalification are embedded as
`layer_norm_and_grid`, `layer_norm_and_grid_campaign_audit` and
`layer_norm_pair_prequalification`. All 169 recorded source hashes and both
features' retained prequalification source hashes match immutable
[candidate-v17](benchmarks/transformer-training-sources/parity-candidate-v17.tar.gz)
(SHA256 `9b9468587b196ba3fb318255507ab48492eabbf65bd20ea2556a4f2cece864f4`).
The candidate binary SHA256 is
`a98b534222ba51a93dbb584d064ab53594f207167173d06feea50706478c7b87`;
the native control is retained V16. Reference/runner hashes remain the actual
strong-control-v1 Python sources above, also present in the V17 snapshot.
All 104 campaign F32 files were stream-hashed with stable prelink fingerprints;
root's verified hardlink pass changes no contents and updates its postlink cache.

Totals are **27 complete campaigns, 712 benchmark attempts, 710 successful workers,
two historical baseline OOMs, 23,680 timed complete updates and 474 passing
first-step benchmark comparisons**. Eight qualified diagnostic captures and two
failed NCU attempts remain excluded. V17 adds no new performance profile: V16
activity percentages cannot be treated as V17's new breakdown. All options
remain default-off, and full-suite parity remains unachieved.

## V18 wider LayerNorm eligibility and bounded Split-K tiles

The next accepted **opt-in benchmark candidate** adds the S512 LayerNorm
W512/W1024, B>=1 eligibility extension and `SplitKBf16TilePolicy::Square64` to
V17. All options remain off by default. Five cases use two fresh processes per
mode, reversed order in trial 1, 32 warmups and 32 measured complete updates.
Controls are the retained V17 binary and fresh model+loss/reduce-overhead SDPA
CUDA-graph workers. This is a five-case extension, not a new eight-case sweep.

| Case | V17 control, ms | V18 candidate, ms | Fresh CG control, ms | Native control/candidate | Candidate tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 55M / S512 / B1 | 10.3519 | 9.8169 | 8.3957 | 1.0545x | 52,155 |
| W512 / S512 / B2 | 6.3771 | 5.8116 | 3.9594 | 1.0973x | 176,199 |
| W512 / S512 / B8 | 15.7141 | 15.6848 | 9.0139 | No-op control | 261,144 |
| W512 / S512 / B16 | 32.9945 | 31.5755 | 15.9194 | 1.0449x | 259,442 |
| W512 / S512 / B32 | 68.2362 | 64.7527 | 29.8107 | 1.0538x | 253,024 |

Step latency is the median of the two process medians; tokens/s is batch tokens
divided by that latency. The wider LayerNorm matcher accepts nine forward and
nine backward pairs in both small-batch cases, with zero runtime fallbacks.
Their model kernel counts fall 450→295 and 479→322. ByGrid remains eight score
programs there. At B8–32 both native arms already have nine LayerNorm pairs and
20 tall/four wide attention programs. Split-K activates zero/16/17 new partial
kernels at B8/B16/B32; four original partial programs remain unchanged. Small
batches also select zero new Split-K kernels, retaining one original program.
These counters are stable through every native qualification, warmup and timed
step. The B8 0.19% latency difference is unattributed process variation.

The LN extension retains closure, typed BF16 widening, all 18 original-plan
boundary checks and primitive forward/VJP order. Its singleton prototype handles
the absent batch-axis broadcast without relaxing the other guards. S128 and
W2048 remain excluded. Split-K changes only proven old 32×32×64 partials after
all original packing decisions: four contiguous partitions, full aligned output
tiles, part-K divisible by 64 and sufficient grid/resource coverage. The new
64×64×32 tile retains each partial's increasing K16 MMA sequence, F32 partial
storage and the existing left-associated four-way reduction with one BF16
rounding. It adds no repacking, partition-count or global-buffer change.

All 30 workers and 20 fixed first-step comparisons pass. All ten native trial
pairs have identical full clipped-gradient and postweight SHA256s. All 650
paired scalar records—qualification plus 32 warmups plus 32 measured updates—
match loss, norm, scale and optimizer step exactly. Each native step retains
replay, zero master copies and norm partials; timed captures have two variants,
two misses and hits 32–63. This does not establish later full-tensor identity
against PyTorch. The worst CG comparisons retain gradient relative L2 0.016815,
postweight relative L2 0.000474 and update relative L2 0.143977, within the
unchanged gates. No foreign compute process was observed.

Frozen qualification passes **188 CPU library and 445 real-device CUDA library
tests**, with four existing CUDA tests ignored, plus strict Clippy, all-target
CPU/CUDA checks and formatting. V18 does not rerun integration tests; V17's
312 integration passes remain historical. Focused LN tests cover actual B1/W1024,
B2/W512 and old B8 nine-pair training, evolving [1,2,1] microbatches, typed BF16,
fallback and exact state/replay. These four-layer state fixtures use hidden 64;
the benchmark independently qualifies full hidden 2752/1408 first-step snapshots
and subsequent scalar trajectories. Its 223 retained PTX modules assemble offline;
25 old W512/B8 candidate modules remain byte-identical to V17. Split qualification
compares every F32 partial before narrowing across five shape families, changed
feeds, nonfinites and signed-zero inputs, then checks mixed/resident gradients,
masters, moments, rollback and recovery. Actual B32 new partials report 80
registers, 18 KiB shared and zero local bytes; API block capacity is three,
versus four for the old 53-register/24-KiB family. This is a resource tradeoff,
not a measurement of achieved occupancy. The four existing 64-square partials
retain 80 registers/35 KiB/two-block API capacity.

### Incremental memory cost

| Case | Owned F32 / steady pool USED_HIGH increase, bytes | Lifecycle pool USED_HIGH increase, bytes | Sampled process peak, V17→V18 MiB |
| --- | ---: | ---: | --- |
| 55M / S512 / B1 | 4,208,640 | 4,208,640 | 2170→2198 |
| W512 / S512 / B2 | 2,129,920 | 5,177,096 | 1018→1050 |
| W512 / S512 / B8 | 0 | 0 | 2000–2032→2064 |
| W512 / S512 / B16 | 0 | 0 | 3472→3376–3442 |
| W512 / S512 / B32 | 0 | 0 | 6288→6064–6288 |

Both small-batch cases add 33,554,432 bytes of steady/lifecycle pool reservation.
High-batch payload and all pool high-water counters are unchanged from V17;
the earlier V17 LayerNorm memory regression persists. Sampled NVML changes
without corresponding pool changes are unattributed, not exact allocation
savings. Pool counters cover the queried pool only; current/default views of the
same pool are not additive. Requested-50-ms NVML whole-process peaks are lower
bounds on instantaneous use. This campaign establishes no new capacity limit.

The complete report/audit are embedded as `layer_norm_width_and_split` and
`layer_norm_width_and_split_campaign_audit`, with separate LN-width and Split-K
prequalification. All 169 recorded report source hashes and 170 freeze-manifest
hashes match [candidate-v18](benchmarks/transformer-training-sources/parity-candidate-v18.tar.gz)
(SHA256 `a21e6ab4f729584593e92ff76253908487b26a3e62d3f7bb2326f1430473a38c`),
including the actual strong-control-v1 Python sources. The retained benchmark
binary SHA256 is `b099bf5b7d177fb021aa71422b7efc773b5deb1b3a9bcdcbf467bb0c24cdfd27`.
All 65 campaign F32 files were stream-hashed with stable prelink fingerprints;
the later deduplication changes no content. Later CE, S128 and four-warp source
preparation is not V18 qualification evidence.

## V18 B32 diagnostic: substantial work remains

A separate retained-V18 B32 capture passes both fixed validations and matches
both unprofiled V18 trials' full first-step gradient/postweight hashes and all
five timed loss/norm/scale records. It records five complete updates, 2,170 CUDA
kernel instances and two capture groups. There are 342 model kernels and 434
actual complete-step kernels per update: the reported 433 omit one scalar-loss
readback materialization. Traced model kernels report zero local memory.

| Semantic model domain | Summed kernel activity per update, ms |
| --- | ---: |
| FFN | 20.0994 |
| Attention core | 15.3360 |
| Head and loss | 9.2570 |
| QKV projection | 8.6927 |
| Normalization and residual | 5.9499 |
| Explicitly ambiguous | 3.0660 |
| Attention output projection | 1.1574 |

Total model activity is 63.5583 ms and complete-step kernel activity 64.5599 ms.
Across complete-step families, ordinary GEMMs account for 25.5357 ms, split-K
partials 8.1441 ms, attention normalization 3.3698/3.3669 ms forward/backward,
and indexed loss 1.4974/3.6876 ms. Explicit LN forward/backward families are
1.2701/1.1189 ms; seven outer residual adds remain separate at 1.0181 ms.
These are diagnostic activity sums, not unprofiled wall-time shares, isolated
speedups, removable-time predictions or achieved occupancy/bandwidth. Separate
V16 domain deltas are retained as descriptive comparisons only. Ambiguous
ownership remains explicit. Raw traces stay under the target profile directory;
bounded mappings, checked role neighbors, SQL queries, hashes, process evidence
and both validations are embedded as `profile_v18_b32`.

Through V18, totals were **28 complete campaigns, 742 benchmark attempts, 740 successful
workers, two historical baseline OOMs, 24,640 timed complete updates and 494
passing first-step benchmark comparisons**. Nine qualified diagnostics and two
failed NCU attempts remain excluded. The five profiled V18 updates are not added
to benchmark timing totals.

## V19 S128 LayerNorm extension and four-warp control

V19 contains two completed campaigns against retained V18. The full eight-case
campaign enables the ordinary BF16 four-warp block together with the S128
LayerNorm extension. Its companion measures only the two S128 cases with
**four-warp off**, isolating LayerNorm. Each mode uses two fresh processes,
32 warmups and 32 synchronized complete timed updates; process order reverses
in trial 1. The reference compiles model and loss with reduce-overhead CUDA
graphs; clipping and fused Adam remain outside that compilation.

| Case | V18 control, ms | V19 LN + four-warp, ms | Fresh CG, ms | Selected four-warp kernels |
| --- | ---: | ---: | ---: | ---: |
| w2048_s128 | 21.5855 | 21.5008 | 21.9733 | 17 |
| w1024_s512 | 9.8594 | 9.9307 | 8.3026 | 13 |
| w512_s512_b8 | 15.6997 | 15.4915 | 9.0395 | 43 |
| w512_s512_b16 | 31.4958 | 31.3346 | 15.9401 | 43 |
| w512_s512_b32 | 64.6544 | 63.8777 | 29.6911 | 42 |
| w512_s128 | 3.3450 | 3.1510 | 2.9917 | 1 |
| w512_s512 | 5.6660 | 5.7033 | 4.0204 | 1 |
| w1024_s128 | 7.7279 | 7.4075 | 6.8835 | 13 |

Four-warp remains experimental. The non-S128 cases are mostly neutral or mixed:
55M/S512 is about 0.7% slower, while B8/B16/B32 are about 1.3%/0.5%/1.2% faster
in this paired screen. These observations do not justify global adoption.
S128 rows above combine two changes and cannot isolate the block contribution.
Both W512 cases select one tied-head gradient kernel, so neither is a no-op.

| Isolated S128 case | V18 control, ms | V19 LN only, ms | Control/candidate | Fresh CG, ms | Fastest qualified recorded CG, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| w512_s128 | 3.31429 | 3.16355 | 1.0476x | 3.14630 | 2.99172 |
| w1024_s128 | 7.75188 | 7.48944 | 1.0350x | 6.90224 | 6.88351 |

The accepted next opt-in baseline is **V19 S128 LayerNorm with four-warp off**.
Eligibility adds only B2/S128/W512 or W1024. Nine forward/backward pairs activate
with zero fallback; W2048 remains outside the LayerNorm domain. All experimental
defaults remain off. The 3.1463 ms fresh W512 control does not replace the faster
qualified 2.9917 ms observation; similarly W1024 retains the 6.8835 ms target.

All 60 workers and 40 fixed first-step comparisons pass. First-step native full
clipped-gradient and postweight hashes match in all 40 paired file comparisons.
Across the qualification update, 32 warmups and 32 timed updates, all 1,300 paired
native loss/norm/scale records are bit-identical. Later full tensor identity is
covered by focused fixtures, not inferred from those scalar trajectories.
Replay stays active; direct-master copies remain zero, with two captured variants
and two misses. All recorded partial-norm paths are active.

Memory is not free: the LN-only W512/W1024 extension adds **2,629,632 /
5,251,072 bytes** of owned F32 payload and exact steady/lifecycle pool USED_HIGH.
W512 pool reservation and sampled whole-process peak stay unchanged (760 MiB).
W1024 reservation rises by 32 MiB and sampled NVML peaks rise from 1,976 to
2,008 MiB in both companion trials. The full campaign has the same exact payload
increments and W1024 sampled peak of 2,010 MiB. Non-S128 full-campaign pool/payload
counts are unchanged, while sampled NVML peaks still increase in some processes:
B8 by 66 MiB in trial 1, B16 by 98 MiB in trial 0, and B32 by 98/226 MiB across
the two trials. This whole-process discrepancy is retained without attributing
its cause or equating it to owned payload. Pool counters are exact for the queried pool, and same-pool views are not additive.
NVML samples remain whole-process lower bounds and are not interchangeable with
framework allocated/reserved counters or complete private CUDA-graph storage.

Qualification includes 191 CPU library tests, 14 focused LN CPU checks, six LN
CUDA filters, ten distinct four-warp checks, final CPU/CUDA all-target checks,
strict Clippy and formatting. No new broad CUDA library or integration run is
claimed. B8/B16 focused resident fixtures retain exact full gradients, masters,
both Adam moments and scalars over [1,2,1] microbatch intervals with changed feeds.
Each selects 43 four-warp kernels alongside nine LN pairs and 24 batched-attention
kernels; B8 has zero SplitSquare kernels and B16 jointly activates 16.

Actual driver resources are 94–124 registers, 18–20 KiB shared, no local memory,
and API capacity four or five blocks for those model fixtures. These are static
resource limits, not achieved occupancy. All 184 offline PTX modules assemble;
86 actual-model baseline/candidate pairs retain eight PTX WMMA and 16 SASS HMMA
instructions and identical shared bytes. SASS here comes from offline cubins,
not driver-JIT export. Whole-plan fallback is proven for conservative source
admission rejection; there is no post-JIT resource fallback.

Frozen V19 source archive SHA256 is
`322d9e06729ce18ef87355db8e3c095535594b6e304e0b7d652e3d382ead16ca`;
retained binary SHA256 is
`61d5e285daf26903245adf109862eccf603284ddbafe3d4b272d26a99bf64886`.
Both raw reports, 173 report source hashes per campaign, frozen source manifest,
focused qualifiers and all 130 F32 snapshot fingerprints are retained. The later
unbuilt cross-phase index fence and CE integration are excluded from V19 evidence.

Through V19, totals were **30 campaigns, 802 benchmark attempts, 800 successful workers,
two historical baseline OOMs, 26,560 timed complete updates and 534 passing
first-step benchmark comparisons**. Nine qualified diagnostics and two failed
NCU attempts remain separate and unchanged.

## V20 ordered cooperative indexed-loss backward

V20 is the accepted next opt-in benchmark baseline, with ordinary four-warp
blocks **off**. Only eligible standalone indexed-loss backward with vocabulary
4096 changes: 256 contributors cooperate on max and exponentials, then lane0
performs the original `+0`, increasing-column F32 sum. The backward per-column
exponential and gradient arithmetic remain unchanged. Forward stays on its old
32-thread path. This preserves sum order; the disconnected parallel-tree and
forward proposals are not measured V20 behavior.

The full eight-case screen uses two fresh processes per mode, 32 warmups and 32
measured complete updates each. All 48 workers and 32 first-step comparisons
pass the unchanged gates. Medians below are medians of independent process
medians; throughput uses the fixed tokens per update. Warmups are excluded from
measured totals.

| Case | V19, ms | V20, ms | Fresh model+loss CG, ms | V20 tokens/s |
|---|---:|---:|---:|---:|
| w2048_s128 | 21.5955 | 21.5696 | 22.1474 | 5,934 |
| w1024_s512 | 9.8339 | 9.7454 | 8.2028 | 52,538 |
| w512_s512_b8 | 15.6664 | 15.0813 | 8.7933 | 271,595 |
| w512_s512_b16 | 31.5081 | 30.0881 | 16.1512 | 272,267 |
| w512_s512_b32 | 64.6359 | 61.9872 | 29.5947 | 264,313 |
| w512_s128 | 3.1571 | 3.1039 | 3.2570 | 82,478 |
| w512_s512 | 5.6758 | 5.7169 | 3.8462 | 179,118 |
| w1024_s128 | 7.4363 | 7.3960 | 6.8441 | 34,613 |

A separate three-case confirmation uses four fresh processes per mode, **256
warmups and 64 measured updates**, for 36 passing workers and 24 passing fixed-gate
comparisons. It retains every main-screen observation, including the two
B2/S512 candidate process medians of about 5.519 and 5.915 ms.

| Confirmed case | V19, ms | V20, ms | Fresh CG, ms | V20 tokens/s |
|---|---:|---:|---:|---:|
| w512_s128 | 2.976777 | 2.927864 | 3.010828 | 87,436 |
| w1024_s128 | 7.448182 | 7.394601 | 6.876625 | 34,620 |
| w512_s512 | 5.685023 | 5.544788 | 3.790161 | 184,678 |

W512/S128 reaches the **case throughput target**: 2.927864 ms is below the fresh
3.010828 ms and strongest historical qualified 2.991720 ms control. The longer
protocol also lowers the V19 median; these runs do not establish warmup as the
cause. W1024/S128 still trails its fresh 6.876625 ms control, and B2/S512 still
trails 3.790161 ms. Full-suite parity remains unmet. The confirmation is a
separate protocol, not additional independent samples of the short screen.

Every candidate qualification, warmup and measured record reports exactly one
selected cooperative backward program. All 65-step main and 321-step confirmation
Native trajectories have exact paired loss, pre-clip norm, gradient scale and
optimizer-step values: 4,892 paired records / 9,784 records inspected. All 28
Native trial pairs have identical first-step gradient and postweight SHA256
values (56 paired files). All measured Native updates replay successfully with
two capture variants. These checks do not replace convergence validation.

The lifetime and steady queried-pool `USED_HIGH` differences are **zero bytes**
for every Native pair; prior memory costs persist. Sampled process peaks vary in
larger cases (B32 baseline/candidate 6,290/6,064 and 6,064/6,288 MiB), so they do
not show an allocation saving. NVML samples at about 50 ms give a whole-process
observed lower bound; exact pool accounting covers that queried pool, excluding
context/modules and other pools. No foreign compute was observed by sampling;
all before/after process snapshots are empty. No capacity expansion is claimed.

Qualification comprises four device-free tests, four focused CUDA tests and 21
affected integration tests across `indexed_loss_rows`, `indexed_loss_forward_rows`,
`ptx_indexed_ops` and `resident_adam`, with zero skips. Parent final checks add 191
CPU library tests, CPU/CUDA all-target checks, strict Clippy, formatting and release
build. This is not a new broad CUDA library/integration-suite run. Focused tests
check exact max/EXP/SUM/gradient bits and NaN classes, near-zero signed-zero behavior,
all 65,536 physical BF16 patterns, changing inputs/targets, cache toggles, replay,
resident state/moments and invalid-update rollback/recovery. Actual driver resources
are 25 registers, 17,412 bytes shared and zero local bytes; reported capacity five
at 256 threads is not measured occupancy. Eight offline PTX modules assemble for
SM89; offline SASS is distinct from driver-JIT SASS.

Frozen V20 archive SHA256 is
`b9e3472edf5de24c682db6117d4c668d3059c3f3b65bb457dd9edcfe55242749`;
retained binary SHA256 is
`d2b257093a1c483b8a52c75e98dbdcec7323dfa97e6c22e193818ca6d6a6e71c`.
Both complete reports, frozen source/Python hashes and qualification audits are
embedded in the raw bundle. All 179 campaign F32 files (29,194,319,872 bytes) were
stream-hashed with stable pre/post file fingerprints before deduplication; those
fingerprints are historical immutable evidence. Later source-only forward CE,
Square128 and SmallSplit work is excluded.

Through V20, totals were **32 campaigns, 886 attempts, 884 successful workers, two
historical baseline OOMs, 30,400 measured complete updates and 590 passing
first-step benchmark comparisons**. Qualification and warmup updates are excluded.
Ten qualified diagnostics (seven Native semantic profiles and three reference
replay verifications) and two failed counter probes remain excluded.

## V20 B32 diagnostic: ordered loss backward is smaller

A separate five-update Nsight capture of the frozen V20 candidate passes both
fixed-gate comparisons against the fresh model+loss control and unprofiled Native.
First-step gradient/postweight SHA256 values and all five measured loss/norm/scale
records exactly match **both** main B32 Native trials. The profile contains 342
model kernels and 434 actual complete-step kernels per update: 433 reported plus
one scalar loss-readback materialization. All 2,170 instances map consistently
across two capture groups; model kernels report zero local bytes. Actual counters
are CE backward one, LN nine forward/nine backward with no fallback, SplitSquare
17 selected/four fallback and ByGrid 24 selected.

Summed kernel activity is **61.5805 ms/update**, including 60.5815 ms in the model
capture. Ordered CE backward contributes 0.8707 ms versus 3.6876 ms in the separate
V18 diagnostic; unchanged forward CE contributes 1.5507 ms. These are descriptive
activity sums from distinct captures, not an isolated causal wall-time reduction,
achievable speedup, hardware utilization or benchmark observation.

| Label-informed model domain | Summed kernel ms/update |
|---|---:|
| FFN | 19.9939 |
| Attention core | 15.1610 |
| QKV projection | 8.6606 |
| Head and loss | 6.6000 |
| Normalization and residual | 5.9257 |
| Ambiguous ownership | 3.0798 |
| Attention output projection | 1.1604 |

The raw bundle embeds derived SQL queries, checked role-neighbor mappings,
commands, resources, both fixed validations, process observations and frozen
source/Python/binary provenance. Raw trace/SQLite remain under
`target/transformer-training/parity-profile-v20-b32`. Sampled process memory and
queried-pool highwater retain their different scopes. This adds one qualified
diagnostic, taking that count from nine to ten, and adds **zero** benchmark
workers, measured updates or first-step benchmark comparisons. The two failed
hardware-counter probes remain separate; no hardware counter claim is added.

## Bounded Square128Streamed qualification; no performance result

The separate default-off `Square128Streamed` experiment is **not an accepted
benchmark policy**. Its scope is ordinary rank2 BF16 MatMul without an arithmetic
epilogue, full 128-wide M/N tiles and 32-wide K tiles, two async stages, 16-byte
copies, proven shared-memory reuse and at least 128 CTAs. Four counted output
loops preserve coordinates, rounding, order and barriers while reducing compiler
register pressure relative to the initial unrolled implementation.

| Actual loaded candidate orientation | Registers | Shared bytes | Local bytes | API block capacity | Admission |
|---|---:|---:|---:|---:|---|
| NN / TT | 166 | 37,888 | 0 | 1 | Rejected |
| NT | 128 | 40,960 | 0 | 2 | Admitted |
| TN | 126 | 34,816 | 0 | 2 | Admitted |

The unchanged gates require at most 128 registers, zero local bytes and capacity
at least two; capacity is not measured occupancy. The initial unrolled NN leaf
used 255 registers and 24 local bytes and was also rejected. Admitted NT/TN leaves
match baseline finite/signed-zero output bits and infinity/NaN classes under
changed feeds and Off/On/Off toggles; NaN payload identity is not promised. NN/TT
rejection and a forced rejection restore the complete baseline plan, PTX and
output bits. The four legacy orientation controls and retained V15
Tall/Wide/TallThenWide PTX remain byte-identical.

Focused qualification has **11 passes and two explicit ignores**: candidate
resident-state and actual-decoder activation remain unqualified. There is no
performance campaign, timing result, new scorecard entry or full-model claim.
The V21 SmallSplit numerical screen explicitly keeps the large-tile policy Off.
The frozen source/binary manifest is
`target/transformer-training/parity-v21-freeze.json`; bounded evidence is
`target/paper-kernels/square128-streamed-qualification/qualification.json`, with
actual loaded PTX/resources under `final/` and counted-loop evidence under
`counted-loop/`. Independently assembled SM89 SASS retains the four epilogue
backedges and K loop; it is not driver-JIT disassembly. This bounded leaf evidence adds no benchmark measurements and does not change
the accepted large-tile policy.

## V21 SmallGridSplitK: modest gains in three active cases

The next accepted opt-in addition is `--small-grid-split-k-bf16` **only for its
qualified active scope**. It remains off by default; Square128Streamed and
ordinary four-warp blocks are off in this campaign. Original long-split priority
is unchanged. SmallGridSplit partitions eligible low-grid rank2 BF16 products
into four contiguous K slices, writes F32 partials and left-folds the four
partials before one BF16 rounding. This changes association versus the unsplit
product; it is not a claim of old-unsplit bit equivalence.

Two fresh processes per mode use 256 warmups and 64 measured complete updates,
with reversed trial execution order. All **18 workers and 12 first-step benchmark
comparisons** pass the unchanged fixed gates. These are medians of independent
process medians, with fixed tokens per update.

| Active case | V20, ms | V21, ms | Latency reduction | Fresh model+loss CG, ms | V21 tokens/s | Selected small splits |
|---|---:|---:|---:|---:|---:|---:|
| w2048_s128 | 21.500727 | 21.152285 | 1.62% | 22.142913 | 6,051 | 17 |
| w512_s128 | 2.928097 | 2.899798 | 0.97% | 3.031517 | 88,282 | 20 |
| w1024_s128 | 7.418373 | 7.357683 | 0.82% | 6.943552 | 34,794 | 8 |

The gains are modest; no broad eight-case V21 performance claim is added. The
strongest qualified recorded controls remain 21.973274 ms for 211M, 2.991721 ms
for W512/S128 and 6.844051 ms for W1024/S128. The first two cases meet their
throughput targets; W1024 remains behind. The suite goal remains **two of eight
cases**, with V20 observations retained for the other five shapes.

All 321 qualification/warmup/measured records in each candidate trial report
17/20/8 selected SmallGridSplit programs with zero fallback for the three active
cases. All 64 measured updates replay, using two capture variants. Across 1,926
paired Native records, summation association changes are visible: loss differs
in 303/321 records for 211M and 320/321 for each smaller case in each trial;
pre-clip norms differ in all 321. None of the twelve paired first-step gradient
or postweight SHA256 values match V20. All fixed **global** first-update gates
pass; no exact old-trajectory or convergence claim is made.

Both trials add **4,194,304 bytes** of owned F32 and lifetime/steady queried-pool
`USED_HIGH` for 211M, and **5,767,168 bytes** for W512/S128. W1024 adds zero measured
owned/pool highwater bytes. Observed process peaks are 6,552→6,552 MiB,
760→792 MiB and 2,008→2,008 MiB respectively. Pool figures are exact for that
queried pool; ~50 ms NVML peaks are whole-process observed lower bounds, not
instantaneous maxima. Prior memory costs persist; no capacity expansion is
claimed. Sampling observed no foreign compute.

A separate recovered **first-update numerical screen** supplies 14 passing fixed
gate comparisons, excluded from benchmark totals. Its original four process
records remain `missing_results`: the harness requested `candidate/results.json`
but looked in `artifacts/results.json`. Every Native process exited zero. Recovery
uses the completed outputs and unchanged validator without rerunning the GPU;
these are not OOMs or Native execution failures. Actual selected counts are
17/20/8 in the three active shapes. W512/S512 has zero selected/four fallback
programs because a saved PV operand map fails the affine proof; it is a fallback
control, not an active-policy qualification.

Focused SmallSplit qualification has ten distinct passing tests, including five
CUDA filters. Each F32 partial and final left-folded BF16 result is checked against
four independently packed ordinary GEMMs, including nonfinite classes. Leaf
unsplit differences are retained. Resident state/moments exactness compares the
same SmallSplit policy with allocation reuse off/on, including changed feeds,
[1,2,1] microbatches and rollback/recovery; it does not establish old-unsplit state
identity. Driver leaves use 76–82 registers, 18–20 KiB shared and zero local bytes,
with API capacity two or three; this is not measured occupancy. Workspace is
16MN bytes per admitted node, bounded by 8 MiB, not a live-memory estimate.

The corrected structural fixture uses W1024 hidden width 2752 rather than 2816;
its earlier metadata is preserved. Shape-only eligibility is distinct from actual
map/JIT selection. The V21 freeze records 197 CPU library tests, CPU/CUDA all-target
checks, strict Clippy and format/diff checks. No new broad CUDA suite is claimed.
Square's two intentional ignored decoder/resident tests remain unqualified.

Evidence pointers are `target/small-grid-split-k-audit/qualification.json`,
`first-update-screen/recovered-qualification.json`, `validations-direct/*.json`
and `target/transformer-training/parity-small-grid-split/report.json`.
Frozen V21 archive SHA256 is
`3ba845b3d7cb1f4e2e3503b323a2b80777a8da119095103f4bcf7f1d0db955bb`;
retained binary SHA256 is
`4704db46a2a4514b63cd6a7a5ed251d98691431f68eb6cb29682157c4148ea3a`.
Numerical-screen failures/recovery remain separate from measured evidence.
The audit verifies all 190 archived frozen source hashes, Python provenance and
47 F32 files totaling 16,991,860,736 logical bytes (including eight numerical-screen
files); pre-dedup fingerprints remain historical evidence. Raw numerical screen,
original failed harness report, 14 validations and performance report are retained
separately in the bundle. Frozen provenance excludes future attention work.

Through V21, totals were **33 campaigns, 904 attempts, 902 successful workers, two
historical baseline OOMs, 31,552 measured complete updates and 602 passing
first-step benchmark comparisons**. The numerical-only screen and all warmups
remain excluded. Ten qualified diagnostics and two failed probes are unchanged.

## V22 ordered cooperative indexed-loss forward

The accepted opt-in addition is `--indexed-loss-cooperative-ordered-forward`.
Eligible rows with vocabulary 4096 use 256 contributors for max and exponentials, then retain
the original increasing-column, +0-initialized F32 sum and scalar loss formula.
All eleven statistics barriers precede the follower-lane return. Existing
cooperative backward remains enabled. The original TileGraph is snapshotted
before either CE rewrite; rejection restores both original kernels and their
launch metadata. All new defaults remain off.

Two campaigns cover all eight cases: five compare against accepted V20, and
three retain the accepted V21 SmallGridSplit addition. They use two fresh workers
per mode, 256 warmups and 64 measured synchronized complete updates, reversing
mode order in trial 1. All **48 workers and 32 fixed first-update comparisons**
pass. Medians are medians of independent process medians.

| Case | Accepted control, ms | V22, ms | Paired reduction | Fresh model+loss CG, ms |
|---|---:|---:|---:|---:|
| w512_s512_b32 | 62.086233 | 61.026139 | 1.71% | 29.899957 |
| w512_s512_b16 | 30.224187 | 29.665774 | 1.85% | 15.947890 |
| w512_s512_b8 | 15.067604 | 14.778371 | 1.92% | 8.838360 |
| w1024_s512 | 9.766495 | 9.658037 | 1.11% | 8.107674 |
| w512_s512 | 5.527296 | 5.442745 | 1.53% | 3.771228 |
| w2048_s128 | 21.205127 | 21.169979 | 0.17% | 22.148406 |
| w512_s128 | 2.890641 | 2.824912 | 2.27% | 3.047345 |
| w1024_s128 | 7.358112 | 7.290264 | 0.92% | 6.930984 |

The 211M result is near-neutral; it does not establish a repeatable small gain.
The other seven paired reductions range from 0.92% to 2.27%. The strongest
qualified controls remain the targets, including faster historical observations;
new equivalent controls are retained when faster. The suite remains **two of
eight cases**, with full parity unproven.

All 5,136 paired Native qualification/warmup/measured records have identical
loss, pre-clip norm and scale. All 32 paired first-update gradient/postweight
snapshot hashes match. Every candidate record selects one cooperative forward
and one cooperative backward program; all 64 measured updates replay with two
capture variants. These are first-update and scalar-trajectory checks, not a
convergence or full-weight-trajectory claim.

Owned F32 and both lifetime/steady queried-pool `USED_HIGH` differences are zero
for every trial. Sampled whole-process first-trial peaks vary: B32 +34 MiB,
B16 −36 MiB andB8 −22 MiB; second-trial deltas and the other cases are zero. These
~50 ms NVML observations are lower bounds on instantaneous peak. Exact queried
pool accounting excludes modules/context/other pools. No allocation saving or
capacity expansion is claimed; prior memory costs persist. No foreign compute
was observed, and all before/after process snapshots are empty.

Focused qualification has four CPU/codegen and six CUDA proofs. The final
selected regression has 38 passes: those ten plus seven existing backward proofs
and 21 affected integration tests, with no selected skips. Parent checks add 198
CPU library passes, CPU/CUDA all-target checks, strict all-feature Clippy,
formatting/diff checks and release build. This is not a new broad CUDA suite.
The actual selected forward uses 24 registers, 17,412 shared bytes and zero local
bytes; API capacity 5 at 256 threads is not measured occupancy. Four exact loaded
PTX modules also assemble offline; that SASS is not driver-JIT disassembly.

`target/transformer-training/v22-campaign-audit/audit.json` verifies the frozen
sources, reports and 104 F32 files totaling 20,668,956,672 logical bytes before
deduplication. Prelink fingerprints remain historical evidence. Raw reports,
audits and prequalification are embedded in the bundle. Frozen archive SHA256:
`6d58d61c7862d5208aeb2663d30c98bda28eb138ef8412158a35ba7ec3d3fa7c`;
binary SHA256:
`85e1a6e4f4f87edf19cf8014b0416096155f7786f8e79cff2f70b2a8e8257905`.
Later attention input-ABI/runtime-rewrite/owner metadata and K64 experiments are
excluded from this binary/source snapshot and these complete-update results.

Through V22, totals were **35 campaigns, 952 attempts, 950 successful workers, two
historical baseline OOMs, 34,624 measured complete updates and 634 passing
first-step benchmark comparisons**. Warmups, numerical-only screens and component
experiments remain separate. The ten qualified complete-update diagnostics and
two failed NCU attempts are unchanged.

## V23 checkpoint-retaining attention forward

The accepted addition is `--attention-training-forward-fusion`, only for
W512/S512/B16 and B32 (BH>=128). It replaces QK, normalization and PV with one
kernel per layer, retaining original QK16, P32, P16 and PV16 checkpoints and
arithmetic order. All four layer unions activate; backward stays unchanged.
BH64 is rejected before extra lifetime reservations, so B8 retains the complete
original kernels and allocation policy. All defaults remain off.

Stable semantic kernel owners resolve the original groups after LayerNorm's
late rewrite. All eight inputs/outputs and allowed early BF16 views are reserved
through PV across every prefix/final reuse. Disjoint byte-range, layout, JIT and
resource checks precede atomic replacement; rejection keeps all original groups.
The actual selected kernel uses 108 registers, 44 KiB shared and zero local
memory; API capacity 2 at256 threads is not measured occupancy. There are no new
input materializations. Active ordinary-program counters exclude replaced
islands; custom attention kernels have a separate region counter.

Two fresh workers per mode, 256 warmups and 64 measured complete updates reverse
mode order in trial 1. All **18 workers and 12 fixed first-update comparisons**
pass, including compiled model+loss CUDA Graph SDPA and retained V22 native.

| Case | V22 native, ms | V23 candidate, ms | Paired reduction | Fresh PyTorch CG, ms |
|---|---:|---:|---:|---:|
| w512_s512_b32 | 61.101438 | 60.270371 | 1.36% | 29.736017 |
| w512_s512_b16 | 29.730993 | 29.360754 | 1.25% | 15.964583 |
| w512_s512_b8 | 14.822377 | 14.752091 | 0.47% | 8.835266 |

Only B16/B32 select fusion. B8's0.47% variation occurs with unchanged kernels and
is not attributed to the option; its accepted V22 observation remains in the
scorecard. B16/B32 improve 1.25–1.36%, but remain behind PyTorch. No additional
case reaches parity.

All 1,926 paired qualification/warmup/measured records have identical loss,
pre-clip norm and clip scale. All 12 first-update clipped-gradient/postweight
SHA256 pairs match. All 64 measured updates replay with two capture variants.
These checks do not establish convergence or a full-weight trajectory.

Owned F32/BF16 bytes and both lifecycle/steady queried-pool USED_HIGH deltas
are zero in every trial. Sampled NVML deltas are B32−226/0 MiB,
B16−96/−32 MiB and fallback B8+64/−64 MiB. These sampled process peaks are lower
bounds and include a different scope from queried-pool counters; no allocation
saving or capacity expansion is claimed. Four quadratic checkpoints remain.
All before/after foreign-compute snapshots are empty.

Focused proofs cover actual resources, exact checkpoints, changing feeds,
[1,2,1] microbatches, direct capture, rejected-update recovery and full fallback.
The successful full-state fixture uses hidden 64/unique embedding indices;
standard hidden 1408 models are separately covered by the full-update campaign.
Parent regression adds **198 CPU library passes and 529 serial CUDA library
passes (11 skipped)** before timing, plus release build. Strict all-feature
Clippy passed in focused qualification. Later test-only GEMM-output and online
oracle scaffolds are excluded from the frozen V23 sources and timing evidence.

The campaign audit verifies 39 F32 files totaling 2,376,671,232 logical bytes,
all recorded provenance, scalar equality, active counters and memory facts:
`target/transformer-training/v23-campaign-audit/audit.json`.
Frozen source SHA256:
`e7135627abdd20f7bed902b73ce54d18e309dc4cedc838f53a4d5436f31bec71`;
retained binary SHA256:
`2310e3d501f3a5f6049734ade61407ebee7d91b3efa2d6aea8a8462205f19bb5`.

Through V23, totals were **36 campaigns, 970 attempts, 968 successful workers, two
historical baseline OOMs, 35,776 measured complete updates and 646 passing
first-update comparisons**. Component experiments remain separate; the ten
qualified complete-update diagnostic profiles and two failed NCU probes are
unchanged. Full-suite parity remains unproven at **two of eight**.

## V24 proven plain BF16 output stores

`--plain-bf16-output` is accepted opt-in for W512/S512/B2, B8, B16 and B32.
It checks the original full 64x64 output's sixteen scalar store mappings exactly,
then removes redundant bounds predicates and uses a common U64 output base plus
constant round offsets. MMA order, input staging, shared exchange, scalar BF16
rounding and launch geometry remain unchanged. All defaults remain off.

Eligibility is plain rank-two BF16 output, exact 64x64/K32/256-thread ordinary
geometry, full M/N tiles, K>=512 aligned 32, proved full-volume NN operands,
two-stage async pipeline and shared reuse. Tails, fused epilogues, batches,
unsupported maps and geometries retain original PTX. The affine proof accepts
only dominating unique bindings, checks bounds/overflow and commits atomically.
A cloned complete baseline is retained only after at least one proof succeeds.
Actual JIT/resource rejection recompiles that complete baseline through the
unchanged CE forward/backward compiler; counters and labels describe the chosen
module. Graph signature version 50 includes the new option; 46,080 combined
identity cases pass.

Three campaigns compare accepted V23 attention in B16/B32 and V22 elsewhere,
retaining SmallGridSplit only in its three previously qualified cases. Two fresh
workers/mode reverse order, with 256 warmups and 64 measured complete updates.
All **48 workers and 32 fixed first-update comparisons** pass.

| Case | Accepted native control, ms | Candidate, ms | Paired reduction | Selected programs | Disposition |
|---|---:|---:|---:|---:|---|
| w512_s512_b32 | 60.442431 | 60.180622 | 0.43% | 21 | accept small gain |
| w512_s512_b16 | 29.440147 | 29.200952 | 0.81% | 21 | accept small gain |
| w512_s512_b8 | 14.832304 | 14.760682 | 0.48% | 21 | accept small gain |
| w1024_s512 | 9.670051 | 9.718528 | -0.50% | 20 | retain V22; mixed |
| w512_s512 | 5.434440 | 5.414603 | 0.37% | 20 | accept small gain |
| w2048_s128 | 21.147583 | 21.144498 | 0.01% | 4 | retain V22; near-neutral |
| w512_s128 | 2.825315 | 2.832471 | -0.25% | 0 | retain V22; inactive |
| w1024_s128 | 7.283193 | 7.312732 | -0.41% | 4 | retain V22; regression |

The four accepted reductions are small (0.36–0.81% aggregate), with the same sign
in both fresh orders. They do not establish precision beyond this protocol or
predict gains in other shapes. W1024/S128 is slower in both trials; W1024/S512
has opposite signs; 211M is near-neutral and W512/S128 selects zero programs.
The prior accepted options remain in those four cases. The stronger fresh
W512/S128 PyTorch aggregate 2.963624 ms tightens that case's target; other stronger
historical controls remain. No new case reaches parity.

All 5,136 paired qualification/warmup/measured records have identical loss,
pre-clip norm and scale. All 32 first-update gradient/postweight SHA256 pairs
match. All 64 measured updates replay with two capture variants. Owned F32/BF16
and both lifecycle/steady queried-pool USED_HIGH deltas are zero in every trial.
Sampled process peaks vary: B32 0/−128 MiB, B16 0/−96 MiB, B8 0/+64 MiB; other deltas
are zero. NVML samples are lower bounds and have a different scope from the
queried pool. No allocation saving or capacity expansion is claimed. No foreign
compute was observed.

Focused qualification has six unique real-device tests and two unique CPU
proofs (one CPU proof repeats in the five-test leaf filter). It covers toggling,
complete module fallback, nested CE forward/backward rejection, unsupported
module identity, and exact actual B8/B16 [1,2,1] state, direct replay and
NaN rejection/recovery. B16 combines both CE options, four accepted attention
unions and nine LayerNorm pairs. Both models select 21 programs at 92 registers,
19,456 shared bytes, zero local memory and API capacity 2; capacity is not
measured occupancy. The first B16 test stopped before updates on stale expected
BMM counts; its log and corrected passing rerun are retained. CPU/CUDA all-target
checks, formatting and strict Clippy pass. This is not a new broad CUDA run.

The frozen model source and binary are verified by
`target/transformer-training/v24-campaign-audit/audit.json`, including 104 F32
files totaling 20,668,956,672 logical bytes. Source SHA256:
`01944344891d4cccf44301264bb05592917e488e2b024a3a7f240e5d32bb536a`;
binary SHA256:
`15e6fd8ad924f61797ae0ef2e97a9d5e08746a200c5f78f7d0a45641fc949bd5`.
A superseded, unbenchmarked prequalification snapshot preserves a test-only
online-prototype mutability failure; the annotation was fixed and all 12 online
CPU proofs/strict Clippy passed before the final snapshot. That prototype is
inactive in the model binary. Later two-step dO proof, Torch leaf/export and
wide-exchange test sources are excluded from these timings. No online-attention
or wide-exchange adoption is claimed.

Through V24, totals were **39 campaigns, 1,018 attempts, 1,016 successful workers, two
historical baseline OOMs, 38,848 measured complete updates and 678 passing
first-update comparisons**. Ten qualified complete-update diagnostics and two
failed NCU probes are unchanged and remain outside benchmark totals. Full-suite
parity remains unproven at **two of eight**.

## Research screens after V24

These experiments guide the next candidate. They do not change the accepted
V24/V22 policy, benchmark totals, or two-of-eight scorecard.

The online attention prototype now has a matched complete forward/backward
component comparison. Its old control includes all six BF16 contractions and
both normalization kernels; at B16/B32 a stronger control retains the accepted
checkpoint-forward fusion. The online pair executes three kernels. Two fresh
processes reverse order, with at least 500 ms warmup and 32 AB/BA pairs per
comparison; each sample contains eight captured complete-pair invocations.

| W512/S512 batch | Strongest applicable old pair, µs, two processes | Online pair, µs, two processes | Old / online |
|---|---:|---:|---:|
| 8 | 786.24 / 784.96 | 884.86 / 885.57 | 0.889x / 0.886x |
| 16 | 1779.71 / 1776.62 | 1602.05 / 1582.66 | 1.111x / 1.123x |
| 32 | 3574.84 / 3570.30 | 3144.64 / 3144.58 | 1.137x / 1.135x |

The Scalar version regresses at B8. Including weaker ordinary controls
at B16/B32, the experiment retains 320 raw pairs, 640 event samples and 5,120
complete-pair invocations. Full outputs and read-only inputs pass pre/post checks.
These are component timings; original allocations remain owned, so no model
speedup or memory saving follows from this result. Vectorized asynchronous input
staging subsequently addresses this component regression in the measurements
below.

The optional 16-byte staging candidate passes four CPU checks and strict
Clippy. Default Scalar generation matches all three revision-4 PTX goldens
byte for byte. Full packet-domain bounds/alignment and waits before shared
reads/alias reuse are checked; any unaligned input selects the whole Scalar
pair. Untimed device tests pass for sequence-major and head-major layouts,
each with eight fixtures including subnormals, signaling NaNs and infinities.
All four outputs and three saved-statistics fields match Scalar at finite-bit
precision and NaN classes, with replay and read-only input checks. Actual Async16
functions use 56/72/106 registers, 49,152 shared bytes and zero local memory;
driver block capacity is two, not measured occupancy. This leaf qualification
does not establish native-model correctness. Raw reports and the matching source/private binary
are retained in the attention qualification directory's `async16` subdirectory.

The subsequent three-mode timing uses the strongest applicable old pair, Scalar
online pair and Async16 online pair in the same process. Two fresh processes
reverse case/compilation/capture construction and initial ABC/CBA order. Each
case has at least 500 ms alternating warmup and 32 ABC/CBA triplets, with eight
captured complete pairs per event sample.

| Batch | Old pair, µs, two processes | Scalar online, µs, two processes | Async16 online, µs, two processes |
|---|---:|---:|---:|
| 8 | 785.07 / 777.28 | 882.29 / 882.30 | 733.18 / 732.42 |
| 16 | 1779.95 / 1809.41 | 1606.83 / 1581.89 | 1322.50 / 1316.42 |
| 32 | 3572.35 / 3572.79 | 3138.61 / 3142.34 | 2614.02 / 2604.16 |

These are medians of component event samples. Async16 reduces pair latency by
about 17% versus Scalar and is faster than the old pair in both orders at all
three sizes. Full four-output and saved-statistics Scalar/Async bit comparisons,
old-pair reference checks, input read-only checks and strict resources pass
before and after each case's timing. Pre/post compute inventories are empty.
The 192 triplets, 576 event samples and 4,608 complete-pair invocations are not
model updates. Original allocations remain owned, and no memory saving or
full-update gain is claimed. The Scalar numerical-screen model binary below
does not include this transport selection yet.

A separate untimed screen runs the actual three online PTX kernels through a
CUDA-driver/autograd bridge inside all four attention layers of the PyTorch
reference decoder. All other model, clipping and Adam work remains PyTorch.
The bridge first reproduces all four BF16 outputs exactly on five retained leaf
fixtures. B8/B16/B32 then pass all six unchanged first-update comparisons against
fresh eager and retained compiled CUDA Graph controls. Against the compiled
controls, clipped-gradient relative L2 is 0.01411–0.01427 and update relative L2
is 0.13145–0.13183, below the existing 0.02 and 0.15 limits. This supports further
integration work; native-project lifetimes, fallback, replay, microbatch state
and invalid-update recovery require separate direct qualification. Near-cancellation
dQ errors and differing NaN propagation in leaf fixtures remain counterevidence;
the global model checks do not establish general numerical equivalence.

The subsequent **full native-project first-update screen passes all six fixed
comparisons** at B8/B16/B32, against both the retained compiled CUDA Graph control
and accepted V24 native snapshots. Each model selects four complete Scalar pairs
(twelve kernels), retains nine LayerNorm forward/backward pairs and four QKV
packs, and reports zero pair fallbacks. Against compiled PyTorch, clipped-gradient
relative L2 is 0.01408–0.01425 and update relative L2 is 0.13151–0.13183. Against
the old native policy they are about 0.01644 and 0.14210; all existing limits,
including own-gradient F64 Adam, remain unchanged.

The explicit `--online-attention-pair-numerical-screen` option is default off
and mutually exclusive with checkpoint-forward fusion. It retains old quadratic
owners and runs a terminal rewrite after allocation reuse. Saved inputs and
new gradient writes are reserved across every reuse pass; typed original-program
metadata resolves fused numeric boundaries without treating local node IDs as
global identities. Split-K shortcuts record their complete emitted groups too.
Missing, ambiguous, unsupported or resource-rejected pairs keep the original
computation. Initial safe-fallback failures and corrected passing evidence are
retained, rather than weakening admission to make the test pass.

This screen includes one subsequent replay diagnostic update per worker, but
adds no throughput campaign or parity case. Full microbatch/state-transaction
qualification remains pending. Queried-pool counters are retained; process NVML
was not sampled here, so no process-peak comparison or allocation saving is
claimed. Source and binary are frozen separately from the accepted V24 policy:
[numerical-screen source](benchmarks/transformer-training-sources/online-attention-screen-v1.tar.gz),
SHA256 `d07bb150b45f8ac13b391bced1c07740be57423d8bc3d1827cc3727c74d6e7b8`;
binary SHA256 `7a24a481e274b235677c2af4c335597b83c78c673e63cc65384b569f91318734`.
The audit under `target/paper-kernels/online-training-attention-native-screen-v1`
verifies 37 provenance files totaling 1,288,289,085 logical bytes. The frozen
creation manifest's pending fields are historical; the separate qualification
report records the completed numerical checks without modifying that manifest.

A test-only wider GEMM exchange improves its older four-warp control on all four
sampled shapes, but loses against the stronger accepted 64x64/plain-output
control on B16 QKV and B8 down-projection gradients. It remains unadopted.
The dominant non-NN paths now have matched actual-contract timings. A generic
CUDA-driver bridge first reproduces the native Rust exports exactly, including
every F32 split-K partial. Two fresh processes then reverse cases, control/capture
construction and initial AB/BA order, using the same preallocated A/B/bias input
addresses. Each comparison has 32 AB/BA pairs, 32 complete operations per event
sample and at least 500 ms alternating warmup.
The FFN and QKV forward shapes are M=16,384, K=512 and N=1,408/1,536;
the FFN weight-gradient shape is M=512, N=1,408, K=16,384.

| Operation | Native, µs, two processes | PyTorch, µs, two processes |
|---|---:|---:|
| FFN fused NT forward | 526.35 / 523.55 | 246.51 / 248.00 |
| QKV fused NT forward | 563.26 / 569.45 | 275.33 / 276.43 |
| FFN TN weight gradient vs single BF16 GEMM | 405.20 / 407.70 | 395.87 / 394.51 |
| FFN TN weight gradient vs matched four F32 partials | 410.42 / 407.49 | 330.88 / 329.78 |

These are per-process medians of component event samples. Native forward is one
kernel, with F32 bias addition before BF16 output rounding; Torch `addmm` matches
its full output bit for bit. Native TN includes its four-part F32 contraction and
ordered final reducer. The matched Torch TN control includes all twelve CUDA
activities: four GEMMs, four workspace memsets, three ordered additions and one
BF16 conversion. The single-GEMM control is a distinct accumulation policy and
is slower here than the matched-partition control. Native TN times differ across
its two independent paired comparisons; they are not interchangeable samples.

All output/partial/input pre/post proofs pass. Pre/post compute-process
inventories are empty. The 256 raw pairs and 16,384 timed component invocations
add zero complete model updates. These results prioritize fused NT forward
linears, where the measured gap is about 2x; the earlier NN proxy result does not
generalize to these model paths. No production selector or parity case changes.

The raw bundle embeds these reports under `research_followup_after_v24`, outside
complete-update totals. Immutable sources, raw component timing pairs, numerical
snapshots and provenance are retained under
`target/paper-kernels/online-training-attention-qualification`,
`online-training-attention-model-screen`, `wide-exchange-qualification` and
`non-nn-calibration`.

## V25 complete online attention with asynchronous staging

The explicit default-off `--online-attention-pair-numerical-screen` option is
accepted in the measured policy for W512/S512/B8, B16 and B32. It replaces each
closed forward/backward attention pair with three native kernels and selects
16-byte asynchronous staging only after full-layout/alignment proofs. Other
cases retain their previously qualified policy. The accepted old
`--attention-training-forward-fusion` flag is removed in these three cases;
the two flags are mutually exclusive. All other accepted per-case flags,
including plain BF16 output stores, stay enabled.

Two campaigns compare the frozen V25 candidate with the accepted V24 binary and
compiled full-model/loss SDPA CUDA Graph control. All modes use two fresh
processes with reversed mode order, 256 warmups and 64 measured complete updates.
All **18 workers, 1,152 measured updates and 12 candidate first-update comparisons**
pass. Six additional offline baseline-versus-fresh-CG checks also pass; they add
no GPU workers/updates and are counted separately from runner comparisons.

| Case | V24 control, ms | V25, ms | Latency reduction | Fresh compiled CG, ms |
|---|---:|---:|---:|---:|
| w512_s512_b8 | 14.744978 | 14.438074 | 2.08% | 8.789715 |
| w512_s512_b16 | 29.235893 | 27.480193 | 6.01% | 15.900018 |
| w512_s512_b32 | 60.157590 | 56.120759 | 6.71% | 29.946573 |

Both fresh trial reductions are positive: B8 1.60/2.56%, B16 6.17/5.84%,
B32 6.71/6.71%. These are descriptive paired observations, not confidence bounds.
B8's fresh compiled aggregate 8.789715 ms tightens its target slightly; stronger
historical aggregates remain the targets in B16/B32. No new case reaches parity.

Every one of the 1,926 qualification/warmup/timed candidate records selects four
Async16 pairs, zero Scalar pairs/fallbacks and no rejection; all nine LayerNorm
pairs, four QKV packs and both ordered-CE options remain active. B8 removes 20
launches and B16/B32 remove 12. All 64 timed steps replay in every native worker,
with two capture variants/misses and zero parameter copies. All 12 candidate
first-step gradient/weight SHA comparisons match the separately prequalified V3
snapshots. This is not equality to the old attention arithmetic: all paired
loss/norm trajectories differ, with maximum observed loss differences of 0.0248,
0.0220 and 0.00845 for B8/B16/B32. The fixed gates qualify only the first update;
no training-convergence or trajectory-equivalence claim is made.

Owned F32 payload and steady queried-pool USED_HIGH increase by exactly 1.5/3/6 MiB
for B8/B16/B32 statistics. BF16 payload is unchanged. Both lifecycle USED_HIGH
and lifecycle/steady RESERVED_HIGH deltas are zero in both trials. All original
quadratic allocation owners remain; this is not a memory-linear allocation
implementation. Sampled process NVML deltas vary: B8 0/+64 MiB, B16 −32/0 MiB,
B32 −64/0 MiB. Samples are lower bounds and cannot replace exact pool accounting;
no memory saving or capacity expansion is claimed. No foreign compute was
observed by the runner.

Prequalification covers actual standard hidden=1408/vocab=4096 B8 and B16 models:
exact losses, clipped gradients, masters and both moments between uncaptured
private-master and captured/direct execution across [1, 2, 1] intervals, NaN
rollback and restart. The initial Scalar test stopped only on an incorrect
last-upload copy-counter assertion; corrected per-call accounting and both
Async state tests pass. A final global-key pruning fix avoids comparing local
F32 view IDs with global node IDs; pure collision/protected-source checks pass,
and final-core B8/B16 audits preserve all 221 kept bindings and twelve kernels.
CPU proofs, formatting and strict Clippy pass; this is not a new broad CUDA run.

Reproducible raw campaigns, all phase counters, pool/NVML samples and unchanged
validator reports are embedded in the raw bundle. The audit streams 39 F32 files,
2,376,671,232 logical bytes, verifying stable fingerprints and every comparison's
provenance. Frozen
[source V25](benchmarks/transformer-training-sources/parity-candidate-v25.tar.gz)
SHA256: `58a6c08812a202bd99e02f0ae3d66554b4978b9e5980c0d2c48c7d6a19d7f061`;
binary SHA256: `a4dcb6cb49532646497c016f85cee1951d2f28add1b3455a664335b53182b43a`.
This is a byte-identical alias of the separately qualified Async V3 release.
V1 Scalar and superseded, unbenchmarked V2 source/binary records remain retained.
The NT128 bias prototype in this snapshot is test-only and inactive here.
Later four-wide qualification sources are retained separately.

Totals after V25 were **41 campaigns, 1,036 attempts, 1,034 successful workers, two
historical baseline OOMs, 40,000 measured complete updates and 690 passing runner
candidate first-update comparisons**, plus six additional control cross-checks.
Ten qualified complete-update diagnostics and two failed NCU probes are unchanged
and excluded. The full-suite goal remains active at **two of eight**.

## Remaining bottlenecks

The accepted V30/V27 per-case policy remains behind the strongest qualified
control in six of eight cases. B32 takes 52.9638 ms versus 29.5947 ms (1.790x
latency); 55M/S512 takes 9.5750 ms versus 8.0944 ms (1.183x). The 211M and
W512/S128 cases meet their targets; that does not establish suite parity.

Profile activity remains a guide to kernels, not isolated causal speedups or
hardware utilization. FFN/GEMM and split-K work, attention, forward indexed loss
and residual/reduction work remain candidates. V25 removes the old probability
computation from execution while retaining its allocation owners. Its checkpoint
change passes the fixed first-update gates; evolving trajectories differ.
Experimental defaults remain off, and prior memory costs persist.

Results from separate experiments must not be multiplied into a predicted model
speedup. New measurements must use the combined native policy and fresh controls,
retaining the fastest qualified measured reference for each case. Historical
model-only/default controls do not justify omitting the stronger model+loss
reference where it performs better. Full-suite parity remains unmet.

## V26 fused NT linear: complete-update measurements

The default-off `--fused-nt128-bias` option replaces eligible single-kernel
linear programs with the qualified eight-warp 128×128×32 implementation. Bias
is added once in F32 before the original BF16 output boundary. It is accepted
opt-in on top of V25 for W512/S512/B8, B16 and B32.

Two fresh trials reverse native baseline, candidate and compiled PyTorch CUDA
graph order, with 256 warmups and 64 measured complete updates per worker:

| Case | Paired V25, ms | V26, ms | Latency reduction | Fresh PyTorch, ms |
|---|---:|---:|---:|---:|
| B8 | 14.5713 | 14.1915 | 2.61% | 8.8374 |
| B16 | 27.4437 | 26.7665 | 2.47% | 15.8961 |
| B32 | 56.1762 | 54.5771 | 2.85% | 29.6424 |

Both trials improve: B8 2.53%/2.68%, B16 2.44%/2.49%, B32 2.86%/2.84%.
All 18 workers pass, recording 1,152 timed complete updates and 12 unchanged
runner first-update comparisons. The scorecard retains faster historical
PyTorch aggregates for B8 and B32; B16's fresh aggregate is the stronger target.
Parity remains two of eight.

Actual B8/B16/B32 descriptor censuses select 13 programs each: four QKV, eight
gate/up, and one tied head. Four attention-output programs require an additional
materialization, and four down projections have rank-three physical inputs;
all eight retain their complete original programs. Selected kernels use 128
registers, 40 KiB shared memory and no local memory, with an API capacity of two
blocks per SM at 256 threads. This capacity query does not measure occupancy.

All six fixed first-update comparisons against accepted V25 native and compiled
PyTorch controls pass without changed gates. All six gradient and post-update
weight snapshot hashes exactly match V25. Actual B8/B16 captured/direct versus
uncaptured/private tests pass exact loss, clipped gradients, master weights and
both Adam moments across `[1,2,1]` microbatch intervals, including NaN rejection
and recovery, zero direct parameter copies and two capture variants. CPU/CUDA
checks, strict Clippy, formatting, two qualified NT PTX identities and twelve
legacy PTX identities pass.

The frozen [V26 source](benchmarks/transformer-training-sources/parity-candidate-v26.tar.gz)
has SHA256 `f3fff637426556c1dce1ac5638b48106f7d17882226b754bcb3d4ef3bf31c850`;
the retained binary has SHA256
`7b54ada64695b8aa13ab75d61c8397626620e98a1c936631e0625c179185bf58`.
All 1,926 qualification/warmup/timed loss, norm and clipping-scale records match
the paired V25 controls exactly. Both native snapshot files match in all six
trial/case pairs. All timed updates replay, with zero direct parameter copies
and two capture variants. Owned F32/BF16 payloads and every queried lifecycle
and steady-state pool field are unchanged. Sampled process maxima are also
unchanged at 2,064/3,472/6,288 MiB for B8/B16/B32; these are lower bounds rather
than allocator measurements. The old quadratic attention owners remain.

Qualification reports and stable snapshot hashes are retained under
`target/paper-kernels/fused-nt128-bias-integration-qualification/`; these new
records and complete reports are included in the raw bundle. The
independent campaign audit is retained in
`target/transformer-training/v26-campaign-audit/`.

## V27 contiguous bias-gradient reshape aliases

The default-off `--contiguous-bias-reshape-aliases` option removes one internal
whole-volume BF16 reshape copy before each eligible bias Sum. Its proof requires
a closed, contiguous, single-consumer copy followed by the original typed
Block128 Sum, with whole aligned bindings and a disjoint internal output.
The reduction PTX, arithmetic order and BF16 boundaries remain unchanged.
Actual kernel pointer uses extend the source lifetime through the reduction;
the copy target is omitted before allocation, and metadata lists only emitted
kernels. Unsupported programs keep their original complete path.

Seven established cases select 20 aliases each. The 211M case uses Subgroup32
reductions for its 128-row inputs, so all 20 copies fall back and its accepted
V22 policy remains unchanged. All eight cases pass 16 unchanged fixed
comparisons against accepted native and strongest compatible PyTorch snapshots;
all 16 gradient/weight snapshot file pairs exactly match the accepted native
references. Two stopped controller attempts are retained: an incorrect expected
alias count for 211M, and decimal JSON values compared before conversion to
their actual F32 settings. Correcting these metadata checks did not change the
numerical validator or its thresholds.

The leaf test covers all 65,536 BF16 patterns across three feeds and two captured
replays, with exact finite output bits, matching NaN classes, unchanged inputs,
and overlapping-root fallback. Actual B8/B16 tests preserve exact loss, clipped
gradients, weights and both moments across `[1,2,1]`, including NaN rollback and
recovery, direct bindings and two capture variants. CPU/CUDA checks, strict
Clippy, formatting and all 14 retained PTX identities pass.

Four campaigns use two fresh reversed-order trials, 256 warmups and 64 measured
complete updates per worker, against each case's prior accepted native binary
and compiled PyTorch CUDA graphs:

| Case | Paired native baseline, ms | V27, ms | Latency reduction | Fresh PyTorch, ms |
|---|---:|---:|---:|---:|
| w1024_s512 | 9.6976 | 9.5750 | 1.26% | 8.0944 |
| w512_s512_b8 | 14.1633 | 13.7542 | 2.89% | 8.7688 |
| w512_s512_b16 | 26.7473 | 25.8524 | 3.35% | 15.9944 |
| w512_s512_b32 | 54.5516 | 52.9638 | 2.91% | 29.7380 |
| w512_s128 | 2.8243 | 2.7872 | 1.31% | 2.8901 |
| w1024_s128 | 7.3097 | 7.2526 | 0.78% | 6.9964 |
| w512_s512 | 5.4214 | 5.3270 | 1.74% | 3.7757 |

All 42 workers pass, with 2,688 measured complete updates and 28 runner fixed
comparisons. Both trials improve in every case. All 4,494 paired
qualification/warmup/timed loss, norm and clipping-scale triples match in their
F64 bits, and all 28 native gradient/weight snapshot file pairs match by SHA256.
Every timed update replays with zero direct parameter copies, two capture
variants and 20 fewer model kernels. Fresh qualified PyTorch aggregates improve
the targets for W1024/S512, B8 and W512/S128; faster historical targets remain
for the other cases. Suite parity remains two of eight.

Nominal skipped temporary allocations range from 11,010,048 to 704,643,072 bytes
per executor before pooling. These are unpooled construction accounts, not VRAM
savings. Owned
F32/BF16 payloads, all queried USED counters, and lifecycle/steady pools remain
unchanged. W1024/S512 setup RESERVED counters decrease by 32 MiB in both trials,
with no remaining difference after qualification. Sampled NVML peak deltas span
−96 to +32 MiB, including a +32 MiB B16 observation; they are lower bounds in a
separate scope and do not establish capacity savings. Old quadratic attention
owners remain allocated.

The frozen [V27 source](benchmarks/transformer-training-sources/parity-candidate-v27.tar.gz)
has SHA256 `b4d153cb361fbcee70f8a37e54dc9f00d9b6a4fadfdc36863d10459024f0eba6`;
the retained binary has SHA256
`dab64dc89a5d4a5d20d4f7ce37f502821a87ca3e3775ae352425e3c1a4b96f37`.
Qualification and retained controller failures are under
`target/paper-kernels/contiguous-bias-reshape-qualification/`. The independent
audit in `target/transformer-training/v27-campaign-audit/audit.json` verifies
91 stable campaign F32 files, 9,689,272,320 logical bytes. The published raw
bundle includes these reports, the new 55M diagnostic, and the explicit-MMA
prototype's retained correctness pass and approximately ninefold component
slowdown. That rejected experiment changed both operand loading and output
stores; it does not isolate the runtime cost of output exchange.

## V28 borrowed BF16 optimizer sources

The default-off `--borrowed-bf16-adam-source` option lets standalone Adam
kernels read eligible live BF16 gradient buffers during a single-microbatch
step. F32 accumulation, norm partials, clipping diagnostics, health checks,
masters and moments remain intact. Batched small parameters retain their old
F32 source. The BF16 kernel recreates the original positive-zero accumulation
boundary before clipping and uses the same Adam arithmetic.

The proof requires the current executor and successful run, matching stream
and context, one successful accumulation, an identity logical map and a typed,
zero-offset owned prefix. Larger pooled allocations are supported without
assuming allocation size equals logical tensor size. Failed, stale, empty,
multiple-microbatch and unsupported-layout paths retain the F32 behavior.
No raw gradient pointer is retained in the optimizer.

Prequalification covers every BF16 bit pattern, tails, both launch geometries,
changed feeds and graph replays. Actual B8 and 55M decoder tests preserve exact
loss, norm, clipping scale, gradients, weights and both moments through
`[1,2,1,0]` accumulation intervals and invalid-update rejection/recovery. The
B8 proof admits 16 matrices and explicitly rejects the four transposed
`out.weight` gradients; 55M admits 20 matrices, including three prefixes of
larger pooled allocations. The 211M case admits 32 standalone parameters,
including eligible large biases. B16 and B32 admit zero and keep V27.

All eight archive screens pass 16 unchanged native/PyTorch numerical
comparisons and 16 native gradient/weight snapshot-file identities. The six
eligible cases then use two fresh processes per mode, 256 warmups and 64
measured synchronized complete updates, with reversed mode order:

| Case | Paired native baseline, ms | V28, ms | Latency reduction | Fresh PyTorch, ms |
|---|---:|---:|---:|---:|
| w2048_s128 | 21.199456 | 20.463468 | 3.47% | 22.114148 |
| w1024_s512 | 9.568433 | 9.401527 | 1.74% | 8.142245 |
| w512_s512_b8 | 13.720727 | 13.687835 | 0.24% | 8.693026 |
| w512_s128 | 2.787638 | 2.751885 | 1.28% | 3.027411 |
| w1024_s128 | 7.222958 | 7.089247 | 1.85% | 6.804052 |
| w512_s512 | 5.319259 | 5.287649 | 0.59% | 3.739350 |

Every paired trial improves, but the B8 and B2 gains are small descriptive
observations, not confidence bounds. The policy is accepted opt-in for these
six cases. The 36 workers record 2,304 measured updates and pass all 24 fixed
first-update comparisons. All 3,852 paired scalar records have bit-identical
loss, norm and clipping scale (11,556 float64 comparisons); all 24 paired
native gradient/weight snapshot files are SHA256-identical.

Owned F32/BF16 tensor payloads, resident master bytes, kernel counts and every
queried current/default allocator pool field and lifecycle stage match the
paired controls. One B8 trial samples a 34 MiB higher process peak; the other
11 native pairs have unchanged sampled peaks. NVML peaks are lower bounds and
have a different scope from allocator counters. There is no allocation or
capacity-saving claim. Logical Adam gradient-read width falls by 404,887,552
bytes for 211M and 101,187,584 bytes for 55M; these are payload differences,
not measured DRAM traffic.

The strongest qualified controls tighten to 8.693026 ms for B8, 6.804052 ms
for W1024/S128 and 3.739350 ms for B2/S512. Faster historical controls remain
the targets elsewhere. The goal remains two of eight; no gate is relaxed.

The frozen [V28 source](benchmarks/transformer-training-sources/parity-candidate-v28.tar.gz)
has SHA256 `18e2457bc2e1b79d25fb570a55a945dc1e843a3a565b4525d8fdcd22758b2cb4`.
The retained candidate binary has SHA256
`fef1008b6482d32fcfd4808c0b567ab4c7421d2cede30d6e03d9063b9f256f5f`.
The independent audit is
`target/transformer-training/v28-campaign-audit/audit.json`. The corrected B8
test differs only in a test child from this release source; production inputs
are byte-identical. The original coverage assertion failure and controller
coverage failure remain in the evidence.

### Matrix component followups

Earlier explicit scalar-load and coalesced-store variants remain rejected:
they are numerically correct but roughly 9–10 times slower than WMMA. A new
owned driver-linked cubin diagnostic identifies repeated 64-bit division
helpers in those newly linked scalar-load variants. Those cubins are not the
identity of the earlier PTX-loaded JIT modules, and the diagnosis does not
establish the sole cause of the earlier slowdown. A normalized scalar-load
variant uses 164 registers and fails the two-CTA resource requirement; it has
no execution or numerical success claim.

The new collective `ldmatrix` prototype passes its strict resource gate
(128/126 registers for scalar/coalesced stores, 40 KiB shared, zero local
memory, two-CTA API capacity). Four cases and eight input patterns match the
qualified WMMA output across replay; unaligned output pointers use the whole
WMMA fallback. An untimed four-mode bridge also matches PyTorch on identical
inputs before, during and after capture.

Two fresh component timing processes show coalesced-store latency reductions
of approximately 23–24% versus WMMA for the two tested NT shapes:

| Shape (M, N, K) | WMMA process medians, µs | Ldmatrix process medians, µs | PyTorch process medians, µs |
|---|---:|---:|---:|
| 16384, 1408, 512 | 395.17 / 387.45 | 299.55 / 298.50 | 242.32 / 243.28 |
| 16384, 1536, 512 | 432.99 / 430.42 | 328.13 / 328.26 | 267.04 / 267.04 |

These cache-hot component results do not establish a model speedup or parity.
The experiment and all prior rejections are retained separately in the raw
bundle and excluded from model worker/update totals. The component result alone did not qualify production integration. V29 below
subsequently validates a narrowly gated B32 integration; it remains outside
V28's historical accepted policy.

## V29 collective loads for fused NT128 linears

The default-off `--fused-nt128-bias-ldmatrix` modifier is accepted for B32 after
complete-update qualification. It applies only to `(M,N,K) = (16384,1408,512)`
and `(16384,1536,512)`, with the existing `--fused-nt128-bias` prerequisites.
The standard B32 model selects eight gate/up and four QKV kernels. Its head
and the eight existing NT fallbacks keep their prior implementations; B8 and
B16 select zero new kernels and retain their accepted policies.

Selection happens after allocation remapping and verifies the retained WMMA
function identity, current typed owners, complete operand ranges, disjoint
output and 128-byte output alignment. Unexpected identity or ABI metadata,
unsupported geometry, compilation/resource rejection or misalignment keeps
the whole original WMMA kernel. The selected kernels use 126 registers,
40 KiB shared memory, no local memory and an API capacity of two CTAs at
256 threads. They preserve the qualified MMA order, bias arithmetic and BF16
rounding. The plain/normalized scalar-load experiments remain rejected.

CPU checks include exact retained PTX identities and negative owner-namespace
collisions. GPU tests prove standard B8/B16/B32 constructor selection,
post-resource and valid-16-byte-but-not-128-byte fallback, and exact selected
B32 evolving state with two direct variants, microbatch accumulation and
invalid-update rollback/recovery. The state fixture uses unique tokens and
vocabulary 16384 to avoid scatter-order ambiguity; it does not replace the
unchanged archived vocabulary-4096 numerical gates. Both archived native and
PyTorch comparisons pass, with exact native first-update snapshots/scalars.

Two fresh processes per mode, 256 warmups and 64 measured synchronized complete
updates, with reversed mode order:

| Case | Paired native baseline, ms | V29, ms | Latency reduction | Fresh compiled PyTorch, ms |
|---|---:|---:|---:|---:|
| w512_s512_b32 | 53.015377 | 51.664600 | 2.55% | 29.499401 |

The individual reductions are 2.59% and 2.50%. All six workers pass, recording
384 measured updates and four unchanged first-update comparisons. Every loss,
norm and clipping scale matches the paired native control across all 642
recorded updates (1,926 float64 bit comparisons). All four paired gradient and
weight snapshot files are SHA256-identical. Owned tensor bytes, every queried
allocator stage and sampled 6,288 MiB process peaks match in both native pairs.
Sampled process peaks remain lower bounds; no capacity saving is claimed.

B32 is still 1.751x the strongest qualified PyTorch latency. The fresh compiled
control tightens the target to 29.499401 ms, and the suite remains at two of
eight. This is a measured model improvement, distinct from the earlier 23–24%
component improvement. No numerical gate changes or shape expansion are
included. Small-parameter norm batching and packed-attention epilogues are disabled in
this V29 measured candidate. V30 separately qualifies norm batching below.

The frozen [V29 source](benchmarks/transformer-training-sources/parity-candidate-v29.tar.gz)
has SHA256 `baf2a165834cbbaf5b5d890bf3c2fbcde24d39148124e9807697f5bb2f981cf9`;
the candidate binary SHA256 is
`70ab5b106197dd0a946bbe02e61363f952c6bd24771ad0b98af733f3db986308`.
The independent audit is `target/transformer-training/v29-campaign-audit/audit.json`.
Complete reports, prequalification and immutable source mapping are retained
in the raw bundle; later working-tree experiments are not these measurements.

## V30 batched small-parameter accumulation and norm

The default-off `--small-accumulation-norm-batches` option composes small
first-overwrite accumulation/norm kernels without changing any member's
arithmetic or reduction tree. It requires first-accumulation overwrite and
norm partials. Each CTA selects one member using grid Y; grid X retains the
exact global 1024-element row span, including unaligned starts. A single
512-byte scratch is reused under mutually exclusive uniform branches. The
unchanged repair and literal second-stage reduction handle boundary rows.

All source/output ranges are checked before any write, with one tracked
mutable global F32 gradient argument and a separate partials argument.
Compilation/resource or alias-proof failures retain the old path. Errors after
enqueue propagate through the existing invalid-attempt handling. Multiple,
empty and failed accumulation intervals preserve their old behavior.

The initial leaf safely rejected a singleton identity because a no-op reshape
canonicalized its map differently. The narrow proof repair uses direct
identity only when source/view shapes already match; it changes no arithmetic,
extent, predicate, alias or resource guard. Both failed attempts and the CPU
reproduction remain in the evidence. All three original norm PTX modules are
byte-identical, along with the existing Adam and matrix module goldens.

The mixed BF16/F32 leaf covers every BF16 bit pattern, changed feeds, tails,
flat F32 writes, partial rows and repaired norms. Actual 55M and B8 decoder
state tests preserve full loss, norm, scale, gradients, weights and both
moments through `[1,2,1,0]`, invalid updates and recovery, including accepted
borrowed sources and two direct variants. All eight archived cases then pass
16 fixed native/PyTorch comparisons, 16 native snapshot-file identities and
24 qualification scalar bit comparisons. B32 retains its twelve V29 kernels.

Observed accumulation dispatch falls from 60 launches to 38 for 211M
(26 small members in four batches plus 34 unchanged large members), and to 27
for the other seven cases (38 small members in five batches plus 22 large).
This is a dispatch reduction, not an allocation or predicted-latency saving.

Two fresh processes per mode, 256 warmups and 64 measured synchronized complete
updates, with reversed mode order:

| Case | Paired native baseline, ms | V30 candidate, ms | Latency reduction | Decision |
|---|---:|---:|---:|---|
| w2048_s128 | 20.438111 | 20.351797 | 0.42% | accept opt-in |
| w1024_s512 | 9.559628 | 9.343373 | 2.26% | accept opt-in |
| w512_s512_b8 | 13.869651 | 13.718435 | 1.09% | accept opt-in |
| w512_s512_b16 | 25.812361 | 25.820058 | -0.03% | retain V27 |
| w512_s512_b32 | 51.717998 | 51.487435 | 0.45% | accept opt-in |
| w512_s128 | 2.754638 | 2.688170 | 2.41% | accept opt-in |
| w1024_s128 | 7.072011 | 6.982544 | 1.27% | accept opt-in |
| w512_s512 | 5.273906 | 5.209722 | 1.22% | accept opt-in |

Seven cases improve in both paired trials and are accepted opt-in. B16 is
neutral/slightly slower in aggregate, with opposite trial directions; its old
policy stays accepted. The small 211M/B32 gains are descriptive observations,
not confidence bounds. B8's fresh candidate median of 13.718 ms is above its
earlier 13.688 ms observation; its matched baseline in this campaign is
13.870 ms. The 1.09% paired result does not establish an invariant latency.

All 48 workers pass, recording 3,072 measured updates and 32 unchanged
first-update comparisons. Every paired loss, norm and scale matches across
all 5,136 recorded native updates (15,408 float64 bit comparisons). All 32
paired native gradient/weight snapshot files are SHA256-identical. The model's
owned F32/BF16 payloads and kernel counts, resident master bytes, and every
queried allocator field/stage match the paired controls. Complete-step launch
counts fall by exactly 22/33 as above.

Sampled process peaks decrease by 64 MiB in one B16 trial and one B32 trial;
the other 14 native pairs are unchanged. These peaks are lower bounds and do
not demonstrate an allocation or capacity saving. B16's memory observation
does not override its neutral timing result. This comparison uses frozen
accepted/candidate revisions, not a same-binary ablation.

Fresh compiled controls tighten the targets to 21.889210 ms for 211M,
8.078641 ms for 55M, and 6.803409 ms for W1024/S128. Faster qualified historical
controls remain targets elsewhere. W1024/S128 measures 6.982544 ms, still
2.63% behind its target; the suite remains two of eight. No numerical gate is
relaxed. Packed-attention and NN experiments remain unadopted research.

The frozen [V30 source](benchmarks/transformer-training-sources/parity-candidate-v30.tar.gz)
has SHA256 `7bd5f808a0045dc8303e057c8a7536ecc58d075bf33ec69256fa35dc7548bc01`;
the retained candidate binary SHA256 is
`3fc71448f76bf7fab32fb504cd70040a740d16202a2f8daaf9e18054d17511de`.
The independent audit is `target/transformer-training/v30-campaign-audit/audit.json`.
Complete source/policy mapping, trajectories, numerical gates, memory samples
and allocator counters remain in the raw bundle.

## Post-V30 Adam256 control retest and S128 profile

This is a control/neutral experiment on the existing immutable V30 binary and
source, not a new compiler revision. All accepted native observations and
per-case policies remain unchanged.

The accepted W1024/S128/**B2** semantic profile maps all 346 model kernels and
405 actual complete-step CUDA kernels per update (404 reported plus scalar loss
readback). Five warmups and five profiled updates yield 7.184739 ms/update of
summed kernel activity, including 2.466160 ms Adam/health, 2.339587 ms GEMM,
0.601821 ms accumulation/norm and 0.585353 ms materialization. Both fixed
validators pass; first-step snapshots and all eleven optimizer-step scalar
records exactly match both accepted trials. This is one additional qualified
native diagnostic, excluded from throughput counts. The
[profile report](transformer-training-profile.md#latest-retained-diagnostics)
keeps exact resources, semantic ambiguity and memory scopes explicit.

Twenty borrowed-BF16 standalone Adam launches and two F32 standalone launches
still use 1024-thread blocks; five small batched launches already use 256.
That current evidence motivated retesting the existing `--adam-256-threads`
flag with no other native difference.

The initial six-worker attempt used an interpreter without PyTorch. Its four
native workers passed and recorded 256 timed updates; both PyTorch controls
failed before CUDA with `ModuleNotFoundError: torch`. Two native-baseline fixed
comparisons passed. These setup failures are retained with full logs and are
separate from historical OOMs or numerical failures. The small positive
native-only differences from this incomplete attempt were not adopted.

The corrected complete campaign uses the established Python interpreter,
explicit CUDA-visible `PYTHONPATH`, and two-thread settings, verified before
freezing its controller. Two fresh reversed-order trials per mode use 256
warmups and 64 synchronized timed complete updates each:

| Trial | Native baseline, ms | Adam256, ms | Latency reduction |
|---|---:|---:|---:|
| 0 | 6.979991 | 6.983693 | -0.0530% |
| 1 | 6.964717 | 6.966430 | -0.0246% |
| Aggregate | 6.972354 | 6.975061 | -0.0388% |

All six workers and all four fixed comparisons pass, recording 384 measured
updates. Adam256 is slightly slower in both native pairs, so the accepted
standalone launch policy stays unchanged. The result does not establish a
confidence interval or a general regression. Fresh qualified PyTorch
CUDA-graph SDPA measures **6.7961947643 ms**, stronger than the previous
6.8034092546 ms target. The accepted native observation remains 6.9825435 ms,
about 2.74% behind the stronger target; the suite remains two of eight.

All 642 paired loss/norm/scale triples (1926 float64 bit comparisons) and four
paired native gradient/postweight files are exact. Twenty borrowed BF16 sources,
27 accumulation launches, two capture variants, replay and zero parameter
copies remain active. Owned tensor payloads, every queried current/default-pool
field at all six stages, and sampled process peak deltas are zero in both
native trials. No allocation/capacity saving is claimed.

Both campaigns, the failed controls, immutable provenance, raw timing samples,
unchanged validators and the profile are retained in the raw bundle. The fresh
compact next-screen policy manifest is
`target/transformer-training/accepted-eight-case-manifest-v30-controls-v1.json`;
its native policies match V30/V27 exactly, with only the stronger qualified
control descriptors updated. The independent corrected audit is
`target/transformer-training/adam256-v30-w1024-s128/retry-v2/audit.json`.

## Evidence and reproduction

Current totals are **62 campaigns, 1,198 attempts, 1,194 successful workers, two
pre-CUDA Python dependency failures, two historical OOMs, 50,240 measured
complete updates and 796 passing runner
candidate first-update comparisons**, plus six historical control cross-checks.
Thirteen qualified diagnostic captures (ten native and three reference) and
two failed NCU attempts are excluded
from benchmark totals. The suite goal remains active at two of eight.

The complete raw evidence is also available as a [local gzip mirror](benchmarks/README.md); decompression reproduces the plain JSON bytes exactly.

[Local raw evidence](benchmarks/README.md) preserves complete
reports, per-update timings, sampled memory, fixed-gate metrics and provenance.
Each campaign maps to its immutable source version in the bundle. Both V5
reports preserve all 124 recorded source hashes, verified against
[candidate-v5](benchmarks/transformer-training-sources/parity-candidate-v5.tar.gz)
(SHA256 `4576863fd5ed0836627266c9c928018366f092b3a5a4ef0e1e5e1833bf851651`).
Their candidate binary SHA256 is
`24beba3d7c538d5985d6f9f4d30f13c033595192e9c437419cfd2264c21018b3`.
The narrow control is V4; the fusion control is the same V5 binary with fusion
disabled. Complete raw reports preserve worker commands, option vectors,
per-update counters and fixed-gate metrics. Later working-tree edits are not
evidence for these binary measurements.
The V6 no-op report maps to
[candidate-v6](benchmarks/transformer-training-sources/parity-candidate-v6.tar.gz)
(SHA256 `7895ed1a867123bd29ad01bab93a5aa648a1e3f08ecd00b4acd29aa924a0564e`);
all 125 recorded source hashes match that archive. Its baseline maps to V5.
The candidate binary SHA256 is
`47e509acc45742cdccd3e65894e717d16c4775b9a85a4c530b74325925c56752`.
The raw bundle identifies this campaign as `transposed_outputs_noop` and retains
all eight zero-counter observations and complete worker reports.
The Adam256 report maps to
[candidate-v7](benchmarks/transformer-training-sources/parity-candidate-v7.tar.gz)
(SHA256 `359bb893ded6022697419cb1816fc2006235298539dc9151252db64834197e22`),
with the retained V6 binary as control. Its candidate binary SHA256 is
`ea14107e875b97119d4f30477faca5d4d87e66ea9c83ed68d738bf4c41c57ace`.
The complete raw report is embedded as `adam_256_threads`.
The corrected transpose and backward-row reports are embedded as
`transposed_outputs_active` and `attention_softmax_backward`. Both map to the
same immutable V7 archive/binary above, with V6 controls; all 127 recorded source
hashes in each report match that archive. Their active counters, complete
per-update replay/payload records, fixed global qualification metrics and
source-option isolation checks are retained in dedicated bundle audits.
The eight-warp report is embedded as `eight_compute_warps`, with V7 as control.
All 130 recorded source hashes match
[candidate-v8](benchmarks/transformer-training-sources/parity-candidate-v8.tar.gz)
(SHA256 `db141b4ee69ddf23d9b83605bd27f03c27af4933bfb6ac119ff6aec523715597`).
Its candidate binary SHA256 is
`788c075851c7110ef73c153f7aa13162cc730e71ca4d22e76b3cfee29ea51218`.
The original retained baseline binary is
`target/release/examples/transformer_training_bench.pre-parity`.

Candidate-v1 passed 312 CPU tests and 333 real-device CUDA-feature library and
selected integration tests (five existing ignored tests), strict Clippy and
formatting. New tests cover full-warp/cohort reductions, tails, batched members,
nonfinite rollback, recovery, immutable publications and graph replay. These
results precede the later first-accumulation experiment.

```sh
python3 integrations/transformer_bench/run.py \
  --output target/transformer-training/new-health-comparison \
  --binary target/release/examples/transformer_training_bench.parity-v1 \
  --baseline-binary target/release/examples/transformer_training_bench.pre-parity \
  --cases w2048_s128,w1024_s512 \
  --modes native_baseline,native,pytorch_matched,pytorch_compiled_sdpa \
  --trials 2 --warmups 8 --repeats 32
```

Use the CUDA-visible Python environment documented in the
[baseline report](transformer-training-performance.md). The runner serializes
workers under the GPU lock and validates against every selected reference.
