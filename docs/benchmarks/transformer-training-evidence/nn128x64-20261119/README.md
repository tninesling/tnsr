# NN128x64 complete-update experiment: no adoption

Six fresh workers completed 384 measured B32 updates with two reversed trials,
256 warmups and 64 samples per worker. Both native arms use the same sealed
binary and source; the candidate adds only `--nn128x64-ldmatrix`. All four
first-update comparisons pass all six unchanged numerical gates.

| Trial | Baseline ms | Candidate ms | Candidate reduction | Sampled NVML delta MiB |
| --- | ---: | ---: | ---: | ---: |
| 0 | 50.8336995 | 50.417336 | +0.8191% | 0 |
| 1 | 50.6380385 | 50.975959 | -0.6673% | +66 |

The direction reverses, so the candidate is not adopted. Aggregate native
medians are 50.735869 ms baseline and 50.696648 ms candidate; the fresh compiled
PyTorch CUDA-graph control is 29.686764 ms. The B32 parity target is not met.
The overall accepted policy and 2/8 goal count remain unchanged.

All four native gradient/weight snapshot pairs match SHA256 exactly. All 642
qualification/warmup/sample scalar triples (1,926 F64 values) match bitwise.
The candidate selects exactly eight NN128x64 programs with 94 registers,
32,768 shared bytes, no local bytes and API capacity two at 256 threads. It
retains four packed online pairs, 12 NT ldmatrix programs and 298 model kernels.
The only payload-info changes are eight plain programs replaced by eight NN
programs and their selection records. All six allocator stages and owned
payload byte counts match exactly. Sampled whole-process NVML peaks are lower
bounds including setup; the +66 MiB observation is not an allocator or causal
memory claim.

The audit verifies source and prerequisite hashes, 2,363 normalized worker
NVML rows against raw inventory, 120 idle desktop pmon rows against raw records,
and zero reported foreign contexts, activity violations or monitor errors.
This sampled guard does not prove idleness between samples; its existing
terminal-tail source assertion cannot be independently reconstructed from the
untimestamped NVML rows. Raw process/log/XML/monitor files and native snapshot
hashes are bound in `audit.json`.

The preceding v1 attempt remains preserved and excluded from adoption: two
PyTorch controls failed because the C compiler was missing, and one native
baseline encountered transient Files GPU activity. V2 records the repaired
Python 3.12.16/compiler environment and updated NVIDIA 595.99.02 driver. No
library source, numerical gate, publication bundle or accepted policy changed.
