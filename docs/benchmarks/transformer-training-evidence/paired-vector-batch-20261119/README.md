# Paired-vector grouping evidence — no adoption

The qualified `timing-v2` experiment has two isolated fresh processes, reversed
case/allocation/construction order, balanced output slots, 1,152 CUDA-event
records and source-bound 1,200 output/240 read-only comparisons. Grouped paired
vector updates regress at all three lengths in both orientations:

| Elements/member | Separate four, µs/group | Grouped four, µs/group | Latency increase |
| --- | ---: | ---: | ---: |
| 1,048,576 | 199.3360 | 208.7600 | 4.7277% |
| 2,818,048 | 562.1760 | 572.4880 | 1.8343% |
| 3,145,728 | 622.7580 | 634.9675 | 1.9606% |

**Do not adopt grouped dispatch.** These repeated static-input/address component
measurements do not establish optimizer-owner, complete-update, cache/DRAM,
capacity or PyTorch parity results. Full-model counts and accepted policies are
unchanged.

`raw-evidence.tar.gz` preserves the ten-case leaf proof (2,250 output/750
read-only checks), both component audits, sealed build metadata, PTX/resources,
all untimed/timing attempts, raw guard CSV/XML/pmon/logs and rejection evidence.
Untimed-v1/v2 and timing-v1 isolation failures remain excluded; only untimed-v3
and timing-v2 qualify as complete campaigns.

`manifest.json` lists original repository-relative paths, SHA256s and sizes.
The archive was reopened and every member verified. Byte-identical artifact
JSON and reloaded PTX copies are represented by 32 alias records pointing to
retained canonical files; recreate those paths from the listed canonical bytes
if restoring the original layout. Executable payloads and Python caches are
excluded; exact binary SHA256/size and source metadata are retained. The two
source archives remain in the durable
[training sources directory](../../transformer-training-sources/), with their
paths and hashes recorded in the manifest.
