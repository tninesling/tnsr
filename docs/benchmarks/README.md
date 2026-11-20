# Benchmark records

The repository includes benchmark tools, human-readable reports, compact result
JSON, and small audit/provenance manifests. The current training comparison is
summarized in [transformer-training-scorecard.json](transformer-training-scorecard.json)
and explained in [the parity report](../transformer-training-parity.md).

Large captures are intentionally excluded from Git: full raw performance/parity
JSON reports and their compressed copies, raw tensor archives, profiler captures,
generated PTX/logs, and duplicate source snapshots. Existing local copies are
preserved. [local-artifacts.json](local-artifacts.json) records the hashes of the
omitted top-level reports. No external artifact upload is implied by a manifest
or checksum.

Audit manifests retain the original filenames, SHA-256 digests, measurement
scope, and limitations. References to excluded archives or ignored `target/`
paths describe locally retained evidence; those files are not supplied by a
fresh checkout. Historical measurements whose raw data were already unavailable
remain explicitly labeled in the reports. Reproduction requires rerunning the
documented tools or separately obtaining the matching artifacts.

See [the decoder-training benchmark instructions](../transformer-training-performance.md)
for the model, numerical gates, dependencies, and reproduction commands.
