# BF16 shadow qualification, November 19

The paired-store leaf passes16 length/kind cases,1536 array comparisons and576 readonly checks. It compares unchanged old Adam state and an independent ordinary GPU materializer with scalar F32, scalar BF16-first and four-value vector variants. Tails, replay, signed zero, rounding ties, subnormals, NaN classes and finite F32 values rounding to BF16 infinity are covered. The existing F32 health gate remains unchanged. No performance or model-memory claim follows.

The corrected actual W1024/S128/B2 census finds21 distinct matrix owners and54,788,096 elements at revisions0 and1. All have exact physical-order view proofs, live materializers and typed target ownership; the tied embedding keeps its F32 consumer. Target BF16 intervals are disjoint and total109,576,192 bytes in each observation. The first census failed because it observed compiler values after pruning; its raw failed attempt is retained. The corrected test observes exact constructor metadata and verifies live pointers after execution.

Two sparse shadow versions imply219,152,384 logical bytes. Removing one logical cast version suggests a net109,576,192-byte (104.5MiB) payload increase; allocator and whole-process peaks must be measured independently. Old readers may retain further versions.

The archive contains raw reports, loaded leaf PTX, test logs, source manifests and retained-binary identities. Complete source snapshots are retained under transformer-training-sources/shadow-proofs-v1-20261119.tar.gz and shadow-proofs-v2-20261119.tar.gz. Production integration and complete-update performance are not established by these tests; accepted policy and parity count remain unchanged at2/8.
