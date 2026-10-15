# Resident training

Issue [91](https://github.com/tninesling/tnsr/issues/91) moves parameters, gradients
and optimizer state to CUDA. The first stack layer supplies shared, revisioned
`ParameterStorage` and a PTX source upload cache.

`Parameter`, expression nodes and graph nodes share the same storage. A host
read takes a guard and synchronizes any pending device reader once. Mutable
access advances the revision when the guard drops. CPU SGD/Adam therefore
invalidate PTX uploads through the same path as `Parameter::set_values`.
Constants use their immutable allocation identity; parameters use identity and
revision, including the storage identity when an explicit parameter ID is reused.

A device owner publishes an expected revision and an owning readback callback.
Publication fails if another writer changed the parameter. Publishing another
device write replaces the pending reader without downloading the old value.
Readers retain ownership of the latest device allocation after a training
session drops. Failed downloads remain retryable. Revisions are wrapping u64
counters; callers must serialize writers and never perform 2^64 mutations
between cache checks.

Normal `Parameter::values`, CPU execution, inference uploads and named-parameter
archive saves all use synchronized host guards. They cannot silently return or
serialize an obsolete host copy. Resident training avoids those reads while
working on the device; explicit host observation is a synchronization boundary.

The graph node's public `data` field now uses `Arc<ParameterStorage<D>>` instead
of `Arc<Mutex<Vec<D>>>`; its `lock` method returns a version-tracking guard and a
contextual error. This is the source-level compatibility change needed to track
mutations through existing graph/optimizer access as well as `Parameter` methods.

Validation includes lazy/retryable readback, superseded publications, revision
conflicts, host mutation, CPU optimizer invalidation, reused IDs with distinct
storage and repeated CUDA executions with no unchanged source uploads. The
first layer passes 182 CPU / 349 all-feature CUDA-required tests and clippy.
