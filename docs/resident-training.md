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

## Resident Adam owner

`ptx::resident::ResidentAdam` registers nonempty f32 parameters with explicit
`AdamGroup` indices. Shared references to the same storage/group are deduplicated;
conflicting IDs/groups are rejected. `executor()` creates compatible model
executors on the owner's context. Each interval calls `begin_step()`, then
`accumulate()` for each weighted-loss microbatch, then `step(max_grad_norm)`.
The loss graph supplies transition weighting; accumulation adds each logical
gradient exactly once through its general virtual index map on CUDA.

The owner imports host edits once at the next interval, retaining moments as an
in-place host edit would. Edits during accumulation/update are rejected. Writers
must be serialized. New owners start fresh moments; named model archives export
model weights, not resumable optimizer state. `gradients()` and `moments()` are
explicit diagnostic downloads; normal updates return only small status values.

Norm reduction uses an explicit deterministic f32 tree over padded flat gradient
storage. It can differ slightly from the old host f64 norm; clipping and named
gradients/weights pass the existing real-model tolerances. Forward/backward
matmul and reduction policies remain those selected on the model executor.
Unrepresentable/nonfinite f32 norms are rejected. Adam uses f32 moments and
`sqrt.rn.f32`; first/second bias correction, epsilon, L2 decay and separate group
rates match the host algorithm. Learning-rate changes retain state/timestep.

Weights and moments are staged in distinct allocations. A device finite check
over every staged weight/moment must pass before publication; a failed check
leaves previous weights, moments and timestep intact. Publishing a successful
update installs owning lazy readers in shared storage. Saved weights, cached
inference, CPU reads and reads after owner destruction therefore see current
values. This layer uses fresh staged storage; stable-address replay is issue 93.

Host-provided index inputs and immutable constant indices are checked from known
host values, avoiding per-operator device readbacks. Weak allocation identity and
per-execution facts prevent stale validation across updates or pooled reuse.
Computed device indices retain the original checked fallback.

The forty-five CUDA trainer uses this owner by default. The CPU path retains the
host optimizer. Full shared/separate-critic two-step PPO comparisons pass, as do
182 CPU / 354 all-feature CUDA-required tests and clippy. Focused lifecycle tests
also pass after the module split. See [measurements](training-resident-performance.md).
