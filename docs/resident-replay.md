# Resident model replay

`ResidentAdam::new_with_graph_replay` opts into private-stream, fixed-shape model
replay. `new` retains the ordinary path. The forty-five CUDA worker exposes
`settings.cuda_graphs=true`; training, validation and benchmark scripts accept
`--cuda-graphs`. Strict/tree reduction ordering remains a separate policy. Both
policies are fixed for a trainer's lifetime.

Each compiled shape bucket owns stable input and parameter-copy buffers. Host
inputs are validated and uploaded before replay. A parameter copy refreshes once
when its resident source allocation changes, including host edits imported at an
interval boundary. Weak allocation identity prevents address reuse without
retaining obsolete generations of GPU weights for every bucket. Partial batches
use their exact shape and transition weight; no padded examples enter the loss.
The trainer's existing eight-entry shape/clip cache bounds capture ownership.

Capture covers the existing forward/backward kernels. Gradient accumulation,
norm/status readbacks, grouped Adam, rates, publication and export stay outside
capture. This preserves dynamic clipping/schedules and the staged finite-update
transaction. Parameters, gradients and Adam moments remain f32. Replay retains
the selected arithmetic/reduction policy and its existing numerical tolerances.

## First use and fallbacks

First use evaluates the pure model twice without accumulating owner gradients.
The first evaluation uploads constants; a second settles workspace which those
uploads may have pinned. Allocation is then frozen while recording the same
model, and its instantiated graph runs once. Only this final evaluation feeds
gradient accumulation and the optimizer. Setup evaluations are counted in
`warmup_executions` and cold GPU work counters; warm timing excludes setup.

The executor retains the graph, module/function handles, source allocations,
output values and pooled workspace. No new CUDA allocation is permitted during
recording. Policy changes, compilation, constant-allocation replacement,
`clear_pool` and explicit gradient release destroy capture before releasing or
reusing its memory. Parameter value changes refresh stable copies without
recompiling or rebinding graph kernel arguments.

Host-backed/immutable constant index facts are checked before every graph launch.
Invalid changed IDs cannot reach an embedding kernel. Computed indices retain
the checked uncaptured validation path. Source bindings which escape as retained
outputs, missing frozen workspace and driver capture failures also fall back.
Failed recording always ends capture, including deferred cudarc guard errors,
before the normal execution runs once. A failed frame or Adam finite check does
not publish parameters/moments or advance the optimizer timestep.

`cuda_graphs` responses report active shapes, captures, graph launches, setup
evaluations, fallback reasons and stable binding bytes. The benchmark requires
active capture without fallbacks when graph mode is requested. Its baseline
requests omit new settings; `--baseline-tree-reductions` selects the same tree
ordering in a compatible older baseline. Per-worker GPU memory is sampled outside
the timed interval, including driver modules, workspace and graph metadata.
Stable binding bytes alone exclude workspace and driver graph metadata.

## Stream and lifetime contract

The resident owner keeps one private execution stream for model uploads,
execution, accumulation and updates. Executors share a gate covering CUDA entry
points, capture and destruction. Device bindings cannot escape this stream
through the public API. The owner's private context disables cudarc's automatic
cross-stream event tracking: waiting on events recorded before capture would
otherwise introduce uncaptured dependencies into it.

Published parameter readers use a separate readback stream. Publication happens
only after the finite-status readback completes all staged writes. Published
weight allocations are immutable and retained through the blocking host copy;
the next optimizer step allocates new weights. Concurrent host readers therefore
need no dependency on model capture and cannot free a weight while reading it.
Other diagnostics use the execution stream. Host writers remain serialized and
revision conflicts retain the existing resident-owner checks.

The capture wrapper binds its context before destruction, and capture is destroyed
before its referenced buffers/modules. This stream/lifetime invariant is the
reason event-free device storage is safe here; ordinary executors keep their
existing event-tracking behavior. CUDA remains optional and the CPU path uses
its existing optimizer.

Tests compare changed inputs, full/partial shapes, host mutations, constant
replacement, group rates, gradients, moments and weights against the CPU/ordinary
CUDA paths. They also check exact IDs above 256, invalid-index recovery, computed
indices, source aliases, cache clearing, concurrent published readers and rejected
nonfinite updates. Both committed shared/separate PPO fixtures require active
full/partial replay while matching their two-step gradients and weights.
