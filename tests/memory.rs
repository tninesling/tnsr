//! Peak-memory harness for executor allocation behavior (#37).
//!
//! See `docs/memory-allocator-plan.md` for the allocator design that builds
//! on these measurements.

use std::alloc::System;
use std::collections::HashMap;

use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use tnsr::graph::TensorGraph;
use tnsr::tensor::TensorExpr;
use tnsr::{Executor, SimpleExecutor};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Vector length for chain nodes; each node buffer is `LEN * 4` bytes.
const LEN: usize = 4096;

/// Builds `x -> neg -> ... -> neg` with `depth` negations.
fn chain_graph(depth: usize) -> TensorGraph<f32> {
    let mut expr = TensorExpr::<f32>::input("x", vec![LEN]);
    for _ in 0..depth {
        expr = -expr;
    }
    expr.into()
}

/// Executes a negation chain and returns the net live bytes allocated during
/// execution (allocations minus deallocations inside the measured region).
fn measure_chain_live_bytes(depth: usize) -> usize {
    let graph = chain_graph(depth);
    let mut executor = SimpleExecutor::new();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![1.0f32; LEN]);

    let region = Region::new(GLOBAL);
    let output = executor.execute(&graph, inputs).unwrap();
    assert_eq!(output.len(), LEN);
    let change = region.change();

    change
        .bytes_allocated
        .saturating_sub(change.bytes_deallocated)
}

#[test]
fn chain_execution_retains_intermediates_baseline() {
    // Baseline for #37: the executor currently retains every intermediate, so
    // live bytes grow with chain depth. Phase 2 of the allocator plan
    // (docs/memory-allocator-plan.md) replaces this with an O(1) assertion.
    let live_50 = measure_chain_live_bytes(50);
    let live_100 = measure_chain_live_bytes(100);

    let node_bytes = LEN * size_of::<f32>();
    assert!(
        live_50 > 25 * node_bytes,
        "expected depth-50 chain to retain most intermediates, got {live_50} bytes"
    );
    assert!(
        live_100 > live_50 + 25 * node_bytes,
        "expected retained bytes to grow with depth: {live_50} vs {live_100}"
    );
}
