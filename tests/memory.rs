//! Peak-memory and buffer-reuse tests for the executor allocator (#37).
//!
//! See `docs/memory-allocator-plan.md` for the allocator design.

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
const NODE_BYTES: usize = LEN * size_of::<f32>();

/// Builds `x -> neg -> ... -> neg` with `depth` negations.
fn chain_graph(depth: usize) -> TensorGraph<f32> {
    let mut expr = TensorExpr::<f32>::input("x", vec![LEN]);
    for _ in 0..depth {
        expr = -expr;
    }
    expr.into()
}

/// Executes a negation chain, returning the executor (for stats) and the
/// bytes allocated from the OS inside the measured region.
fn run_chain(depth: usize) -> (SimpleExecutor<f32>, usize) {
    let graph = chain_graph(depth);
    let mut executor = SimpleExecutor::new();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![1.0f32; LEN]);

    let region = Region::new(GLOBAL);
    let output = executor.execute(&graph, inputs).unwrap();
    assert_eq!(output.len(), LEN);

    (executor, region.change().bytes_allocated)
}

#[test]
fn chain_execution_peak_live_bytes_is_constant() {
    // Only the value being produced and its single live input are held at
    // any point, regardless of chain depth.
    for depth in [50, 100] {
        let (executor, _) = run_chain(depth);
        let peak = executor.stats().peak_live_bytes;
        assert!(
            peak <= 3 * NODE_BYTES,
            "depth {depth}: expected O(1) peak live bytes, got {peak}"
        );
    }
}

#[test]
fn chain_execution_reuses_pooled_buffers() {
    // After the first few nodes every allocation is served by the pool, so
    // OS-level allocation stays flat as depth grows.
    let (_, allocated_50) = run_chain(50);
    let (_, allocated_100) = run_chain(100);

    assert!(
        allocated_100 < 10 * NODE_BYTES,
        "expected pooled reuse, got {allocated_100} bytes allocated"
    );
    assert!(
        allocated_100 <= allocated_50 + NODE_BYTES,
        "allocation should not grow with depth: {allocated_50} vs {allocated_100}"
    );
}
