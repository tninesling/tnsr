//! Peak-memory and buffer-reuse tests for the executor allocator (#37).
//!
//! See `docs/memory-allocator-plan.md` for the allocator design.

use std::alloc::System;
use std::collections::HashMap;

use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use tnsr::graph::TensorGraph;
#[cfg(feature = "cuda")]
use tnsr::ptx::PtxExecutor;
use tnsr::tensor::{Parameter, TensorExpr};
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

#[test]
fn repeated_execution_reuses_output_and_gradient_buffers() {
    const TRAIN_LEN: usize = 4;
    let parameter = Parameter::new(vec![2.0f32; TRAIN_LEN], vec![TRAIN_LEN]);
    let parameter_id = parameter.id();
    let loss = (TensorExpr::from(parameter.clone()) * TensorExpr::from(parameter)).reduce_mean(0);
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);
    let mut executor = SimpleExecutor::new();

    executor.execute(&graph, HashMap::new()).unwrap();
    assert_eq!(
        executor.get_gradients(&graph)[&parameter_id].len(),
        TRAIN_LEN
    );
    let first_misses = executor.stats().pool_misses;

    executor.release_gradients(&graph);
    executor.execute(&graph, HashMap::new()).unwrap();

    assert_eq!(
        executor.stats().pool_misses,
        first_misses,
        "a repeated training step should reuse all computed buffers"
    );
    assert!(executor.stats().pool_hits > 0);
}

#[test]
fn reset_stats_keeps_cached_buffers() {
    let (mut executor, _) = run_chain(10);
    assert!(executor.stats().pool_misses > 0);

    executor.reset_stats();
    assert_eq!(executor.stats().pool_misses, 0);
    assert_eq!(executor.stats().peak_live_bytes, 0);

    let graph = chain_graph(10);
    let inputs = HashMap::from([("x".to_string(), vec![1.0f32; LEN])]);
    executor.execute(&graph, inputs).unwrap();
    assert_eq!(executor.stats().pool_misses, 0);
    assert!(executor.stats().pool_hits > 0);
}

/// Repeated normalization fibers keep multiple physical steps even with
/// pointwise fusion, so this exercises liveness and allocation rather than
/// relying on a unary chain that the tiled planner eliminates.
#[cfg(feature = "cuda")]
fn device_chain_graph(depth: usize, len: usize) -> TensorGraph<f32> {
    let mut expression = TensorExpr::input("x", vec![len / 64, 64]);
    for _ in 0..depth {
        expression = expression.reduce_mean(1).broadcast_axis(1, 64).relu();
    }
    expression.into()
}

#[cfg(feature = "cuda")]
fn ptx_executor() -> Option<PtxExecutor> {
    match PtxExecutor::try_new() {
        Ok(executor) => Some(executor),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            eprintln!("skipping CUDA test: {error:#}");
            None
        }
        Err(error) => panic!("PTX initialization failed: {error:#}"),
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_chain_reuses_device_buffers() {
    let graph = device_chain_graph(20, LEN);
    let Some(mut executor) = ptx_executor() else {
        return;
    };
    let inputs = HashMap::from([("x".to_string(), vec![1.0f32; LEN])]);

    let output = executor.execute(&graph, inputs).unwrap();
    assert_eq!(output, vec![1.0f32; LEN]);
    assert!(executor.execution_metrics().kernel_launches >= 20);

    assert!(executor.stats().pool_hits > 0);
    assert!(executor.stats().peak_live_bytes <= 3 * NODE_BYTES);
}

/// A deep chain whose cumulative intermediate size far exceeds VRAM, but
/// whose live set (2-3 buffers) fits. Without pooling this would OOM;
/// with pooling every allocation after the first few is a pool hit.
#[cfg(feature = "cuda")]
#[test]
fn cuda_deep_chain_fits_in_vram_via_pooling() {
    // The unfused reduction stress graph has three operations per layer.
    // Give recursive graph/compiler traversals more than libtest's default stack.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(check_deep_device_chain)
        .unwrap()
        .join()
        .unwrap();
}

#[cfg(feature = "cuda")]
fn check_deep_device_chain() {
    use cudarc::driver::result;

    // Create the executor first so the CUDA context is initialized.
    let Some(mut executor) = ptx_executor() else {
        return;
    };

    // Query free VRAM and pick a buffer size that occupies ~25% of it,
    // so 3 live buffers (input + output + one spare) stay well under the
    // device limit.
    let (free_bytes, _total_bytes) = result::mem_get_info().unwrap();
    let elem_bytes = std::mem::size_of::<f32>();
    let buf_len = ((free_bytes / 4) / elem_bytes / 64) * 64;
    let buf_bytes = buf_len * elem_bytes;
    let depth = 200;

    let graph = device_chain_graph(depth, buf_len);

    let inputs = HashMap::from([("x".to_string(), vec![1.0f32; buf_len])]);

    executor.execute(&graph, inputs).unwrap();

    // Count actual materializations: virtual broadcasts and fused producers
    // must not turn this into a graph that would fit without liveness reuse.
    assert!(executor.execution_metrics().materialized_bytes > free_bytes);
    assert!(executor.execution_metrics().kernel_launches >= depth);

    // The pool should reuse buffers, keeping peak live bytes at O(1).
    assert!(
        executor.stats().peak_live_bytes <= 3 * buf_bytes,
        "peak live {} should be <= 3 * buf_bytes ({})",
        executor.stats().peak_live_bytes,
        3 * buf_bytes
    );
    assert!(
        executor.stats().pool_hits > 0,
        "pool should have reuse hits"
    );
    // Fresh allocations should be tiny compared to cumulative node count.
    assert!(
        executor.stats().pool_misses < 10,
        "pool misses {} should be minimal for depth {depth}",
        executor.stats().pool_misses
    );
}
