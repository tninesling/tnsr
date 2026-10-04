#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::mem::size_of;

use num_traits::{NumCast, ToPrimitive};
use tnsr::graph::TensorGraph;
use tnsr::ptx::{
    PtxExecutor,
    types::{BF16, CudaDType, F16, F32},
};
use tnsr::tensor::TensorExpr;
use tnsr::tile::{MatMulPipeline, MatMulPrecision, MatMulSharedLayout};
use tnsr::{Executor, SimpleExecutor};

fn values<D: CudaDType>(shape: Vec<usize>, period: usize) -> TensorExpr<D::HostType> {
    let data = (0..shape.iter().product())
        .map(|i| NumCast::from((i % period) as f32 / 32.0 - 0.125).unwrap())
        .collect();
    TensorExpr::constant(data, shape)
}

fn check_views<D: CudaDType>(epsilon: f32) {
    let Ok(mut executor) = PtxExecutor::<D>::try_new_for_dtype() else {
        return;
    };
    let (m, k, n) = (33, 37, 35);
    let graphs: Vec<TensorGraph<D::HostType>> = vec![
        // Contiguous input remains compatible with the staging layout.
        (values::<D>(vec![3, m, k], 7).matmul(values::<D>(vec![k, n], 5))
            + values::<D>(vec![n], 3))
        .relu()
        .into(),
        // An explicit broadcast view stages directly from its single source batch.
        (values::<D>(vec![1, m, k], 7)
            .broadcast_axis(0, 3)
            .matmul(values::<D>(vec![k, n], 5))
            + values::<D>(vec![n], 3))
        .relu()
        .into(),
        // Both matrix axes are transposed, with a broadcast RHS batch.
        (values::<D>(vec![3, k, m], 7)
            .swap_axes(1, 2)
            .matmul(values::<D>(vec![1, n, k], 5).swap_axes(1, 2))
            + values::<D>(vec![n], 3))
        .relu()
        .into(),
        // A batch axis crosses a matrix axis; this is not a trailing transpose.
        (values::<D>(vec![m, 3, k], 7)
            .permute(vec![1, 0, 2])
            .matmul(values::<D>(vec![k, n], 5))
            + values::<D>(vec![n], 3))
        .relu()
        .into(),
        // Independent batch broadcasts in each operand.
        (values::<D>(vec![2, 1, k, m], 7)
            .swap_axes(2, 3)
            .matmul(values::<D>(vec![1, 3, n, k], 5).swap_axes(2, 3))
            + values::<D>(vec![n], 3))
        .relu()
        .into(),
    ];
    for graph in &graphs {
        let expected = SimpleExecutor::<D::HostType>::new()
            .execute(graph, HashMap::new())
            .unwrap();
        for precision in [MatMulPrecision::StrictF32, MatMulPrecision::AllowTf32] {
            executor.set_matmul_precision(precision);
            let actual = executor.execute(graph, HashMap::new()).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(&expected) {
                assert!((a.to_f32().unwrap() - b.to_f32().unwrap()).abs() <= epsilon);
            }
            assert_eq!(executor.execution_metrics().kernel_launches, 1);
            assert_eq!(
                executor.execution_metrics().intermediate_materialized_bytes,
                0
            );
            let source = executor.module_source().unwrap();
            if precision == MatMulPrecision::AllowTf32 {
                assert_eq!(source.matches("wmma.store.d").count(), 1);
                let end = source.find("loop_end_k_tile_").unwrap();
                assert!(source.find("wmma.store.d").unwrap() > end);
            }
        }
    }
    // A physical buffer is used through two different logical layouts.
    let input = values::<D>(vec![33, 33], 7);
    let graph: TensorGraph<D::HostType> = input.clone().matmul(input.transpose()).relu().into();
    let expected = SimpleExecutor::<D::HostType>::new()
        .execute(&graph, HashMap::new())
        .unwrap();
    let actual = executor.execute(&graph, HashMap::new()).unwrap();
    for (a, b) in actual.iter().zip(&expected) {
        assert!((a.to_f32().unwrap() - b.to_f32().unwrap()).abs() <= epsilon);
    }
    assert_eq!(executor.execution_metrics().kernel_launches, 1);
}

#[test]
fn f32_matmul_layout_views() {
    check_views::<F32>(1e-6);
}

#[test]
fn f16_matmul_layout_views() {
    check_views::<F16>(0.001);
}

#[test]
fn bf16_matmul_layout_views() {
    check_views::<BF16>(0.008);
}

// Both expanded schedules must handle partial tiles, transposed operands,
// two-sided batch broadcasts, and scalar epilogues for every storage dtype.
fn check_multiwarp<D: CudaDType>(epsilon: f32) {
    let Ok(mut executor) = PtxExecutor::<D>::try_new_for_dtype() else {
        return;
    };
    for (m, k, n, expected_tile) in [(33, 65, 257, (16, 32)), (257, 65, 259, (32, 32))] {
        let graph: TensorGraph<D::HostType> = (values::<D>(vec![2, 1, k, m], 7)
            .swap_axes(2, 3)
            .matmul(values::<D>(vec![1, 3, n, k], 5).swap_axes(2, 3))
            + values::<D>(vec![n], 3))
        .relu()
        .into();
        let expected = SimpleExecutor::<D::HostType>::new()
            .execute(&graph, HashMap::new())
            .unwrap();
        for precision in [MatMulPrecision::StrictF32, MatMulPrecision::AllowTf32] {
            executor.set_matmul_precision(precision);
            for policy in [
                MatMulSharedLayout::Contiguous,
                MatMulSharedLayout::Padded,
                MatMulSharedLayout::Swizzled,
                MatMulSharedLayout::Auto,
            ] {
                executor.set_matmul_shared_layout(policy);
                for pipeline in [MatMulPipeline::Synchronous, MatMulPipeline::DoubleBuffered] {
                    executor.set_matmul_pipeline(pipeline);
                    let compilations = executor.compilation_count();
                    let actual = executor.execute(&graph, HashMap::new()).unwrap();
                    assert_eq!(executor.compilation_count(), compilations + 1);
                    assert_eq!(actual.len(), expected.len());
                    for (a, b) in actual.iter().zip(&expected) {
                        assert!((a.to_f32().unwrap() - b.to_f32().unwrap()).abs() <= epsilon);
                    }
                    let schedule = executor.execution_plan().unwrap().matmul_regions()[0].schedule;
                    assert_eq!(
                        (schedule.block_tile.m, schedule.block_tile.n),
                        if precision == MatMulPrecision::AllowTf32 {
                            expected_tile
                        } else {
                            (16, 16)
                        }
                    );
                    assert_eq!(executor.execution_metrics().kernel_launches, 1);
                    assert_eq!(
                        executor.execution_metrics().intermediate_materialized_bytes,
                        0
                    );
                }
            }
        }
    }
}

#[test]
fn f32_multiwarp_matmul_views_and_epilogue() {
    check_multiwarp::<F32>(1e-6);
}

#[test]
fn f16_multiwarp_matmul_views_and_epilogue() {
    check_multiwarp::<F16>(0.001);
}

#[test]
fn bf16_multiwarp_matmul_views_and_epilogue() {
    check_multiwarp::<BF16>(0.002);
}

fn check_pipeline<D: CudaDType>(epsilon: f32) {
    let Ok(mut executor) = PtxExecutor::<D>::try_new_for_dtype() else {
        return;
    };
    for k in [0, 1, 8, 16, 17, 31, 32, 33, 64, 65, 66, 96, 98, 128] {
        let n = if k == 128 { 256 } else { 258 };
        let graph: TensorGraph<D::HostType> = (values::<D>(vec![2, 1, 33, k], 7)
            .matmul(values::<D>(vec![1, 3, k, n], 5))
            + values::<D>(vec![n], 3))
        .relu()
        .into();
        let expected = SimpleExecutor::<D::HostType>::new()
            .execute(&graph, HashMap::new())
            .unwrap();
        for precision in [MatMulPrecision::StrictF32, MatMulPrecision::AllowTf32] {
            executor.set_matmul_precision(precision);
            for pipeline in [MatMulPipeline::Synchronous, MatMulPipeline::DoubleBuffered] {
                executor.set_matmul_pipeline(pipeline);
                let compilations = executor.compilation_count();
                let actual = executor.execute(&graph, HashMap::new()).unwrap();
                assert_eq!(executor.compilation_count(), compilations + 1);
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(&expected) {
                    assert!((a.to_f32().unwrap() - b.to_f32().unwrap()).abs() <= epsilon);
                }
                if k == 0 {
                    assert!(
                        executor
                            .execution_plan()
                            .unwrap()
                            .matmul_regions()
                            .is_empty()
                    );
                    assert!(!executor.module_source().unwrap().contains("cp.async"));
                    continue;
                }
                let schedule = executor.execution_plan().unwrap().matmul_regions()[0].schedule;
                assert_eq!(
                    executor.module_source().unwrap().contains("cp.async"),
                    schedule.pipeline_stages == 2
                );
                if pipeline == MatMulPipeline::Synchronous
                    || k <= schedule.block_tile.k
                    || (size_of::<D::HostType>() == 2 && !k.is_multiple_of(2))
                {
                    assert_eq!(schedule.pipeline_stages, 1);
                } else {
                    assert_eq!(schedule.pipeline_stages, 2);
                }
                assert_eq!(executor.execution_metrics().kernel_launches, 1);
                assert_eq!(
                    executor.execution_metrics().intermediate_materialized_bytes,
                    0
                );
            }
        }
    }
}

#[test]
fn f32_pipeline_tails_and_broadcast_epilogue() {
    check_pipeline::<F32>(1e-6);
}
#[test]
fn f16_pipeline_tails_and_broadcast_epilogue() {
    check_pipeline::<F16>(0.001);
}
#[test]
fn bf16_pipeline_tails_and_broadcast_epilogue() {
    check_pipeline::<BF16>(0.008);
}
