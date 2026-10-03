use half::{bf16, f16};
use std::collections::HashMap;
use tnsr::graph::TensorGraph;
use tnsr::tensor::TensorExpr;
use tnsr::tile::{DType, Stmt, TileGraph};
use tnsr::{Executor, SimpleExecutor};

/// Helper to convert f32 slice to generic type
fn to_dtype<D: num_traits::Float>(vals: &[f32]) -> Vec<D> {
    vals.iter().map(|&v| D::from(v).unwrap()).collect()
}

/// Helper to convert generic type back to f32 for comparison
fn to_f32<D: num_traits::Float>(vals: &[D]) -> Vec<f32> {
    vals.iter().map(|&v| v.to_f32().unwrap()).collect()
}

/// Compare results with tolerance appropriate for the dtype
fn assert_close<D: num_traits::Float>(result: &[D], expected: &[f32], tolerance: f32) {
    let result_f32 = to_f32(result);
    assert_eq!(result_f32.len(), expected.len());
    for (i, (&r, &e)) in result_f32.iter().zip(expected.iter()).enumerate() {
        let diff = (r - e).abs();
        assert!(
            diff <= tolerance,
            "Mismatch at index {}: got {}, expected {}, diff {}",
            i,
            r,
            e,
            diff
        );
    }
}

#[test]
fn test_f16_basic_ops() {
    // Create a simple computation: x + 2.0
    let x = TensorExpr::<f16>::input("x", vec![4]);
    let two = TensorExpr::<f16>::constant(to_dtype(&[2.0, 2.0, 2.0, 2.0]), vec![4]);
    let y = x + two;

    let mut executor = SimpleExecutor::<f16>::new();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), to_dtype(&[1.0, 2.0, 3.0, 4.0]));

    let result = executor.execute(&y.into(), inputs).unwrap();

    // f16 has ~3 decimal digits of precision
    assert_close(&result, &[3.0, 4.0, 5.0, 6.0], 0.001);
}

#[test]
fn test_bf16_basic_ops() {
    // Create a simple computation: x * 2.0
    let x = TensorExpr::<bf16>::input("x", vec![4]);
    let two = TensorExpr::<bf16>::constant(to_dtype(&[2.0, 2.0, 2.0, 2.0]), vec![4]);
    let y = x * two;

    let mut executor = SimpleExecutor::<bf16>::new();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), to_dtype(&[1.0, 2.0, 3.0, 4.0]));

    let result = executor.execute(&y.into(), inputs).unwrap();

    // bf16 has ~2-3 decimal digits of precision
    assert_close(&result, &[2.0, 4.0, 6.0, 8.0], 0.01);
}

#[test]
fn test_f16_transcendental() {
    // Test exp on f16
    let x = TensorExpr::<f16>::input("x", vec![3]);
    let y = x.exp();

    let mut executor = SimpleExecutor::<f16>::new();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), to_dtype(&[0.0, 1.0, 2.0]));

    let result = executor.execute(&y.into(), inputs).unwrap();

    // Expected: [e^0, e^1, e^2] = [1.0, 2.718..., 7.389...]
    assert_close(&result, &[1.0, std::f32::consts::E, 7.38906], 0.01);
}

#[test]
fn test_bf16_matmul() {
    // Test 2x2 matmul with bf16
    let a = TensorExpr::<bf16>::input("a", vec![2, 2]);
    let b = TensorExpr::<bf16>::input("b", vec![2, 2]);
    let c = a.matmul(b);

    let mut executor = SimpleExecutor::<bf16>::new();
    let mut inputs = HashMap::new();
    // A = [[1, 2], [3, 4]]
    inputs.insert("a".to_string(), to_dtype(&[1.0, 2.0, 3.0, 4.0]));
    // B = [[5, 6], [7, 8]]
    inputs.insert("b".to_string(), to_dtype(&[5.0, 6.0, 7.0, 8.0]));

    let result = executor.execute(&c.into(), inputs).unwrap();

    // Expected: [[19, 22], [43, 50]]
    assert_close(&result, &[19.0, 22.0, 43.0, 50.0], 0.1);
}

#[test]
fn test_f16_batched_matmul_accumulates_each_batch_independently() {
    let a = TensorExpr::<f16>::input("a", vec![2, 1, 2]);
    let b = TensorExpr::<f16>::input("b", vec![1, 2, 1]);
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), to_dtype(&[1.0, 2.0, 3.0, 4.0]));
    inputs.insert("b".to_string(), to_dtype(&[5.0, 6.0]));

    let result = SimpleExecutor::<f16>::new()
        .execute(&a.matmul(b).into(), inputs)
        .unwrap();

    assert_close(&result, &[17.0, 39.0], 0.01);
}

#[test]
fn test_f16_relu() {
    let x = TensorExpr::<f16>::input("x", vec![6]);
    let y = x.relu();

    let mut executor = SimpleExecutor::<f16>::new();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), to_dtype(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]));

    let result = executor.execute(&y.into(), inputs).unwrap();

    assert_close(&result, &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], 0.001);
}

#[test]
fn test_dtype_comparison() {
    // Same computation in different dtypes
    let input_vals = vec![1.0, 2.0, 3.0, 4.0];

    // f32 (baseline)
    let x_f32 = TensorExpr::<f32>::input("x", vec![4]);
    let y_f32 = x_f32.exp();
    let mut exec_f32 = SimpleExecutor::<f32>::new();
    let mut inputs_f32 = HashMap::new();
    inputs_f32.insert("x".to_string(), input_vals.clone());
    let result_f32 = exec_f32.execute(&y_f32.into(), inputs_f32).unwrap();

    // f16
    let x_f16 = TensorExpr::input("x", vec![4]);
    let y_f16 = x_f16.exp();
    let mut exec_f16 = SimpleExecutor::<f16>::new();
    let mut inputs_f16 = HashMap::new();
    inputs_f16.insert("x".to_string(), to_dtype(&input_vals));
    let result_f16 = exec_f16.execute(&y_f16.into(), inputs_f16).unwrap();

    // bf16
    let x_bf16 = TensorExpr::input("x", vec![4]);
    let y_bf16 = x_bf16.exp();
    let mut exec_bf16 = SimpleExecutor::<bf16>::new();
    let mut inputs_bf16 = HashMap::new();
    inputs_bf16.insert("x".to_string(), to_dtype(&input_vals));
    let result_bf16 = exec_bf16.execute(&y_bf16.into(), inputs_bf16).unwrap();

    // All should be close to f32 baseline, within appropriate tolerances
    assert_close(&result_f16, &result_f32, 0.1);
    assert_close(&result_bf16, &result_f32, 0.1);
}

#[test]
fn test_f16_matmul_accumulates_in_f32() {
    const K: usize = 4096;
    let a = TensorExpr::<f16>::input("a", vec![1, K]);
    let b = TensorExpr::<f16>::input("b", vec![K, 1]);
    let graph = a.matmul(b).into();
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vec![f16::from_f32(1.0); K]);
    inputs.insert("b".to_string(), vec![f16::from_f32(1.0); K]);

    let result = SimpleExecutor::<f16>::new()
        .execute(&graph, inputs)
        .unwrap();

    assert_eq!(result, vec![f16::from_f32(K as f32)]);
}

#[test]
fn test_bf16_reduction_accumulates_in_f32() {
    const N: usize = 1024;
    let x = TensorExpr::<bf16>::input("x", vec![N]);
    let graph = x.reduce_axis_sum(0).into();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![bf16::from_f32(1.0); N]);

    let result = SimpleExecutor::<bf16>::new()
        .execute(&graph, inputs)
        .unwrap();

    assert_eq!(result, vec![bf16::from_f32(N as f32)]);
}

#[test]
fn test_tile_lowering_preserves_f16_storage() {
    let x = TensorExpr::<f16>::input("x", vec![4]);
    let y = TensorExpr::<f16>::input("y", vec![4]);
    let graph: TensorGraph<f16> = (x + y).into();
    let tile_graph = TileGraph::from(graph);

    for tile_ir in tile_graph.graph.node_weights() {
        assert!(
            tile_ir.params.iter().all(|param| param.dtype == DType::F16),
            "all graph boundaries must preserve f16 storage"
        );
    }
}

#[test]
fn test_tile_matmul_uses_f32_accumulator_for_bf16() {
    let a = TensorExpr::<bf16>::input("a", vec![16, 16]);
    let b = TensorExpr::<bf16>::input("b", vec![16, 16]);
    let graph: TensorGraph<bf16> = a.matmul(b).into();
    let tile_graph = TileGraph::from(graph);
    let matmul = tile_graph
        .graph
        .node_weights()
        .find(|tile_ir| tile_ir.kernel_name == "matmul")
        .unwrap();

    assert!(matmul.params.iter().all(|param| param.dtype == DType::BF16));
    assert!(matmul.body.stmts.iter().any(|stmt| {
        matches!(
            stmt,
            Stmt::AllocTile {
                dtype: DType::F32,
                ..
            }
        )
    }));
}

#[cfg(feature = "cuda")]
#[test]
fn test_ptx_lowering_converts_f16_storage_to_f32_compute() {
    use tnsr::ptx::PtxGraph;

    let x = TensorExpr::<f16>::input("x", vec![4]);
    let y = -x;
    let graph: TensorGraph<f16> = y.into();
    let ptx_graph = PtxGraph::from(TileGraph::from(graph));
    let ptx = ptx_graph.to_ptx();

    assert!(ptx.contains("ld.global.b16"));
    assert!(ptx.contains("cvt.f32.f16"));
    assert!(ptx.contains("cvt.rn.f16.f32"));
    assert!(ptx.contains("st.global.b16"));
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_executor_accepts_bf16_graphs() {
    use tnsr::cuda::CudaExecutor;

    let Ok(mut executor) = CudaExecutor::try_new() else {
        return;
    };
    let x = TensorExpr::<bf16>::input("x", vec![4]);
    let two = TensorExpr::constant(vec![bf16::from_f32(2.0); 4], vec![4]);
    let graph = (x + two).into();
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        [1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect(),
    );

    let result = executor.execute(&graph, inputs).unwrap();
    assert_close(&result, &[3.0, 4.0, 5.0, 6.0], 0.01);
}

#[cfg(feature = "cuda")]
#[test]
fn test_ptx_executor_runs_f16_with_f32_compute() {
    use tnsr::ptx::PtxExecutor;
    use tnsr::ptx::types::F16;

    let Ok(mut executor) = PtxExecutor::<F16>::try_new_for_dtype() else {
        return;
    };
    let x = TensorExpr::<f16>::input("x", vec![4]);
    let two = TensorExpr::constant(vec![f16::from_f32(2.0); 4], vec![4]);
    let graph: TensorGraph<f16> = (x * two).into();
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        [1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect(),
    );

    let result = executor.execute(&graph, inputs).unwrap();
    assert_close(&result, &[2.0, 4.0, 6.0, 8.0], 0.01);
}

#[cfg(feature = "cuda")]
#[test]
fn test_ptx_executor_runs_bf16_reduction_with_f32_accumulator() {
    use tnsr::ptx::PtxExecutor;
    use tnsr::ptx::types::BF16;

    let Ok(mut executor) = PtxExecutor::<BF16>::try_new_for_dtype() else {
        return;
    };
    let x = TensorExpr::<bf16>::input("x", vec![512]);
    let graph = x.reduce_axis_sum(0).into();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![bf16::from_f32(1.0); 512]);

    let result = executor.execute(&graph, inputs).unwrap();
    assert_eq!(result, vec![bf16::from_f32(512.0)]);
}

#[cfg(feature = "cuda")]
fn check_half_fused_matmul<D: tnsr::ptx::types::CudaDType>(tolerance: f32) {
    use tnsr::ptx::PtxExecutor;
    use tnsr::tile::MatMulPrecision;
    let Ok(mut executor) = PtxExecutor::<D>::try_new_for_dtype() else {
        return;
    };
    // Odd extents exercise zero padding; distinct batches and broadcast RHS
    // exercise storage-width addressing and batch strides.
    let (m, n, k) = (33, 35, 37);
    let a = TensorExpr::<D::HostType>::input("a", vec![2, m, k]);
    let b = TensorExpr::<D::HostType>::input("b", vec![1, k, n]);
    let bias = TensorExpr::<D::HostType>::input("bias", vec![n]);
    let graph: TensorGraph<D::HostType> = (a.matmul(b) + bias).relu().into();
    let lhs: Vec<f32> = (0..2 * m * k)
        .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
        .collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
    let inputs = HashMap::from([
        ("a".into(), to_dtype::<D::HostType>(&lhs)),
        ("b".into(), to_dtype::<D::HostType>(&rhs)),
        ("bias".into(), to_dtype::<D::HostType>(&vec![0.25; n])),
    ]);
    let expected = SimpleExecutor::<D::HostType>::new()
        .execute(&graph, inputs.clone())
        .unwrap();
    let actual = executor
        .compile_and_execute(&graph, inputs.clone())
        .unwrap();
    assert_close(&actual, &to_f32(&expected), tolerance);
    let source = executor.module_source().unwrap();
    assert!(source.contains("wmma.mma.sync.aligned.m16n16k16"));
    assert_eq!(executor.execution_metrics().kernel_launches, 1);
    assert!(executor.execution_plan().unwrap().matmul_regions().iter().all(|r| r.schedule.storage_dtype == <D::HostType as tnsr::tile::TileDType>::TILE_DTYPE));
    // Switching to strict accumulation invalidates the compiled module.
    executor.set_matmul_precision(MatMulPrecision::StrictF32);
    let strict = executor.compile_and_execute(&graph, inputs).unwrap();
    assert_close(&strict, &to_f32(&expected), tolerance);
    assert_eq!(executor.compilation_count(), 2);
    assert!(!executor.module_source().unwrap().contains("wmma.mma"));
}

#[cfg(feature = "cuda")]
#[test]
fn test_ptx_f16_tensor_core_batched_fused_matmul() {
    check_half_fused_matmul::<tnsr::ptx::types::F16>(0.004);
}

#[cfg(feature = "cuda")]
#[test]
fn test_ptx_bf16_tensor_core_batched_fused_matmul() {
    check_half_fused_matmul::<tnsr::ptx::types::BF16>(0.032);
}

#[cfg(feature = "cuda")]
#[test]
fn test_ptx_half_fused_cooperative_reduction() {
    use tnsr::ptx::types::BF16;
    use tnsr::ptx::{PtxExecutor, PtxReductionMode};
    let Ok(mut executor) =
        PtxExecutor::<BF16>::try_new_with_reduction_mode(PtxReductionMode::DeterministicTree)
    else {
        return;
    };
    let x = TensorExpr::<bf16>::input("x", vec![3, 513]);
    let graph: TensorGraph<bf16> = x.relu().reduce_axis_sum(1).into();
    let inputs = HashMap::from([("x".into(), vec![bf16::from_f32(1.0); 3 * 513])]);
    let actual = executor.compile_and_execute(&graph, inputs).unwrap();
    assert_eq!(actual, vec![bf16::from_f32(513.0); 3]);
    assert_eq!(executor.execution_metrics().kernel_launches, 1);
    assert_eq!(
        executor.execution_metrics().materialized_bytes,
        (3 * 513 + 3) * 2
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_ptx_half_embedding_gradient_accumulates_repeated_ids() {
    use tnsr::ptx::PtxExecutor;
    use tnsr::ptx::types::F16;
    let Ok(mut executor) = PtxExecutor::<F16>::try_new_for_dtype() else {
        return;
    };
    let weight = TensorExpr::parameter(vec![f16::ZERO; 10], vec![5, 2]);
    let weight_id = match weight.kind() {
        tnsr::tensor::ExprKind::Parameter { id, .. } => *id,
        _ => unreachable!(),
    };
    let indices = TensorExpr::constant(to_dtype::<f16>(&[1.0, 1.0, 2.0]), vec![3]);
    let graph: TensorGraph<f16> = weight.embedding(indices).mean_all().into();
    let loss = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss);
    executor
        .compile_and_execute(&graph, HashMap::new())
        .unwrap();
    let expected = [
        0.0,
        0.0,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 6.0,
        1.0 / 6.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    assert_close(
        &executor.get_gradients(&graph)[&weight_id],
        &expected,
        0.001,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_half_runtime_ptx_preserves_gradients_and_reuses_compilation() {
    use tnsr::{Backend, Runtime};
    let Ok(mut runtime) = Runtime::<bf16>::with_backend(Backend::Ptx) else {
        return;
    };
    let weight = TensorExpr::parameter(to_dtype::<bf16>(&[-1.0, 0.5, 2.0]), vec![3]);
    let id = match weight.kind() {
        tnsr::tensor::ExprKind::Parameter { id, .. } => *id,
        _ => unreachable!(),
    };
    let input = TensorExpr::<bf16>::input("x", vec![3]);
    let graph: TensorGraph<bf16> = (weight * input).relu().reduce_axis_sum(0).into();
    let loss = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss);
    for value in [1.0, 2.0] {
        let result = runtime
            .execute(
                &graph,
                HashMap::from([("x".into(), vec![bf16::from_f32(value); 3])]),
            )
            .unwrap();
        assert_close(&result, &[2.5 * value], 0.01);
        assert_close(
            &runtime.get_gradients(&graph)[&id],
            &[0.0, value, value],
            0.01,
        );
    }
    match &runtime {
        Runtime::PtxBF16(executor) => assert_eq!(executor.compilation_count(), 1),
        _ => unreachable!(),
    }
}
