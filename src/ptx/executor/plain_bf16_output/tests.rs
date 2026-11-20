use super::*;
use crate::ptx::BF16;
use crate::{
    tensor::TensorExpr,
    tile::{DType, MatMulCapabilities, MatMulPrecision, MatMulRegion, MatMulSchedule, Stmt},
};
use half::bf16;
fn resources() -> crate::tile::MatMulResources {
    crate::tile::MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..Default::default()
    }
}
fn region() -> Result<MatMulRegion> {
    let graph: TensorGraph<bf16> = TensorExpr::input("a", vec![2048, 512])
        .matmul(TensorExpr::input("b", vec![512, 1024]))
        .into();
    let out = graph
        .graph
        .node_indices()
        .find(|&n| matches!(graph[n], TensorGraphNode::MatMul { .. }))
        .unwrap();
    let mut r = MatMulRegion::from_graph(&graph, out, vec![], vec![out])?;
    r.schedule = MatMulSchedule::select_with_resources(
        2048,
        1024,
        512,
        DType::BF16,
        MatMulCapabilities {
            bf16: true,
            tf32: true,
            f16: true,
        },
        MatMulPrecision::NativeHalf,
        resources(),
    )
    .with_shared_memory_reuse(true);
    r.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        true,
        resources(),
    )?;
    Ok(r)
}
fn tiles(r: MatMulRegion) -> Result<TileGraph> {
    let mut t = TileGraph::from_graph(&TensorGraph::<f32>::new());
    t.add_matmul_regions(&[r])?;
    t.optimize_indices()?;
    Ok(t)
}
#[test]
fn plain_bf16_output_exact_proof_atomic_rejection_and_default_identity() -> Result<()> {
    let r = region()?;
    proof::guard(&r)?;
    let mut t = tiles(r.clone())?;
    let old = t.matmul_region_kernels[0].ir.clone();
    proof::rewrite(
        &mut t.matmul_region_kernels[0].ir,
        proof::Scope { m: r.m, n: r.n },
    )?;
    let new = &t.matmul_region_kernels[0].ir;
    assert_eq!(old.shared_mem_bytes, new.shared_mem_bytes);
    assert_eq!(
        new.body
            .stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Store { .. }))
            .count(),
        16
    );
    assert!(
        !new.body
            .stmts
            .iter()
            .any(|s| matches!(s, Stmt::StoreGlobalPredicated { .. }))
    );
    for mode in 0..5 {
        let mut bad = old.clone();
        match mode {
            0 => bad
                .body
                .stmts
                .retain(|s| !matches!(s, Stmt::StoreGlobalPredicated { .. })),
            1 => {
                if let Some(Stmt::StoreGlobalPredicated { element_index, .. }) = bad
                    .body
                    .stmts
                    .iter_mut()
                    .find(|s| matches!(s, Stmt::StoreGlobalPredicated { .. }))
                {
                    *element_index = crate::tile::Expr::Const(0);
                }
            }
            3 => {
                let duplicate = bad
                    .body
                    .stmts
                    .iter()
                    .find(|s| matches!(s, Stmt::LetIndex { .. }))
                    .unwrap()
                    .clone();
                bad.body.stmts.push(duplicate);
            }
            4 => {
                let store = bad
                    .body
                    .stmts
                    .iter()
                    .find(|s| matches!(s, Stmt::StoreGlobalPredicated { .. }))
                    .unwrap()
                    .clone();
                bad.body.stmts.push(Stmt::ForLoop {
                    loop_var: "unexpected".into(),
                    start: crate::tile::Expr::Const(0),
                    end: crate::tile::Expr::Const(1),
                    carries: vec![],
                    body: crate::tile::Block { stmts: vec![store] },
                });
            }
            _ => {
                bad.params
                    .iter_mut()
                    .find(|p| p.name == "output_0")
                    .unwrap()
                    .dtype = DType::F32;
            }
        }
        let snapshot = format!("{bad:?}");
        assert!(proof::rewrite(&mut bad, proof::Scope { m: r.m, n: r.n }).is_err());
        assert_eq!(snapshot, format!("{bad:?}"));
    }
    for mode in 0..6 {
        let mut bad = r.clone();
        match mode {
            0 => bad.m += 1,
            1 => bad.batch_shape.push(2),
            2 => bad.inputs[0].source_shape[0] -= 1,
            3 => bad.schedule.operand_layouts.0.column_major = true,
            4 => bad.epilogue_operations.push(crate::tile::RegionOp {
                node: bad.outputs[0].node,
                output: crate::tile::RegionValue(99),
                kind: crate::tile::RegionOpKind::Unary {
                    op: crate::tensor::UnaryOp::Neg,
                    input: bad.matmul_value,
                },
            }),
            _ => bad.k = 32,
        }
        assert!(proof::guard(&bad).is_err());
    }
    Ok(())
}
fn executor() -> Result<PtxExecutor<BF16>> {
    let mut e = PtxExecutor::<BF16>::try_new_for_dtype()?;
    e.set_matmul_precision(MatMulPrecision::NativeHalf);
    e.set_matmul_shared_layout(MatMulSharedLayout::Padded);
    e.set_matmul_pipeline(MatMulPipeline::Auto);
    e.set_experimental_shared_memory_reuse(true);
    Ok(e)
}
fn device() -> bool {
    if let Err(e) = CudaContext::new(0) {
        assert!(
            std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1"),
            "{e:#}"
        );
        false
    } else {
        true
    }
}
fn data(phase: usize) -> HashMap<String, Vec<bf16>> {
    HashMap::from([
        (
            "a".into(),
            (0..2048 * 512)
                .map(|i| bf16::from_f32(((i + phase) % 31) as f32 / 64. - 0.2))
                .collect(),
        ),
        (
            "b".into(),
            (0..512 * 1024)
                .map(|i| bf16::from_f32(((i + phase * 3) % 19) as f32 / 32. - 0.1))
                .collect(),
        ),
    ])
}
fn compare(a: &[bf16], b: &[bf16]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        if b.is_nan() {
            assert!(a.is_nan())
        } else {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
#[test]
fn plain_bf16_output_device_toggle_and_complete_module_forced_fallback() -> Result<()> {
    if !device() {
        return Ok(());
    }
    let graph: TensorGraph<bf16> = TensorExpr::input("a", vec![2048, 512])
        .matmul(TensorExpr::input("b", vec![512, 1024]))
        .into();
    let mut e = executor()?;
    e.compile(&graph)?;
    let baseline = e.module_source().unwrap().to_owned();
    let expected = (0..2)
        .map(|p| e.execute_compiled(&graph, data(p)))
        .collect::<Result<Vec<_>>>()?;
    for enabled in [true, false, true] {
        e.set_experimental_plain_bf16_output(enabled);
        e.compile(&graph)?;
        assert_eq!(e.plain_bf16_output_counts(), (usize::from(enabled), 0));
        if !enabled {
            assert_eq!(e.module_source().unwrap(), baseline);
        }
        for (p, x) in expected.iter().enumerate() {
            compare(&e.execute_compiled(&graph, data(p))?, x);
        }
    }
    let mut rejected = executor()?;
    rejected.set_experimental_plain_bf16_output(true);
    force_rejection(|| rejected.compile(&graph))?;
    assert_eq!(rejected.plain_bf16_output_counts(), (0, 1));
    assert_eq!(rejected.module_source().unwrap(), baseline);
    for (p, x) in expected.iter().enumerate() {
        compare(&rejected.execute_compiled(&graph, data(p))?, x);
    }
    Ok(())
}

#[test]
fn plain_bf16_output_ce_forward_backward_nested_fallback_metadata_matches_body() -> Result<()> {
    use crate::tensor::{IndexTensor, Parameter};
    if !device() {
        return Ok(());
    }
    let matmul: TensorGraph<bf16> = TensorExpr::input("a", vec![2048, 512])
        .matmul(TensorExpr::input("b", vec![512, 1024]))
        .into();
    let p = Parameter::<f32>::new(vec![0.; 3 * 4096], vec![3, 4096]);
    let loss: TensorExpr<f32> = TensorExpr::from(p)
        .indexed_cross_entropy_indices(IndexTensor::input("targets", vec![3]))
        .mean_all();
    let mut ce = TensorGraph::new();
    let root = ce.add_expr(&loss, false);
    let ce = ce.with_gradients(root);
    let fwd = ce
        .graph
        .node_indices()
        .find(|&n| matches!(ce[n], TensorGraphNode::IndexedCrossEntropy { .. }))
        .unwrap();
    let bwd = ce
        .graph
        .node_indices()
        .find(|&n| matches!(ce[n], TensorGraphNode::IndexedCrossEntropyBackward { .. }))
        .unwrap();
    let combined = || -> Result<TileGraph> {
        let mut t = tiles(region()?)?;
        t.graph = TileGraph::from_graph(&ce).graph;
        Ok(t)
    };
    for reject_plain in [false, true] {
        for reject_ce in [false, true] {
            let mut e = executor()?;
            e.set_experimental_plain_bf16_output(true);
            e.set_experimental_indexed_loss_cooperative_ordered_forward(true);
            e.set_experimental_indexed_loss_cooperative_ordered_backward(true);
            e.compile(&matmul)?;
            let plan = e.execution_plan().unwrap();
            let run = || e.compile_tile_variant_with_plan(combined().unwrap(), plan);
            let run_ce = || {
                if reject_ce {
                    crate::ptx::graph::with_forced_ordered_ce_resource_rejection(run)
                } else {
                    run()
                }
            };
            let (g, _) = if reject_plain {
                force_rejection(run_ce)
            } else {
                run_ce()
            }?;
            assert_eq!(
                g.plain_bf16_output_counts(),
                if reject_plain { (0, 1) } else { (1, 0) }
            );
            assert_eq!(
                g.indexed_loss_cooperative_ordered_forward_programs(),
                usize::from(!reject_ce)
            );
            assert_eq!(
                g.indexed_loss_cooperative_ordered_backward_programs(),
                usize::from(!reject_ce)
            );
            let mut control = executor()?;
            control.set_experimental_plain_bf16_output(!reject_plain);
            control.set_experimental_indexed_loss_cooperative_ordered_forward(!reject_ce);
            control.set_experimental_indexed_loss_cooperative_ordered_backward(!reject_ce);
            control.compile(&matmul)?;
            let (expected, _) = control
                .compile_tile_variant_with_plan(combined()?, control.execution_plan().unwrap())?;
            assert_eq!(g.module_source(), expected.module_source());
            for (node, len) in [(fwd, 3), (bwd, 3 * 4096)] {
                let (a, b) = (
                    g.indexed_loss_row_launch(node, len)?.unwrap(),
                    expected.indexed_loss_row_launch(node, len)?.unwrap(),
                );
                assert_eq!(
                    (a.block_dim, a.grid_dim, a.shared_mem_bytes),
                    (b.block_dim, b.grid_dim, b.shared_mem_bytes)
                );
            }
        }
    }
    Ok(())
}

#[test]
fn plain_bf16_output_unsupported_modules_remain_byte_identical() -> Result<()> {
    if !device() {
        return Ok(());
    }
    let tail_m =
        TensorExpr::input("a", vec![2049, 512]).matmul(TensorExpr::input("b", vec![512, 1024]));
    let tail_n =
        TensorExpr::input("a", vec![2048, 512]).matmul(TensorExpr::input("b", vec![512, 1025]));
    let tail_k =
        TensorExpr::input("a", vec![2048, 515]).matmul(TensorExpr::input("b", vec![515, 1024]));
    let batched = TensorExpr::input("a", vec![2, 2048, 512])
        .matmul(TensorExpr::input("b", vec![2, 512, 1024]));
    let nt = TensorExpr::input("a", vec![2048, 512])
        .matmul(TensorExpr::input("b", vec![1024, 512]).transpose());
    let fused = TensorExpr::input("a", vec![2048, 512])
        .matmul(TensorExpr::input("b", vec![512, 1024]))
        + TensorExpr::input("bias", vec![2048, 1024]);
    for y in [tail_m, tail_n, tail_k, batched, nt, fused] {
        let graph: TensorGraph<bf16> = y.into();
        let mut e = executor()?;
        e.compile(&graph)?;
        let old = e.module_source().unwrap().to_owned();
        e.set_experimental_plain_bf16_output(true);
        e.compile(&graph)?;
        assert_eq!(e.plain_bf16_output_counts(), (0, 0));
        assert_eq!(e.module_source().unwrap(), old);
    }
    Ok(())
}
