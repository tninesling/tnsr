//! Device-free structural proofs. Optional artifacts are assembled externally.
use super::*;
use crate::{
    graph::{TensorGraph, TensorGraphNode},
    tile::{
        MatMulPipeline, MatMulPrecision, MatMulRegion, MatMulSchedule, MatMulSharedLayout,
        VirtualTensor,
    },
};

fn target() -> PtxTarget {
    let mut target = PtxTarget::sm80();
    target.compute_capability = (8, 9);
    target.matmul_resources.shared_bytes_per_sm = 100 * 1024;
    target.matmul_resources.threads_per_sm = 1536;
    target
}
fn region(orientation: (bool, bool)) -> anyhow::Result<MatMulRegion> {
    let graph: TensorGraph<half::bf16> = crate::tensor::TensorExpr::input("a", vec![2048, 128])
        .matmul(crate::tensor::TensorExpr::input("b", vec![128, 1024]))
        .into();
    let out = graph
        .graph
        .node_indices()
        .find(|&n| matches!(graph[n], TensorGraphNode::MatMul { .. }))
        .unwrap();
    let mut region = MatMulRegion::from_graph(&graph, out, vec![], vec![out])?;
    for (input, transposed) in region.inputs.iter_mut().zip([orientation.0, orientation.1]) {
        if transposed {
            let mut shape = input.source_shape.clone();
            shape.swap(0, 1);
            input.tensor = VirtualTensor::identity(input.tensor.source, shape.clone(), DType::BF16)
                .permute(&[1, 0])?;
            input.source_shape = shape;
        }
    }
    let t = target();
    region.schedule = MatMulSchedule::select_with_resources(
        2048,
        1024,
        128,
        DType::BF16,
        t.matmul_capabilities(),
        MatMulPrecision::NativeHalf,
        t.matmul_resources,
    )
    .with_shared_memory_reuse(true);
    region.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::DoubleBuffered,
        true,
        t.matmul_resources,
    )?;
    Ok(region)
}
fn source(region: MatMulRegion) -> anyhow::Result<String> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(&[region])?;
    tiles.optimize_indices()?;
    Ok(PtxGraph::from(tiles).with_target(target()).module_source())
}
#[test]
fn square128_streamed_four_windows_drain_and_fragment_load_order() -> anyhow::Result<()> {
    for orientation in [(false, false), (false, true), (true, false), (true, true)] {
        let mut region = region(orientation)?;
        let original = region.clone();
        region.schedule = crate::tile::large_tile_policy::square128_streamed::region_candidate(
            &region,
            MatMulSharedLayout::Padded,
            MatMulPipeline::DoubleBuffered,
            true,
            target().matmul_resources,
        )
        .expect("proved contiguous operands");
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.add_matmul_regions(&[region.clone()])?;
        let ir = &tiles.matmul_region_kernels[0].ir;
        ir.validate_layouts()?;
        let body = &ir.body.stmts;
        let windows = body
            .iter()
            .filter_map(|stmt| {
                if let Stmt::StoreAccumulatorWindow {
                    fragment_origin, ..
                } = stmt
                {
                    Some(*fragment_origin)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(windows, [(0, 0), (0, 2), (1, 0), (1, 2)]);
        let output_loops = body
            .iter()
            .filter_map(|stmt| match stmt {
                Stmt::ForLoop {
                    loop_var,
                    start,
                    end,
                    body,
                    ..
                } if loop_var.starts_with("square_output_round_") => Some((start, end, body)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(output_loops.len(), 4);
        for (start, end, body) in output_loops {
            assert_eq!(*start, Expr::Const(0));
            assert_eq!(*end, Expr::Const(16));
            assert_eq!(
                body.stmts
                    .iter()
                    .filter(|s| matches!(s, Stmt::StoreGlobalPredicated { .. }))
                    .count(),
                1
            );
            assert!(!body.stmts.iter().any(|s| matches!(
                s,
                Stmt::Barrier | Stmt::AsyncWait | Stmt::StoreAccumulatorWindow { .. }
            )));
        }
        let view = body
            .iter()
            .position(|s| {
                matches!(
                    s,
                    Stmt::ViewSharedArena {
                        dtype: DType::F32,
                        ..
                    }
                )
            })
            .unwrap();
        assert!(matches!(body[view - 2], Stmt::AsyncWait));
        assert!(matches!(body[view - 1], Stmt::Barrier));
        assert_eq!(ir.shared_mem_bytes, region.schedule.shared_mem_bytes());
        assert!(ir.shared_mem_bytes <= 40 * 1024);
        let baseline = source(original)?;
        let candidate = source(region)?;
        assert_eq!(baseline.matches("wmma.mma.sync").count(), 8);
        assert_eq!(candidate.matches("wmma.mma.sync").count(), 16);
        assert_eq!(candidate.matches("wmma.load.a.sync").count(), 4);
        assert_eq!(candidate.matches("wmma.load.b.sync").count(), 8);
        assert_eq!(candidate.matches("wmma.store.d.sync").count(), 8);
        assert!(candidate.contains("cvt.rn.bf16.f32"));
        assert_eq!(candidate.matches("cvt.rn.bf16.f32").count(), 4);
        for pass in 0..4 {
            assert!(candidate.contains(&format!("loop_body_square_output_round_{pass}")));
        }
        let start = candidate.find("wmma.load.a.sync").unwrap();
        let mut pos = start;
        for _ in 0..2 {
            for _ in 0..2 {
                pos +=
                    candidate[pos..].find("wmma.load.a.sync").unwrap() + "wmma.load.a.sync".len();
            }
            for _ in 0..4 {
                pos +=
                    candidate[pos..].find("wmma.load.b.sync").unwrap() + "wmma.load.b.sync".len();
                for _ in 0..2 {
                    pos += candidate[pos..].find("wmma.mma.sync").unwrap() + "wmma.mma.sync".len();
                }
            }
        }
        if let Some(dir) = std::env::var_os("TNSR_SQUARE128_PTX_DIR") {
            std::fs::create_dir_all(&dir)?;
            for (name, source) in [("baseline", baseline), ("candidate", candidate)] {
                std::fs::write(
                    std::path::PathBuf::from(&dir)
                        .join(format!("{name}-{}-{}.ptx", orientation.0, orientation.1)),
                    source,
                )?;
            }
        }
    }
    Ok(())
}
#[test]
fn square128_streamed_rejects_arithmetic_and_unproved_maps_before_lowering() -> anyhow::Result<()> {
    let mut r = region((false, false))?;
    let resources = target().matmul_resources;
    assert!(
        crate::tile::large_tile_policy::square128_streamed::region_candidate(
            &r,
            MatMulSharedLayout::Padded,
            MatMulPipeline::Synchronous,
            true,
            resources
        )
        .is_none()
    );
    r.inputs[0].source_shape[0] -= 1;
    assert!(
        crate::tile::large_tile_policy::square128_streamed::region_candidate(
            &r,
            MatMulSharedLayout::Padded,
            MatMulPipeline::DoubleBuffered,
            true,
            resources
        )
        .is_none()
    );
    let mut r = region((false, false))?;
    r.outputs[0].value = crate::tile::RegionValue(usize::MAX);
    assert!(
        crate::tile::large_tile_policy::square128_streamed::region_candidate(
            &r,
            MatMulSharedLayout::Padded,
            MatMulPipeline::DoubleBuffered,
            true,
            resources
        )
        .is_none()
    );
    let mut r = region((false, false))?;
    r.epilogue_operations.push(crate::tile::RegionOp {
        node: r.outputs[0].node,
        output: crate::tile::RegionValue(usize::MAX),
        kind: crate::tile::RegionOpKind::Binary {
            op: crate::tensor::BinaryOp::Add,
            lhs: r.matmul_value,
            rhs: r.matmul_value,
        },
    });
    assert!(
        crate::tile::large_tile_policy::square128_streamed::region_candidate(
            &r,
            MatMulSharedLayout::Padded,
            MatMulPipeline::DoubleBuffered,
            true,
            resources
        )
        .is_none()
    );
    Ok(())
}

#[test]
fn square128_streamed_zero_k_speculation_drains_before_window_alias() -> anyhow::Result<()> {
    let r = region((false, false))?;
    let mut schedule = crate::tile::large_tile_policy::square128_streamed::region_candidate(
        &r,
        MatMulSharedLayout::Padded,
        MatMulPipeline::DoubleBuffered,
        true,
        target().matmul_resources,
    )
    .unwrap();
    schedule.logical_shape.k = 0;
    // Production eligibility rejects zero K. Explicit lowering still preserves
    // the final async drain if such a schedule is used as an IR fixture.
    let mut graph = TensorGraph::<half::bf16>::new();
    let a = graph.graph.add_node(TensorGraphNode::Input {
        name: "a",
        shape: vec![2048, 0],
    });
    let b = graph.graph.add_node(TensorGraphNode::Input {
        name: "b",
        shape: vec![0, 1024],
    });
    let out = graph.graph.add_node(TensorGraphNode::MatMul {
        shape: vec![2048, 1024],
    });
    graph.graph.add_edge(a, out, 0);
    graph.graph.add_edge(b, out, 1);
    let mut tiles =
        TileGraph::from_with_matmul_schedules(&graph, true, &HashMap::from([(out, schedule)]));
    tiles.set_physical_nodes(HashSet::from([out]));
    tiles.optimize_indices()?;
    let source = PtxGraph::from(tiles).with_target(target()).module_source();
    let window = source.find("wmma.store.d.sync").unwrap();
    let wait = source[..window].rfind("cp.async.wait_group 0;").unwrap();
    assert!(source[wait..window].contains("bar.sync 0;"));
    if let Some(dir) = std::env::var_os("TNSR_SQUARE128_PTX_DIR") {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            std::path::PathBuf::from(dir).join("candidate-zero-k.ptx"),
            source,
        )?;
    }
    Ok(())
}

#[test]
fn square128_streamed_final_plan_admission_is_actual_and_bounded() -> anyhow::Result<()> {
    for (m, n, k, pipeline, reuse, batched, expected) in [
        (
            2048,
            1024,
            128,
            MatMulPipeline::DoubleBuffered,
            true,
            false,
            1,
        ),
        (2048, 1024, 128, MatMulPipeline::Synchronous, true, false, 0),
        (
            2048,
            1024,
            128,
            MatMulPipeline::DoubleBuffered,
            false,
            false,
            0,
        ),
        (
            2048,
            1024,
            128,
            MatMulPipeline::DoubleBuffered,
            true,
            true,
            0,
        ),
        (
            2049,
            1024,
            128,
            MatMulPipeline::DoubleBuffered,
            true,
            false,
            0,
        ),
        (
            2048,
            1025,
            128,
            MatMulPipeline::DoubleBuffered,
            true,
            false,
            0,
        ),
        (
            2048,
            1024,
            129,
            MatMulPipeline::DoubleBuffered,
            true,
            false,
            0,
        ),
        (
            2048,
            512,
            128,
            MatMulPipeline::DoubleBuffered,
            true,
            false,
            0,
        ),
    ] {
        let a = if batched { vec![1, m, k] } else { vec![m, k] };
        let b = if batched { vec![1, k, n] } else { vec![k, n] };
        let graph: TensorGraph<half::bf16> = crate::tensor::TensorExpr::input("a", a)
            .matmul(crate::tensor::TensorExpr::input("b", b))
            .into();
        let t = target();
        let mut p = crate::ptx::PtxExecutionPlan::build_with_reduction_mode(
            &graph,
            crate::ptx::PtxReductionMode::DeterministicTree,
        )?;
        p.schedule_matmuls_with_tile_policies(
            t.matmul_capabilities(),
            MatMulPrecision::NativeHalf,
            t.matmul_resources,
            MatMulSharedLayout::Padded,
            pipeline,
            true,
            false,
            crate::tile::MediumBf16TilePolicy::Off,
            crate::tile::LargeBf16TilePolicy::Square128Streamed,
            reuse,
        )?;
        p.set_matmul_shared_memory_reuse(reuse);
        let baseline = p.clone();
        assert_eq!(p.large_square128_streamed_count(), 0);
        p.configure_square128_streamed(
            MatMulSharedLayout::Padded,
            pipeline,
            true,
            t.matmul_resources,
        );
        assert_eq!(
            p.large_square128_streamed_count(),
            expected,
            "{m}x{n}x{k}, batched={batched}, reuse={reuse}"
        );
        assert_eq!(p.large_bf16_tile_counts(), (0, 0));
        if expected == 0 {
            assert_eq!(p.matmul_schedules(), baseline.matmul_schedules());
            assert_eq!(p.matmul_regions(), baseline.matmul_regions());
        }
    }
    Ok(())
}
