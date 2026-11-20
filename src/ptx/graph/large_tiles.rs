//! Offline proofs for the opt-in ordinary BF16 rectangles.
use super::*;
use crate::{
    graph::TensorGraph,
    tensor::TensorExpr,
    tile::{LargeBf16TilePolicy as Policy, MatMulPipeline, MatMulPrecision, MatMulSharedLayout},
};

fn plan<G>(
    graph: &TensorGraph<half::bf16, G>,
    policy: Policy,
    reuse: bool,
    pipeline: MatMulPipeline,
) -> anyhow::Result<crate::ptx::PtxExecutionPlan> {
    let target = PtxTarget::sm80();
    let mut plan = crate::ptx::PtxExecutionPlan::build_with_reduction_mode(
        graph,
        crate::ptx::PtxReductionMode::DeterministicTree,
    )?;
    plan.schedule_matmuls_with_tile_policies(
        target.matmul_capabilities(),
        MatMulPrecision::NativeHalf,
        target.matmul_resources,
        MatMulSharedLayout::Auto,
        pipeline,
        true,
        false,
        crate::tile::MediumBf16TilePolicy::Off,
        policy,
        reuse,
    )?;
    plan.set_matmul_shared_memory_reuse(reuse);
    Ok(plan)
}

#[test]
fn large_rectangles_final_pipeline_resource_arena_and_k_order_codegen() -> anyhow::Result<()> {
    use crate::tile::Stmt;
    for (policy, expected) in [
        (Policy::Tall, (1, 0)),
        (Policy::Wide, (0, 1)),
        (Policy::TallThenWide, (1, 0)),
    ] {
        let graph: TensorGraph<half::bf16> = TensorExpr::input("a", vec![4096, 512])
            .matmul(TensorExpr::input("b", vec![512, 1536]))
            .into();
        let plan = plan(&graph, policy, true, MatMulPipeline::Auto)?;
        assert_eq!(plan.large_bf16_tile_counts(), expected);
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.add_matmul_regions(plan.matmul_regions())?;
        for kernel in &tiles.matmul_region_kernels {
            kernel.ir.validate_layouts()?;
            assert_eq!(kernel.ir.shared_mem_bytes, 32768);
            let body = &kernel.ir.body.stmts;
            let exchange = body
                .iter()
                .position(|s| matches!(s, Stmt::ConvertLayout { .. }))
                .expect("fragment exchange");
            assert!(matches!(body[exchange - 3], Stmt::AsyncWait));
            assert!(matches!(body[exchange - 2], Stmt::Barrier));
            assert!(matches!(
                body[exchange - 1],
                Stmt::ViewSharedArena {
                    dtype: DType::F32,
                    byte_offset: 0,
                    ..
                }
            ));
        }
        tiles.optimize_indices()?;
        let source = PtxGraph::from(tiles).module_source();
        assert!(source.contains(".shared .align 32 .b8 shared[32768];"));
        assert_eq!(
            source.matches("wmma.mma.sync").count(),
            8,
            "same two increasing-K16 groups/four accumulator fragments per warp"
        );
        assert_eq!(source.matches("wmma.load.a.sync").count(), 4);
        assert_eq!(source.matches("wmma.load.b.sync").count(), 4);
        let store = source
            .find("wmma.store.d.sync")
            .expect("fragment exchange store");
        let drain = source[..store]
            .rfind("cp.async.wait_group 0;")
            .expect("final drain");
        assert!(source[drain..store].contains("bar.sync 0;"));
        assert!(source.contains("cvt.rn.bf16.f32"));
        if let Some(directory) = std::env::var_os("TNSR_LARGE_TILE_PTX_DIR") {
            std::fs::create_dir_all(&directory)?;
            std::fs::write(
                std::path::PathBuf::from(directory).join(format!("{policy:?}.ptx")),
                source,
            )?;
        }
    }
    Ok(())
}

#[test]
fn large_policy_preserves_original_on_startup_sync_copy_rank_tail_and_reuse_rejection()
-> anyhow::Result<()> {
    for (m, n, k, batched, pipeline, reuse, expected) in [
        (4096, 512, 512, false, MatMulPipeline::Auto, true, (1, 0)),
        (4096, 512, 64, false, MatMulPipeline::Auto, true, (0, 0)),
        (
            4096,
            512,
            64,
            false,
            MatMulPipeline::DoubleBuffered,
            true,
            (1, 0),
        ),
        (
            4096,
            512,
            512,
            false,
            MatMulPipeline::Synchronous,
            true,
            (0, 0),
        ),
        (4096, 512, 512, false, MatMulPipeline::Auto, false, (0, 0)),
        (4097, 512, 512, false, MatMulPipeline::Auto, true, (0, 0)),
        (4096, 513, 512, false, MatMulPipeline::Auto, true, (0, 0)),
        (4096, 512, 513, false, MatMulPipeline::Auto, true, (0, 0)),
        (4096, 512, 512, true, MatMulPipeline::Auto, true, (0, 0)),
    ] {
        let a = if batched { vec![1, m, k] } else { vec![m, k] };
        let b = if batched { vec![1, k, n] } else { vec![k, n] };
        let graph: TensorGraph<half::bf16> = TensorExpr::input("a", a)
            .matmul(TensorExpr::input("b", b))
            .into();
        let baseline = plan(&graph, Policy::Off, reuse, pipeline)?;
        let selected = plan(&graph, Policy::Tall, reuse, pipeline)?;
        assert_eq!(selected.large_bf16_tile_counts(), expected);
        if expected == (0, 0) {
            assert_eq!(baseline.matmul_schedules(), selected.matmul_schedules());
            assert_eq!(
                baseline
                    .matmul_regions()
                    .iter()
                    .map(|r| r.schedule)
                    .collect::<Vec<_>>(),
                selected
                    .matmul_regions()
                    .iter()
                    .map(|r| r.schedule)
                    .collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

#[test]
fn rectangle_shared_arena_zero_k_speculation_drains_before_exchange() -> anyhow::Result<()> {
    use crate::{
        graph::TensorGraphNode,
        tile::{MatMulSchedule, MatMulTile},
    };
    for policy in [Policy::Tall, Policy::Wide] {
        let target = PtxTarget::sm80();
        let base = MatMulSchedule::select_with_resources(
            4096,
            512,
            128,
            DType::BF16,
            target.matmul_capabilities(),
            MatMulPrecision::NativeHalf,
            target.matmul_resources,
        );
        let mut schedule = policy.candidates(base, &[4096, 512], true)[0]
            .with_shared_layout(MatMulSharedLayout::Padded, target.matmul_resources)?
            .with_pipeline(
                MatMulPipeline::DoubleBuffered,
                true,
                true,
                target.matmul_resources,
            );
        assert_eq!(schedule.pipeline_stages, 2);
        schedule.logical_shape = MatMulTile {
            m: 4096,
            n: 512,
            k: 0,
        };
        let mut graph = TensorGraph::<half::bf16>::new();
        let a = graph.graph.add_node(TensorGraphNode::Input {
            name: "a",
            shape: vec![4096, 0],
        });
        let b = graph.graph.add_node(TensorGraphNode::Input {
            name: "b",
            shape: vec![0, 512],
        });
        let out = graph.graph.add_node(TensorGraphNode::MatMul {
            shape: vec![4096, 512],
        });
        graph.graph.add_edge(a, out, 0);
        graph.graph.add_edge(b, out, 1);
        let mut tiles =
            TileGraph::from_with_matmul_schedules(&graph, true, &HashMap::from([(out, schedule)]));
        tiles.set_physical_nodes(HashSet::from([out]));
        tiles.optimize_indices()?;
        let source = PtxGraph::from(tiles).module_source();
        assert!(source.contains(".shared .align 32 .b8 shared[32768];"));
        let exchange = source.find("wmma.store.d.sync").unwrap();
        let drain = source[..exchange].rfind("cp.async.wait_group 0;").unwrap();
        assert!(source[drain..exchange].contains("bar.sync 0;"));
    }
    Ok(())
}
