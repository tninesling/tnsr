//! Offline qualification of the ordinary GEMM lowering, with optional PTX
//! artifacts for an external assembler. No CUDA context or GPU is required.
use std::path::PathBuf;

use anyhow::Result;
use half::{bf16, f16};

use super::{PtxGraph, PtxTarget};
use crate::{
    graph::{TensorGraph, TensorGraphNode},
    tile::{
        MatMulCapabilities, MatMulPipeline, MatMulPrecision, MatMulRegion, MatMulResources,
        MatMulSchedule, MatMulSharedLayout, TileDType, TileGraph, VirtualTensor,
    },
};

fn region<D: TileDType>(
    m: usize,
    k: usize,
    n: usize,
    transposed: (bool, bool),
) -> Result<MatMulRegion> {
    let mut graph = TensorGraph::<D>::new();
    let a = graph.graph.add_node(TensorGraphNode::Input {
        name: "a",
        shape: vec![2, m, k],
    });
    let b = graph.graph.add_node(TensorGraphNode::Input {
        name: "b",
        shape: vec![1, k, n],
    });
    let out = graph.graph.add_node(TensorGraphNode::MatMul {
        shape: vec![2, m, n],
    });
    graph.graph.add_edge(a, out, 0);
    graph.graph.add_edge(b, out, 1);
    let mut region = MatMulRegion::from_graph(&graph, out, vec![], vec![out])?;
    for (input, (source, batch, rows, cols, transpose)) in region
        .inputs
        .iter_mut()
        .zip([(a, 2, m, k, transposed.0), (b, 1, k, n, transposed.1)])
    {
        if transpose {
            let shape = vec![batch, cols, rows];
            input.tensor = VirtualTensor::identity(source, shape.clone(), D::TILE_DTYPE)
                .permute(&[0, 2, 1])?;
            input.source_shape = shape;
        }
    }
    Ok(region)
}

fn source(region: MatMulRegion) -> Result<String> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(&[region])?;
    tiles.optimize_indices()?;
    let mut target = PtxTarget::sm80();
    target.compute_capability = (8, 9);
    Ok(PtxGraph::from(tiles).with_target(target).module_source())
}

fn zero_k_source<D: TileDType>(m: usize, n: usize, schedule: MatMulSchedule) -> Result<String> {
    // Zero K deliberately stays outside MatMulRegion. Its ordinary standalone
    // lowering still has a scalar output and shared operands, but no live loads.
    let mut graph = TensorGraph::<D>::new();
    let a = graph.graph.add_node(TensorGraphNode::Input {
        name: "a",
        shape: vec![2, m, 0],
    });
    let b = graph.graph.add_node(TensorGraphNode::Input {
        name: "b",
        shape: vec![1, 0, n],
    });
    let out = graph.graph.add_node(TensorGraphNode::MatMul {
        shape: vec![2, m, n],
    });
    graph.graph.add_edge(a, out, 0);
    graph.graph.add_edge(b, out, 1);
    let schedules = std::collections::HashMap::from([(out, schedule)]);
    let mut tiles = TileGraph::from_with_matmul_schedules(&graph, true, &schedules);
    tiles.set_physical_nodes(std::collections::HashSet::from([out]));
    tiles.optimize_indices()?;
    let mut target = PtxTarget::sm80();
    target.compute_capability = (8, 9);
    Ok(PtxGraph::from(tiles).with_target(target).module_source())
}

fn qualify<D: TileDType>(format: &str, precision: MatMulPrecision) -> Result<()> {
    let resources = MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        ..MatMulResources::default()
    };
    let caps = MatMulCapabilities {
        tf32: true,
        f16: true,
        bf16: true,
    };
    let directory = std::env::var_os("TNSR_PAPER_PTX_DIR").map(PathBuf::from);
    if let Some(directory) = &directory {
        std::fs::create_dir_all(directory)?;
    }
    let mut manifest = Vec::new();
    // Aligned BF16 selects the current large family; tails and batch broadcasts
    // exercise the existing smaller/native families and scalar staging paths.
    for (label, m, k, n) in [
        ("aligned", 2048, 128, 256),
        ("tails", 257, 65, 259),
        ("zero-k", 33, 0, 35),
    ] {
        for transposed in [(false, false), (true, false), (false, true), (true, true)] {
            for pipeline in [MatMulPipeline::Synchronous, MatMulPipeline::DoubleBuffered] {
                let selected = MatMulSchedule::select_with_resources(
                    m,
                    n,
                    k,
                    D::TILE_DTYPE,
                    caps,
                    precision,
                    resources,
                );
                let (baseline_schedule, baseline, candidate_schedule, candidate) = if k == 0 {
                    let baseline = selected
                        .with_shared_layout(MatMulSharedLayout::Padded, resources)?
                        .with_pipeline(pipeline, true, true, resources);
                    let candidate = baseline.with_shared_memory_reuse(true);
                    (
                        baseline,
                        zero_k_source::<D>(m, n, baseline)?,
                        candidate,
                        zero_k_source::<D>(m, n, candidate)?,
                    )
                } else {
                    let mut region = region::<D>(m, k, n, transposed)?;
                    region.schedule = selected;
                    region.configure_pipeline(
                        MatMulSharedLayout::Padded,
                        pipeline,
                        true,
                        resources,
                    )?;
                    let baseline = region.schedule;
                    let baseline_source = source(region.clone())?;
                    region.schedule = baseline.with_shared_memory_reuse(true);
                    let candidate = region.schedule;
                    (baseline, baseline_source, candidate, source(region)?)
                };
                for opcode in [
                    "wmma.mma.sync",
                    "wmma.load.a.sync",
                    "wmma.load.b.sync",
                    "wmma.store.d.sync",
                    "cp.async",
                    "bar.sync",
                    "ld.global",
                    "st.global",
                ] {
                    assert_eq!(
                        baseline.matches(opcode).count(),
                        candidate.matches(opcode).count(),
                        "{format}/{label}/{transposed:?}/{pipeline:?}: {opcode}"
                    );
                }
                for (reuse, schedule, ptx) in [
                    (false, baseline_schedule, &baseline),
                    (true, candidate_schedule, &candidate),
                ] {
                    assert!(ptx.contains(&format!(
                        ".shared .align 32 .b8 shared[{}];",
                        schedule.shared_mem_bytes()
                    )));
                    if let Some(directory) = &directory {
                        let filename = format!(
                            "arena-{format}-{label}-a{}b{}-{pipeline:?}-reuse{reuse}.ptx",
                            u8::from(transposed.0),
                            u8::from(transposed.1)
                        );
                        std::fs::write(directory.join(&filename), ptx)?;
                        manifest.push(serde_json::json!({
                            "file": filename,
                            "shape_mnk": [m, n, k],
                            "reuse": schedule.reuse_shared_memory,
                            "shared_bytes": schedule.shared_mem_bytes(),
                            "pipeline_stages": schedule.pipeline_stages,
                            "async_operands": [schedule.async_operands.0, schedule.async_operands.1],
                            "block_tile": [schedule.block_tile.m, schedule.block_tile.n, schedule.block_tile.k],
                        }));
                    }
                }
                if k == 0 {
                    assert_eq!(baseline_schedule.plan, crate::tile::MatMulPlan::ScalarF32);
                    assert_eq!(baseline, candidate);
                } else {
                    assert!(candidate_schedule.reuse_shared_memory);
                    assert!(
                        candidate_schedule.shared_mem_bytes()
                            < baseline_schedule.shared_mem_bytes()
                    );
                    assert_eq!(
                        candidate_schedule.with_shared_memory_reuse(false),
                        baseline_schedule
                    );
                    let store = candidate.find("wmma.store.d.sync").unwrap();
                    let loop_end = candidate.rfind("loop_end_k_tile_").unwrap();
                    assert!(store > loop_end);
                    if candidate_schedule.pipeline_stages == 2 {
                        let drain = candidate[..store].rfind("cp.async.wait_group 0;").unwrap();
                        assert!(drain > loop_end);
                        assert!(candidate[drain..store].contains("bar.sync"));
                    }
                }
            }
        }
    }
    if let Some(directory) = directory {
        std::fs::write(
            directory.join(format!("arena-{format}-manifest.json")),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
    }
    Ok(())
}

#[test]
fn shared_memory_reuse_emits_matching_gemm_operations_without_a_device() -> Result<()> {
    qualify::<bf16>("bf16", MatMulPrecision::NativeHalf)?;
    qualify::<f16>("f16", MatMulPrecision::NativeHalf)?;
    qualify::<f32>("tf32x3", MatMulPrecision::Tf32x3)?;
    // Keep an ordinary single-TF32 conversion path covered too.
    qualify::<f32>("tf32", MatMulPrecision::AllowTf32)?;
    Ok(())
}
