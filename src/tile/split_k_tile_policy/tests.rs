//! Policy eligibility, arithmetic-contract and partial-codegen tests.
use super::*;
use crate::{
    graph::{TensorGraph, TensorGraphNode},
    tile::{
        CompareOp, IndexExpr, IndexPredicate, MatMulCapabilities, MatMulPipeline, MatMulPrecision,
        MatMulSharedLayout, ScalarValue,
    },
};
fn resources() -> MatMulResources {
    MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..MatMulResources::default()
    }
}
fn fixture(m: usize, n: usize, k: usize) -> MatMulRegion {
    let mut graph = TensorGraph::<half::bf16>::new();
    let a = graph.graph.add_node(TensorGraphNode::Input {
        name: "a",
        shape: vec![4, m, k],
    });
    let b = graph.graph.add_node(TensorGraphNode::Input {
        name: "b",
        shape: vec![4, k, n],
    });
    let out = graph.graph.add_node(TensorGraphNode::MatMul {
        shape: vec![4, m, n],
    });
    graph.graph.add_edge(a, out, 0);
    graph.graph.add_edge(b, out, 1);
    let mut region = MatMulRegion::from_graph(&graph, out, vec![], vec![out]).unwrap();
    region.schedule = MatMulSchedule::select_with_resources(
        m,
        n,
        k,
        DType::BF16,
        MatMulCapabilities {
            tf32: true,
            f16: true,
            bf16: true,
        },
        MatMulPrecision::NativeHalf,
        resources(),
    );
    region.schedule.block_tile.k = 64;
    let total = 4 * k;
    region.inputs[0].source_shape = vec![total, m];
    region.inputs[0].tensor = VirtualTensor::identity(a, vec![total, m], DType::BF16)
        .permute(&[1, 0])
        .unwrap()
        .reshape(vec![m, 4, k])
        .unwrap()
        .permute(&[1, 0, 2])
        .unwrap();
    region.inputs[1].source_shape = vec![total, n];
    region.inputs[1].tensor = VirtualTensor::identity(b, vec![total, n], DType::BF16)
        .reshape(vec![4, k, n])
        .unwrap();
    // Reproduce the actual old Padded Acol/Brow two-stage family without CUDA.
    region.schedule = region
        .schedule
        .with_operand_orientations(MatMulSharedLayout::Padded, (true, false), resources())
        .unwrap()
        .with_operand_pipeline(MatMulPipeline::Auto, true, true, (true, true), resources());
    region
}

#[test]
fn square64_scope_keeps_existing64_cases_partitions_and_global_buffers_untouched() {
    for (m, n, k) in [
        (256, 512, 512),
        (512, 1408, 4096),
        (512, 1536, 4096),
        (1408, 512, 4096),
        (512, 4096, 4096),
    ] {
        let original = fixture(m, n, k);
        let saved = original.clone();
        assert_eq!(original.schedule.shared_mem_bytes(), 24576);
        assert!(
            SplitKBf16TilePolicy::Off
                .candidate(&original, 4, 4 * k, resources(), true)
                .is_none()
        );
        let candidate = SplitKBf16TilePolicy::Square64
            .candidate(&original, 4, 4 * k, resources(), true)
            .unwrap();
        assert_eq!(candidate.baseline, saved);
        assert_eq!(
            candidate.schedule.block_tile,
            MatMulTile {
                m: 64,
                n: 64,
                k: 32
            }
        );
        assert_eq!(candidate.schedule.warp_topology, (2, 2));
        assert_eq!(candidate.schedule.warp_fragments, (2, 2));
        assert_eq!(candidate.schedule.block_threads, (32, 8, 1));
        assert!(candidate.schedule.reuse_shared_memory);
        assert_eq!(
            original, saved,
            "candidate cannot mutate source maps, packing or baseline schedule"
        );
        for (parts, total, reuse) in [
            (2, 4 * k, true),
            (8, 4 * k, true),
            (4, 4 * k + 64, true),
            (4, 4 * k, false),
        ] {
            assert!(
                SplitKBf16TilePolicy::Square64
                    .candidate(&original, parts, total, resources(), reuse)
                    .is_none()
            );
        }
        let mut already64 = original.clone();
        already64.schedule.block_tile = MatMulTile {
            m: 64,
            n: 64,
            k: 32,
        };
        already64.schedule.warp_fragments = (2, 2);
        assert!(
            SplitKBf16TilePolicy::Square64
                .candidate(&already64, 4, 4 * k, resources(), true)
                .is_none()
        );
    }
    assert_ne!(
        SplitKBf16TilePolicy::Off.codegen_tag(),
        SplitKBf16TilePolicy::Square64.codegen_tag()
    );
}

#[test]
fn square64_rejects_whole_map_offsets_broadcast_predicate_tail_dtype_and_epilogue_changes() {
    let original = fixture(512, 1408, 4096);
    for change in 0..10 {
        let mut r = original.clone();
        match change {
            0 => r.inputs[0].tensor.access.results[0] = IndexExpr::Const(0),
            1 => {
                r.inputs[0].tensor.access.results[0] = IndexExpr::Add(
                    Box::new(r.inputs[0].tensor.access.results[0].clone()),
                    Box::new(IndexExpr::Const(1)),
                )
            }
            2 => {
                r.inputs[0].tensor.predicate = Some(IndexPredicate::Compare {
                    op: CompareOp::Lt,
                    lhs: IndexExpr::IterDim(0),
                    rhs: IndexExpr::Const(4),
                })
            }
            3 => r.inputs[0].tensor.out_of_bounds = Some(ScalarValue::F32(0)),
            4 => r.inputs[1].source_shape[0] += 1,
            5 => r.schedule.storage_dtype = DType::F32,
            6 => r.schedule.pipeline_stages = 1,
            7 => r.schedule.copy_bytes = 4,
            8 => r.schedule.async_operands = (true, false),
            9 => r.schedule.reuse_shared_memory = true,
            _ => unreachable!(),
        }
        assert!(
            SplitKBf16TilePolicy::Square64
                .candidate(&r, 4, 16384, resources(), true)
                .is_none(),
            "change{change}"
        );
    }
    for (m, n, k) in [
        (513, 1408, 4096),
        (512, 1409, 4096),
        (512, 1408, 4097),
        (256, 256, 512),
    ] {
        assert!(
            SplitKBf16TilePolicy::Square64
                .candidate(&fixture(m, n, k), 4, 4 * k, resources(), true)
                .is_none()
        );
    }
}

#[test]
fn square64_whole_batched_source_shapes_and_exact_increasing_k16_sequence() {
    let mut r = fixture(512, 1408, 4096);
    for index in 0..2 {
        let source = r.inputs[index].tensor.source;
        let width = if index == 0 { 512 } else { 1408 };
        let shape = vec![32, 512, width];
        r.inputs[index].source_shape = shape.clone();
        let base = VirtualTensor::identity(source, shape, DType::BF16);
        r.inputs[index].tensor = if index == 0 {
            base.reshape(vec![16384, 512])
                .unwrap()
                .permute(&[1, 0])
                .unwrap()
                .reshape(vec![512, 4, 4096])
                .unwrap()
                .permute(&[1, 0, 2])
                .unwrap()
        } else {
            base.reshape(vec![4, 4096, 1408]).unwrap()
        };
    }
    assert!(
        SplitKBf16TilePolicy::Square64
            .candidate(&r, 4, 16384, resources(), true)
            .is_some()
    );
    for part_k in [64usize, 128, 512, 4096] {
        let old: Vec<_> = (0..part_k / 64)
            .flat_map(|tile| (0..4).map(move |step| tile * 64 + step * 16))
            .collect();
        let new: Vec<_> = (0..part_k / 32)
            .flat_map(|tile| (0..2).map(move |step| tile * 32 + step * 16))
            .collect();
        assert_eq!(old, new);
        for part in 0..4 {
            assert_eq!(
                old.iter().map(|&k| part * part_k + k).collect::<Vec<_>>(),
                new.iter().map(|&k| part * part_k + k).collect::<Vec<_>>()
            );
        }
    }
    let partial = [1e20f32, 1., -1e20, 1.];
    let left = ((partial[0] + partial[1]) + partial[2]) + partial[3];
    let balanced = (partial[0] + partial[1]) + (partial[2] + partial[3]);
    assert_eq!(left, 1.);
    assert_eq!(balanced, 0., "balanced sum is a different experiment");
    let partial = [-0.0f32; 4];
    let left = ((partial[0] + partial[1]) + partial[2]) + partial[3];
    assert_eq!(
        half::bf16::from_f32(left).to_bits(),
        0x8000,
        "no initial+0 is allowed"
    );
    let partial = [1. + 1. / 256., 0.0001, 0., 0.];
    let exact = half::bf16::from_f32(((partial[0] + partial[1]) + partial[2]) + partial[3]);
    let rounded = partial.map(|v| half::bf16::from_f32(v).to_f32());
    assert_ne!(
        exact.to_bits(),
        half::bf16::from_f32(((rounded[0] + rounded[1]) + rounded[2]) + rounded[3]).to_bits(),
        "partial storage must remainF32"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn square64_final_resource_proof_f32_partial_output_and_full_fallback() {
    use crate::tile::{Stmt, TileGraph};
    let r = fixture(512, 1408, 4096);
    let old = r.clone();
    let c = SplitKBf16TilePolicy::Square64
        .candidate(&r, 4, 16384, resources(), true)
        .unwrap();
    let (schedule, ir) = c.partial_ir(&r, resources(), true).unwrap();
    assert_eq!(
        schedule.block_tile,
        MatMulTile {
            m: 64,
            n: 64,
            k: 32
        }
    );
    assert_eq!(schedule.shared_mem_bytes(), 18432);
    assert_eq!(schedule.pipeline_stages, 2);
    assert_eq!(schedule.copy_bytes, 16);
    assert_eq!(schedule.async_operands, (true, true));
    assert_eq!(ir.shared_mem_bytes, 18432);
    assert_eq!(
        ir.params.iter().map(|p| p.dtype).collect::<Vec<_>>(),
        [DType::BF16, DType::BF16, DType::F32]
    );
    let body = &ir.body.stmts;
    let exchange = body
        .iter()
        .position(|s| matches!(s, Stmt::ConvertLayout { .. }))
        .unwrap();
    assert!(matches!(body[exchange - 3], Stmt::AsyncWait));
    assert!(matches!(body[exchange - 2], Stmt::Barrier));
    assert!(c.configure(&r, resources(), false).is_none());
    for budget in [
        MatMulResources {
            shared_bytes_per_block: 18431,
            ..resources()
        },
        MatMulResources {
            shared_bytes_per_sm: 36863,
            ..resources()
        },
        MatMulResources {
            registers_per_sm: 65535,
            ..resources()
        },
        MatMulResources {
            threads_per_sm: 511,
            ..resources()
        },
    ] {
        assert!(c.configure(&r, budget, true).is_none());
    }
    assert_eq!(
        r, old,
        "rejected proof cannot install a synchronous half-candidate"
    );
    let mut changed = r.clone();
    changed.inputs[0].tensor.source = crate::graph::NodeIndex::new(999);
    assert!(
        c.configure(&changed, resources(), true).is_none(),
        "stale source descriptor"
    );
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
    tiles.optimize_indices().unwrap();
    let source = crate::ptx::PtxGraph::from(tiles).module_source();
    assert!(source.contains("st.global.f32"));
    assert!(!source.contains("st.global.b16"));
}

#[cfg(feature = "cuda")]
mod gpu_tests;

#[cfg(feature = "cuda")]
mod rectangle128;
