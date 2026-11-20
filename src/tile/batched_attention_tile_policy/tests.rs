use super::*;
use crate::graph::{TensorGraph, TensorGraphNode};
use crate::tile::{MatMulCapabilities, MatMulPrecision};

fn resources() -> MatMulResources {
    MatMulResources {
        shared_bytes_per_block: 48 * 1024,
        shared_bytes_per_sm: 100 * 1024,
        registers_per_sm: 64 * 1024,
        threads_per_sm: 1536,
    }
}
fn fixture(batch: usize, heads: usize, m: usize, n: usize, k: usize) -> MatMulRegion {
    let mut graph = TensorGraph::<half::bf16>::new();
    let a = graph.graph.add_node(TensorGraphNode::Input {
        name: "a",
        shape: vec![batch, heads, m, k],
    });
    let b = graph.graph.add_node(TensorGraphNode::Input {
        name: "b",
        shape: vec![batch, heads, k, n],
    });
    let output = graph.graph.add_node(TensorGraphNode::MatMul {
        shape: vec![batch, heads, m, n],
    });
    graph.graph.add_edge(a, output, 0);
    graph.graph.add_edge(b, output, 1);
    let mut region = MatMulRegion::from_graph(&graph, output, vec![], vec![output]).unwrap();
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
    region
}
fn model_views(region: &mut MatMulRegion) {
    let (batch, heads) = (region.batch_shape[0], region.batch_shape[1]);
    for input in &mut region.inputs {
        let matrix = &input.tensor.shape[2..];
        let axes = match matrix {
            [512, 64] => [0, 2, 1, 3],
            [64, 512] => [0, 2, 3, 1],
            _ => continue,
        };
        input.source_shape = vec![batch * 512, heads * 64];
        input.tensor = VirtualTensor::identity(input.node, input.source_shape.clone(), DType::BF16)
            .reshape(vec![batch, 512, heads, 64])
            .unwrap()
            .permute(&axes)
            .unwrap();
    }
}
#[test]
fn batched_attention_roles_full_grid_and_k16_rounding_contract() {
    assert_eq!(
        BatchedAttentionBf16TilePolicy::default(),
        BatchedAttentionBf16TilePolicy::Off
    );
    for (m, n, k, role) in [
        (512, 512, 64, AttentionTileRole::Scores),
        (512, 64, 512, AttentionTileRole::QueryWidth),
        (64, 512, 512, AttentionTileRole::WidthQuery),
    ] {
        for (batch, heads) in [(8, 8), (16, 8), (32, 8)] {
            let mut region = fixture(batch, heads, m, n, k);
            model_views(&mut region);
            for policy in [
                BatchedAttentionBf16TilePolicy::TallScores,
                BatchedAttentionBf16TilePolicy::WideScores,
            ] {
                let c = policy.candidate(&region, resources(), true).unwrap();
                assert_eq!(c.role, role);
                assert_eq!(c.schedule.logical_shape, region.schedule.logical_shape);
                assert_eq!(
                    c.schedule.instruction_tile,
                    MatMulTile {
                        m: 16,
                        n: 16,
                        k: 16
                    }
                );
                assert_eq!(c.schedule.storage_dtype, DType::BF16);
                assert_eq!(c.schedule.operand_dtype, DType::BF16);
                assert_eq!(c.schedule.warp_fragments, (2, 2));
                assert_eq!(c.schedule.warp_topology.0 * c.schedule.warp_topology.1, 8);
                let expected = if role == AttentionTileRole::WidthQuery
                    || (role == AttentionTileRole::Scores
                        && policy == BatchedAttentionBf16TilePolicy::WideScores)
                {
                    (64, 128)
                } else {
                    (128, 64)
                };
                assert_eq!((c.schedule.block_tile.m, c.schedule.block_tile.n), expected);
                let ctas = batch * heads * (m / expected.0) * (n / expected.1);
                assert!(ctas >= 128);
                // Tile geometry does not split K or reset an output accumulator:
                // every element still consumes these same increasing K16 chunks.
                let old_k16: Vec<_> = (0..k).step_by(region.schedule.instruction_tile.k).collect();
                let new_k16: Vec<_> = (0..k).step_by(c.schedule.instruction_tile.k).collect();
                assert_eq!(old_k16, new_k16);
            }
            assert!(
                BatchedAttentionBf16TilePolicy::Off
                    .candidate(&region, resources(), true)
                    .is_none()
            );
            assert!(
                BatchedAttentionBf16TilePolicy::TallScores
                    .candidate(&region, resources(), false)
                    .is_none()
            );
        }
    }
}
#[test]
fn batched_attention_rejects_small_batches_tails_rank_and_resource_changes() {
    let policy = BatchedAttentionBf16TilePolicy::TallScores;
    for (b, h, m, n, k) in [
        (1, 8, 512, 64, 512),
        (7, 8, 512, 64, 512),
        (0, 8, 512, 64, 512),
        (8, 8, 511, 64, 512),
        (8, 8, 512, 63, 512),
        (8, 8, 512, 512, 32),
        (8, 8, 512, 512, 128),
        (8, 8, 512, 128, 512),
        (8, 8, 64, 512, 256),
    ] {
        if b == 0 {
            continue;
        } // MatMulRegion itself rejects zero geometry.
        assert!(
            policy
                .candidate(&fixture(b, h, m, n, k), resources(), true)
                .is_none()
        );
    }
    let base = fixture(8, 8, 512, 512, 64);
    let mut zero = base.clone();
    zero.output_shape[0] = 0;
    zero.batch_shape[0] = 0;
    assert!(policy.candidate(&zero, resources(), true).is_none());
    let mut bias = base.clone();
    bias.epilogue_operations.push(crate::tile::RegionOp {
        node: bias.members[0],
        output: crate::tile::RegionValue(99),
        kind: crate::tile::RegionOpKind::Binary {
            op: crate::tensor::BinaryOp::Add,
            lhs: bias.matmul_value,
            rhs: bias.lhs_value,
        },
    });
    assert!(policy.candidate(&bias, resources(), true).is_none());
    for change in 0..8 {
        let mut r = base.clone();
        match change {
            0 => r.output_shape = vec![64, 512, 512],
            1 => r.batch_shape = vec![64],
            2 => r.schedule.storage_dtype = DType::F32,
            3 => r.schedule.operand_dtype = DType::F32,
            4 => r.schedule.plan = MatMulPlan::ScalarF32,
            5 => {
                r.schedule.block_tile = MatMulTile {
                    m: 64,
                    n: 64,
                    k: 32,
                }
            }
            6 => r.schedule.instruction_tile.k = 8,
            7 => r.outputs[0].value = crate::tile::RegionValue(999),
            _ => unreachable!(),
        }
        assert!(
            policy.candidate(&r, resources(), true).is_none(),
            "change {change}"
        );
    }
    for budget in [
        MatMulResources {
            shared_bytes_per_block: 31 * 1024,
            ..resources()
        },
        MatMulResources {
            shared_bytes_per_sm: 63 * 1024,
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
        assert!(policy.candidate(&base, budget, true).is_none());
    }
}
#[test]
fn batched_attention_proves_whole_maps_and_rejects_unsupported_batch_views() {
    let policy = BatchedAttentionBf16TilePolicy::WideScores;
    for (m, n, k) in [(512, 512, 64), (512, 64, 512), (64, 512, 512)] {
        let mut base = fixture(8, 8, m, n, k);
        model_views(&mut base);
        assert!(policy.candidate(&base, resources(), true).is_some());
        if (m, n, k) == (512, 64, 512) {
            let mut transposed = base.clone();
            transposed.inputs[0].tensor =
                transposed.inputs[0].tensor.permute(&[0, 1, 3, 2]).unwrap();
            assert!(policy.candidate(&transposed, resources(), true).is_some());
        }
        for change in 0..8 {
            let mut r = base.clone();
            let input = &mut r.inputs[0];
            match change {
                0 => input.tensor.dtype = DType::F32,
                1 => input.tensor.predicate = Some(crate::tile::IndexPredicate::And(vec![])),
                2 => input.tensor.out_of_bounds = Some(crate::tile::ScalarValue::F32(0)),
                3 => input.tensor = input.tensor.permute(&[1, 0, 2, 3]).unwrap(),
                4 => input.source_shape[0] *= 3,
                5 => input.tensor.access.results[0] = crate::tile::IndexExpr::Const(0),
                6 => {
                    input.tensor.access.results[0] = crate::tile::IndexExpr::Add(
                        Box::new(input.tensor.access.results[0].clone()),
                        Box::new(crate::tile::IndexExpr::Const(1)),
                    )
                }
                7 => {
                    input.tensor.access.results[0] = crate::tile::IndexExpr::Sub(
                        Box::new(input.tensor.access.results[0].clone()),
                        Box::new(crate::tile::IndexExpr::Const(1)),
                    )
                }
                _ => unreachable!(),
            }
            assert!(
                policy.candidate(&r, resources(), true).is_none(),
                "m{m}n{n}k{k} change{change}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn batched_attention_explicit_async_pipeline_is_all_or_nothing() {
    use crate::tile::{MatMulPipeline, MatMulSharedLayout};
    for policy in [
        BatchedAttentionBf16TilePolicy::TallScores,
        BatchedAttentionBf16TilePolicy::WideScores,
    ] {
        for (m, n, k) in [(512, 512, 64), (512, 64, 512), (64, 512, 512)] {
            let mut r = fixture(8, 8, m, n, k);
            model_views(&mut r);
            let c = policy.candidate(&r, resources(), true).unwrap();
            assert_eq!(
                c.schedule.pipeline_stages, 1,
                "raw geometry must not inherit staging"
            );
            for layout in [
                MatMulSharedLayout::Auto,
                MatMulSharedLayout::Contiguous,
                MatMulSharedLayout::Padded,
            ] {
                let s = c
                    .configure(&r, layout, MatMulPipeline::Auto, true, resources())
                    .unwrap();
                assert_eq!(s.pipeline_stages, 2);
                assert_eq!(s.copy_bytes, 16);
                assert_eq!(s.async_operands, (true, true));
                assert!(s.reuse_shared_memory);
                assert!(s.conservative_two_block_fit(resources()));
                let mut selected = r.clone();
                selected.schedule = s;
                let mut tiles = crate::tile::TileGraph::from_graph(&TensorGraph::<f32>::new());
                tiles.add_matmul_regions(&[selected]).unwrap();
                tiles.optimize_indices().unwrap();
                let source = crate::ptx::PtxGraph::from(tiles)
                    .with_target(crate::ptx::PtxTarget::sm80())
                    .module_source();
                assert!(source.contains("cp.async"));
                assert!(source.contains("wmma.mma"));
                assert!(source.contains(&format!("shared[{}]", s.shared_mem_bytes())));
            }
            assert!(
                c.configure(
                    &r,
                    MatMulSharedLayout::Auto,
                    MatMulPipeline::Synchronous,
                    true,
                    resources()
                )
                .is_none()
            );
            assert!(
                c.configure(
                    &r,
                    MatMulSharedLayout::Auto,
                    MatMulPipeline::Auto,
                    false,
                    resources()
                )
                .is_none()
            );
            let budget = MatMulResources {
                shared_bytes_per_sm: 63 * 1024,
                ..resources()
            };
            assert!(
                c.configure(
                    &r,
                    MatMulSharedLayout::Auto,
                    MatMulPipeline::Auto,
                    true,
                    budget
                )
                .is_none()
            );
            let mut changed = r.clone();
            changed.inputs[0].tensor.out_of_bounds = Some(crate::tile::ScalarValue::F32(0));
            assert!(
                c.configure(
                    &changed,
                    MatMulSharedLayout::Auto,
                    MatMulPipeline::Auto,
                    true,
                    resources()
                )
                .is_none()
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn batched_attention_rectangle_gpu_matches_all_bf16_output_bits_and_nonfinite_classes() {
    use crate::ptx::linear::program::Kernel;
    use crate::ptx::{PtxGraph, PtxTarget};
    use crate::tile::{MatMulPipeline, MatMulSharedLayout, TileGraph};
    use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig};
    let device = match CudaContext::new(0) {
        Ok(device) => device,
        Err(error) => {
            assert_ne!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "{error}"
            );
            return;
        }
    };
    let target = PtxTarget::from_context(&device).unwrap();
    assert!(
        target.compute_capability.0 >= 8,
        "BF16 asynchronous rectangle test requires SM80+"
    );
    let stream = device.new_stream().unwrap();
    let limits = target.matmul_resources;
    for (policy, batch, heads, m, n, k) in [
        (
            BatchedAttentionBf16TilePolicy::TallScores,
            8,
            8,
            512,
            512,
            64,
        ),
        (
            BatchedAttentionBf16TilePolicy::WideScores,
            8,
            8,
            512,
            512,
            64,
        ),
        (
            BatchedAttentionBf16TilePolicy::TallScores,
            8,
            8,
            512,
            64,
            512,
        ),
        (
            BatchedAttentionBf16TilePolicy::WideScores,
            8,
            8,
            64,
            512,
            512,
        ),
        (
            BatchedAttentionBf16TilePolicy::TallScoresByGrid,
            1,
            16,
            512,
            512,
            64,
        ),
        (
            BatchedAttentionBf16TilePolicy::TallScoresByGrid,
            2,
            8,
            512,
            512,
            64,
        ),
    ] {
        let mut baseline = fixture(batch, heads, m, n, k);
        model_views(&mut baseline);
        let c = policy.candidate(&baseline, limits, true).unwrap();
        let mut candidate = baseline.clone();
        candidate.schedule = c
            .configure(
                &baseline,
                MatMulSharedLayout::Auto,
                MatMulPipeline::Auto,
                true,
                limits,
            )
            .unwrap();
        baseline
            .configure_pipeline(MatMulSharedLayout::Auto, MatMulPipeline::Auto, true, limits)
            .unwrap();
        baseline.schedule = baseline.schedule.with_shared_memory_reuse(true);
        let mut a = stream
            .alloc_zeros::<half::bf16>(extent(&baseline.inputs[0].source_shape).unwrap())
            .unwrap();
        let mut b = stream
            .alloc_zeros::<half::bf16>(extent(&baseline.inputs[1].source_shape).unwrap())
            .unwrap();
        let output_len = extent(&baseline.output_shape).unwrap();
        let outputs = [
            stream.alloc_zeros::<half::bf16>(output_len).unwrap(),
            stream.alloc_zeros::<half::bf16>(output_len).unwrap(),
        ];
        let kernels: [Kernel; 2] = [&baseline, &candidate]
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
                tiles.add_matmul_regions(std::slice::from_ref(r)).unwrap();
                tiles.optimize_indices().unwrap();
                let graph = PtxGraph::from(tiles).with_target(target);
                let name = graph.matmul_region_kernel_name(0).unwrap();
                let module = device
                    .load_module(cudarc::nvrtc::Ptx::from_src(graph.to_ptx()))
                    .unwrap();
                Kernel {
                    owner: None,
                    label: format!("batched attention correctness {policy:?} {m}x{n}x{k}"),
                    function: module.load_function(name).unwrap(),
                    config: LaunchConfig {
                        grid_dim: (
                            u32::try_from(n / r.schedule.block_tile.n).unwrap(),
                            u32::try_from(m / r.schedule.block_tile.m).unwrap(),
                            u32::try_from(batch * heads).unwrap(),
                        ),
                        block_dim: r.schedule.block_threads,
                        shared_mem_bytes: 0,
                    },
                    pointers: vec![
                        a.device_ptr(&stream).0,
                        b.device_ptr(&stream).0,
                        outputs[i].device_ptr(&stream).0,
                    ],
                    zero_before: None,
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("two compiled kernels"));
        let feed = |len: usize, revision: usize, operand: usize| -> Vec<half::bf16> {
            let special = [
                0x0000, 0x8000, 0x0001, 0x8001, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0x7f81,
            ];
            (0..len)
                .map(|i| {
                    if revision == 2 {
                        return half::bf16::from_bits(if i % 4093 == 0 {
                            special[(i / 4093 + operand) % special.len()]
                        } else if i % 2 == 0 {
                            0
                        } else {
                            0x8000
                        });
                    }
                    let value = if revision == 0 {
                        ((i * 7 + operand * 11) % 33) as f32 / 32. - 0.5
                    } else {
                        ((i * 131 + operand * 19) % 257) as f32 / 97. - 1.23
                    };
                    half::bf16::from_f32(value)
                })
                .collect()
        };
        for revision in 0..3 {
            stream
                .memcpy_htod(&feed(a.len(), revision, 0), &mut a)
                .unwrap();
            stream
                .memcpy_htod(&feed(b.len(), revision, 1), &mut b)
                .unwrap();
            for kernel in &kernels {
                kernel.run(&stream).unwrap();
            }
            stream.synchronize().unwrap();
            let expected = stream.memcpy_dtov(&outputs[0]).unwrap();
            let actual = stream.memcpy_dtov(&outputs[1]).unwrap();
            let mut payload_differences = 0usize;
            for (i, (&a, &b)) in expected.iter().zip(&actual).enumerate() {
                if a.is_nan() || b.is_nan() {
                    assert!(
                        a.is_nan() && b.is_nan(),
                        "NaN class {policy:?} {m}x{n}x{k} revision{revision} index{i}"
                    );
                    payload_differences += usize::from(a.to_bits() != b.to_bits());
                } else {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "BF16 bits {policy:?} {m}x{n}x{k} revision{revision} index{i}"
                    );
                }
            }
            eprintln!(
                "{policy:?} {m}x{n}x{k} revision{revision}: all{output_len} output bits/classes match; NaN payload differences{payload_differences}"
            );
        }
    }
}

#[test]
fn batched_attention_by_grid_keeps_role_grid_threshold_and_old_bh_guards() {
    use BatchedAttentionBf16TilePolicy::{TallScores, TallScoresByGrid, WideScores};
    for (batch, heads) in [
        (1, 1),
        (1, 3),
        (1, 4),
        (2, 8),
        (1, 31),
        (4, 8),
        (7, 9),
        (8, 8),
    ] {
        let batch_heads = batch * heads;
        for (m, n, k, minimum_heads) in [(512, 512, 64, 4), (512, 64, 512, 32), (64, 512, 512, 32)]
        {
            let mut region = fixture(batch, heads, m, n, k);
            model_views(&mut region);
            let new = TallScoresByGrid.candidate(&region, resources(), true);
            assert_eq!(
                new.is_some(),
                batch_heads >= minimum_heads,
                "BH{batch_heads} {m}x{n}x{k} batch={:?} tile={:?} operand_proof={:?}",
                region.batch_shape,
                region.schedule.block_tile,
                operands(&region)
            );
            for old in [TallScores, WideScores] {
                assert_eq!(
                    old.candidate(&region, resources(), true).is_some(),
                    batch_heads >= 64
                );
            }
            if let Some(new) = new {
                assert_eq!(new.schedule.block_tile.m, if m == 64 { 64 } else { 128 });
                assert_eq!(new.schedule.block_tile.n, if m == 64 { 128 } else { 64 });
                #[cfg(feature = "cuda")]
                assert!(
                    new.configure(
                        &region,
                        super::super::MatMulSharedLayout::Auto,
                        super::super::MatMulPipeline::Auto,
                        true,
                        resources()
                    )
                    .is_some()
                );
                #[cfg(feature = "cuda")]
                assert!(
                    new.configure(
                        &region,
                        super::super::MatMulSharedLayout::Auto,
                        super::super::MatMulPipeline::Synchronous,
                        true,
                        resources()
                    )
                    .is_none()
                );
            }
            assert!(
                TallScoresByGrid
                    .candidate(&region, resources(), false)
                    .is_none()
            );
        }
    }
    // Lower BH admission does not bypass full-volume, type, resource or shape guards.
    let base = fixture(1, 16, 512, 512, 64);
    let mut bad = base.clone();
    bad.inputs[0].tensor.dtype = DType::F32;
    assert!(
        TallScoresByGrid
            .candidate(&bad, resources(), true)
            .is_none()
    );
    bad = base.clone();
    bad.inputs[0].tensor.predicate = Some(crate::tile::IndexPredicate::And(vec![]));
    assert!(
        TallScoresByGrid
            .candidate(&bad, resources(), true)
            .is_none()
    );
    let mut too_small = resources();
    too_small.shared_bytes_per_block = 16 * 1024;
    assert!(TallScoresByGrid.candidate(&base, too_small, true).is_none());
    assert!(
        TallScoresByGrid
            .candidate(&fixture(1, 16, 511, 512, 64), resources(), true)
            .is_none()
    );
}
