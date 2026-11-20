use super::*;
use crate::graph::{TensorGraph, TensorGraphNode};
use crate::tile::{
    MatMulCapabilities, MatMulPipeline, MatMulPrecision, MatMulSharedLayout, VirtualTensor,
};

fn resources() -> MatMulResources {
    MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..MatMulResources::default()
    }
}
fn schedule(m: usize, n: usize, k: usize, orientations: (bool, bool)) -> MatMulSchedule {
    MatMulSchedule::select_with_resources(
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
    )
    .with_operand_orientations(MatMulSharedLayout::Auto, orientations, resources())
    .unwrap()
    .with_pipeline(MatMulPipeline::Auto, true, true, resources())
    .with_shared_memory_reuse(true)
}
fn region(orientations: (bool, bool)) -> MatMulRegion {
    let mut g = TensorGraph::<half::bf16>::new();
    let a = g.graph.add_node(TensorGraphNode::Input {
        name: "a",
        shape: vec![2048, 64],
    });
    let b = g.graph.add_node(TensorGraphNode::Input {
        name: "b",
        shape: vec![64, 256],
    });
    let out = g.graph.add_node(TensorGraphNode::MatMul {
        shape: vec![2048, 256],
    });
    g.graph.add_edge(a, out, 0);
    g.graph.add_edge(b, out, 1);
    let mut r = MatMulRegion::from_graph(&g, out, vec![], vec![out]).unwrap();
    r.schedule = schedule(2048, 256, 64, orientations);
    for (input, transposed) in r.inputs.iter_mut().zip([orientations.0, orientations.1]) {
        if transposed {
            input.source_shape = vec![input.tensor.shape[1], input.tensor.shape[0]];
            input.tensor =
                VirtualTensor::identity(input.node, input.source_shape.clone(), DType::BF16)
                    .permute(&[1, 0])
                    .unwrap();
        }
    }
    r
}

#[test]
fn ordinary_four_warp_changes_only_declared_threads_and_preserves_k16_shared_layouts() {
    for orientation in [(false, false), (false, true), (true, false), (true, true)] {
        let r = region(orientation);
        let old = r.schedule;
        let new = region_candidate(&r, resources()).unwrap();
        let mut expected = old;
        expected.block_threads = (32, 4, 1);
        assert_eq!(new, expected);
        assert_eq!(new.shared_mem_bytes(), old.shared_mem_bytes());
        assert_eq!(new.warp_fragments, (2, 2));
        assert_eq!(new.warp_topology, (2, 2));
        assert_eq!(new.thread_count(), 128);
        assert_eq!(
            (0..old.logical_shape.k)
                .step_by(old.instruction_tile.k)
                .collect::<Vec<_>>(),
            (0..new.logical_shape.k)
                .step_by(new.instruction_tile.k)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn ordinary_four_warp_rejects_other_families_tails_pipeline_and_resource_changes() {
    let old = schedule(2048, 256, 64, (false, false));
    for change in 0..15 {
        let mut bad = old;
        match change {
            0 => bad.storage_dtype = DType::F32,
            1 => bad.operand_dtype = DType::F16,
            2 => bad.plan = MatMulPlan::ScalarF32,
            3 => bad.pipeline_stages = 1,
            4 => bad.async_operands = (true, false),
            5 => bad.copy_bytes = 4,
            6 => bad.reuse_shared_memory = false,
            7 => bad.warp_topology = (2, 4),
            8 => bad.warp_fragments = (2, 1),
            9 => bad.block_threads = (16, 16, 1),
            10 => bad.block_tile.m = 128,
            11 => bad.logical_shape.m = 2049,
            12 => bad.logical_shape.n = 255,
            13 => bad.logical_shape.k = 65,
            14 => bad.instruction_tile.k = 8,
            _ => unreachable!(),
        }
        assert!(
            candidate(
                bad,
                &[bad.logical_shape.m, bad.logical_shape.n],
                resources()
            )
            .is_none(),
            "change{change}"
        );
    }
    assert!(candidate(old, &[1, 2048, 256], resources()).is_none());
    assert!(
        candidate(
            schedule(512, 512, 64, (false, false)),
            &[512, 512],
            resources()
        )
        .is_none()
    );
    let mut small = resources();
    small.shared_bytes_per_sm = 30 * 1024;
    assert!(candidate(old, &[2048, 256], small).is_none());
    small = resources();
    small.registers_per_sm = 32 * 1024;
    assert!(candidate(old, &[2048, 256], small).is_none());
    assert!(
        candidate(
            old.with_experimental_bf16_eight_compute_warps(),
            &[2048, 256],
            resources()
        )
        .is_none()
    );
}

#[test]
fn ordinary_four_warp_requires_whole_aligned_nonoverlapping_bounded_maps() {
    let base = region((false, false));
    for change in 0..6 {
        let mut bad = base.clone();
        let a = &mut bad.inputs[0];
        match change {
            0 => a.tensor.predicate = Some(crate::tile::IndexPredicate::And(vec![])),
            1 => a.tensor.out_of_bounds = Some(crate::tile::ScalarValue::F32(0)),
            2 => a.tensor.access.results[0] = crate::tile::IndexExpr::Const(0),
            3 => a.source_shape[0] /= 2,
            4 => {
                a.tensor.access.results[0] = crate::tile::IndexExpr::Sub(
                    Box::new(a.tensor.access.results[0].clone()),
                    Box::new(crate::tile::IndexExpr::Const(1)),
                )
            }
            5 => a.tensor.dtype = DType::F32,
            _ => unreachable!(),
        }
        assert!(region_candidate(&bad, resources()).is_none(), "map{change}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn ordinary_four_warp_generated_ir_covers_copies_stores_and_final_drain() {
    use crate::tile::{Dim, Expr, Stmt, TileGraph};
    fn eval(e: &Expr, x: i64, y: i64) -> i64 {
        match e {
            Expr::Const(v) => *v,
            Expr::ThreadIdx(Dim::X) => x,
            Expr::ThreadIdx(Dim::Y) => y,
            Expr::ThreadIdx(Dim::Z) | Expr::BlockIdx(_) => 0,
            Expr::BlockDim(Dim::X) => 32,
            Expr::BlockDim(Dim::Y) => 4,
            Expr::BlockDim(Dim::Z) => 1,
            Expr::Var(v) if v == "k_tile" => 0,
            Expr::Add(a, b) => eval(a, x, y) + eval(b, x, y),
            Expr::Sub(a, b) => eval(a, x, y) - eval(b, x, y),
            Expr::Mul(a, b) => eval(a, x, y) * eval(b, x, y),
            Expr::FloorDiv(a, b) => eval(a, x, y) / i64::try_from(*b).unwrap(),
            Expr::Mod(a, b) => eval(a, x, y) % i64::try_from(*b).unwrap(),
            Expr::ShiftRight(a, b) => eval(a, x, y) >> *b,
            Expr::BitAnd(a, b) => eval(a, x, y) & i64::try_from(*b).unwrap(),
            _ => panic!("unexpected generated matmul expression {e:?}"),
        }
    }
    for orientation in [(false, false), (false, true), (true, false), (true, true)] {
        let mut r = region(orientation);
        r.schedule = region_candidate(&r, resources()).unwrap();
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.add_matmul_regions(std::slice::from_ref(&r)).unwrap();
        let ir = &tiles.matmul_region_kernels[0].ir;
        ir.validate_layouts().unwrap();
        assert_eq!(ir.shared_mem_bytes, r.schedule.shared_mem_bytes());
        let body = &ir.body.stmts;
        let kloop = body
            .iter()
            .position(|s| matches!(s, Stmt::ForLoop { .. }))
            .unwrap();
        let Stmt::ForLoop { body: mainloop, .. } = &body[kloop] else {
            unreachable!()
        };
        for stage in [&body[..kloop], mainloop.stmts.as_slice()] {
            for (param, rows, cols, column_major) in [
                ("input_0", 64, 32, orientation.0),
                ("input_1", 32, 64, orientation.1),
            ] {
                let mut hits = vec![0_usize; rows * cols];
                let mut statements = 0;
                for statement in stage {
                    let Stmt::AsyncCopy {
                        src_param,
                        tile_row,
                        tile_col,
                        element_index,
                        copy_bytes,
                        ..
                    } = statement
                    else {
                        continue;
                    };
                    if src_param != param {
                        continue;
                    }
                    statements += 1;
                    assert_eq!(*copy_bytes, 16);
                    for y in 0..4 {
                        for x in 0..32 {
                            assert_eq!(eval(element_index, x, y).rem_euclid(8), 0);
                            let row = usize::try_from(eval(tile_row, x, y)).unwrap();
                            let col = usize::try_from(eval(tile_col, x, y)).unwrap();
                            for lane in 0..8 {
                                let (row, col) = if column_major {
                                    (row + lane, col)
                                } else {
                                    (row, col + lane)
                                };
                                assert!(row < rows && col < cols);
                                hits[row * cols + col] += 1;
                            }
                        }
                    }
                }
                assert_eq!(statements, 2, "two vector rounds per operand and thread");
                assert!(
                    hits.iter().all(|&n| n == 1),
                    "every generated operand element exactly once"
                );
            }
        }
        assert!(matches!(body[kloop + 1], Stmt::AsyncWait));
        assert!(matches!(body[kloop + 2], Stmt::Barrier));
        let exchange = body
            .iter()
            .position(|s| matches!(s, Stmt::ConvertLayout { .. }))
            .unwrap();
        assert!(exchange > kloop + 2);
        assert!(matches!(body[exchange + 1], Stmt::Barrier));
        let mut hits = vec![0_usize; 64 * 64];
        let mut stores = 0;
        for statement in body {
            if let Stmt::StoreGlobalPredicated { element_index, .. } = statement {
                stores += 1;
                for y in 0..4 {
                    for x in 0..32 {
                        let flat = usize::try_from(eval(element_index, x, y)).unwrap();
                        let (row, col) = (flat / 256, flat % 256);
                        assert!(row < 64 && col < 64);
                        hits[row * 64 + col] += 1;
                    }
                }
            }
        }
        assert_eq!(stores, 32);
        assert!(hits.iter().all(|&n| n == 1));
        let ptx = crate::ptx::PtxGraph::from(tiles)
            .with_target(crate::ptx::PtxTarget::sm80())
            .to_ptx();
        assert_eq!(ptx.matches("wmma.mma").count(), 8);
        let drain = ptx.rfind("cp.async.wait_group 0;").unwrap();
        let exchange = ptx.find("wmma.store.d.").unwrap();
        assert!(drain < exchange);
    }
}
