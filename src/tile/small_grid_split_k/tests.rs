use super::*;
use crate::{
    graph::{TensorGraph, TensorGraphNode},
    tile::{MatMulCapabilities, MatMulPrecision},
};

fn resources() -> MatMulResources {
    MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..Default::default()
    }
}
fn region(m: usize, n: usize, total_k: usize, orientation: (bool, bool)) -> MatMulRegion {
    let part_k = total_k / 4;
    let mut g = TensorGraph::<half::bf16>::new();
    let a = g.graph.add_node(TensorGraphNode::Input {
        name: "a",
        shape: vec![4, m, part_k],
    });
    let b = g.graph.add_node(TensorGraphNode::Input {
        name: "b",
        shape: vec![4, part_k, n],
    });
    let out = g.graph.add_node(TensorGraphNode::MatMul {
        shape: vec![4, m, n],
    });
    g.graph.add_edge(a, out, 0);
    g.graph.add_edge(b, out, 1);
    let mut r = MatMulRegion::from_graph(&g, out, vec![], vec![out]).unwrap();
    r.schedule = MatMulSchedule::select_with_resources(
        m,
        n,
        part_k,
        DType::BF16,
        MatMulCapabilities {
            tf32: true,
            f16: true,
            bf16: true,
        },
        MatMulPrecision::NativeHalf,
        resources(),
    );
    for (i, column) in [orientation.0, orientation.1].into_iter().enumerate() {
        let shape = match (i, column) {
            (0, false) => vec![m, total_k],
            (0, true) => vec![total_k, m],
            (1, false) => vec![total_k, n],
            (1, true) => vec![n, total_k],
            _ => unreachable!(),
        };
        let input = &mut r.inputs[i];
        input.source_shape = shape.clone();
        let mut view = VirtualTensor::identity(input.node, shape, DType::BF16);
        if column {
            view = view.permute(&[1, 0]).unwrap();
        }
        input.tensor = if i == 0 {
            view.reshape(vec![m, 4, part_k])
                .unwrap()
                .permute(&[1, 0, 2])
                .unwrap()
        } else {
            view.reshape(vec![4, part_k, n]).unwrap()
        };
    }
    r
}

#[test]
fn small_grid_split_shape_domain_preserves_long_and_large_scopes() {
    for (m, n, k) in [
        (64, 64, 512),
        (256, 512, 512),
        (128, 2048, 5504),
        (512, 512, 512),
        (64, 8128, 640),
    ] {
        assert!(SmallGridCandidate::eligible_shape(m, n, k), "{m}/{n}/{k}");
    }
    for (m, n, k) in [
        (63, 64, 512),
        (64, 65, 512),
        (64, 64, 384),
        (64, 64, 513),
        (512, 1024, 512),
        (128, 128, 2048),
        (512, 512, 4096),
        (usize::MAX, 64, 512),
        (64, 64, usize::MAX),
    ] {
        assert!(!SmallGridCandidate::eligible_shape(m, n, k), "{m}/{n}/{k}");
    }
}

#[test]
fn small_grid_split_all_maps_partition_every_physical_element_once() {
    for orientation in [(false, false), (false, true), (true, false), (true, true)] {
        let r = region(128, 64, 512, orientation);
        let original = r.clone();
        let c = SmallGridCandidate::candidate(&r, 512, resources(), true).unwrap();
        assert_eq!(
            c.baseline, r,
            "candidate retains the exact original descriptors"
        );
        assert_eq!(r, original, "proof does not mutate the ordinary bindings");
        assert_eq!(
            c.schedule.block_tile,
            MatMulTile {
                m: 64,
                n: 64,
                k: 32
            }
        );
        assert_eq!(c.schedule.block_threads, (32, 8, 1));
        assert_eq!(
            c.schedule.accumulator_exchange,
            MatMulAccumulatorExchange::FullTile
        );
        assert_eq!(c.schedule.fragment_load, MatMulFragmentLoad::PreloadAll);
        for (which, column) in [orientation.0, orientation.1].into_iter().enumerate() {
            let input = &r.inputs[which];
            let mut hits = vec![0_u8; extent(&input.source_shape).unwrap()];
            for part in 0..4 {
                for row in 0..128 {
                    for col in 0..if which == 0 { 128 } else { 64 } {
                        let coordinates = input.tensor.source_index(&[part, row, col]).unwrap();
                        let actual = coordinates
                            .iter()
                            .zip(&input.source_shape)
                            .fold(0, |x, (&i, &d)| x * d + i);
                        let expected = match (which, column) {
                            (0, false) => row * 512 + part * 128 + col,
                            (0, true) => (part * 128 + col) * 128 + row,
                            (1, false) => (part * 128 + row) * 64 + col,
                            (1, true) => col * 512 + part * 128 + row,
                            _ => unreachable!(),
                        };
                        assert_eq!(actual, expected);
                        hits[actual] += 1;
                    }
                }
            }
            assert!(hits.into_iter().all(|n| n == 1));
        }
    }
}

#[test]
fn small_grid_split_rejects_unproven_maps_and_external_mutation() {
    let original = region(128, 64, 512, (false, true));
    assert!(SmallGridCandidate::candidate(&original, 512, resources(), false).is_none());
    for change in 0..8 {
        let mut bad = original.clone();
        match change {
            0 => bad.inputs[0].tensor.predicate = Some(crate::tile::IndexPredicate::And(vec![])),
            1 => bad.inputs[0].tensor.out_of_bounds = Some(crate::tile::ScalarValue::F32(0)),
            2 => bad.inputs[0].source_shape[0] += 1,
            3 => bad.inputs[0].tensor.access.results[0] = crate::tile::IndexExpr::Const(0),
            4 => bad.inputs[0].tensor.dtype = DType::F32,
            5 => bad.batch_shape = vec![2],
            6 => bad.outputs.push(bad.outputs[0].clone()),
            7 => bad.schedule.plan = MatMulPlan::ScalarF32,
            _ => unreachable!(),
        }
        assert!(
            SmallGridCandidate::candidate(&bad, 512, resources(), true).is_none(),
            "map{change}"
        );
    }
    let mut tiny = resources();
    tiny.shared_bytes_per_sm = 1;
    assert!(SmallGridCandidate::candidate(&original, 512, tiny, true).is_none());
}

#[cfg(feature = "cuda")]
#[test]
fn small_grid_split_final_ir_has_f32_only_partials_and_preserves_k_sentinel() {
    use crate::tile::Stmt;
    for orientation in [(false, false), (false, true), (true, false), (true, true)] {
        let r = region(128, 64, 512, orientation);
        let c = SmallGridCandidate::candidate(&r, 512, resources(), true).unwrap();
        assert!(c.partial_ir(&r, resources(), false).is_none());
        let (s, ir) = c.partial_ir(&r, resources(), true).unwrap();
        assert_eq!(s.async_operands, (true, true));
        assert_eq!(s.copy_bytes, 16);
        assert_eq!(s.pipeline_stages, 2);
        assert!(s.reuse_shared_memory);
        assert!(
            ir.params
                .iter()
                .all(|p| p.dtype == if p.is_input { DType::BF16 } else { DType::F32 })
        );
        ir.validate_layouts().unwrap();
        let kloop = ir
            .body
            .stmts
            .iter()
            .position(|s| matches!(s, Stmt::ForLoop { .. }))
            .unwrap();
        assert!(matches!(ir.body.stmts[kloop + 1], Stmt::AsyncWait));
        assert!(matches!(ir.body.stmts[kloop + 2], Stmt::Barrier));
        let mut changed = r.clone();
        changed.inputs[0].source_shape.reverse();
        assert!(c.partial_ir(&changed, resources(), true).is_none());
    }
}
