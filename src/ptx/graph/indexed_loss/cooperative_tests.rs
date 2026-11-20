use super::*;
use crate::{
    graph::{TensorGraph, TensorGraphNode},
    tile::KernelParam,
};
fn graph(rows: usize, vocabulary: usize, forward: bool) -> (TensorGraph<f32>, NodeIndex) {
    let mut g = TensorGraph::new();
    let x = g.graph.add_node(TensorGraphNode::Input {
        name: "x",
        shape: vec![rows, vocabulary],
    });
    let t = g.graph.add_node(TensorGraphNode::IndexSource {
        source: crate::tensor::IndexTensor::input("t", vec![rows]),
    });
    let dy = g.graph.add_node(TensorGraphNode::Input {
        name: "dy",
        shape: vec![rows],
    });
    let out = g.graph.add_node(if forward {
        TensorGraphNode::IndexedCrossEntropy { shape: vec![rows] }
    } else {
        TensorGraphNode::IndexedCrossEntropyBackward {
            shape: vec![rows, vocabulary],
        }
    });
    g.graph.add_edge(x, out, 0);
    g.graph.add_edge(t, out, 1);
    if !forward {
        g.graph.add_edge(dy, out, 2);
    }
    (g, out)
}
#[test]
fn cooperative_ordered_scope_dtypes_resources_and_forward_default_ptx() {
    for (rows, vocabulary, forward, eligible) in [
        (1, 4096, false, true),
        (17, 4096, false, true),
        (0, 4096, false, false),
        (1, 4095, false, false),
        (1, 4097, false, false),
        (1, 4096, true, false),
    ] {
        let (g, out) = graph(rows, vocabulary, forward);
        let mut tiles = TileGraph::from_graph(&g);
        let before = tiles.graph[out].clone();
        let changed = enable_cooperative_ordered_backward(&mut tiles.graph[out], 48 * 1024);
        assert_eq!(changed, eligible);
        if !eligible {
            assert_eq!(format!("{:?}", tiles.graph[out]), format!("{before:?}"));
        } else {
            tiles.graph[out].validate_layouts().unwrap();
            let s = schedule_for_ir(&tiles.graph[out]).unwrap();
            assert!(s.cooperative_ordered_backward);
            assert_eq!(s.threads, 256);
            assert_eq!(s.output_len, rows * 4096);
        }
    }
    let (g, out) = graph(3, 4096, false);
    for (name, dtype, eligible) in [
        ("logits", DType::BF16, true),
        ("logits", DType::F16, false),
        ("grad_output", DType::BF16, false),
        ("output", DType::BF16, false),
        ("targets", DType::F32, true),
        ("targets", DType::U32, true),
    ] {
        let mut tiles = TileGraph::from_graph(&g);
        tiles.graph[out]
            .params
            .iter_mut()
            .find(|p| p.name == name)
            .unwrap()
            .dtype = dtype;
        assert_eq!(
            enable_cooperative_ordered_backward(&mut tiles.graph[out], 48 * 1024),
            eligible
        );
    }
    let mut ir = TileGraph::from_graph(&g).graph[out].clone();
    assert!(!enable_cooperative_ordered_backward(&mut ir, 17411));
    ir.params.push(KernelParam {
        name: "other".into(),
        dtype: DType::F32,
        is_input: true,
    });
    assert!(!enable_cooperative_ordered_backward(&mut ir, 48 * 1024));
    assert!(cooperative_ordered_resources_fit(
        17412,
        0,
        256,
        1,
        48 * 1024
    ));
    for (shared, local, threads, blocks, limit) in [
        (17411, 0, 256, 1, 49152),
        (17412, 4, 256, 1, 49152),
        (17412, 0, 255, 1, 49152),
        (17412, 0, 256, 0, 49152),
        (17412, 0, 256, 1, 17411),
    ] {
        assert!(!cooperative_ordered_resources_fit(
            shared, local, threads, blocks, limit
        ));
    }
    let (g, out) = graph(3, 4096, true);
    let old = PtxGraph::from(TileGraph::from_graph(&g)).module_source();
    let mut tiles = TileGraph::from_graph(&g);
    assert!(
        PtxGraph::enable_indexed_loss_cooperative_ordered_backward(&mut tiles, 49152).is_none()
    );
    assert_eq!(PtxGraph::from(tiles).module_source(), old);
    assert!(!old.contains("bar.sync"));
    let _ = out;
}
#[test]
fn cooperative_ordered_lowering_has_only_max_tree_and_serial_prefetch_sum() {
    let (g, out) = graph(3, 4096, false);
    let old = PtxGraph::from(TileGraph::from_graph(&g)).module_source();
    let mut tiles = TileGraph::from_graph(&g);
    let fallback =
        PtxGraph::enable_indexed_loss_cooperative_ordered_backward(&mut tiles, 49152).unwrap();
    assert_eq!(PtxGraph::from(fallback).module_source(), old);
    let ptx = PtxGraph::from(tiles);
    assert_eq!(ptx.indexed_loss_cooperative_ordered_backward_programs(), 1);
    assert_eq!(
        ptx.indexed_loss_row_launch(out, 3 * 4096)
            .unwrap()
            .unwrap()
            .block_dim,
        (256, 1, 1)
    );
    let source = ptx.module_source();
    assert_eq!(source.matches("bar.sync").count(), 11);
    assert!(source.contains("[17412]"));
    assert!(source.contains("ce_cooperative_ordered_column"));
    assert!(!source.contains("ce_parallel_sum"));
    assert!(!source.contains("ce_row_broadcast"));
    // The numerical operation after statistics is the same independent-column
    // expression, including a fresh exp, division, indicator subtraction and dy.
    assert_eq!(source.matches("ex2.approx").count(), 2);
    assert_eq!(old.matches("ex2.approx").count(), 2);
}
#[test]
fn cooperative_ordered_columns_and_near_zero_fixture_preserve_left_fold() {
    let mut seen = vec![0; 4096];
    for lane in 0..256 {
        for k in 0..16 {
            seen[lane + 256 * k] += 1;
        }
    }
    assert!(seen.into_iter().all(|n| n == 1));
    let terms = (0..4096)
        .map(|i| if i == 0 { 1. } else { 1e-8f32 })
        .collect::<Vec<_>>();
    let old = terms.iter().fold(0f32, |s, &v| s + v);
    let mut prefetched = 0f32;
    for chunk in terms.chunks_exact(16) {
        let values: Vec<_> = chunk.to_vec();
        for v in values {
            prefetched += v;
        }
    }
    assert_eq!(old.to_bits(), prefetched.to_bits());
    assert_eq!(old, 1.);
    let mut local = [0f32; 256];
    for lane in 0..256 {
        for k in 0..16 {
            local[lane] += terms[lane + 256 * k];
        }
    }
    for stride in [128, 64, 32, 16, 8, 4, 2, 1] {
        for lane in 0..stride {
            local[lane] += local[lane + stride];
        }
    }
    assert!(
        local[0] > old,
        "a parallel SUM tree is a distinct numerical policy"
    );
}
#[cfg(test)]
mod gpu;
