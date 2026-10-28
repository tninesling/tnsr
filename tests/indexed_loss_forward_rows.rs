#![cfg(feature = "cuda")]
use std::collections::HashMap;
use tnsr::{
    Executor,
    graph::{TensorGraph, TensorGraphNode},
    ptx::{PtxExecutor, PtxReductionMode},
    tensor::TensorExpr,
};

fn executor(mode: PtxReductionMode) -> Option<PtxExecutor> {
    match PtxExecutor::try_new_with_reduction_mode(mode) {
        Ok(e) => Some(e),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            None
        }
        Err(error) => panic!("{error:#}"),
    }
}

fn forward_graph(rows: usize, vocabulary: usize) -> TensorGraph<f32> {
    let mut g = TensorGraph::new();
    let x = g.add_expr(&TensorExpr::input("logits", vec![rows, vocabulary]), false);
    let t = g.add_expr(&TensorExpr::input("targets", vec![rows]), false);
    let out = g
        .graph
        .add_node(TensorGraphNode::IndexedCrossEntropy { shape: vec![rows] });
    g.graph.add_edge(x, out, 0);
    g.graph.add_edge(t, out, 1);
    g
}
#[test]
fn forward_loss_rows_match_f64_reference_with_changed_feeds() {
    for mode in [
        PtxReductionMode::Strict,
        PtxReductionMode::DeterministicTree,
        PtxReductionMode::Online,
    ] {
        let Some(mut e) = executor(mode) else { return };
        for (rows, vocabulary) in [
            (1, 255),
            (1, 256),
            (3, 257),
            (33, 1000),
            (17, 4096),
            (512, 4096),
        ] {
            let g = forward_graph(rows, vocabulary);
            for feed in 0..3 {
                let logits: Vec<f32> = (0..rows * vocabulary)
                    .map(|i| {
                        if feed == 0 {
                            0.0
                        } else {
                            (((i * 7 + feed * 13) % 101) as f32 - 50.0) / 13.0
                                + if feed == 2 { 12000.0 } else { 0.0 }
                        }
                    })
                    .collect();
                let targets: Vec<f32> = (0..rows)
                    .map(|r| ((r * 17 + feed * (vocabulary - 1)) % vocabulary) as f32)
                    .collect();
                let expected: Vec<f64> = (0..rows)
                    .map(|r| {
                        let x = &logits[r * vocabulary..(r + 1) * vocabulary];
                        let maximum = x
                            .iter()
                            .map(|&v| f64::from(v))
                            .fold(f64::NEG_INFINITY, f64::max);
                        maximum - f64::from(x[targets[r] as usize])
                            + x.iter()
                                .map(|&v| (f64::from(v) - maximum).exp())
                                .sum::<f64>()
                                .ln()
                    })
                    .collect();
                let inputs =
                    HashMap::from([("logits".into(), logits), ("targets".into(), targets)]);
                let actual = if feed == 0 {
                    e.execute(&g, inputs).unwrap()
                } else {
                    e.execute_compiled(&g, inputs).unwrap()
                };
                // RAW_CAPTURE_FINITE
                assert_eq!(actual.len(), rows);
                for (&a, &b) in actual.iter().zip(&expected) {
                    assert!(
                        (f64::from(a) - b).abs() <= 2e-5 + 2e-6 * b.abs(),
                        "mode={mode:?},rows={rows},vocabulary={vocabulary},feed={feed}: {a}!={b}"
                    );
                }
            }
        }
    }
}
#[test]
fn forward_loss_rows_preserve_nonfinite_classes() {
    let Some(mut e) = executor(PtxReductionMode::Strict) else {
        return;
    };
    for value in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
        let g = forward_graph(3, 257);
        let actual = e
            .execute(
                &g,
                HashMap::from([
                    ("logits".into(), vec![value; 3 * 257]),
                    ("targets".into(), vec![0.0, 256.0, 3.0]),
                ]),
            )
            .unwrap();
        // RAW_CAPTURE_NONFINITE
        assert!(actual.iter().all(|v| v.is_nan()));
    }
}
