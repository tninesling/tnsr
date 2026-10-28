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

fn graph(rows: usize, vocabulary: usize) -> TensorGraph<f32> {
    let mut g = TensorGraph::new();
    let x = g.add_expr(&TensorExpr::input("logits", vec![rows, vocabulary]), false);
    let t = g.add_expr(&TensorExpr::input("targets", vec![rows]), false);
    let dy = g.add_expr(&TensorExpr::input("upstream", vec![rows]), false);
    let out = g
        .graph
        .add_node(TensorGraphNode::IndexedCrossEntropyBackward {
            shape: vec![rows, vocabulary],
        });
    for (slot, n) in [x, t, dy].into_iter().enumerate() {
        g.graph.add_edge(n, out, slot);
    }
    g
}

#[test]
fn ordered_loss_rows_match_independent_reference_and_changed_feeds() {
    for mode in [
        PtxReductionMode::Strict,
        PtxReductionMode::DeterministicTree,
        PtxReductionMode::Online,
    ] {
        let Some(mut e) = executor(mode) else { return };
        for (rows, vocabulary) in [(1, 255), (3, 256), (5, 257), (3, 511), (2, 1025), (7, 4096)] {
            let g = graph(rows, vocabulary);
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
                let upstream: Vec<f32> = (0..rows)
                    .map(|r| [0.0, 1.0, -0.375][(r + feed) % 3])
                    .collect();
                let expected: Vec<f32> = (0..rows)
                    .flat_map(|r| {
                        let x = &logits[r * vocabulary..(r + 1) * vocabulary];
                        let max = x
                            .iter()
                            .map(|v| f64::from(*v))
                            .fold(f64::NEG_INFINITY, f64::max);
                        let sum = x.iter().map(|v| (f64::from(*v) - max).exp()).sum::<f64>();
                        (0..vocabulary)
                            .map(|c| {
                                (((f64::from(x[c]) - max).exp() / sum
                                    - f64::from(u8::from(c == targets[r] as usize)))
                                    * f64::from(upstream[r])) as f32
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let feeds = HashMap::from([
                    ("logits".into(), logits),
                    ("targets".into(), targets),
                    ("upstream".into(), upstream),
                ]);
                let actual = if feed == 0 {
                    e.execute(&g, feeds).unwrap()
                } else {
                    e.execute_compiled(&g, feeds).unwrap()
                };
                for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - b).abs() <= 3e-6 + 3e-3 * b.abs(),
                        "mode={mode:?},rows={rows},vocab={vocabulary},feed={feed},index={i}: {a} != {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn ordered_loss_rows_preserve_nonfinite_classes_and_zero_upstream() {
    let Some(mut e) = executor(PtxReductionMode::Strict) else {
        return;
    };
    for value in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
        let g = graph(3, 257);
        let actual = e
            .execute(
                &g,
                HashMap::from([
                    ("logits".into(), vec![value; 3 * 257]),
                    ("targets".into(), vec![0.0, 256.0, 3.0]),
                    ("upstream".into(), vec![0.0, 1.0, -1.0]),
                ]),
            )
            .unwrap();
        assert!(actual.iter().all(|v| v.is_nan()));
    }
}
