#![cfg(feature = "cuda")]
use std::collections::HashMap;
use tnsr::{
    Executor, SimpleExecutor,
    graph::{TensorGraph, TensorGraphNode},
    ptx::{PtxExecutor, PtxReductionMode},
    tensor::TensorExpr,
};

fn executor(mode: PtxReductionMode) -> Option<PtxExecutor> {
    match PtxExecutor::try_new_with_reduction_mode(mode) {
        Ok(value) => Some(value),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            eprintln!("skipping CUDA: {error:#}");
            None
        }
        Err(error) => panic!("{error:#}"),
    }
}

#[test]
fn cooperative_convolution_weight_gradients_match_host_and_repeat_exactly() {
    let Some(mut tree) = executor(PtxReductionMode::DeterministicTree) else {
        return;
    };
    let Some(mut serial) = executor(PtxReductionMode::Strict) else {
        return;
    };
    // Tail reductions, padded/strided coordinates, both LeNet layers and the
    // output-parallel fallback. Nonuniform dy catches batch/channel mixups.
    let cases = [
        (1, 1, 28, 28, 6, 5, 5, 1, 2),
        (8, 1, 28, 28, 6, 5, 5, 1, 2),
        (64, 1, 28, 28, 6, 5, 5, 1, 2),
        (256, 1, 28, 28, 6, 5, 5, 1, 2),
        (1, 6, 14, 14, 16, 5, 5, 1, 0),
        (8, 6, 14, 14, 16, 5, 5, 1, 0),
        (64, 6, 14, 14, 16, 5, 5, 1, 0),
        (3, 2, 19, 23, 5, 3, 2, 2, 2),
        (1, 17, 19, 19, 17, 4, 4, 1, 0),
    ];
    for (batch, channels, height, width, outputs, kh, kw, stride, padding) in cases {
        let rows = (height + 2 * padding - kh) / stride + 1;
        let columns = (width + 2 * padding - kw) / stride + 1;
        let x: Vec<f32> = (0..batch * channels * height * width)
            .map(|i| (f32::from(u8::try_from((i * 7 + i / 13) % 19).unwrap()) - 9.0) / 32.0)
            .collect();
        let dy: Vec<f32> = (0..batch * outputs * rows * columns)
            .map(|i| (f32::from(u8::try_from((i * 11 + i / 7) % 17).unwrap()) - 8.0) / 64.0)
            .collect();
        let mut graph = TensorGraph::<f32>::new();
        let input = graph.add_expr(
            &TensorExpr::constant(x, vec![batch, channels, height, width]),
            false,
        );
        let upstream = graph.add_expr(
            &TensorExpr::constant(dy, vec![batch, outputs, rows, columns]),
            false,
        );
        let dw = graph.graph.add_node(TensorGraphNode::Conv2dBackwardWeight {
            shape: vec![outputs, channels, kh, kw],
            stride,
            padding,
        });
        graph.graph.add_edge(input, dw, 0);
        graph.graph.add_edge(upstream, dw, 1);
        let expected = SimpleExecutor::new()
            .execute(&graph, HashMap::new())
            .unwrap();
        let strict = serial.execute(&graph, HashMap::new()).unwrap();
        let actual = tree.execute(&graph, HashMap::new()).unwrap();
        assert_eq!(strict, expected, "strict order changed");
        for (index, (&value, &reference)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (value - reference).abs() <= 3e-6 + 3e-3 * reference.abs(),
                "batch={batch} channels={channels} weight={index}: {value} != {reference}"
            );
        }
        let repeated = tree.execute_compiled(&graph, HashMap::new()).unwrap();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            repeated.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn convolution_weight_reduction_policy_is_observable_and_repeatable() {
    let Some(mut serial) = executor(PtxReductionMode::Strict) else {
        return;
    };
    let Some(mut tree) = executor(PtxReductionMode::DeterministicTree) else {
        return;
    };
    let mut values = vec![1.0_f32; 256];
    values[0] = 16_777_216.0;
    values[128] = -16_777_216.0;
    let mut graph = TensorGraph::<f32>::new();
    let x = graph.add_expr(&TensorExpr::constant(values, vec![1, 1, 1, 256]), false);
    let dy = graph.add_expr(
        &TensorExpr::constant(vec![1.0; 256], vec![1, 1, 1, 256]),
        false,
    );
    let dw = graph.graph.add_node(TensorGraphNode::Conv2dBackwardWeight {
        shape: vec![1, 1, 1, 1],
        stride: 1,
        padding: 0,
    });
    graph.graph.add_edge(x, dw, 0);
    graph.graph.add_edge(dy, dw, 1);
    // These are deliberately different numerical policies: an increasing-index
    // f32 sum loses the first 127 unit terms, while the fixed tree retains them.
    assert_eq!(serial.execute(&graph, HashMap::new()).unwrap(), vec![127.0]);
    for mode in [
        PtxReductionMode::DeterministicTree,
        PtxReductionMode::Online,
    ] {
        let Some(mut executor) = executor(mode) else {
            return;
        };
        assert_eq!(
            executor.execute(&graph, HashMap::new()).unwrap(),
            vec![254.0]
        );
    }
    assert_eq!(tree.execute(&graph, HashMap::new()).unwrap(), vec![254.0]);
    assert_eq!(
        tree.execute_compiled(&graph, HashMap::new()).unwrap(),
        vec![254.0]
    );
}

#[test]
fn convolution_weight_tree_matches_independent_wide_accumulation() {
    let Some(mut tree) = executor(PtxReductionMode::DeterministicTree) else {
        return;
    };
    let (batch, channels, height, width, outputs, kh, kw, stride, padding) =
        (3, 2, 19, 23, 5, 3, 2, 2, 2);
    let rows = (height + 2 * padding - kh) / stride + 1;
    let columns = (width + 2 * padding - kw) / stride + 1;
    let x: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| {
            (f32::from(u8::try_from((i * 7 + i / 13) % 29).unwrap()) - 14.0) / 37.0
                * [0.001, 1.0, 100.0][i % 3]
        })
        .collect();
    let dy: Vec<f32> = (0..batch * outputs * rows * columns)
        .map(|i| (f32::from(u8::try_from((i * 11 + i / 7) % 23).unwrap()) - 11.0) / 53.0)
        .collect();
    let mut expected = vec![0.0_f64; outputs * channels * kh * kw];
    for oc in 0..outputs {
        for ic in 0..channels {
            for ky in 0..kh {
                for kx in 0..kw {
                    let weight = ((oc * channels + ic) * kh + ky) * kw + kx;
                    for n in 0..batch {
                        for oy in 0..rows {
                            for ox in 0..columns {
                                let iy = oy * stride + ky;
                                let ix = ox * stride + kx;
                                if iy < padding
                                    || ix < padding
                                    || iy - padding >= height
                                    || ix - padding >= width
                                {
                                    continue;
                                }
                                let input = ((n * channels + ic) * height + iy - padding) * width
                                    + ix
                                    - padding;
                                let upstream = ((n * outputs + oc) * rows + oy) * columns + ox;
                                // Match rounded f32 products; accumulate independently in f64.
                                expected[weight] += f64::from(x[input] * dy[upstream]);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut graph = TensorGraph::<f32>::new();
    let input = graph.add_expr(
        &TensorExpr::constant(x, vec![batch, channels, height, width]),
        false,
    );
    let upstream = graph.add_expr(
        &TensorExpr::constant(dy, vec![batch, outputs, rows, columns]),
        false,
    );
    let dw = graph.graph.add_node(TensorGraphNode::Conv2dBackwardWeight {
        shape: vec![outputs, channels, kh, kw],
        stride,
        padding,
    });
    graph.graph.add_edge(input, dw, 0);
    graph.graph.add_edge(upstream, dw, 1);
    let actual = tree.execute(&graph, HashMap::new()).unwrap();
    for (index, (&value, &reference)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (f64::from(value) - reference).abs() <= 3e-6 + 3e-3 * reference.abs(),
            "weight {index}: {value} != {reference}"
        );
    }
}
