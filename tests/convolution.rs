use std::collections::HashMap;

use tnsr::graph::TensorGraph;
use tnsr::tensor::{Parameter, TensorExpr};
use tnsr::{Executor, Runtime};

const EPSILON: f32 = 1e-4;

fn create_runtime() -> Runtime {
    Runtime::new()
}

/// Naive reference conv2d on the host, used to cross-check the backend.
#[allow(clippy::too_many_arguments)]
fn reference_conv2d(
    input: &[f32],
    in_shape: &[usize],
    weight: &[f32],
    w_shape: &[usize],
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let n = in_shape[0];
    let c_in = in_shape[1];
    let h = in_shape[2];
    let w = in_shape[3];
    let c_out = w_shape[0];
    let kh = w_shape[2];
    let kw = w_shape[3];
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;

    let mut out = vec![0.0f32; n * c_out * h_out * w_out];
    for on in 0..n {
        for oc in 0..c_out {
            for ohi in 0..h_out {
                for owi in 0..w_out {
                    let mut sum = 0.0f32;
                    for ic in 0..c_in {
                        for khi in 0..kh {
                            let ih_raw = ohi * stride + khi;
                            if ih_raw >= padding && ih_raw - padding < h {
                                let ih = ih_raw - padding;
                                for kwi in 0..kw {
                                    let iw_raw = owi * stride + kwi;
                                    if iw_raw >= padding && iw_raw - padding < w {
                                        let iw = iw_raw - padding;
                                        let in_val = input[((on * c_in + ic) * h + ih) * w + iw];
                                        let w_val =
                                            weight[((oc * c_in + ic) * kh + khi) * kw + kwi];
                                        sum += in_val * w_val;
                                    }
                                }
                            }
                        }
                    }
                    out[((on * c_out + oc) * h_out + ohi) * w_out + owi] = sum;
                }
            }
        }
    }
    out
}

fn run_conv2d(
    input: &[f32],
    in_shape: Vec<usize>,
    weight: &[f32],
    w_shape: Vec<usize>,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let mut runtime = create_runtime();
    let x = Parameter::new(input.to_vec(), in_shape);
    let w = Parameter::new(weight.to_vec(), w_shape);
    let node = TensorExpr::from(x).conv2d(w, stride, padding);
    let graph: TensorGraph<f32> = node.into();
    runtime.execute(&graph, HashMap::new()).unwrap()
}

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() < EPSILON,
            "value mismatch: {a:?} vs {b:?} (diff {})",
            (x - y).abs()
        );
    }
}

#[test]
fn conv2d_forward_simple() {
    // 1x1x3x3 input, identity-diagonal 2x2 kernel, stride 1, no padding.
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight = vec![1.0, 0.0, 0.0, 1.0];
    let out = run_conv2d(&input, vec![1, 1, 3, 3], &weight, vec![1, 1, 2, 2], 1, 0);
    assert_close(&out, &[6.0, 8.0, 12.0, 14.0]);
}

#[test]
fn conv2d_forward_with_padding() {
    // 1x1x2x2 input, all-ones 2x2 kernel, stride 1, padding 1 -> 3x3 output.
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let out = run_conv2d(&input, vec![1, 1, 2, 2], &weight, vec![1, 1, 2, 2], 1, 1);
    assert_close(&out, &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]);
}

#[test]
fn conv2d_forward_stride_2() {
    // 1x1x4x4 input, all-ones 2x2 kernel, stride 2, no padding -> 2x2 output.
    let input: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let out = run_conv2d(&input, vec![1, 1, 4, 4], &weight, vec![1, 1, 2, 2], 2, 0);
    assert_close(&out, &[14.0, 22.0, 46.0, 54.0]);
}

#[test]
fn conv2d_forward_multi_channel() {
    // 1x2x3x3 input, 2 output channels each 2x2 over 2 input channels.
    let input: Vec<f32> = (0..18).map(|v| (v as f32) * 0.5).collect();
    let weight: Vec<f32> = (0..16).map(|v| (v as f32) * 0.1 - 0.7).collect();
    let in_shape = vec![1, 2, 3, 3];
    let w_shape = vec![2, 2, 2, 2];
    let out = run_conv2d(&input, in_shape.clone(), &weight, w_shape.clone(), 1, 0);
    let expected = reference_conv2d(&input, &in_shape, &weight, &w_shape, 1, 0);
    assert_close(&out, &expected);
}

#[test]
fn conv2d_forward_batch() {
    // Batch of 2 samples, single channel.
    let input: Vec<f32> = (0..32).map(|v| (v as f32) * 0.25 - 3.0).collect();
    let weight: Vec<f32> = vec![0.2, -0.3, 0.5, 0.1];
    let in_shape = vec![2, 1, 4, 4];
    let w_shape = vec![1, 1, 2, 2];
    let out = run_conv2d(&input, in_shape.clone(), &weight, w_shape.clone(), 1, 0);
    let expected = reference_conv2d(&input, &in_shape, &weight, &w_shape, 1, 0);
    assert_close(&out, &expected);
}

#[test]
fn conv2d_shape_computation() {
    // Verify output shape [N, C_out, H_out, W_out] via total element count.
    // 2x3x8x8 input, 5 output channels, 3x3 kernel, stride 2, padding 1.
    // H_out = (8 + 2 - 3)/2 + 1 = 4, W_out = 4 -> 2*5*4*4 = 160.
    let input: Vec<f32> = (0..(2 * 3 * 8 * 8)).map(|v| (v as f32) * 0.01).collect();
    let weight: Vec<f32> = (0..(5 * 3 * 3 * 3)).map(|v| (v as f32) * 0.01).collect();
    let out = run_conv2d(&input, vec![2, 3, 8, 8], &weight, vec![5, 3, 3, 3], 2, 1);
    assert_eq!(out.len(), 2 * 5 * 4 * 4);
}

#[test]
fn conv2d_with_bias_forward() {
    // conv2d helper with a per-output-channel bias broadcast over the map.
    let mut runtime = create_runtime();
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight = vec![1.0, 0.0, 0.0, 1.0];
    let bias = Parameter::new(vec![10.0f32], vec![1, 1, 1, 1]);
    let x = Parameter::new(input, vec![1, 1, 3, 3]);
    let w = Parameter::new(weight, vec![1, 1, 2, 2]);
    let node = tnsr::nn::conv2d(x, w, 1, 0, Some(bias));
    let graph: TensorGraph<f32> = node.into();
    let out = runtime.execute(&graph, HashMap::new()).unwrap();
    assert_close(&out, &[16.0, 18.0, 22.0, 24.0]);
}

#[test]
fn conv2d_gradient_wrt_weight() {
    let input: Vec<f32> = (0..16).map(|v| (v as f32) * 0.1 + 0.05).collect();
    let weight: Vec<f32> = vec![0.3, -0.2, 0.5, 0.1];
    let in_shape = vec![1, 1, 4, 4];
    let w_shape = vec![1, 1, 2, 2];

    // Analytic gradient of sum(conv) w.r.t. weight.
    let mut runtime = create_runtime();
    let x = Parameter::new(input.clone(), in_shape.clone());
    let w = Parameter::new(weight.clone(), w_shape.clone());
    let w_id = w.id();
    let conv = TensorExpr::from(x).conv2d(w, 1, 0);
    let loss = conv.flatten().reduce_sum(1);
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);
    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);
    let analytic = grads.get(&w_id).expect("weight gradient missing");

    // Numerical gradient via central differences.
    let eps = 1e-3f32;
    for i in 0..weight.len() {
        let mut wp = weight.clone();
        wp[i] += eps;
        let mut wm = weight.clone();
        wm[i] -= eps;
        let lp: f32 = run_conv2d(&input, in_shape.clone(), &wp, w_shape.clone(), 1, 0)
            .iter()
            .sum();
        let lm: f32 = run_conv2d(&input, in_shape.clone(), &wm, w_shape.clone(), 1, 0)
            .iter()
            .sum();
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic[i] - num).abs() < 1e-2,
            "weight grad[{i}] mismatch: analytic {} vs numeric {}",
            analytic[i],
            num
        );
    }
}

#[test]
fn conv2d_gradient_wrt_input() {
    let input: Vec<f32> = (0..16).map(|v| (v as f32) * 0.1 + 0.05).collect();
    let weight: Vec<f32> = vec![0.3, -0.2, 0.5, 0.1];
    let in_shape = vec![1, 1, 4, 4];
    let w_shape = vec![1, 1, 2, 2];

    let mut runtime = create_runtime();
    let x = Parameter::new(input.clone(), in_shape.clone());
    let x_id = x.id();
    let w = Parameter::new(weight.clone(), w_shape.clone());
    let conv = TensorExpr::from(x).conv2d(w, 1, 0);
    let loss = conv.flatten().reduce_sum(1);
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);
    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);
    let analytic = grads.get(&x_id).expect("input gradient missing");

    let eps = 1e-3f32;
    for i in 0..input.len() {
        let mut ip = input.clone();
        ip[i] += eps;
        let mut im = input.clone();
        im[i] -= eps;
        let lp: f32 = run_conv2d(&ip, in_shape.clone(), &weight, w_shape.clone(), 1, 0)
            .iter()
            .sum();
        let lm: f32 = run_conv2d(&im, in_shape.clone(), &weight, w_shape.clone(), 1, 0)
            .iter()
            .sum();
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic[i] - num).abs() < 1e-2,
            "input grad[{i}] mismatch: analytic {} vs numeric {}",
            analytic[i],
            num
        );
    }
}

#[test]
fn max_pool2d_forward_simple() {
    // 1x1x4x4 input, 2x2 pool, stride 2 -> 2x2 output of window maxima.
    let mut runtime = create_runtime();
    let input: Vec<f32> = vec![
        1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let x = Parameter::new(input, vec![1, 1, 4, 4]);
    let node = TensorExpr::from(x).max_pool2d(2, 2);
    let graph: TensorGraph<f32> = node.into();
    let out = runtime.execute(&graph, HashMap::new()).unwrap();
    assert_close(&out, &[6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn max_pool2d_gradient() {
    // Gradient of sum(maxpool) routes 1.0 to each window's max element.
    let input: Vec<f32> = vec![
        1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let mut runtime = create_runtime();
    let x = Parameter::new(input, vec![1, 1, 4, 4]);
    let x_id = x.id();
    let pooled = TensorExpr::from(x).max_pool2d(2, 2);
    let loss = pooled.flatten().reduce_sum(1);
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);
    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);
    let g = grads.get(&x_id).expect("input gradient missing");
    // Maxima are at indices 5 (6), 7 (8), 13 (14), 15 (16).
    let expected = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    ];
    assert_close(g, &expected);
}

#[test]
fn flatten_forward_preserves_data() {
    let mut runtime = create_runtime();
    let input: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = Parameter::new(input.clone(), vec![2, 3, 2, 2]);
    let node = TensorExpr::from(x).flatten();
    assert_eq!(node.shape(), &vec![2, 12]);
    let graph: TensorGraph<f32> = node.into();
    let out = runtime.execute(&graph, HashMap::new()).unwrap();
    assert_close(&out, &input);
}
