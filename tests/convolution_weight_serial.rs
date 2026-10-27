#![cfg(feature = "cuda")]

use num_traits::{Float, ToPrimitive};
use std::collections::HashMap;
use tnsr::{
    Executor,
    graph::{TensorGraph, TensorGraphNode},
    ptx::{BF16, CudaDType, F16, F32, PtxExecutor, PtxReductionMode},
    tensor::TensorExpr,
};

fn executor<D: CudaDType>() -> Option<PtxExecutor<D>> {
    match PtxExecutor::<D>::try_new_with_reduction_mode(PtxReductionMode::Strict) {
        Ok(value) => Some(value),
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

#[derive(Clone, Copy, Debug)]
struct Case {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    o: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
}

fn run<D: CudaDType>(g: Case)
where
    D::HostType: Float,
{
    let Some(mut executor) = executor::<D>() else {
        return;
    };
    let oh = (g.h + 2 * g.pad - g.kh) / g.stride + 1;
    let ow = (g.w + 2 * g.pad - g.kw) / g.stride + 1;
    let mut graph = TensorGraph::<D::HostType>::new();
    let x = graph.add_expr(&TensorExpr::input("x", vec![g.n, g.c, g.h, g.w]), false);
    let dy = graph.add_expr(&TensorExpr::input("dy", vec![g.n, g.o, oh, ow]), false);
    let output = graph.graph.add_node(TensorGraphNode::Conv2dBackwardWeight {
        shape: vec![g.o, g.c, g.kh, g.kw],
        stride: g.stride,
        padding: g.pad,
    });
    graph.graph.add_edge(x, output, 0);
    graph.graph.add_edge(dy, output, 1);
    for feed in 0..2 {
        let x: Vec<D::HostType> = (0..g.n * g.c * g.h * g.w)
            .map(|i| {
                <D::HostType as num_traits::NumCast>::from(
                    (((i * 7 + feed * 3) % 19) as f32 - 9.0) / 16.0,
                )
                .unwrap()
            })
            .collect();
        let dy: Vec<D::HostType> = (0..g.n * g.o * oh * ow)
            .map(|i| {
                <D::HostType as num_traits::NumCast>::from(
                    (((i * 11 + feed * 5) % 17) as f32 - 8.0) / 32.0,
                )
                .unwrap()
            })
            .collect();
        // Independent forward-coordinate scatter. Products/sums are exact
        // dyadics here, so final physical-dtype conversion is also testable.
        let mut expected = vec![0.0_f32; g.o * g.c * g.kh * g.kw];
        for n in 0..g.n {
            for o in 0..g.o {
                for y in 0..oh {
                    for col in 0..ow {
                        for c in 0..g.c {
                            for ky in 0..g.kh {
                                for kx in 0..g.kw {
                                    let iy = y * g.stride + ky;
                                    let ix = col * g.stride + kx;
                                    if iy < g.pad
                                        || ix < g.pad
                                        || iy - g.pad >= g.h
                                        || ix - g.pad >= g.w
                                    {
                                        continue;
                                    }
                                    let xi = ((n * g.c + c) * g.h + iy - g.pad) * g.w + ix - g.pad;
                                    let gi = ((n * g.o + o) * oh + y) * ow + col;
                                    expected[((o * g.c + c) * g.kh + ky) * g.kw + kx] +=
                                        x[xi].to_f32().unwrap() * dy[gi].to_f32().unwrap();
                                }
                            }
                        }
                    }
                }
            }
        }
        let feeds = HashMap::from([("x".into(), x), ("dy".into(), dy)]);
        let actual = if feed == 0 {
            executor.execute(&graph, feeds).unwrap()
        } else {
            executor.execute_compiled(&graph, feeds).unwrap()
        };
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            let expected = <D::HostType as num_traits::NumCast>::from(e)
                .unwrap()
                .to_f32()
                .unwrap();
            assert_eq!(
                a.to_f32().unwrap().to_bits(),
                expected.to_bits(),
                "{g:?},feed={feed},weight={i}"
            );
        }
    }
}

#[test]
fn serial_weight_gradient_rectangular_strided_padding_changed_feeds() {
    let cases = [
        Case {
            n: 3,
            c: 2,
            h: 7,
            w: 9,
            o: 3,
            kh: 3,
            kw: 2,
            stride: 2,
            pad: 2,
        },
        Case {
            n: 2,
            c: 3,
            h: 4,
            w: 7,
            o: 2,
            kh: 1,
            kw: 1,
            stride: 1,
            pad: 0,
        },
        Case {
            n: 1,
            c: 3,
            h: 13,
            w: 13,
            o: 17,
            kh: 3,
            kw: 3,
            stride: 1,
            pad: 1,
        },
        Case {
            n: 1,
            c: 1,
            h: 1,
            w: 1,
            o: 1,
            kh: 3,
            kw: 3,
            stride: 1,
            pad: 2,
        },
    ];
    for g in cases {
        run::<F32>(g);
        run::<F16>(g);
        run::<BF16>(g);
    }
}

#[test]
fn serial_weight_gradient_invalid_nonfinite_terms_are_skipped() {
    let Some(mut executor) = executor::<F32>() else {
        return;
    };
    for sentinel in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut graph = TensorGraph::<f32>::new();
        let x = graph.add_expr(&TensorExpr::input("x", vec![1, 1, 1, 1]), false);
        let dy = graph.add_expr(&TensorExpr::input("dy", vec![1, 1, 1, 1]), false);
        let out = graph.graph.add_node(TensorGraphNode::Conv2dBackwardWeight {
            shape: vec![1, 1, 3, 3],
            stride: 1,
            padding: 1,
        });
        graph.graph.add_edge(x, out, 0);
        graph.graph.add_edge(dy, out, 1);
        let actual = executor
            .execute(
                &graph,
                HashMap::from([("x".into(), vec![2.0]), ("dy".into(), vec![sentinel])]),
            )
            .unwrap();
        for (i, &a) in actual.iter().enumerate() {
            if i == 4 {
                if sentinel.is_nan() {
                    assert!(a.is_nan())
                } else {
                    assert_eq!(a, sentinel)
                }
            } else {
                assert_eq!(a.to_bits(), 0.0_f32.to_bits(), "invalid term at{i}")
            }
        }
    }
}

#[test]
fn serial_weight_gradient_keeps_batch_row_column_cancellation_order() {
    let Some(mut executor) = executor::<F32>() else {
        return;
    };
    let x = vec![
        16_777_216.0,
        1.0,
        -16_777_216.0,
        1.0,
        3.0,
        -3.0,
        0.5,
        0.5,
        16_777_216.0,
        1.0,
        -16_777_216.0,
        1.0,
        2.0,
        -2.0,
        0.25,
        0.25,
    ];
    let expected = x.iter().copied().fold(0.0_f32, |sum, v| sum + v);
    let mut graph = TensorGraph::<f32>::new();
    let source = graph.add_expr(&TensorExpr::constant(x, vec![2, 1, 2, 4]), false);
    let dy = graph.add_expr(
        &TensorExpr::constant(vec![1.0; 16], vec![2, 1, 2, 4]),
        false,
    );
    let out = graph.graph.add_node(TensorGraphNode::Conv2dBackwardWeight {
        shape: vec![1, 1, 1, 1],
        stride: 1,
        padding: 0,
    });
    graph.graph.add_edge(source, out, 0);
    graph.graph.add_edge(dy, out, 1);
    assert_eq!(
        executor.execute(&graph, HashMap::new()).unwrap(),
        vec![expected]
    );
}
