#![cfg(feature = "cuda")]

use num_traits::Float;
use std::collections::HashMap;
use tnsr::{
    graph::{TensorGraph, TensorGraphNode},
    ptx::{
        PtxExecutor, PtxReductionMode,
        types::{BF16, CudaDType, F16, F32},
    },
    tensor::TensorExpr,
};

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

impl Case {
    fn rows(self) -> usize {
        (self.h + 2 * self.pad - self.kh) / self.stride + 1
    }
    fn cols(self) -> usize {
        (self.w + 2 * self.pad - self.kw) / self.stride + 1
    }
    fn input_shape(self) -> Vec<usize> {
        vec![self.n, self.c, self.h, self.w]
    }
    fn output_shape(self) -> Vec<usize> {
        vec![self.n, self.o, self.rows(), self.cols()]
    }
    fn weight_shape(self) -> Vec<usize> {
        vec![self.o, self.c, self.kh, self.kw]
    }
}

fn executor<D: CudaDType>(mode: PtxReductionMode) -> Option<PtxExecutor<D>> {
    match PtxExecutor::<D>::try_new_with_reduction_mode(mode) {
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

fn graph<D: CudaDType>(case: Case, transpose: bool) -> TensorGraph<D::HostType> {
    let mut graph = TensorGraph::<D::HostType>::new();
    let source_shape = if transpose {
        case.output_shape()
    } else {
        case.input_shape()
    };
    let source = graph.add_expr(&TensorExpr::input("source", source_shape), false);
    let weight = graph.add_expr(&TensorExpr::input("weight", case.weight_shape()), false);
    let operation = if transpose {
        TensorGraphNode::ConvTranspose2d {
            shape: case.input_shape(),
            stride: case.stride,
            padding: case.pad,
        }
    } else {
        TensorGraphNode::Conv2d {
            shape: case.output_shape(),
            stride: case.stride,
            padding: case.pad,
        }
    };
    let output = graph.graph.add_node(operation);
    graph.graph.add_edge(source, output, 0);
    graph.graph.add_edge(weight, output, 1);
    graph
}

fn to_host<H: Float>(values: &[f32]) -> Vec<H> {
    values.iter().map(|&v| H::from(v).unwrap()).collect()
}

fn to_f32<H: Float>(values: &[H]) -> Vec<f32> {
    values.iter().map(|v| v.to_f32().unwrap()).collect()
}

fn bits<H: Float>(values: &[H]) -> Vec<u32> {
    to_f32(values).iter().map(|v| v.to_bits()).collect()
}

// Independent scalar references, with explicit logical coordinates. Forward
// sums ic->ky->kx; dX sums oc->ky->kx, matching the existing serial policy.
// This is not a claim about PTXAS multiply/add contraction. Finite comparisons
// retain the existing gradient bound; exactness is required only for replay.
fn reference(case: Case, transpose: bool, source: &[f32], weight: &[f32]) -> Vec<f32> {
    let rows = case.rows();
    let cols = case.cols();
    if !transpose {
        let mut out = vec![0.0_f32; case.n * case.o * rows * cols];
        for n in 0..case.n {
            for oc in 0..case.o {
                for oy in 0..rows {
                    for ox in 0..cols {
                        let mut sum = 0.0_f32;
                        for ic in 0..case.c {
                            for ky in 0..case.kh {
                                for kx in 0..case.kw {
                                    let iy = oy * case.stride + ky;
                                    let ix = ox * case.stride + kx;
                                    if iy < case.pad || ix < case.pad {
                                        continue;
                                    }
                                    let (iy, ix) = (iy - case.pad, ix - case.pad);
                                    if iy >= case.h || ix >= case.w {
                                        continue;
                                    }
                                    let x = ((n * case.c + ic) * case.h + iy) * case.w + ix;
                                    let w = ((oc * case.c + ic) * case.kh + ky) * case.kw + kx;
                                    sum += source[x] * weight[w];
                                }
                            }
                        }
                        out[((n * case.o + oc) * rows + oy) * cols + ox] = sum;
                    }
                }
            }
        }
        out
    } else {
        let mut out = vec![0.0_f32; case.n * case.c * case.h * case.w];
        for n in 0..case.n {
            for ic in 0..case.c {
                for iy in 0..case.h {
                    for ix in 0..case.w {
                        let mut sum = 0.0_f32;
                        for oc in 0..case.o {
                            for ky in 0..case.kh {
                                for kx in 0..case.kw {
                                    // Signed arithmetic expresses the inverse map without
                                    // reproducing the PTX unsigned branch sequence.
                                    let ny = (iy + case.pad) as isize - ky as isize;
                                    let nx = (ix + case.pad) as isize - kx as isize;
                                    let stride = case.stride as isize;
                                    if ny < 0 || nx < 0 || ny % stride != 0 || nx % stride != 0 {
                                        continue;
                                    }
                                    let (oy, ox) = ((ny / stride) as usize, (nx / stride) as usize);
                                    if oy >= rows || ox >= cols {
                                        continue;
                                    }
                                    let dy = ((n * case.o + oc) * rows + oy) * cols + ox;
                                    let w = ((oc * case.c + ic) * case.kh + ky) * case.kw + kx;
                                    sum += source[dy] * weight[w];
                                }
                            }
                        }
                        out[((n * case.c + ic) * case.h + iy) * case.w + ix] = sum;
                    }
                }
            }
        }
        out
    }
}

fn assert_finite_close<H: Float>(actual: &[H], expected: &[H], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (i, (&actual, &expected)) in to_f32(actual).iter().zip(&to_f32(expected)).enumerate() {
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= 3e-6 + 3e-3 * expected.abs(),
            "{context} element {i}: {actual} != {expected}"
        );
    }
}

fn check_case<D: CudaDType>(executor: &mut PtxExecutor<D>, case: Case, transpose: bool) {
    let source_len = if transpose {
        case.output_shape().iter().product()
    } else {
        case.input_shape().iter().product()
    };
    let source: Vec<f32> = (0..source_len)
        .map(|i| ((i * 7 + i / 11) % 23) as f32 / 32.0 - 11.0 / 32.0)
        .collect();
    let weight: Vec<f32> = (0..case.weight_shape().iter().product())
        .map(|i| ((i * 11 + i / 7) % 19) as f32 / 64.0 - 9.0 / 64.0)
        .collect();
    let graph = graph::<D>(case, transpose);
    let before = executor.compilation_count();
    executor.compile_owned(graph.clone()).unwrap();
    assert_eq!(executor.compilation_count(), before + 1);
    let mut original = None;
    for changed in [false, true] {
        // Both named feeds change; the second result is -0.5 times the first
        // before output quantization, catching stale source and weight feeds.
        let source: Vec<_> = source
            .iter()
            .map(|v| if changed { -*v } else { *v })
            .collect();
        let weight: Vec<_> = weight
            .iter()
            .map(|v| if changed { *v * 0.5 } else { *v })
            .collect();
        let source = to_host::<D::HostType>(&source);
        let weight = to_host::<D::HostType>(&weight);
        let expected = reference(case, transpose, &to_f32(&source), &to_f32(&weight));
        let expected = to_host::<D::HostType>(&expected);
        let feeds = HashMap::from([("source".into(), source), ("weight".into(), weight)]);
        let actual = executor.execute_compiled(&graph, feeds.clone()).unwrap();
        assert_finite_close(
            &actual,
            &expected,
            &format!("{case:?} transpose={transpose} changed={changed}"),
        );
        let repeat = executor.execute_compiled(&graph, feeds).unwrap();
        assert_eq!(bits(&actual), bits(&repeat), "replay changed {case:?}");
        if changed {
            assert_ne!(
                original.as_ref().unwrap(),
                &bits(&actual),
                "changed feeds were ignored {case:?}"
            );
        } else {
            original = Some(bits(&actual));
        }
        assert_eq!(executor.compilation_count(), before + 1);
    }
}

const CASES: [Case; 7] = [
    // Multibatch, rectangular kernels, stride divisibility and block tails.
    Case {
        n: 2,
        c: 3,
        h: 7,
        w: 9,
        o: 5,
        kh: 3,
        kw: 2,
        stride: 2,
        pad: 1,
    },
    Case {
        n: 2,
        c: 2,
        h: 9,
        w: 13,
        o: 3,
        kh: 2,
        kw: 3,
        stride: 4,
        pad: 2,
    },
    // Actual LeNet layer geometries, including area25 unroll boundary.
    Case {
        n: 1,
        c: 1,
        h: 28,
        w: 28,
        o: 6,
        kh: 5,
        kw: 5,
        stride: 1,
        pad: 2,
    },
    Case {
        n: 1,
        c: 6,
        h: 14,
        w: 14,
        o: 16,
        kh: 5,
        kw: 5,
        stride: 1,
        pad: 0,
    },
    Case {
        n: 1,
        c: 3,
        h: 5,
        w: 7,
        o: 2,
        kh: 1,
        kw: 1,
        stride: 2,
        pad: 0,
    },
    // Exactly25, and area26 rectangular fallback.
    Case {
        n: 1,
        c: 2,
        h: 8,
        w: 11,
        o: 3,
        kh: 5,
        kw: 5,
        stride: 1,
        pad: 2,
    },
    Case {
        n: 1,
        c: 2,
        h: 9,
        w: 10,
        o: 3,
        kh: 2,
        kw: 13,
        stride: 1,
        pad: 6,
    },
];

fn check_finite_suite<D: CudaDType>(cases: &[Case]) {
    for mode in [
        PtxReductionMode::Strict,
        PtxReductionMode::DeterministicTree,
    ] {
        let Some(mut executor) = executor::<D>(mode) else {
            return;
        };
        for &case in cases {
            for transpose in [false, true] {
                check_case(&mut executor, case, transpose);
            }
        }
    }
}

#[test]
fn spatial_convolution_f32_ordered_reference_and_changed_feeds() {
    check_finite_suite::<F32>(&CASES);
}

#[test]
fn spatial_convolution_f16_storage_and_changed_feeds() {
    // Dyadic fixtures are exactly representable on input; the reference rounds
    // once on output, matching f32 accumulation with physical half storage.
    check_finite_suite::<F16>(&[CASES[0], CASES[4], CASES[6]]);
}

#[test]
fn spatial_convolution_bf16_storage_and_changed_feeds() {
    check_finite_suite::<BF16>(&[CASES[1], CASES[4], CASES[5]]);
}

fn check_invalid_taps<D: CudaDType>() {
    let case = Case {
        n: 1,
        c: 1,
        h: 1,
        w: 1,
        o: 1,
        kh: 3,
        kw: 3,
        stride: 1,
        pad: 1,
    };
    let Some(mut executor) = executor::<D>(PtxReductionMode::Strict) else {
        return;
    };
    for transpose in [false, true] {
        let graph = graph::<D>(case, transpose);
        executor.compile_owned(graph.clone()).unwrap();
        for sentinel in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut weight = vec![sentinel; 9];
            weight[4] = 3.0;
            let feeds = HashMap::from([
                ("source".into(), to_host::<D::HostType>(&[2.0])),
                ("weight".into(), to_host::<D::HostType>(&weight)),
            ]);
            let actual = executor.execute_compiled(&graph, feeds.clone()).unwrap();
            assert_eq!(
                to_f32(&actual),
                vec![6.0],
                "invalid nonfinite tap was evaluated, transpose={transpose}"
            );
            assert_eq!(
                bits(&actual),
                bits(&executor.execute_compiled(&graph, feeds).unwrap())
            );
        }
        // A valid NaN must still propagate: the implementation cannot simply
        // discard all nonfinite products while handling padding.
        let feeds = HashMap::from([
            ("source".into(), to_host::<D::HostType>(&[2.0])),
            ("weight".into(), to_host::<D::HostType>(&[f32::NAN; 9])),
        ]);
        let actual = executor.execute_compiled(&graph, feeds).unwrap();
        assert!(to_f32(&actual)[0].is_nan());
    }
}

#[test]
fn spatial_convolution_invalid_nonfinite_taps_are_skipped() {
    check_invalid_taps::<F32>();
    check_invalid_taps::<F16>();
    check_invalid_taps::<BF16>();
}

#[test]
fn spatial_convolution_cancellation_repeats_across_reduction_modes() {
    // This fixture exercises cancellation but intentionally has no rigid
    // analytic f32 answer: PTXAS may contract plain mul.f32/add.f32 differently.
    // Exact candidate-vs-frozen-baseline comparison belongs in the private
    // experiment; this test checks replay and reduction-policy invariance.
    let case = Case {
        n: 1,
        c: 4,
        h: 1,
        w: 1,
        o: 4,
        kh: 1,
        kw: 1,
        stride: 1,
        pad: 0,
    };
    for transpose in [false, true] {
        let graph = graph::<F32>(case, transpose);
        let feeds = HashMap::from([
            ("source".into(), vec![16_777_216.0, 1.0, -16_777_216.0, 1.0]),
            ("weight".into(), vec![1.0; 16]),
        ]);
        let mut strict_bits = None;
        for mode in [
            PtxReductionMode::Strict,
            PtxReductionMode::DeterministicTree,
        ] {
            let Some(mut executor) = executor::<F32>(mode) else {
                return;
            };
            executor.compile_owned(graph.clone()).unwrap();
            let actual = executor.execute_compiled(&graph, feeds.clone()).unwrap();
            assert!(actual.iter().all(|v| v.is_finite()));
            assert_eq!(
                bits(&actual),
                bits(&executor.execute_compiled(&graph, feeds.clone()).unwrap())
            );
            if let Some(expected) = &strict_bits {
                assert_eq!(
                    &bits(&actual),
                    expected,
                    "convolution changed reduction policy"
                );
            } else {
                strict_bits = Some(bits(&actual));
            }
        }
    }
}
