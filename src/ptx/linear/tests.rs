use super::*;
fn round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn fixture(s: LinearShape, branches: usize) -> LinearInputs {
    let values = |len: usize, period: usize| {
        (0..len)
            .map(|i| ((i % period) as f32 - (period / 2) as f32) / 16.)
            .collect()
    };
    LinearInputs {
        x: values(s.rows * s.inputs, 7),
        weight: values(s.outputs * s.inputs, 5),
        bias: values(s.outputs, 3),
        upstream: (0..branches)
            .map(|branch| {
                vec![
                    if branch == 0 {
                        1.0 + 1.0 / 256.0
                    } else {
                        1.0 / 256.0
                    };
                    s.rows * s.outputs
                ]
            })
            .collect(),
    }
}
fn reference(s: LinearShape, v: &LinearInputs) -> [Vec<f32>; 4] {
    let x = v.x.iter().copied().map(round).collect::<Vec<_>>();
    let w = v.weight.iter().copied().map(round).collect::<Vec<_>>();
    let b = v.bias.iter().copied().map(round).collect::<Vec<_>>();
    let g = (0..s.rows * s.outputs)
        .map(|i| {
            v.upstream
                .iter()
                .map(|v| round(v[i]))
                .reduce(|a, b| round(a + b))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut y = vec![0.; s.rows * s.outputs];
    let mut dx = vec![0.; s.rows * s.inputs];
    let mut dw = vec![0.; s.outputs * s.inputs];
    let mut db = vec![0.; s.outputs];
    for r in 0..s.rows {
        for n in 0..s.outputs {
            let mut sum = 0.;
            for k in 0..s.inputs {
                sum += x[r * s.inputs + k] * w[n * s.inputs + k];
            }
            y[r * s.outputs + n] = round(sum + b[n]);
        }
    }
    for r in 0..s.rows {
        for k in 0..s.inputs {
            let mut sum = 0.;
            for n in 0..s.outputs {
                sum += g[r * s.outputs + n] * w[n * s.inputs + k];
            }
            dx[r * s.inputs + k] = round(sum);
        }
    }
    for n in 0..s.outputs {
        for k in 0..s.inputs {
            let mut sum = 0.;
            for r in 0..s.rows {
                sum += g[r * s.outputs + n] * x[r * s.inputs + k];
            }
            dw[n * s.inputs + k] = round(sum);
        }
        let mut sum = 0.;
        for r in 0..s.rows {
            sum += g[r * s.outputs + n];
        }
        db[n] = round(sum);
    }
    [y, dx, dw, db]
}
fn available() -> bool {
    match CudaContext::new(0) {
        Ok(device) => {
            crate::ptx::PtxTarget::from_context(&device)
                .unwrap()
                .compute_capability
                .0
                >= 8
        }
        Err(error) => {
            assert_ne!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "{error}"
            );
            false
        }
    }
}
#[test]
fn physical_bf16_linear_rounds_fanout_before_backward_and_accumulates_in_f32() {
    if !available() {
        return;
    }
    for shape in [
        LinearShape {
            rows: 3,
            inputs: 5,
            outputs: 4,
        },
        LinearShape {
            rows: 33,
            inputs: 65,
            outputs: 35,
        },
    ] {
        for replay in [false, true] {
            let mut linear = LinearTraining::new(shape, LinearPrecision::Bf16, 2, replay).unwrap();
            let input = fixture(shape, 2);
            let expected = reference(shape, &input);
            assert!(linear.run().is_err());
            linear.upload(&input).unwrap();
            linear.run().unwrap();
            let actual = linear.readback().unwrap();
            for (a, b) in [
                &actual.output,
                &actual.input_gradient,
                &actual.weight_gradient,
                &actual.bias_gradient,
            ]
            .into_iter()
            .zip(&expected)
            {
                assert_eq!(a, b);
            }
            assert_eq!(actual.accumulated_weight_gradient, expected[2]);
            assert_eq!(actual.accumulated_bias_gradient, expected[3]);
            assert_eq!(
                linear.info().bf16_buffer_bytes,
                (2 * shape.rows * shape.inputs
                    + 2 * shape.outputs * shape.inputs
                    + 2 * shape.outputs
                    + 4 * shape.rows * shape.outputs)
                    * 2
            );
            let mut changed = input;
            changed.weight.iter_mut().for_each(|x| *x *= -2.);
            changed.x.iter_mut().for_each(|x| *x *= -0.5);
            let changed_expected = reference(shape, &changed);
            linear.upload(&changed).unwrap();
            linear.run().unwrap();
            let next = linear.readback().unwrap();
            assert_eq!(next.output, changed_expected[0]);
            assert_eq!(next.input_gradient, changed_expected[1]);
            assert_eq!(next.weight_gradient, changed_expected[2]);
            assert_eq!(
                next.accumulated_weight_gradient,
                expected[2]
                    .iter()
                    .zip(&changed_expected[2])
                    .map(|(a, b)| a + b)
                    .collect::<Vec<_>>()
            );
            linear.clear_gradients().unwrap();
            linear.run().unwrap();
            assert_eq!(
                linear.readback().unwrap().accumulated_weight_gradient,
                changed_expected[2]
            );
        }
    }
}
#[test]
fn linear_rejects_invalid_shapes_and_uploads_before_mutation() {
    assert!(
        LinearShape {
            rows: 0,
            inputs: 2,
            outputs: 3
        }
        .lengths()
        .is_err()
    );
    assert!(
        LinearShape {
            rows: usize::MAX,
            inputs: 2,
            outputs: 3
        }
        .lengths()
        .is_err()
    );
    if !available() {
        return;
    }
    let s = LinearShape {
        rows: 3,
        inputs: 5,
        outputs: 4,
    };
    let mut linear = LinearTraining::new(s, LinearPrecision::Bf16, 1, true).unwrap();
    let mut v = fixture(s, 1);
    linear.upload(&v).unwrap();
    linear.run().unwrap();
    let expected = linear.readback().unwrap().output;
    v.x[0] = 100.;
    v.bias.clear();
    assert!(linear.upload(&v).is_err());
    linear.run().unwrap();
    assert_eq!(linear.readback().unwrap().output, expected);
}

#[test]
fn bf16_linear_fuses_bias_rounding_and_recovers_after_nonfinite_input() {
    if !available() {
        return;
    }
    for width in [1, 17] {
        let s = LinearShape {
            rows: width,
            inputs: width,
            outputs: width,
        };
        let mut v = fixture(s, 1);
        v.x.fill(0.);
        v.weight.fill(0.);
        v.bias.fill(-1.015625);
        for row in 0..width {
            v.x[row * width] = 1.0078125;
            v.weight[row * width] = 1.0078125;
        }
        let mut linear = LinearTraining::new(s, LinearPrecision::Bf16, 1, true).unwrap();
        linear.upload(&v).unwrap();
        linear.run().unwrap();
        let first = linear.readback().unwrap();
        // Rounding GEMM before bias would give zero instead of this exact residual.
        assert!(first.output.iter().all(|x| *x == 1.0 / 16384.0));
        v.x[0] = f32::NAN;
        linear.upload(&v).unwrap();
        linear.run().unwrap();
        assert!(linear.readback().unwrap().output[0].is_nan());
        v.x[0] = 1.0078125;
        v.upstream[0].fill(1e-43);
        linear.clear_gradients().unwrap();
        linear.upload(&v).unwrap();
        linear.run().unwrap();
        let next = linear.readback().unwrap();
        assert_eq!(next.output, first.output);
        assert!(
            next.bias_gradient
                .iter()
                .chain(&next.accumulated_weight_gradient)
                .all(|x| *x == 0.)
        );
    }
}
