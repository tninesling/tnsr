use super::*;

fn data(shape: Shape, salt: usize) -> Vec<f64> {
    (0..shape.elements().unwrap())
        .map(|i| ((i * 13 + salt) % 31) as f64 / 29. - 0.43)
        .collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[test]
fn online_training_f64_vjp_matches_independent_directional_derivatives() -> Result<()> {
    let shape = Shape::new(2, 5, 3)?;
    let (q, k, v, g) = (
        data(shape, 1),
        data(shape, 3),
        data(shape, 7),
        data(shape, 11),
    );
    let reference = evaluate(
        shape,
        Inputs {
            query: &q,
            key: &k,
            value: &v,
            incoming: &g,
        },
    )?;
    let direction = [data(shape, 17), data(shape, 19), data(shape, 23)];
    let epsilon = 1e-5;
    // This is a fixed mathematical oracle check, not a training tolerance.
    for enabled in [
        [true, false, false],
        [false, true, false],
        [false, false, true],
        [true, true, true],
    ] {
        let mut plus = [q.clone(), k.clone(), v.clone()];
        let mut minus = plus.clone();
        for tensor in 0..3 {
            if enabled[tensor] {
                for i in 0..plus[tensor].len() {
                    plus[tensor][i] += epsilon * direction[tensor][i];
                    minus[tensor][i] -= epsilon * direction[tensor][i];
                }
            }
        }
        let positive = evaluate(
            shape,
            Inputs {
                query: &plus[0],
                key: &plus[1],
                value: &plus[2],
                incoming: &g,
            },
        )?;
        let negative = evaluate(
            shape,
            Inputs {
                query: &minus[0],
                key: &minus[1],
                value: &minus[2],
                incoming: &g,
            },
        )?;
        let numerical = (objective(&positive, &g)? - objective(&negative, &g)?) / (2. * epsilon);
        let expected = [&reference.d_query, &reference.d_key, &reference.d_value]
            .iter()
            .enumerate()
            .filter(|(i, _)| enabled[*i])
            .map(|(i, x)| dot(x, &direction[i]))
            .sum::<f64>();
        let tolerance = 1e-8 + 1e-7 * expected.abs();
        assert!(
            (numerical - expected).abs() <= tolerance,
            "direction {enabled:?}: finite difference{numerical}, analytic{expected}"
        );
    }
    assert!(
        reference
            .score_gradient_row_sum
            .iter()
            .all(|x| x.abs() < 1e-13)
    );
    Ok(())
}

#[test]
fn online_training_f64_single_key_and_causal_exclusion() -> Result<()> {
    let shape = Shape::new(1, 1, 4)?;
    let q = data(shape, 1);
    let k = data(shape, 2);
    let v = data(shape, 3);
    let g = data(shape, 4);
    let r = evaluate(
        shape,
        Inputs {
            query: &q,
            key: &k,
            value: &v,
            incoming: &g,
        },
    )?;
    assert_eq!(r.output, v);
    assert_eq!(r.d_value, g);
    assert_eq!(r.delta, vec![dot(&v, &g)]);
    assert!(r.d_query.iter().chain(&r.d_key).all(|x| *x == 0.));
    let shape = Shape::new(1, 4, 3)?;
    let q = data(shape, 1);
    let mut k = data(shape, 2);
    let mut v = data(shape, 3);
    let mut g = vec![0.; shape.elements()?];
    g[..3].copy_from_slice(&[0.3, -0.5, 0.7]);
    let before = evaluate(
        shape,
        Inputs {
            query: &q,
            key: &k,
            value: &v,
            incoming: &g,
        },
    )?;
    for x in &mut k[3..] {
        *x += 100.;
    }
    for x in &mut v[3..] {
        *x -= 70.;
    }
    let after = evaluate(
        shape,
        Inputs {
            query: &q,
            key: &k,
            value: &v,
            incoming: &g,
        },
    )?;
    assert_eq!(&before.output[..3], &after.output[..3]);
    assert!(
        after.d_key[3..]
            .iter()
            .chain(&after.d_value[3..])
            .all(|x| *x == 0.)
    );
    Ok(())
}

#[test]
fn online_training_f64_head_streaming_uses_one_square_workspace() -> Result<()> {
    let shape = Shape::new(3, 7, 5)?;
    let (q, k, v, g) = (
        data(shape, 1),
        data(shape, 2),
        data(shape, 3),
        data(shape, 4),
    );
    let whole = evaluate(
        shape,
        Inputs {
            query: &q,
            key: &k,
            value: &v,
            incoming: &g,
        },
    )?;
    assert_eq!(whole.quadratic_workspace_elements, 7 * 7);
    let n = 7 * 5;
    for head in 0..3 {
        let range = head * n..(head + 1) * n;
        let one = evaluate(
            Shape::new(1, 7, 5)?,
            Inputs {
                query: &q[range.clone()],
                key: &k[range.clone()],
                value: &v[range.clone()],
                incoming: &g[range.clone()],
            },
        )?;
        for (a, b) in [
            (&whole.output, &one.output),
            (&whole.d_query, &one.d_query),
            (&whole.d_key, &one.d_key),
            (&whole.d_value, &one.d_value),
        ] {
            assert_eq!(&a[range.clone()], b);
        }
    }
    Ok(())
}

#[test]
fn online_training_f64_nonfinite_and_large_finite_diagnostics_are_explicit() -> Result<()> {
    let shape = Shape::new(1, 4, 2)?;
    let q = vec![0.; 8];
    let k = q.clone();
    let max_bf16 = (2_f64 - 2_f64.powi(-7)) * 2_f64.powi(127);
    let v = vec![max_bf16; 8];
    let g = vec![1e-30; 8];
    let r = evaluate(
        shape,
        Inputs {
            query: &q,
            key: &k,
            value: &v,
            incoming: &g,
        },
    )?;
    assert!(r.output.iter().all(|x| x.is_finite() && *x > 1e38));
    assert_eq!(r.denominator, vec![1., 2., 3., 4.]);
    assert_eq!(r.maximum, vec![0.; 4]);
    let mut bad = q.clone();
    bad[0] = f64::NAN;
    assert!(
        evaluate(
            shape,
            Inputs {
                query: &bad,
                key: &k,
                value: &v,
                incoming: &g
            }
        )
        .is_err()
    );
    assert_eq!(
        classify(&[0., -0., f64::INFINITY, f64::NEG_INFINITY, f64::NAN]),
        Classes {
            finite: 2,
            positive_infinity: 1,
            negative_infinity: 1,
            nan: 1
        }
    );
    assert!(Shape::new(0, 4, 2).is_err());
    assert!(Shape::new(1, 513, 2).is_err());
    let nonfinite = error_metrics(&[0.], &[f64::NAN])?;
    assert!(!nonfinite.finite && nonfinite.max_absolute.is_nan() && nonfinite.relative_l2.is_nan());
    let metrics = error_metrics(&[0., 2.], &[0., 2.])?;
    assert!(metrics.finite);
    assert_eq!(metrics.max_absolute, 0.);
    assert_eq!(metrics.relative_l2, 0.);
    assert_eq!(metrics.reference_l2, 2.);
    Ok(())
}
