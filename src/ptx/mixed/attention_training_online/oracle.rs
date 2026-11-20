//! Independent head-streamed F64 attention and analytic VJP.
//!
//! No TileIR, generated PTX, MMA emulation, old softmax derivative, or GPU
//! implementation participates in this reference. The only quadratic workspace
//! is one head's probability matrix. Inputs/returned tensors are linear in BHSD.
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    pub batch_heads: usize,
    pub sequence: usize,
    pub width: usize,
}
impl Shape {
    pub(super) fn new(batch_heads: usize, sequence: usize, width: usize) -> Result<Self> {
        ensure!(
            batch_heads > 0 && (1..=512).contains(&sequence) && (1..=64).contains(&width),
            "oracle shape must have BH>0, 1<=S<=512 and 1<=D<=64"
        );
        let shape = Self {
            batch_heads,
            sequence,
            width,
        };
        shape.elements()?;
        Ok(shape)
    }
    pub(super) fn elements(self) -> Result<usize> {
        self.batch_heads
            .checked_mul(self.sequence)
            .and_then(|n| n.checked_mul(self.width))
            .ok_or_else(|| anyhow::anyhow!("oracle linear tensor extent overflow"))
    }
    fn rows(self) -> Result<usize> {
        self.batch_heads
            .checked_mul(self.sequence)
            .ok_or_else(|| anyhow::anyhow!("oracle row extent overflow"))
    }
}

pub(super) struct Inputs<'a> {
    pub query: &'a [f64],
    pub key: &'a [f64],
    pub value: &'a [f64],
    pub incoming: &'a [f64],
}

#[derive(Debug)]
pub(super) struct Reference {
    pub output: Vec<f64>,
    pub d_query: Vec<f64>,
    /// Physical head-major [BH,S,D]; transpose only at the leaf ABI boundary.
    pub d_key: Vec<f64>,
    pub d_value: Vec<f64>,
    pub maximum: Vec<f64>,
    pub denominator: Vec<f64>,
    /// Ideal F64 O*dO delta, deliberately not a rounded-O implementation.
    pub delta: Vec<f64>,
    pub score_gradient_row_sum: Vec<f64>,
    pub quadratic_workspace_elements: usize,
}

fn validate(shape: Shape, inputs: &Inputs<'_>) -> Result<usize> {
    Shape::new(shape.batch_heads, shape.sequence, shape.width)?;
    let len = shape.elements()?;
    for (name, values) in [
        ("Q", inputs.query),
        ("K", inputs.key),
        ("V", inputs.value),
        ("dO", inputs.incoming),
    ] {
        ensure!(
            values.len() == len,
            "oracle {name} length differs from BHSD extent"
        );
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "ideal F64 oracle requires finite {name}; classify nonfinite diagnostics separately"
        );
    }
    Ok(len)
}

pub(super) fn evaluate(shape: Shape, inputs: Inputs<'_>) -> Result<Reference> {
    let len = validate(shape, &inputs)?;
    let rows = shape.rows()?;
    let (s, d) = (shape.sequence, shape.width);
    let scale = 1.0 / (d as f64).sqrt();
    let mut result = Reference {
        output: vec![0.; len],
        d_query: vec![0.; len],
        d_key: vec![0.; len],
        d_value: vec![0.; len],
        maximum: vec![0.; rows],
        denominator: vec![0.; rows],
        delta: vec![0.; rows],
        score_gradient_row_sum: vec![0.; rows],
        quadratic_workspace_elements: s * s,
    };
    // One head is complete before its probability workspace is reused.
    let mut probability = vec![0_f64; s * s];
    for head in 0..shape.batch_heads {
        probability.fill(0.);
        let base = head * s * d;
        for i in 0..s {
            let mut maximum = f64::NEG_INFINITY;
            for j in 0..=i {
                let mut dot = 0_f64;
                for feature in 0..d {
                    dot +=
                        inputs.query[base + i * d + feature] * inputs.key[base + j * d + feature];
                }
                let score = dot * scale;
                ensure!(
                    score.is_finite(),
                    "F64 score overflow in oracle head{head}/row{i}/key{j}"
                );
                probability[i * s + j] = score;
                maximum = maximum.max(score);
            }
            let mut denominator = 0_f64;
            for j in 0..=i {
                let weight = (probability[i * s + j] - maximum).exp();
                probability[i * s + j] = weight;
                denominator += weight;
            }
            ensure!(
                denominator.is_finite() && denominator > 0.,
                "invalid oracle denominator"
            );
            result.maximum[head * s + i] = maximum;
            result.denominator[head * s + i] = denominator;
            for j in 0..=i {
                probability[i * s + j] /= denominator;
            }
            for feature in 0..d {
                let mut output = 0_f64;
                for j in 0..=i {
                    output += probability[i * s + j] * inputs.value[base + j * d + feature];
                }
                result.output[base + i * d + feature] = output;
            }
        }
        for i in 0..s {
            let mut delta = 0_f64;
            for feature in 0..d {
                delta +=
                    result.output[base + i * d + feature] * inputs.incoming[base + i * d + feature];
            }
            result.delta[head * s + i] = delta;
            for j in 0..=i {
                let mut d_probability = 0_f64;
                for feature in 0..d {
                    d_probability += inputs.incoming[base + i * d + feature]
                        * inputs.value[base + j * d + feature];
                }
                let p = probability[i * s + j];
                let d_score = p * (d_probability - delta) * scale;
                result.score_gradient_row_sum[head * s + i] += d_score;
                for feature in 0..d {
                    result.d_query[base + i * d + feature] +=
                        d_score * inputs.key[base + j * d + feature];
                    result.d_key[base + j * d + feature] +=
                        d_score * inputs.query[base + i * d + feature];
                    result.d_value[base + j * d + feature] +=
                        p * inputs.incoming[base + i * d + feature];
                }
            }
        }
    }
    ensure!(
        [
            &result.output,
            &result.d_query,
            &result.d_key,
            &result.d_value
        ]
        .iter()
        .all(|v| v.iter().all(|x| x.is_finite())),
        "F64 output/VJP overflow; oracle fixture is outside reference range"
    );
    Ok(result)
}

pub(super) fn objective(reference: &Reference, incoming: &[f64]) -> Result<f64> {
    ensure!(
        reference.output.len() == incoming.len(),
        "oracle objective length mismatch"
    );
    Ok(reference
        .output
        .iter()
        .zip(incoming)
        .map(|(a, b)| a * b)
        .sum())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Classes {
    pub finite: usize,
    pub positive_infinity: usize,
    pub negative_infinity: usize,
    pub nan: usize,
}
pub(super) fn classify(values: &[f64]) -> Classes {
    let mut result = Classes::default();
    for &x in values {
        if x.is_nan() {
            result.nan += 1;
        } else if x == f64::INFINITY {
            result.positive_infinity += 1;
        } else if x == f64::NEG_INFINITY {
            result.negative_infinity += 1;
        } else {
            result.finite += 1;
        }
    }
    result
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ErrorMetrics {
    pub finite: bool,
    pub max_absolute: f64,
    pub relative_l2: f64,
    pub reference_l2: f64,
}
/// Descriptive leaf metrics only. This supplies no new full-model pass threshold.
pub(super) fn error_metrics(expected: &[f64], actual: &[f64]) -> Result<ErrorMetrics> {
    ensure!(
        expected.len() == actual.len(),
        "oracle metric length mismatch"
    );
    let finite = expected.iter().chain(actual).all(|x| x.is_finite());
    if !finite {
        return Ok(ErrorMetrics {
            finite: false,
            max_absolute: f64::NAN,
            relative_l2: f64::NAN,
            reference_l2: f64::NAN,
        });
    }
    let mut error = 0_f64;
    let mut norm = 0_f64;
    let mut maximum = 0_f64;
    for (&a, &b) in expected.iter().zip(actual) {
        let delta = a - b;
        error += delta * delta;
        norm += a * a;
        maximum = maximum.max(delta.abs());
    }
    let reference_l2 = norm.sqrt();
    Ok(ErrorMetrics {
        finite,
        max_absolute: maximum,
        relative_l2: error.sqrt() / reference_l2.max(1e-12),
        reference_l2,
    })
}

#[cfg(test)]
mod tests;
