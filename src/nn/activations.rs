//! Model activations expressed using the differentiable tensor primitives.
use crate::tensor::TensorExpr;

/// Numerically stable f32 sigmoid, including large finite inputs.
pub fn sigmoid(x: impl Into<TensorExpr<f32>>) -> TensorExpr<f32> {
    let x = x.into();
    let negative = (-x.clone()).gt(0.0);
    let nonnegative = TensorExpr::from(1.0) - negative.clone();
    let magnitude = x * (TensorExpr::from(1.0) - negative.clone() * 2.0);
    let z = (-magnitude).exp();
    let denominator = z.clone() + 1.0;
    (TensorExpr::from(1.0) / denominator.clone()).mask(nonnegative)
        + (z / denominator).mask(negative)
}

/// SiLU (the gate used by SwiGLU): `x * sigmoid(x)`.
pub fn silu(x: impl Into<TensorExpr<f32>>) -> TensorExpr<f32> {
    let x = x.into();
    x.clone() * sigmoid(x)
}

/// Erf-form GELU, `x * Phi(x)`, evaluated with a rational-exponential f32 CDF.
///
/// Uses the error-function approximation in Abramowitz and Stegun 7.1.26,
/// rather than the common tanh GELU approximation. Both values and symbolic
/// gradients are tested against PyTorch's default GELU. Floating-point and
/// approximation error are subject to the tolerances in those tests; this is
/// not a bitwise implementation of a platform's `erf` routine.
pub fn gelu(x: impl Into<TensorExpr<f32>>) -> TensorExpr<f32> {
    let x = x.into();
    let negative = (-x.clone()).gt(0.0);
    let nonnegative = TensorExpr::from(1.0) - negative.clone();
    let magnitude = x.clone()
        * (TensorExpr::from(1.0) - negative.clone() * 2.0)
        * std::f32::consts::FRAC_1_SQRT_2;
    let t = TensorExpr::from(1.0) / (magnitude.clone() * 0.327_591_1 + 1.0);
    let polynomial = ((((t.clone() * 1.061_405_4 - 1.453_152_1) * t.clone() + 1.421_413_8)
        * t.clone()
        - 0.284_496_72)
        * t.clone()
        + 0.254_829_6)
        * t;
    let tail = polynomial * (-(magnitude.clone() * magnitude)).exp() * 0.5;
    // Evaluate the negative tail directly to avoid cancellation near zero CDF.
    let cdf = (TensorExpr::from(1.0) - tail.clone()).mask(nonnegative) + tail.mask(negative);
    x * cdf
}

/// Elementwise minimum. At ties, split gradients equally between operands.
pub fn minimum(a: impl Into<TensorExpr<f32>>, b: impl Into<TensorExpr<f32>>) -> TensorExpr<f32> {
    let a = a.into();
    let b = b.into();
    let a_greater = a.clone().gt(b.clone());
    let b_greater = b.clone().gt(a.clone());
    let equal = TensorExpr::from(1.0) - a_greater.clone() - b_greater.clone();
    a.clone().mask(b_greater) + b.clone().mask(a_greater) + ((a + b) * 0.5).mask(equal)
}

/// Elementwise maximum. At ties, split gradients equally between operands.
pub fn maximum(a: impl Into<TensorExpr<f32>>, b: impl Into<TensorExpr<f32>>) -> TensorExpr<f32> {
    let a = a.into();
    let b = b.into();
    -minimum(-a, -b)
}

/// Clamp finite values to `[lower, upper]`, preserving gradients at the bounds.
pub fn clamp(x: impl Into<TensorExpr<f32>>, lower: f32, upper: f32) -> TensorExpr<f32> {
    assert!(lower.is_finite() && upper.is_finite() && lower <= upper);
    let x = x.into();
    let below = TensorExpr::from(lower).gt(x.clone());
    let above = x.clone().gt(upper);
    let inside = TensorExpr::from(1.0) - below.clone() - above.clone();
    x.mask(inside) + TensorExpr::from(lower).mask(below) + TensorExpr::from(upper).mask(above)
}

#[cfg(test)]
mod tests;
