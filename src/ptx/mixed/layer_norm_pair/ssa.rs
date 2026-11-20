//! Executable CPU specification of the original row arithmetic, not a compact
//! derivative. Device codegen must retain its own original approximate unary
//! instructions and audited contraction checkpoints; this is a tree/order oracle.
use super::*;
#[cfg(test)]
pub(super) const LANES: usize = 128;
#[cfg(test)]
pub(super) fn sum(values: &[f32]) -> f32 {
    assert!(matches!(values.len(), 512 | 1024));
    let mut partials = [0_f32; LANES];
    for (lane, a) in partials.iter_mut().enumerate() {
        for column in (lane..values.len()).step_by(LANES) {
            *a += values[column];
        }
    }
    let mut stride = LANES / 2;
    while stride > 0 {
        // A separate read/write snapshot mirrors one uniform shared barrier per
        // stride; no masked lane exits before a subsequent collective.
        let previous = partials;
        for lane in 0..stride {
            partials[lane] = previous[lane] + previous[lane + stride];
        }
        stride /= 2;
    }
    partials[0]
}
#[cfg(test)]
pub(super) struct Forward {
    pub mean: f32,
    pub variance_plus_epsilon: f32,
    pub inverse_std: f32,
    pub normalized: Vec<f32>,
    pub output: Vec<f32>,
}
#[cfg(test)]
pub(super) fn forward(x: &[f32], gamma: &[f32], beta: &[f32], epsilon: f32) -> Forward {
    let width = x.len();
    assert!(matches!(width, 512 | 1024));
    assert_eq!(gamma.len(), width);
    assert_eq!(beta.len(), width);
    let mean = sum(x) / width as f32;
    let centered: Vec<_> = x.iter().map(|x| x - mean).collect();
    let square: Vec<_> = centered.iter().map(|c| c * c).collect();
    let variance = sum(&square) / width as f32;
    let variance_plus_epsilon = variance + epsilon;
    let inverse_std = (variance_plus_epsilon.ln() * -0.5).exp();
    let normalized: Vec<_> = centered.iter().map(|c| c * inverse_std).collect();
    let output = normalized
        .iter()
        .zip(gamma)
        .zip(beta)
        .map(|((&n, &g), &b)| (n * g) + b)
        .collect();
    Forward {
        mean,
        variance_plus_epsilon,
        inverse_std,
        normalized,
        output,
    }
}
#[cfg(test)]
pub(super) fn backward(x: &[f32], gamma: &[f32], incoming: &[f32], saved: &Forward) -> Vec<f32> {
    let width = x.len();
    assert!(matches!(width, 512 | 1024));
    assert_eq!(gamma.len(), width);
    assert_eq!(incoming.len(), width);
    let centered: Vec<_> = x.iter().map(|x| x - saved.mean).collect();
    let g: Vec<_> = incoming.iter().zip(gamma).map(|(g, w)| g * w).collect();
    let direct: Vec<_> = g.iter().map(|g| g * saved.inverse_std).collect();
    let q: Vec<_> = g
        .iter()
        .zip(&centered)
        .map(|(&g, &c)| {
            let e = g * c;
            let exp_vjp = e * saved.inverse_std;
            let scaled = exp_vjp * -0.5;
            scaled / saved.variance_plus_epsilon
        })
        .collect();
    let dv = sum(&q);
    let ds = dv * (1_f32 / width as f32);
    let dc: Vec<_> = direct
        .iter()
        .zip(centered)
        .map(|(&u, c)| {
            let p1 = ds * c;
            let first = u + p1;
            let p2 = ds * c;
            first + p2
        })
        .collect();
    let neg: Vec<_> = dc.iter().map(|v| -v).collect();
    let dm = sum(&neg);
    let mean_term = dm * (1_f32 / width as f32);
    dc.iter().map(|v| v + mean_term).collect()
}
/// These are former standalone Mul output boundaries in the observed B32
/// baseline. The final mean-term checkpoint depends on the ACTUAL selected
/// baseline pointwise region; the source DAG alone cannot resolve it.
#[derive(Clone, Debug)]
pub(super) struct CheckpointCandidates {
    pub direct: NodeIndex,
    pub first_square_vjp: NodeIndex,
    pub second_square_vjp: NodeIndex,
    pub mean_term: NodeIndex,
    pub requires_baseline_contraction_audit: bool,
}
pub(super) fn checkpoint_candidates(pair: &LayerNormPair) -> Option<CheckpointCandidates> {
    let by_node: HashMap<_, _> = pair.program.iter().map(|op| (op.node, op)).collect();
    let root = by_node.get(&pair.backward)?;
    if root.kind != Kind::Binary(BinaryOp::Add) {
        return None;
    }
    let cc2 = pair.program.get(*root.inputs.first()?)?;
    let mean = pair.program.get(*root.inputs.get(1)?)?;
    let cc1 = pair.program.get(*cc2.inputs.first()?)?;
    let p2 = pair.program.get(*cc2.inputs.get(1)?)?;
    let direct = pair.program.get(*cc1.inputs.first()?)?;
    let p1 = pair.program.get(*cc1.inputs.get(1)?)?;
    if cc1.kind != Kind::Binary(BinaryOp::Add)
        || cc2.kind != Kind::Binary(BinaryOp::Add)
        || [p1, p2, direct, mean]
            .iter()
            .any(|op| op.kind != Kind::Binary(BinaryOp::Mul))
    {
        return None;
    }
    Some(CheckpointCandidates {
        direct: direct.node,
        first_square_vjp: p1.node,
        second_square_vjp: p2.node,
        mean_term: mean.node,
        requires_baseline_contraction_audit: true,
    })
}
