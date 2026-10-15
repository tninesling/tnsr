use anyhow::{Result, ensure};

/// Adam hyperparameters for one parameter group. Arithmetic/state remain f32.
#[derive(Clone, Debug)]
pub struct AdamGroup {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}
impl AdamGroup {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate >= 0.0,
            "invalid Adam learning rate"
        );
        ensure!(
            self.beta1.is_finite()
                && (0.0..1.0).contains(&self.beta1)
                && self.beta2.is_finite()
                && (0.0..1.0).contains(&self.beta2),
            "invalid Adam betas"
        );
        ensure!(
            self.epsilon.is_finite()
                && self.epsilon > 0.0
                && self.weight_decay.is_finite()
                && self.weight_decay >= 0.0,
            "invalid Adam epsilon/decay"
        );
        Ok(())
    }
}

/// Small status returned after an update; full gradients/moments are opt-in reads.
#[derive(Debug)]
pub struct ResidentStep {
    pub gradient_norm: f64,
    pub gradient_scale: f32,
    pub timestep: usize,
    pub kernel_launches: usize,
}

/// Explicitly downloaded first and second Adam moments.
#[derive(Debug, PartialEq)]
pub struct AdamMoments {
    pub first: Vec<f32>,
    pub second: Vec<f32>,
}
