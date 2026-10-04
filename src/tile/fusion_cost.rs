//! Region-level profitability scoring. Legality and numerical policies are
//! checked by the planner before candidates reach this interface.

use super::DType;

/// Compiler-visible features for one candidate kernel. Counts are estimates,
/// not measurements of physical register allocation or cache behavior.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FusionFeatures {
    pub storage_dtype: Option<DType>,
    pub logical_shape: Vec<usize>,
    pub launches: usize,
    pub traffic_bytes: usize,
    pub intermediate_bytes: usize,
    /// Simple operation equivalents; transcendental/division operations carry
    /// more weight. Includes repeated work rather than adding it twice.
    pub weighted_operations: usize,
    pub repeated_operations: usize,
    pub address_operations: usize,
    pub estimated_registers_per_thread: usize,
    pub shared_bytes_per_block: usize,
    pub threads_per_block: usize,
    pub active_threads: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FusionCost {
    pub launch_ns: usize,
    pub traffic_ns: usize,
    pub compute_ns: usize,
    pub pressure_ns: usize,
}

impl FusionCost {
    pub fn total_ns(self) -> usize {
        self.launch_ns
            .saturating_add(self.traffic_ns)
            .saturating_add(self.compute_ns)
            .saturating_add(self.pressure_ns)
    }
}

/// Score a complete candidate's kernels together. A learned scorer can model
/// interactions between regions; this interface does not require additive
/// e-node costs. Lower cost wins; ties retain the existing fusion.
pub trait FusionScorer {
    fn score(&self, kernels: &[FusionFeatures]) -> FusionCost;
}

/// A conservative initial model, with explicit coefficients for calibration.
/// These estimates rank candidates; they are not predicted wall-clock timings.
#[derive(Debug, Clone, Copy)]
pub struct AnalyticalFusionScorer {
    pub launch_ns: usize,
    pub bytes_per_ns: usize,
    pub operations_per_ns: usize,
    pub multiprocessor_count: usize,
    pub registers_per_sm: usize,
    pub shared_bytes_per_sm: usize,
    pub threads_per_sm: usize,
}

impl Default for AnalyticalFusionScorer {
    fn default() -> Self {
        Self {
            launch_ns: 3_000,
            bytes_per_ns: 500,
            operations_per_ns: 32_000,
            multiprocessor_count: 1,
            registers_per_sm: 64 * 1024,
            shared_bytes_per_sm: 64 * 1024,
            threads_per_sm: 1_536,
        }
    }
}

impl FusionScorer for AnalyticalFusionScorer {
    fn score(&self, kernels: &[FusionFeatures]) -> FusionCost {
        let mut cost = FusionCost::default();
        for kernel in kernels {
            cost.launch_ns = cost
                .launch_ns
                .saturating_add(kernel.launches.saturating_mul(self.launch_ns));
            let traffic = kernel.traffic_bytes.div_ceil(self.bytes_per_ns.max(1));
            let compute = kernel
                .weighted_operations
                .saturating_add(kernel.address_operations)
                .div_ceil(self.operations_per_ns.max(1));
            cost.traffic_ns = cost.traffic_ns.saturating_add(traffic);
            cost.compute_ns = cost.compute_ns.saturating_add(compute);
            let threads = kernel.threads_per_block.max(1);
            let resident = (self.registers_per_sm
                / threads
                    .saturating_mul(kernel.estimated_registers_per_thread.max(1))
                    .max(1))
            .min(self.shared_bytes_per_sm / kernel.shared_bytes_per_block.max(1))
            .min(self.threads_per_sm / threads)
            .max(1);
            let parallelism = resident
                .saturating_mul(threads)
                .saturating_mul(self.multiprocessor_count.max(1))
                .min(kernel.active_threads.max(1));
            // Bound uncertainty in occupancy and underfilled reduction grids.
            let slowdown = self
                .threads_per_sm
                .saturating_mul(self.multiprocessor_count.max(1))
                .div_ceil(parallelism.max(1))
                .clamp(1, 64);
            cost.pressure_ns = cost.pressure_ns.saturating_add(
                compute
                    .saturating_mul(slowdown - 1)
                    .saturating_add(traffic.saturating_mul(slowdown.min(4) - 1)),
            );
        }
        cost
    }
}
