//! Region-level profitability scoring. Legality and numerical policies are
//! checked by the planner before candidates reach this interface.

use super::DType;

/// Compiler-visible features for one candidate kernel. Counts are estimates,
/// not measurements of physical register allocation or cache behavior.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
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
    /// Estimated input working set for one serial reduction block, accounting
    /// for smaller broadcast storage. Zero for other kernel classes.
    pub serial_working_set_bytes: usize,
    /// Logical elements visited along a fused reduction axis; zero otherwise.
    pub reduction_extent: usize,
    /// Row-major logical stride of the reduction axis. Composed virtual maps
    /// may change physical accesses; this is not a measured memory stride.
    pub reduction_stride: usize,
    /// Threads cooperating on one output fiber: one for strict serial regions,
    /// subgroup/block width for cooperative regions, zero for other steps.
    pub reduction_lanes: usize,
    /// Full-domain visits required by a reduction region (one, or two when a
    /// broadcast-back epilogue recomputes producers); zero for other steps.
    pub full_domain_passes: usize,
    /// Number of nonidentity or masked virtual inputs bound to this region.
    pub non_identity_inputs: usize,
    /// Structural integer operations in input access maps. This is distinct
    /// from address_operations, which estimates dynamic work over all elements.
    pub input_index_operations: usize,
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
            // Small rows can reuse SM-local cache despite an underfilled grid.
            // Wide multi-input serial rows cannot: use the SM shared-memory
            // budget as a conservative cache-capacity proxy for that case.
            // Cache demand belongs to concurrently resident blocks on each SM,
            // not to one block alone. A smaller block must not look free when
            // many such blocks run together on a large output grid.
            let blocks = kernel.active_threads.div_ceil(threads).max(1);
            let active_sms = blocks.min(self.multiprocessor_count.max(1));
            let concurrent_blocks = resident.min(blocks.div_ceil(active_sms));
            let working_set = kernel
                .serial_working_set_bytes
                .saturating_mul(concurrent_blocks);
            let memory_slowdown = if working_set > self.shared_bytes_per_sm {
                slowdown
            } else {
                slowdown.min(4)
            };
            cost.pressure_ns = cost.pressure_ns.saturating_add(
                compute
                    .saturating_mul(slowdown - 1)
                    .saturating_add(traffic.saturating_mul(memory_slowdown - 1)),
            );
        }
        cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_cache_cost_accounts_for_concurrently_resident_blocks() {
        let scorer = AnalyticalFusionScorer {
            multiprocessor_count: 76,
            shared_bytes_per_sm: 100 * 1024,
            ..AnalyticalFusionScorer::default()
        };
        let small = FusionFeatures {
            launches: 1,
            traffic_bytes: 256 * 256 * 4,
            weighted_operations: 256 * 256,
            estimated_registers_per_thread: 32,
            threads_per_block: 32,
            active_threads: 256,
            serial_working_set_bytes: 32 * 256 * 4,
            reduction_lanes: 1,
            ..FusionFeatures::default()
        };
        let mut wide = small.clone();
        wide.threads_per_block = 1024;
        wide.serial_working_set_bytes = 256 * 256 * 4;
        assert!(
            scorer.score(std::slice::from_ref(&small)).total_ns()
                < scorer.score(std::slice::from_ref(&wide)).total_ns()
        );
        // Once many blocks are resident on each SM, their combined working set
        // no longer fits merely because each individual 32-thread block fits.
        let mut crowded_small = small;
        crowded_small.active_threads = 8192;
        let mut crowded_wide = wide;
        crowded_wide.active_threads = 8192;
        assert_eq!(
            scorer.score(&[crowded_small]),
            scorer.score(&[crowded_wide])
        );
    }
}
