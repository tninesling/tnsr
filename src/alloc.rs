//! Memory allocation tracking and buffer pooling for executors.
//!
//! This module is the foundation for the tensor memory allocator described in
//! `docs/memory-allocator-plan.md` (#37). It currently provides allocation
//! statistics; buffer pooling is added in a later phase.

/// Memory allocation statistics gathered during graph execution.
///
/// Counters are cumulative across executions unless [`AllocStats::reset`] is
/// called. All byte counts are element bytes (elements x element size).
#[derive(Debug, Clone, Default)]
pub struct AllocStats {
    /// Total bytes allocated for node values across all executions.
    pub bytes_allocated: usize,
    /// Total number of node value buffers allocated across all executions.
    pub buffers_allocated: usize,
    /// Peak live bytes held in the executor's value map at any single point
    /// during execution.
    pub peak_live_bytes: usize,
}

impl AllocStats {
    /// Record the allocation of a single buffer of `bytes`.
    pub(crate) fn record_alloc(&mut self, bytes: usize) {
        self.bytes_allocated += bytes;
        self.buffers_allocated += 1;
    }

    /// Record the current live byte total, updating the peak.
    pub(crate) fn record_live(&mut self, live_bytes: usize) {
        self.peak_live_bytes = self.peak_live_bytes.max(live_bytes);
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
