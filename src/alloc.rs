//! Memory allocation tracking and buffer pooling for executors.
//!
//! This module implements the tensor memory allocator described in
//! `docs/memory-allocator-plan.md` (#37): executors return intermediate value
//! buffers to a [`BufferPool`] once liveness analysis shows they are dead,
//! and later allocations reuse them.

use std::collections::HashMap;

/// Memory allocation statistics gathered during graph execution.
///
/// Counters are cumulative across executions unless [`AllocStats::reset`] is
/// called. All byte counts are element bytes (elements x element size).
#[derive(Debug, Clone, Default)]
pub struct AllocStats {
    /// Total bytes allocated from the OS across all executions (pool misses).
    pub bytes_allocated: usize,
    /// Total buffers allocated from the OS across all executions (pool misses).
    pub buffers_allocated: usize,
    /// Peak live bytes held in the executor's value map at any single point
    /// during execution.
    pub peak_live_bytes: usize,
    /// Pool allocations satisfied by a reused buffer.
    pub pool_hits: usize,
    /// Pool allocations that required a fresh buffer.
    pub pool_misses: usize,
}

impl AllocStats {
    /// Record the allocation of a single buffer of `bytes`.
    // Used by the CUDA executor, which does not have a buffer pool yet.
    #[allow(dead_code)]
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

/// Reusable buffer store for executor value allocations.
///
/// Buffers are bucketed by exact length: training loops run the same shapes
/// every step, so exact match gives near-perfect reuse. Power-of-two
/// bucketing is a follow-up if fragmentation appears.
///
/// Reused buffers contain stale data; callers must fully overwrite them.
#[derive(Debug)]
pub struct BufferPool<D> {
    free: HashMap<usize, Vec<Vec<D>>>,
    stats: PoolStats,
}

impl<D> Default for BufferPool<D> {
    fn default() -> Self {
        Self {
            free: HashMap::new(),
            stats: PoolStats::default(),
        }
    }
}

/// Cumulative reuse counters for a [`BufferPool`].
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    /// Allocations satisfied by a reused buffer.
    pub hits: usize,
    /// Allocations that required a fresh buffer.
    pub misses: usize,
    /// Total bytes allocated from the OS (misses only).
    pub fresh_bytes: usize,
}

impl<D> BufferPool<D> {
    /// Take a buffer of exactly `len` elements, reusing a pooled buffer when
    /// possible.
    pub fn take(&mut self, len: usize) -> Vec<D>
    where
        D: num_traits::Float,
    {
        if let Some(buf) = self.free.get_mut(&len).and_then(Vec::pop) {
            self.stats.hits += 1;
            buf
        } else {
            self.stats.misses += 1;
            self.stats.fresh_bytes += len * std::mem::size_of::<D>();
            vec![D::zero(); len]
        }
    }

    /// Return a buffer to the pool for future reuse.
    pub fn give(&mut self, buf: Vec<D>) {
        self.free.entry(buf.len()).or_default().push(buf);
    }

    /// Cumulative reuse counters for this pool.
    pub fn stats(&self) -> PoolStats {
        self.stats
    }
}
