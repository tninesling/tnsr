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
    /// Record the current live byte total, updating the peak.
    pub(crate) fn record_live(&mut self, live_bytes: usize) {
        self.peak_live_bytes = self.peak_live_bytes.max(live_bytes);
    }

    /// Return live-memory counters combined with current pool counters.
    pub(crate) fn with_pool_stats(&self, pool: PoolStats) -> Self {
        Self {
            bytes_allocated: pool.fresh_bytes,
            buffers_allocated: pool.misses,
            peak_live_bytes: self.peak_live_bytes,
            pool_hits: pool.hits,
            pool_misses: pool.misses,
        }
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
    inner: ExactSizePool<Vec<D>>,
}

impl<D> Default for BufferPool<D> {
    fn default() -> Self {
        Self {
            inner: ExactSizePool::default(),
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

/// Backend-agnostic exact-size buffer cache.
///
/// Allocation is supplied by the caller, so the same pool supports host
/// vectors, CUDA device allocations, and future storage backends.
#[derive(Debug)]
struct ExactSizePool<B> {
    free: HashMap<usize, Vec<B>>,
    stats: PoolStats,
}

impl<B> Default for ExactSizePool<B> {
    fn default() -> Self {
        Self {
            free: HashMap::new(),
            stats: PoolStats::default(),
        }
    }
}

impl<B> ExactSizePool<B> {
    fn take_or_try_with<E>(
        &mut self,
        len: usize,
        bytes: usize,
        allocate: impl FnOnce() -> Result<B, E>,
    ) -> Result<B, E> {
        if let Some(buf) = self.free.get_mut(&len).and_then(Vec::pop) {
            self.stats.hits += 1;
            Ok(buf)
        } else {
            let buf = allocate()?;
            self.stats.misses += 1;
            self.stats.fresh_bytes += bytes;
            Ok(buf)
        }
    }

    fn give(&mut self, len: usize, buf: B) {
        self.free.entry(len).or_default().push(buf);
    }

    fn stats(&self) -> PoolStats {
        self.stats
    }

    fn reset_stats(&mut self) {
        self.stats = PoolStats::default();
    }

    fn clear(&mut self) {
        self.free.clear();
    }
}

impl<D> BufferPool<D> {
    /// Take a buffer of exactly `len` elements, reusing a pooled buffer when
    /// possible.
    pub fn take(&mut self, len: usize) -> Vec<D>
    where
        D: num_traits::Float,
    {
        self.inner
            .take_or_try_with(len, len * std::mem::size_of::<D>(), || {
                Ok::<_, std::convert::Infallible>(vec![D::zero(); len])
            })
            .unwrap()
    }

    /// Return a buffer to the pool for future reuse.
    pub fn give(&mut self, buf: Vec<D>) {
        self.inner.give(buf.len(), buf);
    }

    /// Cumulative reuse counters for this pool.
    pub fn stats(&self) -> PoolStats {
        self.inner.stats()
    }

    /// Reset cumulative reuse counters without discarding cached buffers.
    pub fn reset_stats(&mut self) {
        self.inner.reset_stats();
    }

    /// Discard every cached buffer.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

/// Reusable CUDA buffer store, bucketed by exact element count.
#[cfg(feature = "cuda")]
#[derive(Debug, Default)]
pub struct CudaBufferPool {
    inner: ExactSizePool<cudarc::driver::CudaSlice<f32>>,
}

#[cfg(feature = "cuda")]
impl CudaBufferPool {
    /// Take a device buffer of exactly `len` elements.
    pub fn take(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        len: usize,
    ) -> anyhow::Result<cudarc::driver::CudaSlice<f32>> {
        self.inner
            .take_or_try_with(len, len * std::mem::size_of::<f32>(), || {
                if len == 0 {
                    stream.null::<f32>().map_err(anyhow::Error::from)
                } else {
                    stream.alloc_zeros::<f32>(len).map_err(anyhow::Error::from)
                }
            })
    }

    /// Return a device buffer to the pool for reuse on the same stream.
    pub fn give(&mut self, buf: cudarc::driver::CudaSlice<f32>) {
        self.inner.give(buf.len(), buf);
    }

    /// Cumulative reuse counters for this pool.
    pub fn stats(&self) -> PoolStats {
        self.inner.stats()
    }

    /// Reset cumulative reuse counters without discarding cached buffers.
    pub fn reset_stats(&mut self) {
        self.inner.reset_stats();
    }

    /// Discard every cached device buffer.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}
