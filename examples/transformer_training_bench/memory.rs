//! Example-only telemetry of the process's current/default CUDA memory pools.
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, sys};
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize)]
pub struct PoolBytes {
    pub used_current: u64,
    pub used_high: u64,
    pub reserved_current: u64,
    pub reserved_high: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub current_pool: PoolBytes,
    pub default_pool: PoolBytes,
    pub same_pool: bool,
}
pub struct Memory {
    context: Arc<CudaContext>,
    current: sys::CUmemoryPool,
    default: sys::CUmemoryPool,
}
impl Memory {
    pub fn new() -> Result<Self> {
        let context = CudaContext::new(0)?;
        context.bind_to_thread()?;
        let mut current = std::ptr::null_mut();
        let mut default = std::ptr::null_mut();
        // SAFETY: live CUDA context/device; both output pointers are writable
        // CUmemoryPool handles, and the context remains owned by this helper.
        unsafe {
            sys::cuDeviceGetMemPool(&mut current, context.cu_device()).result()?;
            sys::cuDeviceGetDefaultMemPool(&mut default, context.cu_device()).result()?;
        }
        ensure!(
            !current.is_null() && !default.is_null(),
            "CUDA memory pool is unavailable"
        );
        Ok(Self {
            context,
            current,
            default,
        })
    }
    fn pool(&self, pool: sys::CUmemoryPool) -> Result<PoolBytes> {
        let get = |attribute| -> Result<u64> {
            let mut value = 0u64;
            // SAFETY: pool belongs to this process/device and remains live;
            // these four memory-accounting attributes write exactly one u64.
            unsafe {
                sys::cuMemPoolGetAttribute(pool, attribute, std::ptr::from_mut(&mut value).cast())
                    .result()?;
            }
            Ok(value)
        };
        Ok(PoolBytes {
            used_current: get(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
            used_high: get(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
            reserved_current: get(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
            reserved_high: get(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?,
        })
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        self.context.bind_to_thread()?;
        Ok(Snapshot {
            current_pool: self.pool(self.current)?,
            default_pool: self.pool(self.default)?,
            same_pool: self.current == self.default,
        })
    }
    /// Reset only the high-water accounting of these process-owned pools.
    /// Call after all benchmark streams synchronize; no allocation is released.
    pub fn reset_highs(&self) -> Result<()> {
        self.context.bind_to_thread()?;
        for pool in [
            Some(self.current),
            (self.default != self.current).then_some(self.default),
        ]
        .into_iter()
        .flatten()
        {
            for attribute in [
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            ] {
                let mut zero = 0u64;
                // SAFETY: live process-owned pool and writable u64 zero of the
                // exact type required for resetting these two high-water marks.
                unsafe {
                    sys::cuMemPoolSetAttribute(
                        pool,
                        attribute,
                        std::ptr::from_mut(&mut zero).cast(),
                    )
                    .result()?;
                }
            }
        }
        Ok(())
    }
}
