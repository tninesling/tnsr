//! Bounded unpublished Adam state, separate from the current committed version.
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub(super) struct Staging {
    pub moment: CudaSlice<f32>,
    pub variance: CudaSlice<f32>,
    pub health: Arc<CudaSlice<f32>>,
    pub settings: Vec<CudaSlice<f32>>,
}

impl Staging {
    pub fn new(stream: &Arc<CudaStream>, len: usize, groups: usize) -> Result<Self> {
        // Every valid entry is overwritten before checking/publishing. Padding
        // is never written by Adam kernels and retains these initial zeros.
        Ok(Self {
            moment: stream.alloc_zeros(len)?,
            variance: stream.alloc_zeros(len)?,
            health: Arc::new(stream.alloc_zeros(len)?),
            settings: (0..groups)
                .map(|_| stream.alloc_zeros(10).map_err(Into::into))
                .collect::<Result<_>>()?,
        })
    }
}
