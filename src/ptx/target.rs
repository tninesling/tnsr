use anyhow::{Context, Result};
use cudarc::driver::CudaContext;

/// CUDA target properties that affect generated PTX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtxTarget {
    pub compute_capability: (u32, u32),
}

impl PtxTarget {
    pub(crate) fn from_context(context: &CudaContext) -> Result<Self> {
        let (major, minor) = context
            .compute_capability()
            .context("Failed to query CUDA compute capability")?;
        Ok(Self {
            compute_capability: (
                u32::try_from(major).context("CUDA compute capability major is negative")?,
                u32::try_from(minor).context("CUDA compute capability minor is negative")?,
            ),
        })
    }

    pub const fn supports_tf32(self) -> bool {
        self.compute_capability.0 >= 8
    }

    pub(crate) const fn sm80() -> Self {
        Self {
            compute_capability: (8, 0),
        }
    }
}
