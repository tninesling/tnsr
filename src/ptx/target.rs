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

    pub const fn matmul_capabilities(self) -> crate::tile::MatMulCapabilities {
        crate::tile::MatMulCapabilities {
            tf32: self.supports_tf32(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{MatMulPlan, MatMulPrecision, MatMulSchedule};

    #[test]
    fn synthetic_target_profiles_select_supported_f32_instructions() {
        for (compute_capability, expected) in [
            ((6, 1), MatMulPlan::ScalarF32),
            ((7, 5), MatMulPlan::ScalarF32),
            ((8, 0), MatMulPlan::TensorCoreTf32),
            ((8, 9), MatMulPlan::TensorCoreTf32),
        ] {
            let target = PtxTarget { compute_capability };
            assert_eq!(
                MatMulSchedule::select(
                    128,
                    128,
                    128,
                    target.matmul_capabilities(),
                    MatMulPrecision::AllowTf32
                )
                .plan,
                expected
            );
            assert_eq!(
                MatMulSchedule::select(
                    128,
                    128,
                    128,
                    target.matmul_capabilities(),
                    MatMulPrecision::StrictF32
                )
                .plan,
                MatMulPlan::ScalarF32
            );
        }
    }
}
