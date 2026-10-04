use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, sys::CUdevice_attribute};

use crate::tile::MatMulResources;

/// CUDA target properties that affect generated PTX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtxTarget {
    pub multiprocessor_count: usize,
    pub compute_capability: (u32, u32),
    pub matmul_resources: MatMulResources,
}

impl PtxTarget {
    pub(crate) fn from_context(context: &CudaContext) -> Result<Self> {
        let (major, minor) = context
            .compute_capability()
            .context("Failed to query CUDA compute capability")?;
        let attribute = |attribute| -> Result<usize> {
            usize::try_from(context.attribute(attribute)?)
                .context("CUDA resource budget is negative")
        };
        Ok(Self {
            multiprocessor_count: attribute(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )?,
            matmul_resources: MatMulResources {
                shared_bytes_per_block: attribute(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                )?,
                shared_bytes_per_sm: attribute(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                )?,
                registers_per_sm: attribute(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                )?,
                threads_per_sm: attribute(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                )?,
            },
            compute_capability: (
                u32::try_from(major).context("CUDA compute capability major is negative")?,
                u32::try_from(minor).context("CUDA compute capability minor is negative")?,
            ),
        })
    }

    pub const fn matmul_capabilities(self) -> crate::tile::MatMulCapabilities {
        crate::tile::MatMulCapabilities {
            tf32: self.supports_tf32(),
            f16: self.compute_capability.0 >= 7,
            bf16: self.compute_capability.0 >= 8,
        }
    }

    pub const fn supports_async_copy(self) -> bool {
        self.compute_capability.0 >= 8
    }

    pub const fn supports_tf32(self) -> bool {
        self.compute_capability.0 >= 8
    }

    pub(crate) const fn sm80() -> Self {
        Self {
            multiprocessor_count: 1,
            compute_capability: (8, 0),
            matmul_resources: MatMulResources {
                shared_bytes_per_block: 48 * 1024,
                shared_bytes_per_sm: 64 * 1024,
                registers_per_sm: 64 * 1024,
                threads_per_sm: 1024,
            },
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
            let target = PtxTarget {
                multiprocessor_count: 1,
                compute_capability,
                matmul_resources: MatMulResources::default(),
            };
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

#[cfg(test)]
mod half_tests {
    use super::*;
    use crate::tile::{DType, MatMulPlan, MatMulPrecision, MatMulSchedule};

    #[test]
    fn half_schedules_respect_target_and_precision() {
        for (sm, f16, bf16) in [
            ((6, 1), MatMulPlan::ScalarF32, MatMulPlan::ScalarF32),
            ((7, 0), MatMulPlan::TensorCoreF16, MatMulPlan::ScalarF32),
            ((7, 5), MatMulPlan::TensorCoreF16, MatMulPlan::ScalarF32),
            (
                (8, 0),
                MatMulPlan::TensorCoreF16,
                MatMulPlan::TensorCoreBF16,
            ),
            (
                (8, 9),
                MatMulPlan::TensorCoreF16,
                MatMulPlan::TensorCoreBF16,
            ),
        ] {
            let caps = PtxTarget {
                multiprocessor_count: 1,
                compute_capability: sm,
                matmul_resources: MatMulResources::default(),
            }
            .matmul_capabilities();
            for (dtype, expected) in [(DType::F16, f16), (DType::BF16, bf16)] {
                let schedule = MatMulSchedule::select_for_dtype(
                    33,
                    35,
                    37,
                    dtype,
                    caps,
                    MatMulPrecision::AllowTf32,
                );
                assert_eq!(schedule.plan, expected);
                assert_eq!(schedule.storage_dtype, dtype);
                assert_eq!(
                    MatMulSchedule::select_for_dtype(
                        33,
                        35,
                        37,
                        dtype,
                        caps,
                        MatMulPrecision::StrictF32
                    )
                    .plan,
                    MatMulPlan::ScalarF32
                );
                assert_eq!(
                    MatMulSchedule::select_for_dtype(
                        33,
                        35,
                        0,
                        dtype,
                        caps,
                        MatMulPrecision::AllowTf32
                    )
                    .plan,
                    MatMulPlan::ScalarF32
                );
            }
        }
    }
}
