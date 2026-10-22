use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, sys::CUdevice_attribute};

use crate::tile::{MatMulResources, ReductionSchedule};

/// Independent elementwise lanes do not need a full 1,024-thread block.
/// Smaller blocks leave room for more resident blocks when register use rises.
pub(crate) fn pointwise_launch_config(elements: u32) -> LaunchConfig {
    const THREADS: u32 = 256;
    LaunchConfig {
        grid_dim: (elements.div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Compiled resource usage can require fewer independent warps per block.
/// Reducing the number of whole warps preserves each fiber's arithmetic and mapping.
pub(crate) fn reduction_launch_config(
    schedule: ReductionSchedule,
    rows: u32,
    function: &CudaFunction,
) -> Result<LaunchConfig> {
    let threads = match schedule {
        ReductionSchedule::Serial => return Ok(LaunchConfig::for_num_elems(rows)),
        ReductionSchedule::Subgroup { width } => width,
        ReductionSchedule::Block { threads } => {
            return Ok(LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            });
        }
    };
    let limit = u32::try_from(function.max_threads_per_block()?)
        .context("Reduction function thread limit is negative")?;
    let mut fibers = schedule.fibers_per_block();
    loop {
        let block_threads = threads
            .checked_mul(fibers)
            .context("Reduction block thread count overflow")?;
        if block_threads <= limit
            && function.occupancy_max_active_blocks_per_multiprocessor(block_threads, 0, None)? > 0
        {
            return Ok(LaunchConfig {
                grid_dim: (rows.div_ceil(fibers), 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: 0,
            });
        }
        anyhow::ensure!(
            fibers > 1,
            "Reduction fiber does not fit compiled CUDA resources"
        );
        fibers /= 2;
    }
}

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

#[cfg(test)]
mod reduction_launch_tests {
    use super::*;
    use cudarc::nvrtc::Ptx;

    #[test]
    fn packed_reduction_launch_respects_compiled_function_limit() {
        let device = match CudaContext::new(0) {
            Ok(device) => device,
            Err(error) if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1") => {
                eprintln!("skipping CUDA test: {error:#}");
                return;
            }
            Err(error) => panic!("CUDA initialization: {error:#}"),
        };
        let (major, minor) = device.compute_capability().unwrap();
        let source = format!(
            ".version 8.6\n.target sm_{major}{minor}\n.address_size 64\n.visible .entry bounded() .maxntid 128, 1, 1 {{ ret; }}"
        );
        let module = device.load_module(Ptx::from_src(source)).unwrap();
        let function = module.load_function("bounded").unwrap();
        for schedule in [
            ReductionSchedule::Subgroup { width: 32 },
            ReductionSchedule::Block { threads: 128 },
        ] {
            let config = reduction_launch_config(schedule, 9, &function).unwrap();
            assert_eq!(config.block_dim, (128, 1, 1));
            assert_eq!(
                config.grid_dim,
                (
                    if matches!(schedule, ReductionSchedule::Block { .. }) {
                        9
                    } else {
                        3
                    },
                    1,
                    1
                )
            );
        }
    }
}
