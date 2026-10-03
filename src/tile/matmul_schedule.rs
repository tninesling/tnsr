use super::{DType, MatMulPlan, TileLayout};

/// Arithmetic policy, independent of the target's available instructions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MatMulPrecision {
    /// Use increasing-K scalar f32 accumulation for every storage dtype.
    StrictF32,
    /// Permit tensor cores: TF32 for f32 storage, native FP16/BF16 for half storage.
    #[default]
    AllowTf32,
}

/// Tensor-core operand formats supported by a CUDA target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulCapabilities {
    pub tf32: bool,
    pub f16: bool,
    pub bf16: bool,
}

/// Resource budgets used for conservative occupancy estimates. The register
/// estimate reserves 128 registers per thread; JIT allocation can vary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatMulResources {
    pub shared_bytes_per_block: usize,
    pub shared_bytes_per_sm: usize,
    pub registers_per_sm: usize,
    pub threads_per_sm: usize,
}

impl Default for MatMulResources {
    fn default() -> Self {
        Self {
            shared_bytes_per_block: 48 * 1024,
            shared_bytes_per_sm: 64 * 1024,
            registers_per_sm: 64 * 1024,
            threads_per_sm: 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulTile {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatMulOperandLayout {
    RowMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatMulAccumulatorLayout {
    ThreadScalar,
    WarpFragment,
}

/// Single source of truth for lowering and launch geometry. These first schedules
/// use synchronous shared-memory staging; layouts and deeper pipelines can evolve
/// without changing the logical region or encoding a GPU tile in its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulSchedule {
    pub plan: MatMulPlan,
    pub logical_shape: MatMulTile,
    pub block_tile: MatMulTile,
    pub instruction_tile: MatMulTile,
    pub block_threads: (u32, u32, u32),
    /// Warps that compute output fragments; remaining block warps stage operands.
    pub warp_topology: (usize, usize),
    pub operand_layouts: (MatMulOperandLayout, MatMulOperandLayout),
    pub operand_dtype: DType,
    /// Storage format at global-memory graph boundaries.
    pub storage_dtype: DType,
    pub accumulator_layout: MatMulAccumulatorLayout,
    pub pipeline_stages: usize,
}

impl MatMulSchedule {
    pub fn thread_count(&self) -> usize {
        self.block_threads.0 as usize
            * self.block_threads.1 as usize
            * self.block_threads.2 as usize
    }

    /// Representation supplied to epilogue layout propagation.
    pub fn accumulator_tile_layout(&self) -> TileLayout {
        match self.accumulator_layout {
            MatMulAccumulatorLayout::ThreadScalar => TileLayout::ThreadScalar,
            MatMulAccumulatorLayout::WarpFragment => TileLayout::WarpAccumulator {
                operand_dtype: self.operand_dtype,
                block_width: self.block_threads.0,
                warp_topology: self.warp_topology,
            },
        }
    }

    pub fn select(
        m: usize,
        n: usize,
        k: usize,
        capabilities: MatMulCapabilities,
        precision: MatMulPrecision,
    ) -> Self {
        Self::select_for_dtype(m, n, k, DType::F32, capabilities, precision)
    }

    pub fn select_for_dtype(
        m: usize,
        n: usize,
        k: usize,
        storage_dtype: DType,
        capabilities: MatMulCapabilities,
        precision: MatMulPrecision,
    ) -> Self {
        Self::select_with_resources(
            m,
            n,
            k,
            storage_dtype,
            capabilities,
            precision,
            MatMulResources::default(),
        )
    }

    pub fn select_with_resources(
        m: usize,
        n: usize,
        k: usize,
        storage_dtype: DType,
        capabilities: MatMulCapabilities,
        precision: MatMulPrecision,
        resources: MatMulResources,
    ) -> Self {
        let large_half =
            m >= 16 && n >= 16 && k >= 16 && m.saturating_mul(n).saturating_mul(k) >= 32 * 32 * 32;
        let plan = match storage_dtype {
            DType::F16
                if large_half && capabilities.f16 && precision != MatMulPrecision::StrictF32 =>
            {
                MatMulPlan::TensorCoreF16
            }
            DType::BF16
                if large_half && capabilities.bf16 && precision != MatMulPrecision::StrictF32 =>
            {
                MatMulPlan::TensorCoreBF16
            }
            DType::F32 => MatMulPlan::for_shape_with_tf32(
                m,
                n,
                k,
                capabilities.tf32 && precision == MatMulPrecision::AllowTf32,
            ),
            _ => MatMulPlan::ScalarF32,
        };
        let tensor_core = plan != MatMulPlan::ScalarF32;
        // Keep small and narrow work on the original tile. Larger tiles reuse
        // operands across two or four computing warps without growing the block.
        let block_tile = if tensor_core && m >= 256 && n >= 256 && k >= 64 {
            MatMulTile {
                m: 32,
                n: 32,
                k: 32,
            }
        } else if tensor_core && m >= 32 && n >= 256 && n / m >= 2 && k >= 64 {
            MatMulTile {
                m: 16,
                n: 32,
                k: 32,
            }
        } else {
            MatMulTile {
                m: 16,
                n: 16,
                k: 16,
            }
        };
        let shared_bytes = (block_tile.m * block_tile.k + block_tile.k * block_tile.n)
            * storage_dtype.size_bytes()
            + block_tile.m * block_tile.n * 4;
        let resident_blocks = (resources.threads_per_sm / 256)
            .min(resources.registers_per_sm / (128 * 256))
            .min(resources.shared_bytes_per_sm / shared_bytes);
        let block_tile = if shared_bytes <= resources.shared_bytes_per_block && resident_blocks >= 2
        {
            block_tile
        } else {
            MatMulTile {
                m: 16,
                n: 16,
                k: 16,
            }
        };
        let expanded = block_tile.n != 16;

        Self {
            plan,
            logical_shape: MatMulTile { m, n, k },
            block_tile,
            instruction_tile: if tensor_core {
                MatMulTile {
                    m: 16,
                    n: 16,
                    k: if storage_dtype == DType::F32 { 8 } else { 16 },
                }
            } else {
                MatMulTile { m: 1, n: 1, k: 1 }
            },
            block_threads: if expanded { (32, 8, 1) } else { (16, 16, 1) },
            warp_topology: if tensor_core {
                (block_tile.m / 16, block_tile.n / 16)
            } else {
                (8, 1)
            },
            operand_layouts: (MatMulOperandLayout::RowMajor, MatMulOperandLayout::RowMajor),
            operand_dtype: if plan == MatMulPlan::TensorCoreTf32 {
                DType::TF32
            } else {
                storage_dtype
            },
            storage_dtype,
            accumulator_layout: if tensor_core {
                MatMulAccumulatorLayout::WarpFragment
            } else {
                MatMulAccumulatorLayout::ThreadScalar
            },
            pipeline_stages: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_and_precision_gate_tensor_core_schedule() {
        for tf32 in [false, true] {
            for precision in [MatMulPrecision::StrictF32, MatMulPrecision::AllowTf32] {
                let schedule = MatMulSchedule::select(
                    128,
                    128,
                    128,
                    MatMulCapabilities {
                        tf32,
                        f16: false,
                        bf16: false,
                    },
                    precision,
                );
                assert_eq!(
                    schedule.plan == MatMulPlan::TensorCoreTf32,
                    tf32 && precision == MatMulPrecision::AllowTf32
                );
                assert_eq!(schedule.block_threads, (16, 16, 1));
                assert_eq!(schedule.pipeline_stages, 1);
            }
        }
    }

    #[test]
    fn small_and_zero_k_shapes_use_scalar_schedule() {
        for (m, n, k) in [(15, 17, 16), (1, 1, 512), (128, 128, 0)] {
            assert_eq!(
                MatMulSchedule::select(
                    m,
                    n,
                    k,
                    MatMulCapabilities {
                        tf32: true,
                        f16: false,
                        bf16: false
                    },
                    MatMulPrecision::AllowTf32
                )
                .plan,
                MatMulPlan::ScalarF32
            );
        }
    }

    #[test]
    fn shape_families_select_tiles_for_each_supported_format() {
        let caps = MatMulCapabilities {
            tf32: true,
            f16: true,
            bf16: true,
        };
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (m, n, k, tile, warps) in [
                (64, 64, 64, (16, 16, 16), (1, 1)),
                (128, 768, 256, (16, 32, 32), (1, 2)),
                (33, 257, 65, (16, 32, 32), (1, 2)),
                (257, 259, 65, (32, 32, 32), (2, 2)),
                (128, 128, 1024, (16, 16, 16), (1, 1)),
            ] {
                let schedule = MatMulSchedule::select_for_dtype(
                    m,
                    n,
                    k,
                    dtype,
                    caps,
                    MatMulPrecision::AllowTf32,
                );
                assert_eq!(
                    (
                        schedule.block_tile.m,
                        schedule.block_tile.n,
                        schedule.block_tile.k
                    ),
                    tile
                );
                assert_eq!(schedule.warp_topology, warps);
                assert_eq!(
                    schedule.instruction_tile.k,
                    if dtype == DType::F32 { 8 } else { 16 }
                );
                let strict = MatMulSchedule::select_for_dtype(
                    m,
                    n,
                    k,
                    dtype,
                    caps,
                    MatMulPrecision::StrictF32,
                );
                assert_eq!(strict.plan, MatMulPlan::ScalarF32);
                assert_eq!(strict.block_threads, (16, 16, 1));
            }
        }
    }

    #[test]
    fn constrained_targets_keep_the_original_tile() {
        let caps = MatMulCapabilities {
            tf32: true,
            f16: true,
            bf16: true,
        };
        for resources in [
            MatMulResources {
                shared_bytes_per_block: 4096,
                ..Default::default()
            },
            MatMulResources {
                shared_bytes_per_sm: 12 * 1024,
                ..Default::default()
            },
            MatMulResources {
                registers_per_sm: 32 * 1024,
                ..Default::default()
            },
            MatMulResources {
                threads_per_sm: 256,
                ..Default::default()
            },
        ] {
            let schedule = MatMulSchedule::select_with_resources(
                512,
                512,
                512,
                DType::F32,
                caps,
                MatMulPrecision::AllowTf32,
                resources,
            );
            assert_eq!(
                schedule.block_tile,
                MatMulTile {
                    m: 16,
                    n: 16,
                    k: 16
                }
            );
            assert_eq!(schedule.warp_topology, (1, 1));
        }
    }
}
