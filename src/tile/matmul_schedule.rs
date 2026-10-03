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
    /// Representation supplied to epilogue layout propagation.
    pub fn accumulator_tile_layout(&self) -> TileLayout {
        match self.accumulator_layout {
            MatMulAccumulatorLayout::ThreadScalar => TileLayout::ThreadScalar,
            MatMulAccumulatorLayout::WarpFragment => TileLayout::WarpAccumulator {
                operand_dtype: self.operand_dtype,
                block_width: self.block_threads.0,
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
        Self {
            plan,
            logical_shape: MatMulTile { m, n, k },
            block_tile: MatMulTile {
                m: 16,
                n: 16,
                k: 16,
            },
            instruction_tile: if tensor_core {
                MatMulTile {
                    m: 16,
                    n: 16,
                    k: if storage_dtype == DType::F32 { 8 } else { 16 },
                }
            } else {
                MatMulTile { m: 1, n: 1, k: 1 }
            },
            block_threads: (16, 16, 1),
            warp_topology: if tensor_core { (1, 1) } else { (8, 1) },
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
}
