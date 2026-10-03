use super::{DType, MatMulPlan};

/// Arithmetic policy, independent of the target's available instructions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MatMulPrecision {
    /// Preserve F32 operands and increasing-K scalar accumulation.
    StrictF32,
    /// Permit TF32 operand rounding on capable targets.
    #[default]
    AllowTf32,
}

/// Capabilities relevant to the currently implemented F32 schedules.
/// FP16 tensor cores alone (for example on SM75) cannot accelerate these operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulCapabilities {
    pub tf32: bool,
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
    pub accumulator_layout: MatMulAccumulatorLayout,
    pub pipeline_stages: usize,
}

impl MatMulSchedule {
    pub fn select(
        m: usize,
        n: usize,
        k: usize,
        capabilities: MatMulCapabilities,
        precision: MatMulPrecision,
    ) -> Self {
        let plan = MatMulPlan::for_shape_with_tf32(
            m,
            n,
            k,
            capabilities.tf32 && precision == MatMulPrecision::AllowTf32,
        );
        let tensor_core = plan == MatMulPlan::TensorCoreTf32;
        Self {
            plan,
            logical_shape: MatMulTile { m, n, k },
            block_tile: MatMulTile {
                m: 16,
                n: 16,
                k: 16,
            },
            instruction_tile: if tensor_core {
                MatMulTile { m: 16, n: 16, k: 8 }
            } else {
                MatMulTile { m: 1, n: 1, k: 1 }
            },
            block_threads: (16, 16, 1),
            warp_topology: if tensor_core { (1, 1) } else { (8, 1) },
            operand_layouts: (MatMulOperandLayout::RowMajor, MatMulOperandLayout::RowMajor),
            operand_dtype: if tensor_core { DType::TF32 } else { DType::F32 },
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
                let schedule =
                    MatMulSchedule::select(128, 128, 128, MatMulCapabilities { tf32 }, precision);
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
                    MatMulCapabilities { tf32: true },
                    MatMulPrecision::AllowTf32
                )
                .plan,
                MatMulPlan::ScalarF32
            );
        }
    }
}
