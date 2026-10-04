use anyhow::{Result, anyhow, ensure};

use super::{DType, MatMulPlan, SharedLayout, TileLayout};

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

/// Deterministic shared-memory layout selection, with explicit alternatives for
/// controlled comparisons. Swizzles preserve whole WMMA fragments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MatMulSharedLayout {
    #[default]
    Auto,
    Contiguous,
    Padded,
    Swizzled,
}

/// Bound operand movement to two shared-memory stages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MatMulPipeline {
    #[default]
    Auto,
    Synchronous,
    DoubleBuffered,
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
pub enum MatMulAccumulatorLayout {
    ThreadScalar,
    WarpFragment,
}

/// Single source of truth for launch geometry, shared layouts, and pipeline stages.
/// Scheduling stays independent of the logical region and its tensor shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulSchedule {
    pub plan: MatMulPlan,
    pub logical_shape: MatMulTile,
    pub block_tile: MatMulTile,
    pub instruction_tile: MatMulTile,
    pub block_threads: (u32, u32, u32),
    /// Warps that compute output fragments; remaining block warps stage operands.
    pub warp_topology: (usize, usize),
    pub operand_layouts: (SharedLayout, SharedLayout),
    /// One synchronous stage or two asynchronously copied stages.
    pub pipeline_stages: usize,
    /// Raw bytes moved per asynchronous copy group.
    pub copy_bytes: usize,
    pub operand_dtype: DType,
    /// Storage format at global-memory graph boundaries.
    pub storage_dtype: DType,
    pub accumulator_layout: MatMulAccumulatorLayout,
}

impl MatMulSchedule {
    pub fn with_shared_layout(
        mut self,
        policy: MatMulSharedLayout,
        resources: MatMulResources,
    ) -> Result<Self> {
        let tensor_core = self.plan != MatMulPlan::ScalarF32;
        let policy = match policy {
            MatMulSharedLayout::Auto
                if tensor_core && self.warp_topology.0 * self.warp_topology.1 > 1 =>
            {
                MatMulSharedLayout::Padded
            }
            MatMulSharedLayout::Auto => MatMulSharedLayout::Contiguous,
            explicit => explicit,
        };
        let layout = |cols| -> Result<SharedLayout> {
            match policy {
                MatMulSharedLayout::Padded => SharedLayout::padded(
                    cols,
                    if !tensor_core {
                        1
                    } else {
                        SharedLayout::wmma_alignment(self.operand_dtype)
                            / self.operand_dtype.size_bytes()
                    },
                ),
                MatMulSharedLayout::Swizzled => {
                    SharedLayout::swizzled(cols, if tensor_core { 16 } else { 1 })
                }
                _ => Ok(SharedLayout::row_major(cols)),
            }
        };
        ensure!(
            self.block_tile.m > 0 && self.block_tile.n > 0 && self.block_tile.k > 0,
            "matmul block tile dimensions must be positive"
        );
        let (a, b) = (layout(self.block_tile.k)?, layout(self.block_tile.n)?);
        a.validate(self.block_tile.m, self.block_tile.k, self.operand_dtype)?;
        b.validate(self.block_tile.k, self.block_tile.n, self.operand_dtype)?;
        let c = SharedLayout::row_major(self.block_tile.n);
        c.validate(self.block_tile.m, self.block_tile.n, DType::F32)?;
        self.operand_layouts = (
            SharedLayout::row_major(self.block_tile.k),
            SharedLayout::row_major(self.block_tile.n),
        );
        let accumulator_bytes = if tensor_core {
            c.allocation_bytes(self.block_tile.m, DType::F32)
        } else {
            0
        };
        let shared_bytes = a
            .allocation_bytes(self.block_tile.m, self.operand_dtype)
            .checked_add(b.allocation_bytes(self.block_tile.k, self.operand_dtype))
            .and_then(|bytes| bytes.checked_add(accumulator_bytes))
            .ok_or_else(|| anyhow!("matmul shared allocation size overflows usize"))?;
        if shared_bytes <= resources.shared_bytes_per_block
            && resources.shared_bytes_per_sm / shared_bytes >= 2
        {
            self.operand_layouts = (a, b);
        }
        Ok(self)
    }

    /// Select after operand layouts. `contiguous` proves aligned source groups
    /// can follow the logical matrix columns without crossing a physical row.
    pub fn with_pipeline(
        mut self,
        policy: MatMulPipeline,
        supports_async: bool,
        contiguous: bool,
        resources: MatMulResources,
    ) -> Self {
        self.pipeline_stages = 1;
        self.copy_bytes = 4;
        let tile = self.block_tile;
        let pairs_aligned = self.storage_dtype.size_bytes() == 4
            || (contiguous
                && self.logical_shape.k.is_multiple_of(2)
                && self.logical_shape.n.is_multiple_of(2)
                && [self.operand_layouts.0, self.operand_layouts.1]
                    .iter()
                    .all(|layout| layout.row_stride.is_multiple_of(2) && layout.xor_mask & 1 == 0));
        let operands = self
            .operand_layouts
            .0
            .allocation_bytes(tile.m, self.storage_dtype)
            + self
                .operand_layouts
                .1
                .allocation_bytes(tile.k, self.storage_dtype);
        let accumulator = if self.plan == MatMulPlan::ScalarF32 {
            0
        } else {
            SharedLayout::row_major(tile.n).allocation_bytes(tile.m, DType::F32)
        };
        let bytes = 2 * operands + accumulator;
        if supports_async
            && pairs_aligned
            && policy != MatMulPipeline::Synchronous
            && self.logical_shape.k > tile.k
            && (policy == MatMulPipeline::DoubleBuffered || self.plan != MatMulPlan::ScalarF32)
            && bytes <= resources.shared_bytes_per_block
            && resources.shared_bytes_per_sm / bytes >= 2
        {
            self.pipeline_stages = 2;
            let group = 16 / self.storage_dtype.size_bytes();
            if contiguous
                && self.logical_shape.k.is_multiple_of(group)
                && self.logical_shape.n.is_multiple_of(group)
                && [self.operand_layouts.0, self.operand_layouts.1]
                    .iter()
                    .all(|layout| {
                        layout.row_stride.is_multiple_of(group)
                            && layout.xor_mask & (group - 1) == 0
                    })
            {
                self.copy_bytes = 16;
            }
            // Auto amortizes pipeline startup only on vectorized copies with
            // several K tiles and computing warps. Explicit mode exercises the other paths.
            if policy == MatMulPipeline::Auto
                && (self.copy_bytes != 16
                    || self.logical_shape.k.div_ceil(tile.k) < 4
                    || self.warp_topology.0 * self.warp_topology.1 == 1)
            {
                self.pipeline_stages = 1;
                self.copy_bytes = 4;
            }
        }
        self
    }

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
            operand_layouts: (
                SharedLayout::row_major(block_tile.k),
                SharedLayout::row_major(block_tile.n),
            ),
            pipeline_stages: 1,
            copy_bytes: 4,
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
    #[test]
    fn pipeline_selection_respects_target_tail_alignment_and_resources() {
        let caps = MatMulCapabilities {
            tf32: true,
            f16: true,
            bf16: true,
        };
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for k in [8, 16, 17, 32, 33, 64, 66, 96] {
                let base = MatMulSchedule::select_for_dtype(
                    33,
                    258,
                    k,
                    dtype,
                    caps,
                    MatMulPrecision::AllowTf32,
                )
                .with_shared_layout(MatMulSharedLayout::Auto, MatMulResources::default())
                .unwrap();
                let selected = base.with_pipeline(
                    MatMulPipeline::DoubleBuffered,
                    true,
                    true,
                    MatMulResources::default(),
                );
                let expected =
                    k > base.block_tile.k && (dtype == DType::F32 || k.is_multiple_of(2));
                assert_eq!(selected.pipeline_stages, if expected { 2 } else { 1 });
                for (policy, supported, resources) in [
                    (
                        MatMulPipeline::DoubleBuffered,
                        false,
                        MatMulResources::default(),
                    ),
                    (
                        MatMulPipeline::Synchronous,
                        true,
                        MatMulResources::default(),
                    ),
                    (
                        MatMulPipeline::DoubleBuffered,
                        true,
                        MatMulResources {
                            shared_bytes_per_block: 1024,
                            ..Default::default()
                        },
                    ),
                ] {
                    assert_eq!(
                        base.with_pipeline(policy, supported, true, resources)
                            .pipeline_stages,
                        1
                    );
                }
            }
            let single_warp = MatMulSchedule::select_for_dtype(
                33,
                128,
                128,
                dtype,
                caps,
                MatMulPrecision::AllowTf32,
            )
            .with_pipeline(
                MatMulPipeline::Auto,
                true,
                true,
                MatMulResources::default(),
            );
            assert_eq!(single_warp.pipeline_stages, 1);
            let base = MatMulSchedule::select_for_dtype(
                128,
                768,
                256,
                dtype,
                caps,
                MatMulPrecision::AllowTf32,
            )
            .with_shared_layout(MatMulSharedLayout::Auto, MatMulResources::default())
            .unwrap();
            let vector = base.with_pipeline(
                MatMulPipeline::DoubleBuffered,
                true,
                true,
                MatMulResources::default(),
            );
            assert_eq!(vector.pipeline_stages, 2);
            assert_eq!(vector.copy_bytes, 16);
            assert_eq!(
                base.with_pipeline(MatMulPipeline::Auto, true, true, MatMulResources::default())
                    .pipeline_stages,
                2
            );
            assert_eq!(
                base.with_pipeline(
                    MatMulPipeline::Auto,
                    true,
                    false,
                    MatMulResources::default()
                )
                .pipeline_stages,
                1
            );
            assert_eq!(
                base.with_pipeline(
                    MatMulPipeline::DoubleBuffered,
                    true,
                    false,
                    MatMulResources::default()
                )
                .copy_bytes,
                4
            );
        }
    }

    #[test]
    fn layout_selection_respects_instruction_alignment_and_memory_budgets() {
        let caps = MatMulCapabilities {
            tf32: true,
            f16: true,
            bf16: true,
        };
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let base = MatMulSchedule::select_for_dtype(
                512,
                512,
                512,
                dtype,
                caps,
                MatMulPrecision::AllowTf32,
            );
            for policy in [MatMulSharedLayout::Padded, MatMulSharedLayout::Swizzled] {
                let selected = base
                    .with_shared_layout(policy, MatMulResources::default())
                    .unwrap();
                assert!(
                    selected
                        .operand_layouts
                        .0
                        .supports_wmma(selected.operand_dtype)
                );
                assert!(
                    selected
                        .operand_layouts
                        .1
                        .supports_wmma(selected.operand_dtype)
                );
                let constrained = base
                    .with_shared_layout(
                        policy,
                        MatMulResources {
                            shared_bytes_per_block: 1024,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(constrained.operand_layouts, base.operand_layouts);
            }
        }
    }
}
