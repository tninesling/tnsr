use anyhow::{Result, anyhow, ensure};

use super::{DType, Dim, Expr, MatMulPlan, SharedLayout, TileLayout};

/// Arithmetic policy, independent of the target's available instructions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MatMulPrecision {
    /// Use increasing-K scalar f32 accumulation for every storage dtype.
    StrictF32,
    /// Keep f32 operands on ordered scalar accumulation while allowing native
    /// FP16/BF16 tensor cores with f32 accumulation for half-storage operands.
    NativeHalf,
    /// F32 storage uses three TF32 products (hi*hi + hi*lo + lo*hi) with f32 accumulation.
    /// Retains substantially more operand precision than a single TF32 product;
    /// accumulation order and rounding still differ from StrictF32.
    /// Half storage uses its native tensor-core format, as with AllowTf32.
    Tf32x3,
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
    ThreadArray { elements: usize },
    WarpFragment,
}

/// Single source of truth for launch geometry, shared layouts, and pipeline stages.
/// Scheduling stays independent of the logical region and its tensor shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulSchedule {
    pub plan: MatMulPlan,
    pub logical_shape: MatMulTile,
    /// Number of neighboring output row tiles visited before advancing columns.
    /// One retains the ordinary row-major CUDA grid traversal.
    pub grid_group_m: usize,
    pub block_tile: MatMulTile,
    pub instruction_tile: MatMulTile,
    pub block_threads: (u32, u32, u32),
    /// Warps that compute output fragments; remaining block warps stage operands.
    pub warp_topology: (usize, usize),
    /// Number of 16x16 accumulator fragments owned by each computing warp.
    pub warp_fragments: (usize, usize),
    pub operand_layouts: (SharedLayout, SharedLayout),
    /// One synchronous stage or two asynchronously copied stages.
    pub pipeline_stages: usize,
    /// Raw bytes moved per asynchronous copy group.
    pub copy_bytes: usize,
    /// Operands copied asynchronously; others use predicated scalar staging.
    pub async_operands: (bool, bool),
    pub operand_dtype: DType,
    /// Storage format at global-memory graph boundaries.
    pub storage_dtype: DType,
    pub accumulator_layout: MatMulAccumulatorLayout,
}

impl MatMulSchedule {
    /// Experimental SIMT outer-product tile. Defaults never call this selector.
    /// Small output grids, including PPO weight gradients, retain their schedule.
    /// Apply this before selecting operand layouts and asynchronous staging.
    pub fn with_experimental_register_tile(mut self, resources: MatMulResources) -> Self {
        if self.plan != MatMulPlan::ScalarF32
            || self.storage_dtype != DType::F32
            || self.logical_shape.m < 1024
            || self.logical_shape.n < 256
            || self.logical_shape.k < 128
        {
            return self;
        }
        // No shared accumulator exchange is needed. Budget two resident CTAs
        // at 128 registers/thread; the actual JIT resource usage must be measured.
        let shared_bytes = (64 * 16 + 16 * 64) * 4;
        if resources.shared_bytes_per_block < shared_bytes
            || resources.shared_bytes_per_sm < 2 * shared_bytes
            || resources.registers_per_sm < 2 * 128 * 256
            || resources.threads_per_sm < 512
        {
            return self;
        }
        self.block_tile = MatMulTile {
            m: 64,
            n: 64,
            k: 16,
        };
        self.block_threads = (16, 16, 1);
        self.warp_topology = (8, 1);
        self.warp_fragments = (1, 1);
        self.accumulator_layout = MatMulAccumulatorLayout::ThreadArray { elements: 16 };
        self.operand_layouts = (SharedLayout::row_major(16), SharedLayout::row_major(64));
        self.pipeline_stages = 1;
        self.copy_bytes = 4;
        self.async_operands = (false, false);
        self
    }

    /// Per-thread output shape for supported scalar register-array schedules.
    /// The flattened accumulator slot is row-major within this microtile.
    pub fn register_thread_tile(&self) -> Option<(usize, usize)> {
        if self.plan != MatMulPlan::ScalarF32
            || self.storage_dtype != DType::F32
            || self.block_threads != (16, 16, 1)
        {
            return None;
        }
        match (self.block_tile, self.accumulator_layout) {
            (
                MatMulTile {
                    m: 32,
                    n: 32,
                    k: 16 | 32,
                },
                MatMulAccumulatorLayout::ThreadArray { elements: 4 },
            ) => Some((4, 1)),
            (
                MatMulTile {
                    m: 64,
                    n: 64,
                    k: 16,
                },
                MatMulAccumulatorLayout::ThreadArray { elements: 16 },
            ) => Some((4, 4)),
            _ => None,
        }
    }

    /// Authoritative output ownership shared by scalar lowering and epilogues.
    pub fn register_accumulator_coordinates(&self, slot: usize) -> Option<(Expr, Expr)> {
        let (thread_rows, thread_cols) = self.register_thread_tile()?;
        if slot >= thread_rows * thread_cols {
            return None;
        }
        let lane_rows = self.block_tile.m / thread_rows;
        let lane_cols = self.block_tile.n / thread_cols;
        let linear =
            Expr::ThreadIdx(Dim::Y) * self.block_threads.0 as usize + Expr::ThreadIdx(Dim::X);
        let row = Expr::ShiftRight(
            Box::new(linear.clone()),
            u64::from(lane_cols.trailing_zeros()),
        ) + Expr::Const(((slot / thread_cols) * lane_rows) as i64);
        let col = Expr::BitAnd(Box::new(linear), (lane_cols - 1) as u64)
            + Expr::Const(((slot % thread_cols) * lane_cols) as i64);
        Some((row, col))
    }

    pub fn validate_grid_group_m(group: usize) -> Result<()> {
        ensure!(
            group.is_power_of_two() && group <= 32,
            "matmul grid row group must be a power of two from 1 through 32"
        );
        Ok(())
    }

    /// Change output-tile launch order only; each output retains its K order.
    pub fn with_grid_group_m(mut self, group: usize) -> Result<Self> {
        Self::validate_grid_group_m(group)?;
        self.grid_group_m = group;
        Ok(self)
    }

    /// Partial output-row groups require more expensive index selection. Keep
    /// a power-of-two group dividing the tile grid; odd row counts fall back
    /// to row-major traversal. Logical element tails still use normal bounds.
    pub fn effective_grid_group_m(&self) -> usize {
        let rows = self.logical_shape.m.div_ceil(self.block_tile.m);
        let mut group = self.grid_group_m;
        while group > 1 && !rows.is_multiple_of(group) {
            group /= 2;
        }
        group
    }

    pub fn with_shared_layout(
        self,
        policy: MatMulSharedLayout,
        resources: MatMulResources,
    ) -> Result<Self> {
        self.with_operand_orientations(policy, (false, false), resources)
    }

    pub(crate) fn with_operand_orientations(
        mut self,
        policy: MatMulSharedLayout,
        column_major: (bool, bool),
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
        let mut a = layout(if column_major.0 {
            self.block_tile.m
        } else {
            self.block_tile.k
        })?;
        let mut b = layout(if column_major.1 {
            self.block_tile.k
        } else {
            self.block_tile.n
        })?;
        a.column_major = column_major.0;
        b.column_major = column_major.1;
        a.validate(self.block_tile.m, self.block_tile.k, self.operand_dtype)?;
        b.validate(self.block_tile.k, self.block_tile.n, self.operand_dtype)?;
        let c = SharedLayout::row_major(self.block_tile.n);
        c.validate(self.block_tile.m, self.block_tile.n, DType::F32)?;
        self.operand_layouts = (
            SharedLayout::row_major(self.block_tile.k),
            SharedLayout::row_major(self.block_tile.n),
        );
        let accumulator_bytes = if tensor_core {
            c.allocation_bytes(self.block_tile.m, self.block_tile.n, DType::F32)
        } else {
            0
        };
        let shared_bytes = a
            .allocation_bytes(self.block_tile.m, self.block_tile.k, self.operand_dtype)
            .checked_add(b.allocation_bytes(
                self.block_tile.k,
                self.block_tile.n,
                self.operand_dtype,
            ))
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
    /// along each layout's physical minor axis without crossing a physical row.
    pub fn with_pipeline(
        self,
        policy: MatMulPipeline,
        supports_async: bool,
        contiguous: bool,
        resources: MatMulResources,
    ) -> Self {
        self.with_operand_pipeline(policy, supports_async, contiguous, (true, true), resources)
    }

    pub(crate) fn with_operand_pipeline(
        mut self,
        policy: MatMulPipeline,
        supports_async: bool,
        contiguous: bool,
        async_operands: (bool, bool),
        resources: MatMulResources,
    ) -> Self {
        self.pipeline_stages = 1;
        self.copy_bytes = 4;
        self.async_operands = (false, false);
        let tile = self.block_tile;
        let copy_extents = [
            if self.operand_layouts.0.column_major {
                self.logical_shape.m
            } else {
                self.logical_shape.k
            },
            if self.operand_layouts.1.column_major {
                self.logical_shape.k
            } else {
                self.logical_shape.n
            },
        ];
        let active = [async_operands.0, async_operands.1];
        let layouts = [self.operand_layouts.0, self.operand_layouts.1];
        let aligned = |group: usize| {
            copy_extents
                .iter()
                .zip(layouts)
                .zip(active)
                .all(|((&extent, layout), enabled)| {
                    !enabled
                        || (extent.is_multiple_of(group)
                            && layout.row_stride.is_multiple_of(group)
                            && layout.xor_mask & (group - 1) == 0)
                })
        };
        let pairs_aligned = self.storage_dtype.size_bytes() == 4 || (contiguous && aligned(2));
        let operands = self
            .operand_layouts
            .0
            .allocation_bytes(tile.m, tile.k, self.storage_dtype)
            + self
                .operand_layouts
                .1
                .allocation_bytes(tile.k, tile.n, self.storage_dtype);
        let accumulator = if self.plan == MatMulPlan::ScalarF32 {
            0
        } else {
            SharedLayout::row_major(tile.n).allocation_bytes(tile.m, tile.n, DType::F32)
        };
        let bytes = 2 * operands + accumulator;
        if supports_async
            && active.iter().any(|enabled| *enabled)
            && pairs_aligned
            && policy != MatMulPipeline::Synchronous
            && self.logical_shape.k > tile.k
            && (policy == MatMulPipeline::DoubleBuffered || self.plan != MatMulPlan::ScalarF32)
            && bytes <= resources.shared_bytes_per_block
            && resources.shared_bytes_per_sm / bytes >= 2
        {
            self.pipeline_stages = 2;
            self.async_operands = async_operands;
            let group = 16 / self.storage_dtype.size_bytes();
            if contiguous && aligned(group) {
                self.copy_bytes = 16;
            }
            // Large BF16 register tiles amortize vectorized, two-operand staging
            // with only two K tiles. Hybrid copies and smaller tiles retain the
            // longer startup threshold.
            let short_register_pipeline = self.plan == MatMulPlan::TensorCoreBF16
                && tile
                    == (MatMulTile {
                        m: 64,
                        n: 64,
                        k: 32,
                    })
                && self.warp_topology == (2, 2)
                && self.warp_fragments == (2, 2)
                && async_operands == (true, true);
            if policy == MatMulPipeline::Auto
                && (self.copy_bytes != 16
                    || (self.logical_shape.k.div_ceil(tile.k) < 4 && !short_register_pipeline)
                    || self.warp_topology.0 * self.warp_topology.1 == 1)
            {
                self.pipeline_stages = 1;
                self.copy_bytes = 4;
                self.async_operands = (false, false);
            }
        }
        self
    }

    pub(crate) fn copy_group(&self, lhs: bool) -> usize {
        let enabled = if lhs {
            self.async_operands.0
        } else {
            self.async_operands.1
        };
        if self.pipeline_stages == 2 && enabled {
            self.copy_bytes / self.storage_dtype.size_bytes()
        } else {
            1
        }
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
            MatMulAccumulatorLayout::ThreadArray { elements } => {
                TileLayout::ThreadArray { elements }
            }
            MatMulAccumulatorLayout::WarpFragment => TileLayout::WarpAccumulator {
                operand_dtype: self.operand_dtype,
                block_width: self.block_threads.0,
                warp_topology: self.warp_topology,
                warp_fragments: self.warp_fragments,
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
                capabilities.tf32
                    && matches!(
                        precision,
                        MatMulPrecision::AllowTf32 | MatMulPrecision::Tf32x3
                    ),
            ),
            _ => MatMulPlan::ScalarF32,
        };
        let plan = if plan == MatMulPlan::TensorCoreTf32 && precision == MatMulPrecision::Tf32x3 {
            MatMulPlan::TensorCoreTf32x3
        } else {
            plan
        };
        let tensor_core = plan != MatMulPlan::ScalarF32;
        // Keep small and narrow work on the original tile. Larger tiles reuse
        // operands across two or four computing warps without growing the block.
        let register_tiled = plan == MatMulPlan::ScalarF32
            && storage_dtype == DType::F32
            && m >= 128
            && n >= 256
            && k >= 128;
        // A wider staging tile halves the outer load/barrier iterations without
        // changing the increasing-K multiply/add sequence. Include accumulator
        // storage in the budget and retain two resident blocks; tighter targets
        // keep the existing 16-wide staging tile.
        let scalar_staging_k = if resources.shared_bytes_per_block >= 12 * 1024
            && resources.shared_bytes_per_sm >= 24 * 1024
        {
            32
        } else {
            16
        };
        let block_tile = if register_tiled {
            MatMulTile {
                m: 32,
                n: 32,
                k: scalar_staging_k,
            }
        } else if plan == MatMulPlan::TensorCoreBF16 && m >= 2048 && n >= 256 && k >= 64 {
            MatMulTile {
                m: 64,
                n: 64,
                k: 32,
            }
        } else if tensor_core && m >= 256 && n >= 256 && k >= 64 {
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
        let multi_fragment = block_tile.m == 64 && block_tile.n == 64;

        Self {
            plan,
            logical_shape: MatMulTile { m, n, k },
            grid_group_m: 1,
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
            block_threads: if expanded && tensor_core {
                (32, 8, 1)
            } else {
                (16, 16, 1)
            },
            warp_topology: if multi_fragment {
                (2, 2)
            } else if tensor_core {
                (block_tile.m / 16, block_tile.n / 16)
            } else {
                (8, 1)
            },
            warp_fragments: if multi_fragment { (2, 2) } else { (1, 1) },
            operand_layouts: (
                SharedLayout::row_major(block_tile.k),
                SharedLayout::row_major(block_tile.n),
            ),
            pipeline_stages: 1,
            copy_bytes: 4,
            async_operands: (false, false),
            operand_dtype: if matches!(
                plan,
                MatMulPlan::TensorCoreTf32 | MatMulPlan::TensorCoreTf32x3
            ) {
                DType::TF32
            } else {
                storage_dtype
            },
            storage_dtype,
            accumulator_layout: if tensor_core {
                MatMulAccumulatorLayout::WarpFragment
            } else if block_tile.m * block_tile.n == 1024 {
                MatMulAccumulatorLayout::ThreadArray { elements: 4 }
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
    fn experimental_register_tile_is_opt_in_and_preserves_gradient_fallbacks() {
        let caps = MatMulCapabilities {
            tf32: true,
            f16: true,
            bf16: true,
        };
        let resources = MatMulResources::default();
        let base = MatMulSchedule::select(4096, 683, 256, caps, MatMulPrecision::StrictF32);
        assert_eq!(base.register_thread_tile(), Some((4, 1)));
        let experimental = base.with_experimental_register_tile(resources);
        assert_eq!(experimental.register_thread_tile(), Some((4, 4)));
        assert_eq!(
            experimental.block_tile,
            MatMulTile {
                m: 64,
                n: 64,
                k: 16
            }
        );
        assert_eq!(experimental.pipeline_stages, 1);
        for (m, n, k) in [
            (683, 256, 4096),
            (768, 256, 4096),
            (512, 683, 256),
            (4096, 128, 256),
            (1024, 256, 127),
        ] {
            let base = MatMulSchedule::select(m, n, k, caps, MatMulPrecision::StrictF32);
            assert_eq!(base.with_experimental_register_tile(resources), base);
        }
        for resources in [
            MatMulResources {
                registers_per_sm: 32 * 1024,
                ..resources
            },
            MatMulResources {
                shared_bytes_per_block: 4096,
                ..resources
            },
            MatMulResources {
                shared_bytes_per_sm: 8192,
                ..resources
            },
            MatMulResources {
                threads_per_sm: 256,
                ..resources
            },
        ] {
            assert_eq!(base.with_experimental_register_tile(resources), base);
        }
        for precision in [MatMulPrecision::AllowTf32, MatMulPrecision::Tf32x3] {
            let base = MatMulSchedule::select(4096, 683, 256, caps, precision);
            assert_eq!(base.with_experimental_register_tile(resources), base);
        }
        let half = MatMulSchedule::select_for_dtype(
            4096,
            683,
            256,
            DType::BF16,
            caps,
            MatMulPrecision::StrictF32,
        );
        assert_eq!(half.with_experimental_register_tile(resources), half);
    }

    #[test]
    fn register_accumulator_ownership_is_a_bijection() {
        fn value(expression: &Expr, x: u64, y: u64) -> u64 {
            match expression {
                Expr::Const(v) => *v as u64,
                Expr::ThreadIdx(Dim::X) => x,
                Expr::ThreadIdx(Dim::Y) => y,
                Expr::Add(a, b) => value(a, x, y) + value(b, x, y),
                Expr::Mul(a, b) => value(a, x, y) * value(b, x, y),
                Expr::ShiftRight(a, shift) => value(a, x, y) >> shift,
                Expr::BitAnd(a, mask) => value(a, x, y) & mask,
                _ => panic!("unexpected ownership expression"),
            }
        }
        let base = MatMulSchedule::select(
            4096,
            683,
            256,
            MatMulCapabilities {
                tf32: false,
                f16: false,
                bf16: false,
            },
            MatMulPrecision::StrictF32,
        );
        for schedule in [
            base,
            base.with_experimental_register_tile(MatMulResources::default()),
        ] {
            let (rows, cols) = schedule.register_thread_tile().unwrap();
            let mut seen = std::collections::HashSet::new();
            for slot in 0..rows * cols {
                let (r, c) = schedule.register_accumulator_coordinates(slot).unwrap();
                for y in 0..16 {
                    for x in 0..16 {
                        let coordinate = (value(&r, x, y), value(&c, x, y));
                        assert!(coordinate.0 < schedule.block_tile.m as u64);
                        assert!(coordinate.1 < schedule.block_tile.n as u64);
                        assert!(seen.insert(coordinate), "duplicate {coordinate:?}");
                    }
                }
            }
            assert_eq!(seen.len(), schedule.block_tile.m * schedule.block_tile.n);
            assert!(
                schedule
                    .register_accumulator_coordinates(rows * cols)
                    .is_none()
            );
        }
    }

    #[test]
    fn experimental_register_tile_layouts_validate_all_supported_staging_policies() {
        use crate::tile::{MatMulLayout, TileIRBuilder};
        let resources = MatMulResources::default();
        let base = MatMulSchedule::select(
            1025,
            683,
            129,
            MatMulCapabilities {
                tf32: false,
                f16: false,
                bf16: false,
            },
            MatMulPrecision::StrictF32,
        )
        .with_experimental_register_tile(resources);
        for policy in [
            MatMulSharedLayout::Contiguous,
            MatMulSharedLayout::Padded,
            MatMulSharedLayout::Swizzled,
        ] {
            for orientations in [(false, false), (true, false), (false, true), (true, true)] {
                for pipeline in [MatMulPipeline::Synchronous, MatMulPipeline::DoubleBuffered] {
                    let schedule = base
                        .with_operand_orientations(policy, orientations, resources)
                        .unwrap()
                        .with_pipeline(pipeline, true, true, resources);
                    let mut builder = TileIRBuilder::new();
                    builder.start_kernel("experimental_register_tile_layout_validation");
                    let a =
                        builder.alloc_shared_layout(DType::F32, 64, 16, schedule.operand_layouts.0);
                    let b =
                        builder.alloc_shared_layout(DType::F32, 16, 64, schedule.operand_layouts.1);
                    let dest = builder.alloc_register_array(64, 64, 16);
                    builder.zero(dest);
                    builder.matmul(dest, a, b, MatMulLayout::NN, schedule);
                    for slot in 0..16 {
                        let element = builder.alloc_register(DType::F32, 1, 1);
                        builder.extract_register(element, dest, slot);
                    }
                    builder.finish().validate_layouts().unwrap();
                }
            }
        }
    }

    #[test]
    fn large_bf16_register_tiles_respect_resources_and_keep_f32_schedule() {
        let capabilities = MatMulCapabilities {
            tf32: true,
            f16: true,
            bf16: true,
        };
        let select = |dtype, resources| {
            MatMulSchedule::select_with_resources(
                4096,
                768,
                256,
                dtype,
                capabilities,
                MatMulPrecision::AllowTf32,
                resources,
            )
        };
        let bf16 = select(DType::BF16, MatMulResources::default());
        assert_eq!(
            bf16.block_tile,
            MatMulTile {
                m: 64,
                n: 64,
                k: 32
            }
        );
        assert_eq!(bf16.warp_topology, (2, 2));
        assert_eq!(bf16.warp_fragments, (2, 2));
        let f32 = select(DType::F32, MatMulResources::default());
        assert_eq!(
            f32.block_tile,
            MatMulTile {
                m: 32,
                n: 32,
                k: 32
            }
        );
        assert_eq!(f32.warp_fragments, (1, 1));
        let constrained = select(
            DType::BF16,
            MatMulResources {
                registers_per_sm: 32 * 1024,
                ..MatMulResources::default()
            },
        );
        assert_eq!(
            constrained.block_tile,
            MatMulTile {
                m: 16,
                n: 16,
                k: 16
            }
        );
        assert_eq!(constrained.warp_fragments, (1, 1));
    }

    #[test]
    fn native_half_preserves_f32_and_gates_half_tensor_cores() {
        for supported in [false, true] {
            let capabilities = MatMulCapabilities {
                tf32: true,
                f16: supported,
                bf16: supported,
            };
            for (dtype, native) in [
                (DType::F32, MatMulPlan::ScalarF32),
                (DType::F16, MatMulPlan::TensorCoreF16),
                (DType::BF16, MatMulPlan::TensorCoreBF16),
            ] {
                let schedule = MatMulSchedule::select_for_dtype(
                    128,
                    128,
                    128,
                    dtype,
                    capabilities,
                    MatMulPrecision::NativeHalf,
                );
                assert_eq!(
                    schedule.plan,
                    if supported {
                        native
                    } else {
                        MatMulPlan::ScalarF32
                    }
                );
            }
        }
    }

    #[test]
    fn capabilities_and_precision_gate_tensor_core_schedule() {
        for tf32 in [false, true] {
            for precision in [
                MatMulPrecision::StrictF32,
                MatMulPrecision::AllowTf32,
                MatMulPrecision::Tf32x3,
            ] {
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
                    schedule.plan != MatMulPlan::ScalarF32,
                    tf32 && precision != MatMulPrecision::StrictF32
                );
                assert_eq!(
                    schedule.plan == MatMulPlan::TensorCoreTf32x3,
                    tf32 && precision == MatMulPrecision::Tf32x3
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

    fn short_k_schedule(m: usize, k: usize, n: usize, dtype: DType) -> MatMulSchedule {
        MatMulSchedule::select_with_resources(
            m,
            n,
            k,
            dtype,
            MatMulCapabilities {
                tf32: true,
                f16: true,
                bf16: true,
            },
            MatMulPrecision::AllowTf32,
            short_k_resources(),
        )
        .with_shared_layout(MatMulSharedLayout::Auto, short_k_resources())
        .unwrap()
    }

    fn short_k_resources() -> MatMulResources {
        MatMulResources {
            shared_bytes_per_sm: 100 * 1024,
            threads_per_sm: 1536,
            ..MatMulResources::default()
        }
    }

    #[test]
    fn short_k_bf16_register_pipeline_matches_explicit_double_buffering() {
        let resources = short_k_resources();
        for (k, n) in [(64, 256), (64, 768), (64, 1024), (96, 768)] {
            for orientations in [(false, false), (false, true)] {
                let base = short_k_schedule(2048, k, n, DType::BF16)
                    .with_operand_orientations(MatMulSharedLayout::Auto, orientations, resources)
                    .unwrap();
                let selected = base.with_pipeline(MatMulPipeline::Auto, true, true, resources);
                let explicit =
                    base.with_pipeline(MatMulPipeline::DoubleBuffered, true, true, resources);
                assert_eq!(selected, explicit);
                assert_eq!(selected.pipeline_stages, 2);
                assert_eq!(selected.copy_bytes, 16);
                assert_eq!(selected.async_operands, (true, true));
                assert_eq!(selected.warp_fragments, (2, 2));
            }
        }
    }

    #[test]
    fn short_k_bf16_pipeline_keeps_hybrid_tail_and_small_tile_fallbacks() {
        let resources = short_k_resources();
        for (m, k, n, dtype, active) in [
            (2048, 64, 768, DType::BF16, (true, false)),
            (2048, 64, 768, DType::BF16, (false, true)),
            (2048, 64, 770, DType::BF16, (true, true)),
            (2048, 65, 768, DType::BF16, (true, true)),
            (256, 64, 768, DType::BF16, (true, true)),
            (16, 64, 768, DType::BF16, (true, true)),
            (2048, 64, 768, DType::F16, (true, true)),
            (2048, 64, 768, DType::F32, (true, true)),
        ] {
            let selected = short_k_schedule(m, k, n, dtype).with_operand_pipeline(
                MatMulPipeline::Auto,
                true,
                true,
                active,
                resources,
            );
            assert_eq!(
                selected.pipeline_stages, 1,
                "unexpected short pipeline for {m}x{k}x{n}, {dtype:?}, {active:?}"
            );
            assert_eq!(selected.async_operands, (false, false));
        }
    }

    #[test]
    fn short_k_bf16_pipeline_keeps_capability_alignment_and_resource_guards() {
        let resources = short_k_resources();
        let base = short_k_schedule(2048, 64, 768, DType::BF16);
        for (policy, supports_async, contiguous, budget) in [
            (MatMulPipeline::Synchronous, true, true, resources),
            (MatMulPipeline::Auto, false, true, resources),
            (MatMulPipeline::Auto, true, false, resources),
            (
                MatMulPipeline::Auto,
                true,
                true,
                MatMulResources {
                    shared_bytes_per_block: 32 * 1024,
                    ..resources
                },
            ),
            (
                MatMulPipeline::Auto,
                true,
                true,
                MatMulResources {
                    shared_bytes_per_sm: 64 * 1024,
                    ..resources
                },
            ),
        ] {
            let selected = base.with_pipeline(policy, supports_async, contiguous, budget);
            assert_eq!(selected.pipeline_stages, 1);
            assert_eq!(selected.async_operands, (false, false));
        }
        let constrained = MatMulSchedule::select_with_resources(
            2048,
            768,
            64,
            DType::BF16,
            MatMulCapabilities {
                tf32: true,
                f16: true,
                bf16: true,
            },
            MatMulPrecision::AllowTf32,
            MatMulResources {
                registers_per_sm: 32 * 1024,
                ..resources
            },
        );
        assert_eq!(constrained.warp_fragments, (1, 1));
    }

    #[test]
    fn short_k_bf16_pipeline_keeps_unmeasured_register_geometries_synchronous() {
        let resources = short_k_resources();
        let mut base = short_k_schedule(2048, 64, 768, DType::BF16);
        base.block_tile = MatMulTile {
            m: 128,
            n: 32,
            k: 32,
        };
        base.warp_topology = (4, 1);
        let base = base
            .with_shared_layout(MatMulSharedLayout::Auto, resources)
            .unwrap();
        let explicit = base.with_pipeline(MatMulPipeline::DoubleBuffered, true, true, resources);
        assert_eq!(explicit.pipeline_stages, 2);
        assert_eq!(explicit.copy_bytes, 16);
        assert_eq!(explicit.warp_fragments, (2, 2));
        let selected = base.with_pipeline(MatMulPipeline::Auto, true, true, resources);
        assert_eq!(selected.pipeline_stages, 1);
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
    #[test]
    fn hybrid_pipeline_only_requires_alignment_of_async_operand() {
        let resources = MatMulResources::default();
        let caps = MatMulCapabilities {
            tf32: true,
            f16: true,
            bf16: true,
        };
        for (m, k, n, column_major, mask) in [
            (264, 136, 283, (true, false), (true, false)),
            (267, 136, 280, (true, false), (false, true)),
            (264, 137, 280, (false, false), (false, true)),
        ] {
            let base = MatMulSchedule::select_for_dtype(
                m,
                n,
                k,
                DType::BF16,
                caps,
                MatMulPrecision::AllowTf32,
            )
            .with_operand_orientations(MatMulSharedLayout::Padded, column_major, resources)
            .unwrap();
            let selected =
                base.with_operand_pipeline(MatMulPipeline::Auto, true, true, mask, resources);
            assert_eq!(selected.pipeline_stages, 2);
            assert_eq!(selected.async_operands, mask);
            assert_eq!(selected.copy_group(true), if mask.0 { 8 } else { 1 });
            assert_eq!(selected.copy_group(false), if mask.1 { 8 } else { 1 });
            for (supported, budget) in [
                (false, resources),
                (
                    true,
                    MatMulResources {
                        shared_bytes_per_sm: 1,
                        ..resources
                    },
                ),
            ] {
                let fallback =
                    base.with_operand_pipeline(MatMulPipeline::Auto, supported, true, mask, budget);
                assert_eq!(fallback.pipeline_stages, 1);
                assert_eq!(fallback.async_operands, (false, false));
            }
        }
    }
    #[test]
    fn scalar_register_tiles_respect_precision_shapes_and_resource_fallbacks() {
        let caps = MatMulCapabilities {
            tf32: true,
            f16: true,
            bf16: true,
        };
        let resources = MatMulResources::default();
        let selected = MatMulSchedule::select_with_resources(
            129,
            257,
            129,
            DType::F32,
            caps,
            MatMulPrecision::StrictF32,
            resources,
        );
        assert_eq!(
            selected.accumulator_tile_layout(),
            TileLayout::ThreadArray { elements: 4 }
        );
        assert_eq!(selected.block_threads, (16, 16, 1));
        assert_eq!(selected.block_tile.k, 32);
        let constrained_staging = MatMulSchedule::select_with_resources(
            129,
            257,
            129,
            DType::F32,
            caps,
            MatMulPrecision::StrictF32,
            MatMulResources {
                shared_bytes_per_block: 8 * 1024,
                ..resources
            },
        );
        assert_eq!(
            constrained_staging.block_tile,
            MatMulTile {
                m: 32,
                n: 32,
                k: 16
            }
        );
        for (m, n, k, dtype, precision, budget) in [
            (
                127,
                257,
                129,
                DType::F32,
                MatMulPrecision::StrictF32,
                resources,
            ),
            (
                129,
                255,
                129,
                DType::F32,
                MatMulPrecision::StrictF32,
                resources,
            ),
            (
                129,
                257,
                127,
                DType::F32,
                MatMulPrecision::StrictF32,
                resources,
            ),
            (
                129,
                257,
                129,
                DType::F32,
                MatMulPrecision::AllowTf32,
                resources,
            ),
            (
                129,
                257,
                129,
                DType::BF16,
                MatMulPrecision::StrictF32,
                resources,
            ),
            (
                129,
                257,
                129,
                DType::F16,
                MatMulPrecision::StrictF32,
                resources,
            ),
            (
                129,
                257,
                129,
                DType::F32,
                MatMulPrecision::StrictF32,
                MatMulResources {
                    shared_bytes_per_block: 4096,
                    ..resources
                },
            ),
        ] {
            let fallback =
                MatMulSchedule::select_with_resources(m, n, k, dtype, caps, precision, budget);
            assert!(!matches!(
                fallback.accumulator_tile_layout(),
                TileLayout::ThreadArray { .. }
            ));
        }
    }

    #[test]
    fn register_array_extraction_rejects_wrong_slots_and_scalar_consumption() {
        use crate::tile::{MatMulLayout, TileIRBuilder};
        for index in [0, 3, 4] {
            let mut b = TileIRBuilder::new();
            b.start_kernel("register_array_validation");
            let array = b.alloc_register_array(32, 32, 4);
            let scalar = b.alloc_register(DType::F32, 1, 1);
            b.extract_register(scalar, array, index);
            assert_eq!(b.finish().validate_layouts().is_ok(), index < 4);
        }
        let mut b = TileIRBuilder::new();
        b.start_kernel("array_scalar_rejection");
        let array = b.alloc_register_array(32, 32, 4);
        let scalar = b.alloc_register(DType::F32, 1, 1);
        b.add(scalar, array, scalar);
        assert!(b.finish().validate_layouts().is_err());
        let schedule = MatMulSchedule::select(
            129,
            257,
            129,
            MatMulCapabilities {
                tf32: true,
                f16: true,
                bf16: true,
            },
            MatMulPrecision::StrictF32,
        );
        let mut b = TileIRBuilder::new();
        b.start_kernel("array_matrix_layout");
        let a = b.alloc_shared(DType::F32, 32, schedule.block_tile.k);
        let rhs = b.alloc_shared(DType::F32, schedule.block_tile.k, 32);
        let dest = b.alloc_register_array(32, 32, 4);
        b.zero(dest);
        b.matmul(dest, a, rhs, MatMulLayout::NN, schedule);
        assert!(b.finish().validate_layouts().is_ok());
    }
}
