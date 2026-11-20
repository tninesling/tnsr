//! Disconnected dense S512/D64 training-attention forward prototype.
//!
//! QK is rounded to BF16 before scale/mask/normalization. P32 and P16 are
//! checkpoints; only normalized P16 enters PV. No online numerator/rescaling,
//! causal skipping, zero shortcut or changed K16 accumulation order is used.
//! Leaf inputs use an explicitly proven dense head-major or sequence-major BF16
//! descriptor. Outputs retain dense BHxx order. No runtime integration exists.
use super::*;
use crate::tile::{
    Block, Dim, Expr, KernelParam, MatMulAccumulatorExchange, MatMulAccumulatorLayout,
    MatMulFragmentLoad, MatMulLayout, MatMulPlan, MatMulSchedule, MatMulTile, ScalarBuilder,
    ScalarMemory, SharedLayout, Stmt, TileIR, TileVar,
};

#[cfg(test)]
pub(super) mod binding_observer;

mod input_layout;
use input_layout::InputLayout;

mod recognize;
pub(super) fn online_pair_forward_seeds<G>(graph: &TensorGraph<f32, G>) -> Vec<recognize::Region> {
    recognize::recognize(graph)
}

pub(super) mod rewrite;
#[cfg(test)]
mod tests;

const SEQUENCE: usize = 512;
const WIDTH: usize = 64;
const KEYS: usize = 64;
const WARPS: usize = 8;
const THREADS: usize = WARPS * 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Geometry {
    pub batch_heads: usize,
    pub query_tile: usize,
}
impl Geometry {
    pub(super) fn new(batch_heads: usize, query_tile: usize) -> Result<Self> {
        ensure!(
            batch_heads > 0 && batch_heads <= u16::MAX as usize && matches!(query_tile, 16 | 32),
            "unsupported dense attention prototype geometry"
        );
        for tail in [SEQUENCE * WIDTH * 2, SEQUENCE * SEQUENCE * 4] {
            ensure!(
                batch_heads
                    .checked_mul(tail)
                    .is_some_and(|bytes| bytes <= isize::MAX as usize),
                "dense attention binding extent overflow"
            );
        }
        Ok(Self {
            batch_heads,
            query_tile,
        })
    }
    pub(super) fn shared_bytes(self) -> usize {
        self.score_bytes() + self.query_bytes() + KEYS * WIDTH * 2
    }
    fn score_bytes(self) -> usize {
        self.query_tile * SEQUENCE * 2
    }
    fn query_bytes(self) -> usize {
        self.query_tile * WIDTH * 2
    }
    fn key_offset(self) -> usize {
        self.score_bytes() + self.query_bytes()
    }
    pub(super) fn launch(self) -> Result<LaunchConfig> {
        Ok(LaunchConfig {
            grid_dim: (
                u32::try_from(SEQUENCE / self.query_tile)?,
                u32::try_from(self.batch_heads)?,
                1,
            ),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        })
    }
}

fn ix(v: usize) -> Expr {
    Expr::Const(v as i64)
}
fn div(v: Expr, d: usize) -> Expr {
    Expr::FloorDiv(Box::new(v), d)
}
fn rem(v: Expr, d: usize) -> Expr {
    Expr::Mod(Box::new(v), d)
}
fn sm(var: TileVar) -> ScalarMemory {
    ScalarMemory::Shared(var)
}
fn global(name: &str) -> ScalarMemory {
    ScalarMemory::Global(name.into())
}
fn view(
    b: &mut ScalarBuilder,
    arena: TileVar,
    offset: usize,
    dtype: DType,
    rows: usize,
    cols: usize,
    layout: SharedLayout,
) -> TileVar {
    let dest = b.var();
    b.stmts.push(Stmt::ViewSharedArena {
        dest,
        arena,
        byte_offset: offset,
        dtype,
        rows,
        cols,
        layout,
    });
    dest
}
fn schedule(query_tile: usize, lhs: SharedLayout, rhs: SharedLayout) -> MatMulSchedule {
    let tile = MatMulTile {
        m: query_tile,
        n: WIDTH,
        k: KEYS,
    };
    MatMulSchedule {
        plan: MatMulPlan::TensorCoreBF16,
        logical_shape: tile,
        grid_group_m: 1,
        block_tile: tile,
        instruction_tile: MatMulTile {
            m: 16,
            n: 16,
            k: 16,
        },
        block_threads: (32, 8, 1),
        warp_topology: (query_tile / 16, 4),
        warp_fragments: (1, 1),
        accumulator_exchange: MatMulAccumulatorExchange::FullTile,
        fragment_load: MatMulFragmentLoad::PreloadAll,
        operand_layouts: (lhs, rhs),
        pipeline_stages: 1,
        reuse_shared_memory: false,
        copy_bytes: 0,
        async_operands: (false, false),
        operand_dtype: DType::BF16,
        storage_dtype: DType::BF16,
        accumulator_layout: MatMulAccumulatorLayout::WarpFragment,
    }
}
fn fragment(b: &mut ScalarBuilder, s: MatMulSchedule) -> TileVar {
    let var = b.var();
    b.stmts.push(Stmt::AllocTile {
        var,
        space: crate::tile::MemorySpace::Fragment,
        layout: s.accumulator_tile_layout(),
        dtype: DType::F32,
        rows: s.block_tile.m,
        cols: s.block_tile.n,
    });
    var
}
fn mma(b: &mut ScalarBuilder, dest: TileVar, a: TileVar, rhs: TileVar, schedule: MatMulSchedule) {
    b.stmts.push(Stmt::MatMul {
        dest,
        a,
        b: rhs,
        layout: MatMulLayout::NN,
        schedule,
    });
}
fn exchange(b: &mut ScalarBuilder, dest: TileVar, src: TileVar) {
    b.stmts.push(Stmt::ConvertLayout {
        dest,
        src,
        coordinates: None,
    });
    b.barrier();
}
fn distribute(
    b: &mut ScalarBuilder,
    count: usize,
    prefix: &str,
    f: impl FnOnce(&mut ScalarBuilder, Expr) -> Result<()>,
) -> Result<()> {
    ensure!(
        count.is_multiple_of(THREADS),
        "prototype has no scalar staging tail"
    );
    let name = b.loop_name(prefix);
    b.fold(name, ix(count / THREADS), vec![], |b, round| {
        let index = round * THREADS + Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X);
        f(b, index)
    })
}

/// Actual compiled leaf resource limits, never achieved occupancy. Neither the
/// diagnostic QK-dump variant nor a different query tile proves another fits.
#[cfg(test)]
pub(super) fn admitted(
    function: &cudarc::driver::CudaFunction,
    geometry: Geometry,
) -> Result<bool> {
    Ok((1..=192).contains(&function.num_regs()?)
        && function.local_size_bytes()? == 0
        && usize::try_from(function.shared_size_bytes()?)? == geometry.shared_bytes()
        && function.max_threads_per_block()? >= 256
        && usize::try_from(function.shared_size_bytes()?)? <= 48 * 1024
        && function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? >= 1)
}

#[cfg(test)]
pub(super) fn ir(geometry: Geometry, include_qk_checkpoint: bool) -> Result<TileIR> {
    ir_with_input_layout(geometry, include_qk_checkpoint, InputLayout::DenseHeadMajor)
}

pub(super) fn ir_with_input_layout(
    geometry: Geometry,
    include_qk_checkpoint: bool,
    input_layout: InputLayout,
) -> Result<TileIR> {
    input_layout.validate(geometry)?;
    ensure!(
        matches!(input_layout, InputLayout::DenseHeadMajor) || include_qk_checkpoint,
        "sequence-major prototype retains all four checkpoints"
    );
    let Geometry { query_tile, .. } = geometry;
    let mut b = ScalarBuilder::default();
    let arena = b.var();
    b.stmts.push(Stmt::AllocSharedArena {
        arena,
        bytes: geometry.shared_bytes(),
    });
    let scores = view(
        &mut b,
        arena,
        0,
        DType::BF16,
        query_tile,
        SEQUENCE,
        SharedLayout::row_major(SEQUENCE),
    );
    let q = view(
        &mut b,
        arena,
        geometry.score_bytes(),
        DType::BF16,
        query_tile,
        WIDTH,
        SharedLayout::row_major(WIDTH),
    );
    let key_layout = SharedLayout {
        row_stride: WIDTH,
        column_major: true,
        xor_mask: 0,
    };
    let k = view(
        &mut b,
        arena,
        geometry.key_offset(),
        DType::BF16,
        WIDTH,
        KEYS,
        key_layout,
    );
    let v = view(
        &mut b,
        arena,
        geometry.key_offset(),
        DType::BF16,
        KEYS,
        WIDTH,
        SharedLayout::row_major(WIDTH),
    );
    let scratch = view(
        &mut b,
        arena,
        geometry.key_offset(),
        DType::F32,
        query_tile,
        WIDTH,
        SharedLayout::row_major(WIDTH),
    );
    let qk_schedule = schedule(query_tile, SharedLayout::row_major(WIDTH), key_layout);
    let qk = fragment(&mut b, qk_schedule);
    let head = Expr::BlockIdx(Dim::Y);
    let query_base = Expr::BlockIdx(Dim::X) * query_tile;
    distribute(&mut b, query_tile * WIDTH, "stage_query", |b, index| {
        let query = query_base.clone() + div(index.clone(), WIDTH);
        let feature = rem(index.clone(), WIDTH);
        let value = b.load(
            global("query"),
            input_layout.index(head.clone(), query, feature),
        );
        b.store(sm(q), index, value);
        Ok(())
    })?;
    b.barrier();
    let key_loop = b.loop_name("qk_key_tiles");
    b.fold(key_loop, ix(SEQUENCE / KEYS), vec![], |b, chunk| {
        let key_base = chunk * KEYS;
        b.stmts.push(Stmt::Zero { tile: qk });
        distribute(b, KEYS * WIDTH, "stage_key", |b, index| {
            let key = key_base.clone() + div(index.clone(), WIDTH);
            let feature = rem(index.clone(), WIDTH);
            let value = b.load(
                global("key"),
                input_layout.index(head.clone(), key, feature),
            );
            // Physical column-major [feature,key] equals packed [key,feature].
            b.store(sm(k), index, value);
            Ok(())
        })?;
        b.barrier();
        mma(b, qk, q, k, qk_schedule);
        // ALL K readers retire before its 8KiB bytes become F32 exchange.
        b.barrier();
        exchange(b, scratch, qk);
        distribute(b, query_tile * KEYS, "round_qk_bf16", |b, index| {
            let row = div(index.clone(), KEYS);
            let col = key_base.clone() + rem(index.clone(), KEYS);
            let value = b.load(sm(scratch), index);
            b.store(sm(scores), row.clone() * SEQUENCE + col.clone(), value);
            if include_qk_checkpoint {
                let global_row = head.clone() * SEQUENCE + query_base.clone() + row;
                b.store(global("qk"), global_row * SEQUENCE + col, value);
            }
            Ok(())
        })?;
        // Publish rounded scores and retire scratch readers before next K load.
        b.barrier();
        Ok(())
    })?;
    let rows = b.loop_name("normalize_rows");
    b.fold(rows, ix(query_tile / WARPS), vec![], |b, round| {
        let row = round * WARPS + Expr::ThreadIdx(Dim::Y);
        let query = query_base.clone() + row.clone();
        let global_row = head.clone() * SEQUENCE + query.clone();
        super::attention_softmax_warp::forward_row(
            b,
            super::attention_softmax_warp::ForwardRowBindings {
                scores: sm(scores),
                score_row: row.clone(),
                mask: global("mask"),
                mask_row: query,
                probability: global("probability"),
                probability_row: global_row.clone(),
                narrowed: sm(scores),
                narrowed_row: row,
                narrowed_checkpoint: Some((global("narrowed"), global_row)),
            },
        );
        Ok(())
    })?;
    // Each warp loaded all16 old scores before its inplace writes. Rows are
    // independent; this final CTA barrier publishes all P16 before PV readers.
    b.barrier();
    let pv_schedule = schedule(
        query_tile,
        SharedLayout::row_major(SEQUENCE),
        SharedLayout::row_major(WIDTH),
    );
    let pv = fragment(&mut b, pv_schedule);
    b.stmts.push(Stmt::Zero { tile: pv });
    for chunk in 0..SEQUENCE / KEYS {
        // Full padded span may overlap <=896 bytes of DEAD Q, not live V.
        // Actual WMMA accesses remain entirely inside scores/P16's32KiB.
        let p = view(
            &mut b,
            arena,
            chunk * KEYS * 2,
            DType::BF16,
            query_tile,
            KEYS,
            SharedLayout::row_major(SEQUENCE),
        );
        distribute(&mut b, KEYS * WIDTH, "stage_value", |b, index| {
            let key = ix(chunk * KEYS) + div(index.clone(), WIDTH);
            let feature = rem(index.clone(), WIDTH);
            let value = b.load(
                global("value"),
                input_layout.index(head.clone(), key, feature),
            );
            b.store(sm(v), index, value);
            Ok(())
        })?;
        b.barrier();
        mma(&mut b, pv, p, v, pv_schedule);
        // Every V reader retires before next chunk or final F32 exchange.
        b.barrier();
    }
    exchange(&mut b, scratch, pv);
    distribute(
        &mut b,
        query_tile * WIDTH,
        "round_output_bf16",
        |b, index| {
            let row = query_base.clone() + div(index.clone(), WIDTH);
            let feature = rem(index.clone(), WIDTH);
            let value = b.load(sm(scratch), index);
            b.store(
                global("output"),
                head.clone() * (SEQUENCE * WIDTH) + row * WIDTH + feature,
                value,
            );
            Ok(())
        },
    )?;
    let mut params = vec![
        KernelParam {
            name: "query".into(),
            dtype: DType::BF16,
            is_input: true,
        },
        KernelParam {
            name: "key".into(),
            dtype: DType::BF16,
            is_input: true,
        },
        KernelParam {
            name: "value".into(),
            dtype: DType::BF16,
            is_input: true,
        },
        KernelParam {
            name: "mask".into(),
            dtype: DType::F32,
            is_input: true,
        },
    ];
    if include_qk_checkpoint {
        params.push(KernelParam {
            name: "qk".into(),
            dtype: DType::BF16,
            is_input: false,
        });
    }
    params.extend([
        KernelParam {
            name: "probability".into(),
            dtype: DType::F32,
            is_input: false,
        },
        KernelParam {
            name: "narrowed".into(),
            dtype: DType::BF16,
            is_input: false,
        },
        KernelParam {
            name: "output".into(),
            dtype: DType::BF16,
            is_input: false,
        },
    ]);
    let ir = TileIR {
        kernel_name: match input_layout {
            InputLayout::DenseHeadMajor => format!(
                "attention_training_dense_q{query_tile}_qkdump{}",
                usize::from(include_qk_checkpoint)
            ),
            InputLayout::SequenceMajor { heads } => format!(
                "attention_training_sequence_h{heads}_q{query_tile}_qkdump{}",
                usize::from(include_qk_checkpoint)
            ),
        },
        params,
        body: Block { stmts: b.stmts },
        shared_mem_bytes: geometry.shared_bytes(),
    };
    ir.validate_layouts()?;
    Ok(ir)
}
