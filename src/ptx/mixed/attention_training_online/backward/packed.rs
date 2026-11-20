//! Qualified exact packed epilogues; runtime admission remains fail-closed. Original WMMA rows stay opaque.
//! dKT is transposed by explicit scalar shared stores, never by relabeling a
//! row-only WMMA store as column-major. Runtime ownership is proved separately.
use super::*;
#[cfg(test)]
use crate::ptx::mixed::qkv_grad_pack;
#[cfg(test)]
use crate::{
    graph::{NodeIndex, TensorGraph},
    ptx::mixed::{Binding, Value},
    tensor::{TensorExpr, ValueDType},
};
use crate::{
    ptx::mixed::qkv_grad_pack::{Join, QkvGradPack},
    tensor::BinaryOp,
};
use anyhow::{Context, ensure};
#[cfg(test)]
use std::collections::HashMap;

const TRANSPOSE_OFFSET: usize = 16_384;
const TRANSPOSE_PITCH: usize = 65;
const ROUND_OFFSET: usize = 40_960;
const ROUND_ELEMENTS: usize = 512;

fn packed_index(shape: Shape, sequence: Expr, feature: Expr, window: usize) -> Expr {
    let bh = Expr::BlockIdx(Dim::Y);
    ((div(bh.clone(), shape.heads) * SEQUENCE + sequence) * (3 * shape.heads * TILE))
        + ix(window * shape.heads * TILE)
        + rem(bh, shape.heads) * TILE
        + feature
}
fn distributed(name: &str) -> Expr {
    Expr::Var(name.into()) * THREADS + Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X)
}
fn top_var(block: &Block) -> usize {
    let mut n = 0;
    for statement in &block.stmts {
        let v = match statement {
            Stmt::AllocSharedArena { arena, .. } => Some(*arena),
            Stmt::AllocTile { var, .. } => Some(*var),
            Stmt::ViewSharedArena { dest, .. }
            | Stmt::SetScalar { dest, .. }
            | Stmt::LoadScalar { dest, .. }
            | Stmt::WarpReduce { dest, .. }
            | Stmt::WarpBroadcastLeader { dest, .. } => Some(*dest),
            _ => None,
        };
        if let Some(v) = v {
            n = n.max(v.0);
        }
        match statement {
            Stmt::If { body, .. } | Stmt::ForLoop { body, .. } => n = n.max(top_var(body)),
            _ => {}
        }
    }
    n
}
fn new_builder(ir: &TileIR) -> ScalarBuilder {
    let mut b = ScalarBuilder::default();
    for _ in 0..=top_var(&ir.body) {
        b.var();
    }
    b
}
fn arena_id(ir: &TileIR) -> Result<TileVar> {
    match ir.body.stmts.first() {
        Some(Stmt::AllocSharedArena { arena, bytes }) if *bytes == SHARED_BYTES => Ok(*arena),
        _ => anyhow::bail!("qualified48KiB arena prefix changed"),
    }
}
fn typed_view(
    dest: TileVar,
    arena: TileVar,
    offset: usize,
    dtype: DType,
    rows: usize,
    cols: usize,
    pitch: usize,
) -> Stmt {
    Stmt::ViewSharedArena {
        dest,
        arena,
        byte_offset: offset,
        dtype,
        rows,
        cols,
        layout: SharedLayout::row_major(pitch),
    }
}
fn round(b: &mut ScalarBuilder, scratch: TileVar, value: TileVar) -> TileVar {
    let slot = (Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X)) * 2usize;
    b.store(sm(scratch), slot.clone(), value);
    b.load(sm(scratch), slot)
}
fn join(
    b: &mut ScalarBuilder,
    region: &QkvGradPack,
    tree: &Join,
    window: usize,
    active: TileVar,
    scratch: TileVar,
    root: bool,
) -> TileVar {
    match tree {
        Join::Input(i) => {
            if region.inputs[*i].window == window {
                active
            } else {
                b.value(ScalarExpr::Constant(0.))
            }
        }
        Join::Add(a, c) => {
            let a = join(b, region, a, window, active, scratch, false);
            let c = join(b, region, c, window, active, scratch, false);
            let sum = b.value(ScalarExpr::binary(
                ScalarBinaryOp::Arithmetic(BinaryOp::Add),
                a,
                c,
            ));
            if root { sum } else { round(b, scratch, sum) }
        }
    }
}
fn packed_store(
    b: &mut ScalarBuilder,
    region: &QkvGradPack,
    shape: Shape,
    window: usize,
    coordinate: (Expr, Expr),
    value: TileVar,
    scratch: TileVar,
) {
    // First round replaces the removed raw BF16 global checkpoint. Second
    // round is the original packer's selected padded CastBF16, including sNaNs.
    let raw = round(b, scratch, value);
    let padded = round(b, scratch, raw);
    let result = join(b, region, &region.tree, window, padded, scratch, true);
    b.store(
        global("packed_gradient"),
        packed_index(shape, coordinate.0, coordinate.1, window),
        result,
    );
}
fn validate_region(region: &QkvGradPack, shape: Shape) -> Result<()> {
    ensure!(
        region.shape == [shape.batch, SEQUENCE, 3 * shape.heads * TILE],
        "pack shape mismatch"
    );
    let mut windows = region.inputs.iter().map(|i| i.window).collect::<Vec<_>>();
    windows.sort_unstable();
    ensure!(
        windows == [0, 1, 2],
        "pack mustownthree distinct complete windows"
    );
    fn leaves(tree: &Join, out: &mut Vec<usize>) {
        match tree {
            Join::Input(i) => out.push(*i),
            Join::Add(a, b) => {
                leaves(a, out);
                leaves(b, out);
            }
        }
    }
    let mut ids = Vec::new();
    leaves(&region.tree, &mut ids);
    ids.sort_unstable();
    ensure!(
        ids == [0, 1, 2],
        "pack Join mustuseeachrawsourceexactlyonce"
    );
    Ok(())
}
fn loop_parts(stmt: &Stmt, prefix: &str) -> Result<(String, TileVar, TileVar, Expr)> {
    let Stmt::ForLoop {
        loop_var,
        start,
        end,
        carries,
        body,
    } = stmt
    else {
        anyhow::bail!("qualifiedstoreloopabsent")
    };
    ensure!(
        loop_var.starts_with(prefix)
            && *start == ix(0)
            && *end == ix(TILE * TILE / THREADS)
            && carries.is_empty(),
        "storeloopdomain changed"
    );
    let [
        Stmt::LoadScalar {
            dest,
            source: ScalarMemory::Shared(shared),
            index,
        },
        Stmt::StoreScalar { value, .. },
    ] = body.stmts.as_slice()
    else {
        anyhow::bail!("storeloopbody changed")
    };
    ensure!(
        dest == value && *index == distributed(loop_var),
        "storeloopreader changed"
    );
    Ok((loop_var.clone(), *dest, *shared, index.clone()))
}
fn loop_with(name: String, body: Block) -> Stmt {
    Stmt::ForLoop {
        loop_var: name,
        start: ix(0),
        end: ix(TILE * TILE / THREADS),
        carries: vec![],
        body,
    }
}

pub(in crate::ptx::mixed::attention_training_online) fn dquery_ir(
    shape: Shape,
    layout: ReadLayout,
    region: &QkvGradPack,
) -> Result<TileIR> {
    validate_region(region, shape)?;
    let mut ir = super::d_query_ir_with_transport(shape, layout, layout, Transport::Async16)?;
    let mut b = new_builder(&ir);
    let scratch = b.var();
    let arena = arena_id(&ir)?;
    let old = ir.body.stmts.pop().context("dQstoreabsent")?;
    let (name, value, exchange, index) = loop_parts(&old, "store_dquery")?;
    let Stmt::ForLoop { body, .. } = &old else {
        unreachable!()
    };
    let Stmt::StoreScalar {
        target: ScalarMemory::Global(param),
        index: address,
        ..
    } = &body.stmts[1]
    else {
        anyhow::bail!("dQtarget changed")
    };
    ensure!(
        param == "d_query"
            && *address
                == dense_index(
                    Expr::BlockIdx(Dim::X) * TILE + div(index.clone(), TILE),
                    rem(index.clone(), TILE)
                ),
        "dQaddress changed"
    );
    let body = b.block(|b| {
        b.stmts.push(Stmt::LoadScalar {
            dest: value,
            source: sm(exchange),
            index: index.clone(),
        });
        packed_store(
            b,
            region,
            shape,
            0,
            (
                Expr::BlockIdx(Dim::X) * TILE + div(index.clone(), TILE),
                rem(index, TILE),
            ),
            value,
            scratch,
        );
        Ok(())
    })?;
    ir.body.stmts.insert(
        1,
        typed_view(
            scratch,
            arena,
            ROUND_OFFSET,
            DType::BF16,
            1,
            ROUND_ELEMENTS,
            ROUND_ELEMENTS,
        ),
    );
    ir.body.stmts.push(loop_with(name, body));
    ir.kernel_name.push_str("_PackedQkv");
    let output = ir
        .params
        .iter_mut()
        .find(|p| p.name == "d_query")
        .context("dQparamabsent")?;
    output.name = "packed_gradient".into();
    ir.validate_layouts()?;
    Ok(ir)
}

pub(in crate::ptx::mixed::attention_training_online) fn dkeyvalue_ir(
    shape: Shape,
    layout: ReadLayout,
    region: &QkvGradPack,
) -> Result<TileIR> {
    validate_region(region, shape)?;
    let mut ir = super::d_key_value_ir_with_transport(shape, layout, layout, Transport::Async16)?;
    let mut b = new_builder(&ir);
    let scratch = b.var();
    let transpose = b.var();
    let arena = arena_id(&ir)?;
    ensure!(ir.body.stmts.len() >= 7, "DKVsuffixabsent");
    let suffix = ir.body.stmts.split_off(ir.body.stmts.len() - 7);
    let [
        first,
        Stmt::Barrier,
        key_loop,
        Stmt::Barrier,
        last,
        Stmt::Barrier,
        value_loop,
    ] = suffix.as_slice()
    else {
        anyhow::bail!("DKVphases changed")
    };
    let (name, value, exchange, index) = loop_parts(key_loop, "store_dkey_transposed")?;
    let Stmt::ConvertLayout {
        dest,
        coordinates: None,
        ..
    } = first
    else {
        anyhow::bail!("DKexchange changed")
    };
    ensure!(*dest == exchange, "DKexchange target changed");
    let Stmt::ForLoop { body, .. } = key_loop else {
        unreachable!()
    };
    let Stmt::StoreScalar {
        target: ScalarMemory::Global(param),
        index: address,
        ..
    } = &body.stmts[1]
    else {
        anyhow::bail!("DKstore changed")
    };
    ensure!(
        param == "d_key_transposed"
            && *address
                == (Expr::BlockIdx(Dim::Y) * TILE + div(index.clone(), TILE)) * SEQUENCE
                    + (Expr::BlockIdx(Dim::X) * TILE + rem(index.clone(), TILE)),
        "DKaddress changed"
    );
    let transpose_body = b.block(|b| {
        b.stmts.push(Stmt::LoadScalar {
            dest: value,
            source: sm(exchange),
            index: index.clone(),
        });
        b.store(
            sm(transpose),
            rem(index.clone(), TILE) * TRANSPOSE_PITCH + div(index, TILE),
            value,
        );
        Ok(())
    })?;
    let packed_name = format!("{name}_packed");
    let pindex = distributed(&packed_name);
    let key_body = b.block(|b| {
        let value = b.load(
            sm(transpose),
            div(pindex.clone(), TILE) * TRANSPOSE_PITCH + rem(pindex.clone(), TILE),
        );
        packed_store(
            b,
            region,
            shape,
            1,
            (
                Expr::BlockIdx(Dim::X) * TILE + div(pindex.clone(), TILE),
                rem(pindex, TILE),
            ),
            value,
            scratch,
        );
        Ok(())
    })?;
    let (vname, vvalue, vexchange, vindex) = loop_parts(value_loop, "store_dvalue")?;
    let Stmt::ConvertLayout {
        dest,
        coordinates: None,
        ..
    } = last
    else {
        anyhow::bail!("DVexchange changed")
    };
    ensure!(
        *dest == vexchange && vexchange == exchange,
        "DVexchange targetchanged"
    );
    let Stmt::ForLoop { body, .. } = value_loop else {
        unreachable!()
    };
    let Stmt::StoreScalar {
        target: ScalarMemory::Global(param),
        index: address,
        ..
    } = &body.stmts[1]
    else {
        anyhow::bail!("DVstorechanged")
    };
    ensure!(
        param == "d_value"
            && *address
                == dense_index(
                    Expr::BlockIdx(Dim::X) * TILE + div(vindex.clone(), TILE),
                    rem(vindex.clone(), TILE)
                ),
        "DVaddress changed"
    );
    let value_body = b.block(|b| {
        b.stmts.push(Stmt::LoadScalar {
            dest: vvalue,
            source: sm(vexchange),
            index: vindex.clone(),
        });
        packed_store(
            b,
            region,
            shape,
            2,
            (
                Expr::BlockIdx(Dim::X) * TILE + div(vindex.clone(), TILE),
                rem(vindex, TILE),
            ),
            vvalue,
            scratch,
        );
        Ok(())
    })?;
    ir.body.stmts.insert(
        1,
        typed_view(
            transpose,
            arena,
            TRANSPOSE_OFFSET,
            DType::F32,
            TILE,
            TILE,
            TRANSPOSE_PITCH,
        ),
    );
    ir.body.stmts.insert(
        2,
        typed_view(
            scratch,
            arena,
            ROUND_OFFSET,
            DType::BF16,
            1,
            ROUND_ELEMENTS,
            ROUND_ELEMENTS,
        ),
    );
    ir.body.stmts.extend([
        first.clone(),
        Stmt::Barrier,
        loop_with(name, transpose_body),
        Stmt::Barrier,
        loop_with(packed_name, key_body),
        Stmt::Barrier,
        last.clone(),
        Stmt::Barrier,
        loop_with(vname, value_body),
    ]);
    ir.params
        .retain(|p| p.name != "d_key_transposed" && p.name != "d_value");
    ir.params.push(super::super::common::param(
        "packed_gradient",
        DType::BF16,
        false,
    ));
    ir.kernel_name.push_str("_PackedQkv");
    ir.validate_layouts()?;
    Ok(ir)
}

#[cfg(test)]
fn fixture(shape: Shape) -> Result<(QkvGradPack, HashMap<NodeIndex, Value>)> {
    let (b, h, s, d) = (shape.batch, shape.heads, SEQUENCE, TILE);
    let w = h * d;
    let q = TensorExpr::input("q", vec![b, h, s, d])
        .cast(ValueDType::BF16)
        .permute(vec![0, 2, 1, 3])
        .reshape(vec![b, s, w]);
    let k = TensorExpr::input("k", vec![b, h, d, s])
        .cast(ValueDType::BF16)
        .transpose()
        .permute(vec![0, 2, 1, 3])
        .reshape(vec![b, s, w]);
    let v = TensorExpr::input("v", vec![b, h, s, d])
        .cast(ValueDType::BF16)
        .permute(vec![0, 2, 1, 3])
        .reshape(vec![b, s, w]);
    let z = |n| TensorExpr::constant(vec![0.], vec![]).broadcast(vec![b, s, n]);
    let q = TensorExpr::concat(&[q, z(2 * w)], 2).cast(ValueDType::BF16);
    let k = TensorExpr::concat(&[z(w), k, z(w)], 2).cast(ValueDType::BF16);
    let v = TensorExpr::concat(&[z(2 * w), v], 2).cast(ValueDType::BF16);
    let graph: TensorGraph<f32> = ((v + k) + q).into();
    let mut regions = qkv_grad_pack::recognize(&graph);
    ensure!(regions.len() == 1, "exactbaselinepacknotrecognized");
    let region = regions.remove(0);
    let values = region
        .inputs
        .iter()
        .map(|input| {
            let shape = graph[input.source].shape().clone();
            (
                input.source,
                Value {
                    binding: Binding {
                        pointer: 1,
                        len: shape.iter().product(),
                        dtype: DType::BF16,
                    },
                    view: VirtualTensor::identity(input.source, shape.clone(), DType::BF16),
                    source_shape: shape,
                    chain: vec![],
                },
            )
        })
        .collect();
    Ok((region, values))
}

#[cfg(test)]
mod gpu;
#[cfg(test)]
mod tests;
