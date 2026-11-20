//! Four-contiguous-values/thread Adam with a default-off, checked owner selector.
mod dispatch;
use super::*;
use crate::tile::ir::MatrixLayout;
use crate::tile::{
    Block, DType, Dim, Expr, MemorySpace, ScalarExpr, ScalarMemory, ScalarPredicate, Stmt, TileIR,
    TileLayout, TileVar,
};
use anyhow::{bail, ensure};
pub(super) use dispatch::{State, bindings_admitted};
pub use dispatch::{VectorBf16FirstAdamResources, VectorBf16FirstAdamStatus};

#[derive(Clone, Copy, Debug)]
pub(super) struct Span {
    pub(super) address: usize,
    pub(super) bytes: usize,
}
/// W/M/V/G,nextW/nextM/nextV,health. Live ownership stays with the caller.
pub(super) fn admitted(len: usize, spans: &[Span; 8], fresh_bf16: bool) -> bool {
    if !fresh_bf16 || len == 0 || !len.is_multiple_of(1024) {
        return false;
    }
    let Some(f32bytes) = len.checked_mul(4) else {
        return false;
    };
    let Some(bf16bytes) = len.checked_mul(2) else {
        return false;
    };
    let expected = [
        f32bytes,
        f32bytes,
        f32bytes,
        bf16bytes,
        f32bytes,
        f32bytes,
        f32bytes,
        len / 64,
    ];
    for (i, (span, bytes)) in spans.iter().zip(expected).enumerate() {
        let alignment = if i == 3 {
            8
        } else if i == 7 {
            4
        } else {
            16
        };
        if span.address == 0
            || !span.address.is_multiple_of(alignment)
            || span.bytes != bytes
            || span.address.checked_add(bytes).is_none()
        {
            return false;
        }
    }
    for i in 4..8 {
        for j in 0..8 {
            if i == j {
                continue;
            }
            let a = spans[i];
            let b = spans[j];
            if a.address < b.address + b.bytes && b.address < a.address + a.bytes {
                return false;
            }
        }
    }
    true
}
fn admitted_with_settings(len: usize, spans: &[Span; 8], settings: Span, fresh: bool) -> bool {
    if settings.address == 0
        || settings.bytes != 40
        || !settings.address.is_multiple_of(4)
        || settings.address.checked_add(settings.bytes).is_none()
        || !admitted(len, spans, fresh)
    {
        return false;
    }
    spans[4..].iter().all(|output| {
        output.address >= settings.address + settings.bytes
            || settings.address >= output.address + output.bytes
    })
}
fn linear() -> Expr {
    Expr::Mul(
        Box::new(Expr::BlockIdx(Dim::X)),
        Box::new(Expr::BlockDim(Dim::X)),
    ) + Expr::ThreadIdx(Dim::X)
}
fn remap_expr(value: &ScalarExpr, offset: usize) -> Result<ScalarExpr> {
    Ok(match value {
        ScalarExpr::Constant(v) => ScalarExpr::Constant(*v),
        ScalarExpr::Var(v) => ScalarExpr::Var(TileVar(v.0 + offset)),
        ScalarExpr::Unary { op, value } => ScalarExpr::Unary {
            op: *op,
            value: Box::new(remap_expr(value, offset)?),
        },
        ScalarExpr::Binary { op, lhs, rhs } => ScalarExpr::Binary {
            op: *op,
            lhs: Box::new(remap_expr(lhs, offset)?),
            rhs: Box::new(remap_expr(rhs, offset)?),
        },
        ScalarExpr::Select { .. } => bail!("selected Adam arithmetic is unsupported"),
    })
}
fn remap(stmt: &Stmt, offset: usize, index: &Expr) -> Result<Stmt> {
    let v = |x: TileVar| TileVar(x.0 + offset);
    Ok(match stmt {
        Stmt::AllocTile {
            var,
            space,
            layout,
            dtype,
            rows,
            cols,
        } => Stmt::AllocTile {
            var: v(*var),
            space: *space,
            layout: *layout,
            dtype: *dtype,
            rows: *rows,
            cols: *cols,
        },
        Stmt::LoadGlobalPredicated {
            dest,
            src_param,
            element_index,
            row,
            col,
            layout,
        } => {
            ensure!(
                src_param == "settings"
                    || ["weight", "gradient", "moment", "variance"].contains(&src_param.as_str()),
                "unknown Adam input"
            );
            Stmt::LoadGlobalPredicated {
                dest: v(*dest),
                src_param: src_param.clone(),
                element_index: if src_param == "settings" {
                    element_index.clone()
                } else {
                    index.clone()
                },
                row: row.clone(),
                col: if src_param == "settings" {
                    col.clone()
                } else {
                    index.clone()
                },
                layout: *layout,
            }
        }
        Stmt::SetScalar { dest, value } => {
            ensure!(
                !matches!(value, ScalarExpr::Select { .. }),
                "unknown selected Adam arithmetic"
            );
            Stmt::SetScalar {
                dest: v(*dest),
                value: remap_expr(value, offset)?,
            }
        }
        Stmt::Add { dest, a, b } => Stmt::Add {
            dest: v(*dest),
            a: v(*a),
            b: v(*b),
        },
        Stmt::Sub { dest, a, b } => Stmt::Sub {
            dest: v(*dest),
            a: v(*a),
            b: v(*b),
        },
        Stmt::Mul { dest, a, b } => Stmt::Mul {
            dest: v(*dest),
            a: v(*a),
            b: v(*b),
        },
        Stmt::Div { dest, a, b } => Stmt::Div {
            dest: v(*dest),
            a: v(*a),
            b: v(*b),
        },
        Stmt::Sqrt { dest, src } => Stmt::Sqrt {
            dest: v(*dest),
            src: v(*src),
        },
        Stmt::StoreGlobalPredicated {
            dest_param,
            src,
            row,
            layout,
            ..
        } => Stmt::StoreGlobalPredicated {
            dest_param: dest_param.clone(),
            src: v(*src),
            element_index: index.clone(),
            row: row.clone(),
            col: index.clone(),
            layout: *layout,
        },
        _ => bail!("unrecognized original Adam statement: {stmt:?}"),
    })
}
fn alloc(stmts: &mut Vec<Stmt>, next: &mut usize, shared: bool, cols: usize) -> TileVar {
    let var = TileVar(*next);
    *next += 1;
    stmts.push(Stmt::AllocTile {
        var,
        space: if shared {
            MemorySpace::Shared
        } else {
            MemorySpace::Register
        },
        layout: if shared {
            TileLayout::Shared(crate::tile::SharedLayout::row_major(cols))
        } else {
            TileLayout::ThreadScalar
        },
        dtype: DType::F32,
        rows: 1,
        cols,
    });
    var
}
fn when(lhs: Expr, rhs: usize, stmts: Vec<Stmt>) -> Stmt {
    Stmt::If {
        condition: ScalarPredicate::IndexLt {
            lhs,
            rhs: Expr::Const(rhs as i64),
        },
        body: Block { stmts },
    }
}
pub(super) fn ir(len: usize) -> Result<TileIR> {
    ensure!(
        len > 0 && len.is_multiple_of(1024) && i64::try_from(len).is_ok(),
        "vector Adam requires whole 1024-element CTAs"
    );
    let old = kernels::adam_bf16_first(len)?;
    let zeros = old
        .body
        .stmts
        .iter()
        .filter_map(|s| {
            if let Stmt::Zero { tile } = s {
                Some(*tile)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let [zero] = zeros.as_slice() else {
        bail!("old Adam initializer changed")
    };
    let bodies = old
        .body
        .stmts
        .iter()
        .filter_map(|s| {
            if let Stmt::If {
                condition:
                    ScalarPredicate::IndexLt {
                        lhs,
                        rhs: Expr::Const(end),
                    },
                body,
            } = s
            {
                if *lhs == linear() && *end == len as i64 {
                    Some(body)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let [body] = bodies.as_slice() else {
        bail!("old Adam body changed")
    };
    // Strict remapping walks every supported definition/use, rejects new patterns.
    for s in &body.stmts {
        remap(s, 0, &linear())?;
    }
    let max = body
        .stmts
        .iter()
        .filter_map(|s| {
            if let Stmt::AllocTile { var, .. } = s {
                Some(var.0)
            } else {
                None
            }
        })
        .max()
        .context("no old Adam scalar definitions")?;
    let stride = max + 1;
    let mut stmts = Vec::new();
    for j in 0..4 {
        stmts.push(Stmt::AllocTile {
            var: TileVar(zero.0 + j * stride),
            space: MemorySpace::Register,
            layout: TileLayout::ThreadScalar,
            dtype: DType::F32,
            rows: 1,
            cols: 1,
        });
        stmts.push(Stmt::Zero {
            tile: TileVar(zero.0 + j * stride),
        });
        for s in &body.stmts {
            if matches!(s, Stmt::AllocTile { .. }) {
                stmts.push(remap(s, j * stride, &linear())?);
            }
        }
    }
    for name in ["weight", "moment", "variance"] {
        let loads = body
            .stmts
            .iter()
            .filter_map(|s| {
                if let Stmt::LoadGlobalPredicated {
                    dest, src_param, ..
                } = s
                {
                    if src_param == name { Some(*dest) } else { None }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let [dest] = loads.as_slice() else {
            bail!("old Adam vector input multiplicity changed")
        };
        stmts.push(Stmt::LoadGlobalF32x4 {
            dests: std::array::from_fn(|j| TileVar(dest.0 + j * stride)),
            param: name.into(),
            group_index: linear(),
            extent: len,
        });
    }
    for j in 0..4 {
        let index = linear() * 4usize + Expr::Const(j as i64);
        for s in &body.stmts {
            if matches!(
                s,
                Stmt::AllocTile { .. } | Stmt::StoreGlobalPredicated { .. }
            ) {
                continue;
            }
            if matches!(s,Stmt::LoadGlobalPredicated{src_param,..}if ["weight","moment","variance"].contains(&src_param.as_str()))
            {
                continue;
            }
            stmts.push(remap(s, j * stride, &index)?);
        }
    }
    for name in ["next_weight", "next_moment", "next_variance"] {
        let stores = body
            .stmts
            .iter()
            .filter_map(|s| {
                if let Stmt::StoreGlobalPredicated {
                    src, dest_param, ..
                } = s
                {
                    if dest_param == name { Some(*src) } else { None }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let [src] = stores.as_slice() else {
            bail!("old Adam output multiplicity changed")
        };
        stmts.push(Stmt::StoreGlobalF32x4 {
            values: std::array::from_fn(|j| TileVar(src.0 + j * stride)),
            param: name.into(),
            group_index: linear(),
            extent: len,
        });
    }
    let mut next = 4 * stride;
    let h01 = alloc(&mut stmts, &mut next, false, 1);
    let h23 = alloc(&mut stmts, &mut next, false, 1);
    let health = alloc(&mut stmts, &mut next, false, 1);
    stmts.extend([
        Stmt::Add {
            dest: h01,
            a: *zero,
            b: TileVar(zero.0 + stride),
        },
        Stmt::Add {
            dest: h23,
            a: TileVar(zero.0 + 2 * stride),
            b: TileVar(zero.0 + 3 * stride),
        },
        Stmt::Add {
            dest: health,
            a: h01,
            b: h23,
        },
    ]);
    let shared = alloc(&mut stmts, &mut next, true, 8);
    let warp_sum = alloc(&mut stmts, &mut next, false, 1);
    let group_value = alloc(&mut stmts, &mut next, false, 1);
    let group_sum = alloc(&mut stmts, &mut next, false, 1);
    let lane = Expr::BitAnd(Box::new(Expr::ThreadIdx(Dim::X)), 31);
    let warp = Expr::ShiftRight(Box::new(Expr::ThreadIdx(Dim::X)), 5);
    let group = Expr::ShiftRight(Box::new(Expr::ThreadIdx(Dim::X)), 6);
    let dst = Expr::BlockIdx(Dim::X) * 4usize + group.clone();
    stmts.extend([
        Stmt::WarpReduce {
            dest: warp_sum,
            src: health,
            op: crate::tensor::ReduceOp::Sum,
            broadcast: false,
        },
        when(
            lane.clone(),
            1,
            vec![Stmt::StoreScalar {
                target: ScalarMemory::Shared(shared),
                index: warp,
                value: warp_sum,
            }],
        ),
        Stmt::Barrier,
        when(
            Expr::BitAnd(Box::new(Expr::ThreadIdx(Dim::X)), 63),
            32,
            vec![
                Stmt::Zero { tile: group_value },
                when(
                    lane.clone(),
                    2,
                    vec![Stmt::LoadScalar {
                        dest: group_value,
                        source: ScalarMemory::Shared(shared),
                        index: group * 2usize + lane.clone(),
                    }],
                ),
                Stmt::WarpReduce {
                    dest: group_sum,
                    src: group_value,
                    op: crate::tensor::ReduceOp::Sum,
                    broadcast: false,
                },
                when(
                    lane,
                    1,
                    vec![Stmt::StoreGlobalPredicated {
                        dest_param: "health".into(),
                        src: group_sum,
                        element_index: dst.clone(),
                        row: Expr::Const(0),
                        col: dst,
                        layout: MatrixLayout {
                            rows: 1,
                            cols: len / 256,
                            row_stride: len / 256,
                        },
                    }],
                ),
            ],
        ),
    ]);
    let candidate = TileIR {
        kernel_name: "resident_test_vector4_bf16_adam".into(),
        params: old.params,
        body: Block { stmts },
        shared_mem_bytes: 32,
    };
    candidate.validate_layouts()?;
    Ok(candidate)
}
#[cfg(test)]
mod tests;

#[cfg(test)]
mod batched;
