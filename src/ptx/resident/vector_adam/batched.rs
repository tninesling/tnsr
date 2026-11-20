//! Qualification-only same-length, <=4-member paired-vector Adam batches.
//! One CTA belongs to exactly one parameter. No owner selection or timing API.
use super::*;
use crate::ptx::resident::bf16_shadow::{AdamKind, paired_ir};
use crate::tile::ir::KernelParam;

const SHARED: TileVar = TileVar(0);
const MAX_MEMBERS: usize = 4;
const PARAMETER_BYTES_LIMIT: usize = 4096;
const ABI: [&str; 10] = [
    "weight",
    "gradient",
    "moment",
    "variance",
    "settings",
    "next_weight",
    "next_moment",
    "next_variance",
    "health",
    "next_shadow",
];

#[derive(Clone, Copy)]
struct Rewrite {
    member: usize,
    lower: usize,
    offset: usize,
    inverse: bool,
}
impl Rewrite {
    fn var(self, v: TileVar) -> Result<TileVar> {
        Ok(TileVar(if self.inverse {
            v.0.checked_sub(self.offset)
                .context("batch inverse variable underflow")?
        } else {
            v.0.checked_add(self.offset)
                .context("batch variable overflow")?
        }))
    }
    fn param(self, name: &str) -> Result<String> {
        if self.inverse {
            let original = name
                .strip_suffix(&format!("_{}", self.member))
                .context("batch inverse parameter suffix mismatch")?;
            ensure!(ABI.contains(&original), "batch inverse unknown parameter");
            Ok(original.into())
        } else {
            ensure!(ABI.contains(&name), "batch unknown original parameter");
            Ok(format!("{name}_{}", self.member))
        }
    }
    fn index(self, e: &Expr) -> Result<Expr> {
        if self.inverse
            && let Expr::Sub(lhs, rhs) = e
            && **lhs == Expr::BlockIdx(Dim::X)
            && **rhs == Expr::Const(i64::try_from(self.lower)?)
        {
            return Ok(Expr::BlockIdx(Dim::X));
        }
        Ok(match e {
            Expr::BlockIdx(Dim::X) if !self.inverse => Expr::Sub(
                Box::new(e.clone()),
                Box::new(Expr::Const(i64::try_from(self.lower)?)),
            ),
            Expr::Const(_) | Expr::BlockIdx(_) | Expr::BlockDim(_) | Expr::ThreadIdx(_) => {
                e.clone()
            }
            Expr::Add(a, b) => Expr::Add(Box::new(self.index(a)?), Box::new(self.index(b)?)),
            Expr::Sub(a, b) => Expr::Sub(Box::new(self.index(a)?), Box::new(self.index(b)?)),
            Expr::Mul(a, b) => Expr::Mul(Box::new(self.index(a)?), Box::new(self.index(b)?)),
            Expr::FloorDiv(a, n) => Expr::FloorDiv(Box::new(self.index(a)?), *n),
            Expr::Mod(a, n) => Expr::Mod(Box::new(self.index(a)?), *n),
            Expr::ShiftRight(a, n) => Expr::ShiftRight(Box::new(self.index(a)?), *n),
            Expr::BitAnd(a, n) => Expr::BitAnd(Box::new(self.index(a)?), *n),
            Expr::Var(_) => bail!("unexpected lexical index in canonical paired vector body"),
        })
    }
    fn scalar(self, e: &ScalarExpr) -> Result<ScalarExpr> {
        Ok(match e {
            ScalarExpr::Constant(_) => e.clone(),
            ScalarExpr::Var(v) => ScalarExpr::Var(self.var(*v)?),
            ScalarExpr::Unary { op, value } => ScalarExpr::Unary {
                op: *op,
                value: Box::new(self.scalar(value)?),
            },
            ScalarExpr::Binary { op, lhs, rhs } => ScalarExpr::Binary {
                op: *op,
                lhs: Box::new(self.scalar(lhs)?),
                rhs: Box::new(self.scalar(rhs)?),
            },
            ScalarExpr::Select {
                condition,
                then_value,
                else_value,
            } => ScalarExpr::Select {
                condition: Box::new(self.predicate(condition)?),
                then_value: Box::new(self.scalar(then_value)?),
                else_value: Box::new(self.scalar(else_value)?),
            },
        })
    }
    fn predicate(self, p: &ScalarPredicate) -> Result<ScalarPredicate> {
        Ok(match p {
            ScalarPredicate::IndexLt { lhs, rhs } => ScalarPredicate::IndexLt {
                lhs: self.index(lhs)?,
                rhs: self.index(rhs)?,
            },
            ScalarPredicate::Equal { lhs, rhs } => ScalarPredicate::Equal {
                lhs: self.scalar(lhs)?,
                rhs: self.scalar(rhs)?,
            },
        })
    }
    fn memory(self, m: &ScalarMemory) -> Result<ScalarMemory> {
        Ok(match m {
            ScalarMemory::Shared(v) => ScalarMemory::Shared(self.var(*v)?),
            ScalarMemory::Global(p) => ScalarMemory::Global(self.param(p)?),
        })
    }
    fn block(self, b: &Block) -> Result<Block> {
        Ok(Block {
            stmts: b
                .stmts
                .iter()
                .map(|s| self.stmt(s))
                .collect::<Result<_>>()?,
        })
    }
    fn stmt(self, s: &Stmt) -> Result<Stmt> {
        let v = |v| self.var(v);
        Ok(match s {
            Stmt::AllocTile {
                var,
                space: MemorySpace::Shared,
                layout,
                dtype,
                rows,
                cols,
            } if !self.inverse => {
                ensure!(
                    *dtype == DType::F32
                        && *rows == 1
                        && *cols == 8
                        && *layout == TileLayout::Shared(crate::tile::SharedLayout::row_major(8)),
                    "canonical paired vector shared scratch changed"
                );
                Stmt::AliasShared {
                    dest: v(*var)?,
                    src: SHARED,
                    layout: crate::tile::SharedLayout::row_major(8),
                    rows: 1,
                    cols: 8,
                }
            }
            Stmt::AliasShared {
                dest,
                src,
                layout,
                rows,
                cols,
            } if self.inverse => {
                ensure!(
                    *src == SHARED
                        && *rows == 1
                        && *cols == 8
                        && *layout == crate::tile::SharedLayout::row_major(8),
                    "batch inverse shared alias changed"
                );
                Stmt::AllocTile {
                    var: v(*dest)?,
                    space: MemorySpace::Shared,
                    layout: TileLayout::Shared(*layout),
                    dtype: DType::F32,
                    rows: *rows,
                    cols: *cols,
                }
            }
            Stmt::AllocTile {
                var,
                space,
                layout,
                dtype,
                rows,
                cols,
            } => Stmt::AllocTile {
                var: v(*var)?,
                space: *space,
                layout: *layout,
                dtype: *dtype,
                rows: *rows,
                cols: *cols,
            },
            Stmt::Zero { tile } => Stmt::Zero { tile: v(*tile)? },
            Stmt::SetScalar { dest, value } => Stmt::SetScalar {
                dest: v(*dest)?,
                value: self.scalar(value)?,
            },
            Stmt::LoadGlobalPredicated {
                dest,
                src_param,
                element_index,
                row,
                col,
                layout,
            } => Stmt::LoadGlobalPredicated {
                dest: v(*dest)?,
                src_param: self.param(src_param)?,
                element_index: self.index(element_index)?,
                row: self.index(row)?,
                col: self.index(col)?,
                layout: *layout,
            },
            Stmt::StoreGlobalPredicated {
                dest_param,
                src,
                element_index,
                row,
                col,
                layout,
            } => Stmt::StoreGlobalPredicated {
                dest_param: self.param(dest_param)?,
                src: v(*src)?,
                element_index: self.index(element_index)?,
                row: self.index(row)?,
                col: self.index(col)?,
                layout: *layout,
            },
            Stmt::LoadGlobalF32x4 {
                dests,
                param,
                group_index,
                extent,
            } => Stmt::LoadGlobalF32x4 {
                dests: [v(dests[0])?, v(dests[1])?, v(dests[2])?, v(dests[3])?],
                param: self.param(param)?,
                group_index: self.index(group_index)?,
                extent: *extent,
            },
            Stmt::StoreGlobalF32x4 {
                values,
                param,
                group_index,
                extent,
            } => Stmt::StoreGlobalF32x4 {
                values: [v(values[0])?, v(values[1])?, v(values[2])?, v(values[3])?],
                param: self.param(param)?,
                group_index: self.index(group_index)?,
                extent: *extent,
            },
            Stmt::Add { dest, a, b } => Stmt::Add {
                dest: v(*dest)?,
                a: v(*a)?,
                b: v(*b)?,
            },
            Stmt::Sub { dest, a, b } => Stmt::Sub {
                dest: v(*dest)?,
                a: v(*a)?,
                b: v(*b)?,
            },
            Stmt::Mul { dest, a, b } => Stmt::Mul {
                dest: v(*dest)?,
                a: v(*a)?,
                b: v(*b)?,
            },
            Stmt::Div { dest, a, b } => Stmt::Div {
                dest: v(*dest)?,
                a: v(*a)?,
                b: v(*b)?,
            },
            Stmt::Sqrt { dest, src } => Stmt::Sqrt {
                dest: v(*dest)?,
                src: v(*src)?,
            },
            Stmt::WarpReduce {
                dest,
                src,
                op,
                broadcast,
            } => Stmt::WarpReduce {
                dest: v(*dest)?,
                src: v(*src)?,
                op: *op,
                broadcast: *broadcast,
            },
            Stmt::LoadScalar {
                dest,
                source,
                index,
            } => Stmt::LoadScalar {
                dest: v(*dest)?,
                source: self.memory(source)?,
                index: self.index(index)?,
            },
            Stmt::StoreScalar {
                target,
                index,
                value,
            } => Stmt::StoreScalar {
                target: self.memory(target)?,
                index: self.index(index)?,
                value: v(*value)?,
            },
            Stmt::If { condition, body } => Stmt::If {
                condition: self.predicate(condition)?,
                body: self.block(body)?,
            },
            Stmt::Barrier => Stmt::Barrier,
            _ => bail!("unsupported canonical paired vector statement in batch remapping: {s:?}"),
        })
    }
}
fn stride(block: &Block) -> Result<usize> {
    let mut maximum = None;
    fn visit(block: &Block, maximum: &mut Option<usize>) {
        for statement in &block.stmts {
            match statement {
                Stmt::AllocTile { var, .. } => *maximum = Some(maximum.unwrap_or(0).max(var.0)),
                Stmt::If { body, .. } => visit(body, maximum),
                _ => {}
            }
        }
    }
    visit(block, &mut maximum);
    maximum
        .context("canonical paired vector has no variables")?
        .checked_add(1)
        .context("batch variable stride overflow")
}
fn region_body(statement: &Stmt) -> Result<&Block> {
    let Stmt::If { body, .. } = statement else {
        bail!("batch upper region missing");
    };
    let [Stmt::If { body, .. }] = body.stmts.as_slice() else {
        bail!("batch lower region missing");
    };
    Ok(body)
}
fn ir(len: usize, members: usize) -> Result<TileIR> {
    ensure!(
        (1..=MAX_MEMBERS).contains(&members),
        "paired vector batch supports one to four members"
    );
    ensure!(
        len > 0 && len.is_multiple_of(1024) && u32::try_from(len).is_ok(),
        "batch requires full representable vector CTAs"
    );
    let blocks = len / 1024;
    let total_blocks = blocks
        .checked_mul(members)
        .context("batch CTA range overflow")?;
    ensure!(
        u32::try_from(total_blocks).is_ok(),
        "batch CUDA grid overflow"
    );
    let original = paired_ir(len, AdamKind::Vector4)?;
    ensure!(
        original.params.len() == 10 && original.params.iter().map(|p| p.name.as_str()).eq(ABI),
        "paired vector ABI changed"
    );
    let stride = stride(&original.body)?;
    let mut params = Vec::new();
    let mut body = Block {
        stmts: vec![Stmt::AllocTile {
            var: SHARED,
            space: MemorySpace::Shared,
            layout: TileLayout::Shared(crate::tile::SharedLayout::row_major(8)),
            dtype: DType::F32,
            rows: 1,
            cols: 8,
        }],
    };
    for member in 0..members {
        let lower = blocks
            .checked_mul(member)
            .context("batch region start overflow")?;
        let upper = lower
            .checked_add(blocks)
            .context("batch region end overflow")?;
        let rewrite = Rewrite {
            member,
            lower,
            offset: member
                .checked_mul(stride)
                .and_then(|n| n.checked_add(1))
                .context("batch variable offset overflow")?,
            inverse: false,
        };
        for param in &original.params {
            params.push(KernelParam {
                name: rewrite.param(&param.name)?,
                dtype: param.dtype,
                is_input: param.is_input,
            });
        }
        body.stmts.push(Stmt::If {
            condition: ScalarPredicate::IndexLt {
                lhs: Expr::BlockIdx(Dim::X),
                rhs: Expr::Const(i64::try_from(upper)?),
            },
            body: Block {
                stmts: vec![Stmt::If {
                    condition: ScalarPredicate::IndexLt {
                        lhs: Expr::Const(i64::try_from(lower)?),
                        rhs: Expr::BlockIdx(Dim::X) + Expr::Const(1),
                    },
                    body: rewrite.block(&original.body)?,
                }],
            },
        });
    }
    ensure!(
        params
            .len()
            .checked_mul(8)
            .is_some_and(|bytes| bytes <= PARAMETER_BYTES_LIMIT),
        "batch ABI exceeds 4KB kernel parameter limit"
    );
    let batch = TileIR {
        kernel_name: "resident_test_batched_paired_vector_adam".into(),
        params,
        body,
        shared_mem_bytes: 32,
    };
    batch.validate_layouts()?;
    Ok(batch)
}
/// Raw addresses are freshly supplied by the caller. Only readonly operands may
/// alias; every staged state/health/shadow output is disjoint from all other spans.
fn admitted(len: usize, members: &[[Span; 10]]) -> bool {
    if !(1..=MAX_MEMBERS).contains(&members.len()) || len == 0 || !len.is_multiple_of(1024) {
        return false;
    }
    let Some(f32_bytes) = len.checked_mul(4) else {
        return false;
    };
    let Some(bf16_bytes) = len.checked_mul(2) else {
        return false;
    };
    let expected = [
        f32_bytes,
        bf16_bytes,
        f32_bytes,
        f32_bytes,
        40,
        f32_bytes,
        f32_bytes,
        f32_bytes,
        len / 64,
        bf16_bytes,
    ];
    let alignment = [16, 8, 16, 16, 4, 16, 16, 16, 4, 2];
    for spans in members {
        for (index, span) in spans.iter().enumerate() {
            if span.bytes != expected[index]
                || span.address == 0
                || !span.address.is_multiple_of(alignment[index])
                || span.address.checked_add(span.bytes).is_none()
            {
                return false;
            }
        }
    }
    for (member, spans) in members.iter().enumerate() {
        for (output, &a) in spans.iter().enumerate().skip(5) {
            for (other, operands) in members.iter().enumerate() {
                for (operand, b) in operands.iter().enumerate() {
                    if member == other && output == operand {
                        continue;
                    }
                    if a.address < b.address + b.bytes && b.address < a.address + a.bytes {
                        return false;
                    }
                }
            }
        }
    }
    true
}
mod tests;

mod scoped;
