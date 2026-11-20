//! Strict canonical transform of canonical row math, never a second formula.
use super::*;
use anyhow::{Context, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};

fn definitions(block: &Block, output: &mut BTreeSet<usize>) -> Result<()> {
    for stmt in &block.stmts {
        match stmt {
            Stmt::AllocSharedArena { arena, .. } => {
                output.insert(arena.0);
            }
            Stmt::ViewSharedArena { dest, .. }
            | Stmt::SetScalar { dest, .. }
            | Stmt::LoadScalar { dest, .. }
            | Stmt::WarpReduce { dest, .. }
            | Stmt::WarpBroadcastLeader { dest, .. } => {
                output.insert(dest.0);
            }
            Stmt::AllocTile { var, .. } => {
                output.insert(var.0);
            }
            Stmt::If { body, .. } => definitions(body, output)?,
            Stmt::ForLoop { carries, body, .. } => {
                output.extend(carries.iter().map(|c| c.var.0));
                definitions(body, output)?;
            }
            Stmt::StoreScalar { .. }
            | Stmt::Barrier
            | Stmt::Zero { .. }
            | Stmt::MatMul { .. }
            | Stmt::ConvertLayout { .. }
            | Stmt::AsyncCopy { .. }
            | Stmt::AsyncCommit
            | Stmt::AsyncWait => {}
            _ => bail!("unsupported canonical forward definition: {stmt:?}"),
        }
    }
    Ok(())
}
fn fresh(next: &mut usize) -> Result<TileVar> {
    let var = TileVar(*next);
    *next = next
        .checked_add(1)
        .context("register statistics variable overflow")?;
    Ok(var)
}
#[derive(Clone)]
struct Remap {
    vars: BTreeMap<usize, TileVar>,
    old: Expr,
    new: Expr,
    forbidden_var: Option<String>,
}
impl Remap {
    fn var(&self, v: TileVar) -> TileVar {
        self.vars.get(&v.0).copied().unwrap_or(v)
    }
    fn index(&self, e: &Expr) -> Result<Expr> {
        if *e == self.old {
            return Ok(self.new.clone());
        }
        Ok(match e {
            Expr::Var(name) => {
                ensure!(
                    self.forbidden_var.as_ref() != Some(name),
                    "unmapped canonical row/output index"
                );
                e.clone()
            }
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
        })
    }
    fn scalar(&self, e: &ScalarExpr) -> Result<ScalarExpr> {
        Ok(match e {
            ScalarExpr::Constant(_) => e.clone(),
            ScalarExpr::Var(v) => ScalarExpr::Var(self.var(*v)),
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
    fn predicate(&self, p: &ScalarPredicate) -> Result<ScalarPredicate> {
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
    fn memory(&self, m: &ScalarMemory) -> ScalarMemory {
        match m {
            ScalarMemory::Global(_) => m.clone(),
            ScalarMemory::Shared(v) => ScalarMemory::Shared(self.var(*v)),
        }
    }
    fn statement(&self, s: &Stmt) -> Result<Stmt> {
        Ok(match s {
            Stmt::SetScalar { dest, value } => Stmt::SetScalar {
                dest: self.var(*dest),
                value: self.scalar(value)?,
            },
            Stmt::LoadScalar {
                dest,
                source,
                index,
            } => Stmt::LoadScalar {
                dest: self.var(*dest),
                source: self.memory(source),
                index: self.index(index)?,
            },
            Stmt::StoreScalar {
                target,
                index,
                value,
            } => Stmt::StoreScalar {
                target: self.memory(target),
                index: self.index(index)?,
                value: self.var(*value),
            },
            Stmt::WarpReduce {
                dest,
                src,
                op,
                broadcast,
            } => Stmt::WarpReduce {
                dest: self.var(*dest),
                src: self.var(*src),
                op: *op,
                broadcast: *broadcast,
            },
            Stmt::WarpBroadcastLeader { dest, src } => Stmt::WarpBroadcastLeader {
                dest: self.var(*dest),
                src: self.var(*src),
            },
            Stmt::If { condition, body } => Stmt::If {
                condition: self.predicate(condition)?,
                body: self.block(body)?,
            },
            Stmt::Barrier => Stmt::Barrier,
            _ => bail!("unsupported canonical forward row/output statement: {s:?}"),
        })
    }
    fn block(&self, b: &Block) -> Result<Block> {
        Ok(Block {
            stmts: b
                .stmts
                .iter()
                .map(|s| self.statement(s))
                .collect::<Result<_>>()?,
        })
    }
    #[cfg(test)]
    fn inverse(&self) -> Self {
        Self {
            vars: self.vars.iter().map(|(a, b)| (b.0, TileVar(*a))).collect(),
            old: self.new.clone(),
            new: self.old.clone(),
            forbidden_var: None,
        }
    }
}
fn remapping(
    block: &Block,
    old: Expr,
    new: Expr,
    forbidden: String,
    next: &mut usize,
) -> Result<Remap> {
    let mut defs = BTreeSet::new();
    definitions(block, &mut defs)?;
    let vars = defs
        .into_iter()
        .map(|v| Ok((v, fresh(next)?)))
        .collect::<Result<_>>()?;
    Ok(Remap {
        vars,
        old,
        new,
        forbidden_var: Some(forbidden),
    })
}
fn stats_accesses(block: &Block) -> (Vec<Stmt>, Vec<Stmt>) {
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    fn walk(b: &Block, reads: &mut Vec<Stmt>, writes: &mut Vec<Stmt>) {
        for s in &b.stmts {
            match s {
                Stmt::LoadScalar {
                    source: ScalarMemory::Global(p),
                    ..
                } if p == "statistics" => reads.push(s.clone()),
                Stmt::StoreScalar {
                    target: ScalarMemory::Global(p),
                    ..
                } if p == "statistics" => writes.push(s.clone()),
                Stmt::If { body, .. } | Stmt::ForLoop { body, .. } => walk(body, reads, writes),
                _ => {}
            }
        }
    }
    walk(block, &mut reads, &mut writes);
    (reads, writes)
}
#[cfg(test)]
struct RowProof {
    original: Block,
    rewritten: Block,
    remap: Remap,
    reads: Vec<Stmt>,
    writes: Vec<Stmt>,
    carry: [TileVar; 2],
}
#[cfg(test)]
struct OutputProof {
    original: Block,
    rewritten: Block,
    remap: Remap,
    denominator: Stmt,
    broadcast: TileVar,
}
#[cfg(test)]
struct Proof {
    rows: Vec<RowProof>,
    outputs: Vec<OutputProof>,
    carries: Vec<LoopCarry>,
}
struct Built {
    ir: TileIR,
    #[cfg(test)]
    proof: Proof,
}
fn build_impl(mut ir: TileIR) -> Result<Built> {
    ensure!(
        ir.shared_mem_bytes == SHARED_BYTES,
        "canonical forward shared arena changed"
    );
    ensure!(
        ir.params.iter().map(|p| p.name.as_str()).eq([
            "query",
            "key",
            "value",
            "output_0",
            "statistics"
        ]),
        "canonical forward ABI changed"
    );
    ensure!(
        ir.params.iter().map(|p| (p.dtype, p.is_input)).eq([
            (DType::BF16, true),
            (DType::BF16, true),
            (DType::BF16, true),
            (DType::BF16, false),
            (DType::F32, false)
        ]),
        "canonical typed forward ABI changed"
    );
    let mut defs = BTreeSet::new();
    definitions(&ir.body, &mut defs)?;
    let mut next = defs
        .last()
        .context("canonical forward has no variables")?
        .checked_add(1)
        .context("forward variable overflow")?;
    let mut carried = Vec::new();
    let mut carries = Vec::new();
    for _ in 0..8 {
        let m = fresh(&mut next)?;
        let l = fresh(&mut next)?;
        carried.push([m, l]);
        carries.extend([
            LoopCarry {
                var: m,
                initial: ScalarExpr::Constant(f32::NEG_INFINITY),
            },
            LoopCarry {
                var: l,
                initial: ScalarExpr::Constant(0.),
            },
        ]);
    }
    let key_pos = ir
        .body
        .stmts
        .iter()
        .position(|s| matches!(s,Stmt::ForLoop {loop_var,..}if loop_var.starts_with("causal_keys")))
        .context("canonical key loop absent")?;
    let Stmt::ForLoop {
        body: key_body,
        carries: key_carries,
        start: key_start,
        end: key_end,
        ..
    } = &mut ir.body.stmts[key_pos]
    else {
        bail!("canonical key loop changed")
    };
    ensure!(
        key_carries.is_empty() && *key_start == ix(0) && *key_end == Expr::BlockIdx(Dim::X) + ix(1),
        "canonical causal key order/extent/state changed"
    );
    *key_carries = carries.clone();
    let row_pos = key_body
        .stmts
        .iter()
        .position(|s| matches!(s,Stmt::ForLoop {loop_var,..}if loop_var.starts_with("online_rows")))
        .context("canonical row loop absent")?;
    let Stmt::ForLoop {
        loop_var,
        start,
        end,
        carries: row_carries,
        body: old_row,
    } = &key_body.stmts[row_pos]
    else {
        bail!("canonical row loop changed")
    };
    ensure!(
        *start == ix(0) && *end == ix(8) && row_carries.is_empty(),
        "canonical row loop geometry changed"
    );
    let old_row = old_row.clone();
    let row_name = loop_var.clone();
    let row_expr = Expr::Var(row_name.clone()) * 8usize + Expr::ThreadIdx(Dim::Y);
    let (reads, writes) = stats_accesses(&old_row);
    ensure!(
        reads.len() == 2 && writes.len() == 2,
        "canonical row statistics multiplicity changed"
    );
    let canonical_stats = stats_index(Expr::BlockIdx(Dim::X) * TILE + row_expr.clone());
    for (slot, (read, write)) in reads.iter().zip(&writes).enumerate() {
        let Stmt::LoadScalar {
            index: read_index, ..
        } = read
        else {
            bail!("statistics read changed")
        };
        let Stmt::StoreScalar {
            index: write_index, ..
        } = write
        else {
            bail!("statistics write changed")
        };
        let expected = if slot == 0 {
            canonical_stats.clone()
        } else {
            canonical_stats.clone() + ix(1)
        };
        ensure!(
            *read_index == expected && *write_index == expected,
            "canonical m/l slot/index order changed"
        );
    }
    let mut row_maps = Vec::new();
    #[cfg(test)]
    let mut row_proofs = Vec::new();
    let mut unrolled = Vec::new();
    fn replace(
        block: &mut Block,
        reads: &[Stmt],
        writes: &[Stmt],
        carry: [TileVar; 2],
        remap: &Remap,
        counts: &mut [usize; 4],
    ) -> Result<()> {
        for s in &mut block.stmts {
            if let Stmt::If { body, .. } = s {
                replace(body, reads, writes, carry, remap, counts)?;
                continue;
            }
            for j in 0..2 {
                let read = remap.statement(&reads[j])?;
                let write = remap.statement(&writes[j])?;
                if format!("{s:?}") == format!("{read:?}") {
                    let Stmt::LoadScalar { dest, .. } = read else {
                        bail!("statistics read changed")
                    };
                    *s = Stmt::SetScalar {
                        dest,
                        value: ScalarExpr::Var(carry[j]),
                    };
                    counts[j] += 1;
                    break;
                }
                if format!("{s:?}") == format!("{write:?}") {
                    let Stmt::StoreScalar { value, .. } = write else {
                        bail!("statistics write changed")
                    };
                    *s = Stmt::SetScalar {
                        dest: carry[j],
                        value: ScalarExpr::Var(value),
                    };
                    counts[2 + j] += 1;
                    break;
                }
            }
        }
        Ok(())
    }
    for (round, &carry) in carried.iter().enumerate() {
        let row = ix(round) * 8usize + Expr::ThreadIdx(Dim::Y);
        let map = remapping(&old_row, row_expr.clone(), row, row_name.clone(), &mut next)?;
        let mut rewritten = map.block(&old_row)?;
        let mut counts = [0; 4];
        replace(&mut rewritten, &reads, &writes, carry, &map, &mut counts)?;
        ensure!(
            counts == [1; 4],
            "register row failed exact statistics substitution"
        );
        unrolled.extend(rewritten.stmts.clone());
        row_maps.push(map.clone());
        #[cfg(test)]
        row_proofs.push(RowProof {
            original: old_row.clone(),
            rewritten,
            remap: map,
            reads: reads.clone(),
            writes: writes.clone(),
            carry,
        });
    }
    key_body.stmts.splice(row_pos..row_pos + 1, unrolled);
    // Reg mode defines m/l via loop carry initials, never incoming global data.
    let init_pos=ir.body.stmts.iter().position(|s|matches!(s,Stmt::ForLoop {loop_var,..}if loop_var.starts_with("initialize_statistics"))).context("canonical statistics initializer absent")?;
    let init = ir.body.stmts.remove(init_pos);
    let Stmt::ForLoop {
        loop_var: init_name,
        start: init_start,
        end: init_end,
        carries: init_carries,
        body: init_body,
    } = init
    else {
        bail!("statistics initializer changed")
    };
    ensure!(
        init_start == ix(0) && init_end == ix(8) && init_carries.is_empty(),
        "statistics initializer geometry changed"
    );
    let [Stmt::If { condition, body }] = init_body.stmts.as_slice() else {
        bail!("statistics initializer leader body changed")
    };
    ensure!(
        *condition
            == ScalarPredicate::IndexLt {
                lhs: Expr::ThreadIdx(Dim::X),
                rhs: ix(1)
            },
        "statistics initializer lane owner changed"
    );
    let [
        Stmt::SetScalar {
            dest: m,
            value: ScalarExpr::Constant(m_value),
        },
        Stmt::SetScalar {
            dest: l,
            value: ScalarExpr::Constant(l_value),
        },
        Stmt::StoreScalar {
            value: stored_m, ..
        },
        Stmt::StoreScalar {
            value: stored_l, ..
        },
    ] = body.stmts.as_slice()
    else {
        bail!("statistics initial constants/stores changed")
    };
    ensure!(
        m_value.to_bits() == f32::NEG_INFINITY.to_bits()
            && l_value.to_bits() == 0.0f32.to_bits()
            && stored_m == m
            && stored_l == l,
        "statistics initializer values changed"
    );
    let (init_reads, init_writes) = stats_accesses(&init_body);
    ensure!(
        init_reads.is_empty() && init_writes.len() == 2,
        "statistics initializer contract changed"
    );
    let initial_index = stats_index(
        Expr::BlockIdx(Dim::X) * TILE + (Expr::Var(init_name) * 8usize + Expr::ThreadIdx(Dim::Y)),
    );
    for (slot, write) in init_writes.iter().enumerate() {
        let Stmt::StoreScalar { index, .. } = write else {
            bail!("statistics initializer store changed")
        };
        ensure!(
            *index
                == if slot == 0 {
                    initial_index.clone()
                } else {
                    initial_index.clone() + ix(1)
                },
            "statistics initializer slot/index changed"
        );
    }
    let output_pos = ir
        .body
        .stmts
        .iter()
        .position(
            |s| matches!(s,Stmt::ForLoop {loop_var,..}if loop_var.starts_with("store_output")),
        )
        .context("canonical final output loop absent")?;
    let Stmt::ForLoop {
        loop_var,
        start,
        end,
        carries: output_carries,
        body: old_output,
    } = &ir.body.stmts[output_pos]
    else {
        bail!("canonical output loop changed")
    };
    ensure!(
        *start == ix(0) && *end == ix(16) && output_carries.is_empty(),
        "canonical output loop geometry changed"
    );
    let old_output = old_output.clone();
    let output_name = loop_var.clone();
    let old_i = Expr::Var(output_name.clone()) * THREADS
        + Expr::ThreadIdx(Dim::Y) * 32usize
        + Expr::ThreadIdx(Dim::X);
    let (den_reads, den_writes) = stats_accesses(&old_output);
    ensure!(
        den_reads.len() == 1 && den_writes.is_empty(),
        "canonical denominator changed"
    );
    let Stmt::LoadScalar {
        index: den_index, ..
    } = &den_reads[0]
    else {
        bail!("canonical denominator absent")
    };
    ensure!(
        *den_index == stats_index(Expr::BlockIdx(Dim::X) * TILE + div(old_i.clone(), TILE)) + ix(1),
        "canonical final denominator index changed"
    );
    let mut final_stmts = Vec::new();
    #[cfg(test)]
    let mut output_proofs = Vec::new();
    for (round, &carry) in carried.iter().enumerate() {
        let map = &row_maps[round];
        let mut exports = Vec::new();
        for (j, write) in writes.iter().enumerate() {
            let Stmt::StoreScalar { target, index, .. } = map.statement(write)? else {
                bail!("statistics export changed")
            };
            exports.push(Stmt::StoreScalar {
                target,
                index,
                value: carry[j],
            });
        }
        final_stmts.push(Stmt::If {
            condition: ScalarPredicate::IndexLt {
                lhs: Expr::ThreadIdx(Dim::X),
                rhs: ix(1),
            },
            body: Block { stmts: exports },
        });
        let broadcast = fresh(&mut next)?;
        final_stmts.push(Stmt::WarpBroadcastLeader {
            dest: broadcast,
            src: carry[1],
        });
        for half in 0..2 {
            // Replace the WHOLE canonical element index. Substituting its round
            // alone would retain the old two-warps-per-row ownership and fail.
            let new_i = (ix(round) * 8usize + Expr::ThreadIdx(Dim::Y)) * TILE
                + Expr::ThreadIdx(Dim::X)
                + ix(half * 32);
            let map = remapping(
                &old_output,
                old_i.clone(),
                new_i,
                output_name.clone(),
                &mut next,
            )?;
            let mut rewritten = map.block(&old_output)?;
            let denominator = map.statement(&den_reads[0])?;
            let mut count = 0;
            for s in &mut rewritten.stmts {
                if format!("{s:?}") == format!("{denominator:?}") {
                    let Stmt::LoadScalar { dest, .. } = s else {
                        bail!("output denominator changed")
                    };
                    *s = Stmt::SetScalar {
                        dest: *dest,
                        value: ScalarExpr::Var(broadcast),
                    };
                    count += 1;
                }
            }
            ensure!(count == 1, "output denominator substitution not unique");
            final_stmts.extend(rewritten.stmts.clone());
            #[cfg(test)]
            output_proofs.push(OutputProof {
                original: old_output.clone(),
                rewritten,
                remap: map,
                denominator: den_reads[0].clone(),
                broadcast,
            });
        }
    }
    ir.body
        .stmts
        .splice(output_pos..output_pos + 1, final_stmts);
    ir.kernel_name.push_str("_register_statistics");
    ir.validate_layouts()?;
    Ok(Built {
        ir,
        #[cfg(test)]
        proof: Proof {
            rows: row_proofs,
            outputs: output_proofs,
            carries,
        },
    })
}
#[cfg(test)]
fn build(ir: TileIR) -> Result<(TileIR, Proof)> {
    let built = build_impl(ir)?;
    Ok((built.ir, built.proof))
}
pub(super) fn transform(ir: TileIR) -> Result<TileIR> {
    Ok(build_impl(ir)?.ir)
}

#[cfg(test)]
mod tests;
