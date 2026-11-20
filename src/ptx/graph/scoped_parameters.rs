//! CFG(test) explicit uniform-region pointer loads. Default lowering is untouched.
use super::*;
use crate::tile::{Block, ScalarMemory, ScalarPredicate};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub(crate) struct Scope {
    /// Indices of the outer and inner If statements; loads belong to the full
    /// selected inner body, never to a nested first-use/thread predicate.
    pub path: Vec<usize>,
    pub parameters: Vec<String>,
}
struct Region {
    root: usize,
    inner: usize,
    parameters: Vec<String>,
}
fn resolve(expr: &Expr, names: &HashMap<String, Expr>, depth: usize) -> Result<Expr> {
    ensure!(depth < 128, "cyclic/deep scoped index binding");
    Ok(match expr {
        Expr::Var(name) => resolve(
            names.get(name).context("unknown scoped guard index")?,
            names,
            depth + 1,
        )?,
        Expr::Add(a, b) => Expr::Add(
            Box::new(resolve(a, names, depth + 1)?),
            Box::new(resolve(b, names, depth + 1)?),
        ),
        Expr::Sub(a, b) => Expr::Sub(
            Box::new(resolve(a, names, depth + 1)?),
            Box::new(resolve(b, names, depth + 1)?),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Box::new(resolve(a, names, depth + 1)?),
            Box::new(resolve(b, names, depth + 1)?),
        ),
        Expr::FloorDiv(a, n) => Expr::FloorDiv(Box::new(resolve(a, names, depth + 1)?), *n),
        Expr::Mod(a, n) => Expr::Mod(Box::new(resolve(a, names, depth + 1)?), *n),
        Expr::ShiftRight(a, n) => Expr::ShiftRight(Box::new(resolve(a, names, depth + 1)?), *n),
        Expr::BitAnd(a, n) => Expr::BitAnd(Box::new(resolve(a, names, depth + 1)?), *n),
        _ => expr.clone(),
    })
}
fn prefix(stmt: &Stmt, names: &mut HashMap<String, Expr>, shared: bool) -> Result<()> {
    match stmt {
        Stmt::LetIndex { name, value } => {
            ensure!(!names.contains_key(name), "duplicate scoped index binding");
            let value = resolve(value, names, 0)?;
            names.insert(name.clone(), value);
        }
        Stmt::AllocTile {
            space: MemorySpace::Shared,
            ..
        } if shared => {}
        _ => bail!("scoped prefix has unknown statement or memory use"),
    }
    Ok(())
}
fn uses(block: &Block, names: &mut BTreeSet<String>) -> Result<()> {
    for stmt in &block.stmts {
        match stmt {
            Stmt::LoadGlobalF32x4 { param, .. } | Stmt::StoreGlobalF32x4 { param, .. } => {
                names.insert(param.clone());
            }
            Stmt::LoadGlobalPredicated { src_param, .. } => {
                names.insert(src_param.clone());
            }
            Stmt::StoreGlobalPredicated { dest_param, .. } => {
                names.insert(dest_param.clone());
            }
            Stmt::LoadScalar {
                source: ScalarMemory::Global(param),
                ..
            }
            | Stmt::StoreScalar {
                target: ScalarMemory::Global(param),
                ..
            } => {
                names.insert(param.clone());
            }
            Stmt::If { body, .. } => uses(body, names)?,
            Stmt::LetIndex { .. }
            | Stmt::AllocTile { .. }
            | Stmt::AliasShared { .. }
            | Stmt::Zero { .. }
            | Stmt::SetScalar { .. }
            | Stmt::Add { .. }
            | Stmt::Sub { .. }
            | Stmt::Mul { .. }
            | Stmt::Div { .. }
            | Stmt::Sqrt { .. }
            | Stmt::WarpReduce { .. }
            | Stmt::LoadScalar {
                source: ScalarMemory::Shared(_),
                ..
            }
            | Stmt::StoreScalar {
                target: ScalarMemory::Shared(_),
                ..
            }
            | Stmt::Barrier => {}
            _ => bail!("unknown statement in scoped parameter-use proof"),
        }
    }
    Ok(())
}
fn prove(ir: &TileIR, scopes: &[Scope]) -> Result<Vec<Region>> {
    ir.validate_layouts()?;
    ensure!(
        (1..=4).contains(&scopes.len()) && ir.params.len() == 10 * scopes.len(),
        "scoped prototype requires <=4 ten-pointer regions"
    );
    let declared = ir
        .params
        .iter()
        .map(|p| p.name.clone())
        .collect::<BTreeSet<_>>();
    ensure!(
        declared.len() == ir.params.len(),
        "duplicate scoped root parameter"
    );
    let mut requested = BTreeMap::new();
    for scope in scopes {
        ensure!(
            scope.path.len() == 2,
            "only fully selected uniform regions may load parameters; nested first-use loads rejected"
        );
        ensure!(
            requested.insert(scope.path[0], scope).is_none(),
            "duplicate scoped root region"
        );
    }
    let mut seen = BTreeSet::new();
    let mut names = HashMap::new();
    let mut regions = Vec::new();
    let mut lower_expected = 0i64;
    for (root, stmt) in ir.body.stmts.iter().enumerate() {
        if let Stmt::If { condition, body } = stmt {
            let scope = requested
                .get(&root)
                .context("unclaimed scoped root conditional")?;
            let ScalarPredicate::IndexLt { lhs, rhs } = condition else {
                bail!("outer scope guard is not uniform index bound");
            };
            let Expr::Const(upper) = resolve(rhs, &names, 0)? else {
                bail!("outer scope bound is not constant");
            };
            ensure!(
                resolve(lhs, &names, 0)? == Expr::BlockIdx(Dim::X)
                    && upper > lower_expected
                    && u32::try_from(upper).is_ok(),
                "outer scope is not a bounded uniform CTA region"
            );
            let inner = scope.path[1];
            ensure!(
                inner.checked_add(1) == Some(body.stmts.len()),
                "scope must select complete inner body with no trailing statements"
            );
            let mut inner_names = names.clone();
            for prefix_stmt in &body.stmts[..inner] {
                prefix(prefix_stmt, &mut inner_names, false)?;
            }
            let Stmt::If {
                condition,
                body: inner_body,
            } = body.stmts.get(inner).context("inner scope missing")?
            else {
                bail!("inner scope is not If");
            };
            let ScalarPredicate::IndexLt { lhs, rhs } = condition else {
                bail!("inner scope guard is not uniform index bound");
            };
            let lower = resolve(lhs, &inner_names, 0)?;
            let rhs = resolve(rhs, &inner_names, 0)?;
            let cta_plus_one = Expr::BlockIdx(Dim::X) + Expr::Const(1);
            let one_plus_cta = Expr::Const(1) + Expr::BlockIdx(Dim::X);
            ensure!(
                lower == Expr::Const(lower_expected)
                    && (rhs == cta_plus_one || rhs == one_plus_cta),
                "inner scope is not the matching lower CTA bound; thread-dependent predicates rejected"
            );
            let mut referenced = BTreeSet::new();
            uses(inner_body, &mut referenced)?;
            let claimed = scope.parameters.iter().cloned().collect::<BTreeSet<_>>();
            ensure!(
                claimed.len() == 10
                    && claimed.len() == scope.parameters.len()
                    && referenced == claimed,
                "parameter use is cross-scope, unknown or not dominated by the full selected region"
            );
            ensure!(
                claimed
                    .iter()
                    .all(|p| declared.contains(p) && seen.insert(p.clone())),
                "root parameter has unknown or multiple scoped ownership"
            );
            regions.push(Region {
                root,
                inner,
                parameters: scope.parameters.clone(),
            });
            lower_expected = upper;
        } else {
            prefix(stmt, &mut names, true)?;
        }
    }
    ensure!(
        regions.len() == scopes.len() && seen == declared,
        "scope partition does not cover every root parameter"
    );
    Ok(regions)
}
pub(crate) fn validate(ir: &TileIR, scopes: &[Scope]) -> Result<()> {
    prove(ir, scopes).map(|_| ())
}
fn begin_if<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    condition: &ScalarPredicate,
) -> Result<&'a str> {
    let ScalarPredicate::IndexLt { lhs, rhs } = condition else {
        bail!("proved uniform guard changed");
    };
    let lhs = lower_expr(func, ctx, lhs);
    let rhs = lower_expr(func, ctx, rhs);
    let pred = func.add_predicate_register();
    func.add_inst(Inst::setp_lt_u64(pred.clone(), lhs, rhs));
    let yes = bumpalo::format!(in ctx.arena,"scoped_if_body_{}",ctx.label_counter).into_bump_str();
    ctx.label_counter += 1;
    let end = bumpalo::format!(in ctx.arena,"scoped_if_end_{}",ctx.label_counter).into_bump_str();
    ctx.label_counter += 1;
    func.add_inst(Inst::Bra {
        condition: pred,
        target: yes,
    });
    func.add_inst(Inst::BraUni { target: end });
    func.add_inst(Inst::Label(yes));
    Ok(end)
}
fn lower<'a>(
    ir: &TileIR,
    scopes: &[Scope],
    arena: &'a bumpalo::Bump,
    index: usize,
) -> Result<Function<'a>> {
    let regions = prove(ir, scopes)?;
    let kernel_name = bumpalo::format!(in arena,"{}_{}",ir.kernel_name,index).into_bump_str();
    let mut func = Function::new(kernel_name, arena);
    let mut ctx = LoweringContext::new(arena);
    let mut declarations = HashMap::new();
    // ABI declarations stay at root; no pointer value is loaded here.
    for param in &ir.params {
        let name = bumpalo::format!(in arena,"{}",param.name).into_bump_str();
        declarations.insert(param.name.clone(), func.add_param_u64(name));
        ctx.param_dtypes.insert(param.name.clone(), param.dtype);
    }
    if ir.shared_mem_bytes > 0 {
        func.add_shared_memory(arena.alloc_str("shared"), ir.shared_mem_bytes);
    }
    for (root, stmt) in ir.body.stmts.iter().enumerate() {
        let Some(region) = regions.iter().find(|r| r.root == root) else {
            lower_stmt(&mut func, &mut ctx, stmt);
            continue;
        };
        let Stmt::If { condition, body } = stmt else {
            bail!("proved outer If changed");
        };
        let outer_end = begin_if(&mut func, &mut ctx, condition)?;
        let outer_indices = ctx.index_vars.clone();
        for prefix in &body.stmts[..region.inner] {
            lower_stmt(&mut func, &mut ctx, prefix);
        }
        let Stmt::If { condition, body } = &body.stmts[region.inner] else {
            bail!("proved inner If changed");
        };
        let inner_end = begin_if(&mut func, &mut ctx, condition)?;
        let old_pointers = std::mem::take(&mut ctx.param_ptrs);
        for name in &region.parameters {
            let value = func.add_u64_register();
            func.add_inst(Inst::LdParamU64 {
                dst: value.clone(),
                addr: declarations[name].clone(),
            });
            let address = func.add_u64_register();
            func.add_inst(Inst::ConvertToGlobal {
                dst: address.clone(),
                src: value,
            });
            ctx.param_ptrs.insert(name.clone(), address);
        }
        // Every pointer use has been proved to belong to this selected region.
        // Hoisting to its entry dominates even nested per-thread predicates.
        lower_block(&mut func, &mut ctx, body);
        ctx.param_ptrs = old_pointers;
        func.add_inst(Inst::Label(inner_end));
        ctx.index_vars = outer_indices;
        func.add_inst(Inst::Label(outer_end));
    }
    func.add_inst(Inst::Label(ctx.return_label));
    func.add_inst(Inst::Ret);
    Ok(func)
}
/// Explicit CFG(test) entry point. Ordinary entries use the unchanged eager
/// lowerer; only the separately supplied/proved entry gets scoped loads.
pub(crate) fn module(
    ordinary: Vec<TileIR>,
    scoped: TileIR,
    scopes: &[Scope],
    target: PtxTarget,
) -> Result<(String, Vec<String>)> {
    let arena = bumpalo::Bump::new();
    let mut module = Module::new_with_target(target.compute_capability);
    let mut names = Vec::new();
    let index = ordinary.len();
    for (index, ir) in ordinary.into_iter().enumerate() {
        let function = tile_ir_to_function(ir, &arena, index);
        names.push(function.name.to_owned());
        module.add_function(function);
    }
    let function = lower(&scoped, scopes, &arena, index)?;
    names.push(function.name.to_owned());
    module.add_function(function);
    Ok((module.to_string(), names))
}
