//! Restricted composition of the unchanged per-member scalar norm-row IR.
use super::*;
use crate::tile::{
    Block, Dim, Expr, ScalarBuilder, ScalarExpr, ScalarMemory, ScalarPredicate, Stmt, TileIR,
    TileVar,
};
use std::collections::HashMap;

fn value(v: &ScalarExpr, vars: &HashMap<TileVar, TileVar>) -> Result<ScalarExpr> {
    Ok(match v {
        ScalarExpr::Constant(x) => ScalarExpr::Constant(*x),
        ScalarExpr::Var(v) => {
            ScalarExpr::Var(*vars.get(v).context("batch scalar use before definition")?)
        }
        ScalarExpr::Unary { op, value: a } => ScalarExpr::Unary {
            op: *op,
            value: Box::new(value(a, vars)?),
        },
        ScalarExpr::Binary { op, lhs, rhs } => ScalarExpr::Binary {
            op: *op,
            lhs: Box::new(value(lhs, vars)?),
            rhs: Box::new(value(rhs, vars)?),
        },
        ScalarExpr::Select {
            condition,
            then_value,
            else_value,
        } => ScalarExpr::Select {
            condition: Box::new(predicate(condition, vars)?),
            then_value: Box::new(value(then_value, vars)?),
            else_value: Box::new(value(else_value, vars)?),
        },
    })
}
fn predicate(p: &ScalarPredicate, vars: &HashMap<TileVar, TileVar>) -> Result<ScalarPredicate> {
    Ok(match p {
        ScalarPredicate::IndexLt { lhs, rhs } => ScalarPredicate::IndexLt {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        },
        ScalarPredicate::Equal { lhs, rhs } => ScalarPredicate::Equal {
            lhs: value(lhs, vars)?,
            rhs: value(rhs, vars)?,
        },
    })
}
fn memory(m: &ScalarMemory, vars: &HashMap<TileVar, TileVar>, name: &str) -> Result<ScalarMemory> {
    Ok(match m {
        ScalarMemory::Shared(v) => {
            ScalarMemory::Shared(*vars.get(v).context("batch shared allocation absent")?)
        }
        ScalarMemory::Global(n) if n == "gradient" => ScalarMemory::Global(name.into()),
        ScalarMemory::Global(n) if n == "accumulated" || n == "partials" => m.clone(),
        _ => anyhow::bail!("unrecognized norm-row global argument"),
    })
}
fn destination(
    old: TileVar,
    b: &mut ScalarBuilder,
    vars: &mut HashMap<TileVar, TileVar>,
) -> TileVar {
    *vars.entry(old).or_insert_with(|| b.var())
}
pub(super) fn body(
    block: &Block,
    b: &mut ScalarBuilder,
    vars: &mut HashMap<TileVar, TileVar>,
    name: &str,
    base: usize,
) -> Result<Block> {
    let mut out = Vec::new();
    for s in &block.stmts {
        out.push(match s {
            Stmt::SetScalar{dest,value:v}=>{
                let next=destination(*dest,b,vars);
                Stmt::SetScalar{dest:next,value:value(v,vars)?}
            },
            Stmt::LoadScalar{dest,source,index}=>{
                ensure!(matches!(source,ScalarMemory::Shared(_))||matches!(source,ScalarMemory::Global(n)if n=="gradient"),"norm-row read target contract changed");
                let next=destination(*dest,b,vars);
                Stmt::LoadScalar{dest:next,source:memory(source,vars,name)?,index:index.clone()}
            },
            Stmt::StoreScalar{target,index,value:v}=>{
                ensure!(matches!(target,ScalarMemory::Shared(_))||matches!(target,ScalarMemory::Global(n)if n=="accumulated"||n=="partials"),"norm-row write target contract changed");
                Stmt::StoreScalar{
                target:memory(target,vars,name)?,
                index:if matches!(target,ScalarMemory::Global(n)if n=="accumulated") {Expr::Const(i64::try_from(base)?)+index.clone()}else{index.clone()},
                value:*vars.get(v).context("batch store use before definition")?,
            }},
            Stmt::If{condition,body:child}=>Stmt::If{condition:predicate(condition,vars)?,body:body(child,b,vars,name,base)?},
            Stmt::Barrier=>Stmt::Barrier,
            _=>anyhow::bail!("unrecognized norm-row statement; preserve standalone fallback"),
        });
    }
    Ok(Block { stmts: out })
}
/// Grid Y selects one member; X covers the maximum EXACT global-row span.
/// Every branch containing barriers depends only on block indices/constants.
pub(super) fn batch_ir(
    members: &[(VirtualTensor, Shape, Range<usize>)],
    padded_len: usize,
) -> Result<TileIR> {
    ensure!(
        (2..=BATCH_MEMBERS).contains(&members.len()),
        "small norm batch cardinality"
    );
    let mut b = ScalarBuilder::default();
    let scratch = b.var();
    let mut params = Vec::new();
    let mut branches = Vec::new();
    let mut hoisted = None;
    for (i, (view, shape, range)) in members.iter().enumerate() {
        ensure!(
            range.end <= padded_len,
            "batch member exceeds global gradient argument"
        );
        let q = span(range)?;
        let mut old = norm_partials::accumulate_ir(view, shape, range.clone())?;
        ensure!(
            old.shared_mem_bytes == norm_partials::ROW_THREADS * 4,
            "norm-row scratch changed"
        );
        ensure!(!old.body.stmts.is_empty(), "norm-row body empty");
        let allocation = old.body.stmts.remove(0);
        let Stmt::AllocTile {
            var,
            space,
            layout,
            dtype,
            rows,
            cols,
        } = allocation
        else {
            anyhow::bail!("norm-row scratch absent")
        };
        ensure!(
            space == crate::tile::MemorySpace::Shared
                && dtype == crate::tile::DType::F32
                && rows == 1
                && cols == norm_partials::ROW_THREADS,
            "norm-row scratch contract changed"
        );
        if hoisted.is_none() {
            hoisted = Some(Stmt::AllocTile {
                var: scratch,
                space,
                layout,
                dtype,
                rows,
                cols,
            });
        }
        let name = format!("gradient_{i}");
        params.push(crate::tile::KernelParam {
            name: name.clone(),
            dtype: view.dtype,
            is_input: true,
        });
        let mut vars = HashMap::from([(var, scratch)]);
        let member = body(&old.body, &mut b, &mut vars, &name, range.start)?;
        // Wrapping-u64 (Y-i)<1 is true exactly for Y==i in bounded grid Y.
        branches.push(Stmt::If {
            condition: ScalarPredicate::IndexLt {
                lhs: Expr::Sub(
                    Box::new(Expr::BlockIdx(Dim::Y)),
                    Box::new(Expr::Const(i64::try_from(i)?)),
                ),
                rhs: Expr::Const(1),
            },
            body: Block {
                stmts: vec![Stmt::If {
                    condition: ScalarPredicate::IndexLt {
                        lhs: Expr::BlockIdx(Dim::X),
                        rhs: Expr::Const(i64::try_from(q)?),
                    },
                    body: member,
                }],
            },
        });
    }
    params.extend(
        ["accumulated", "partials"].map(|name| crate::tile::KernelParam {
            name: name.into(),
            dtype: crate::tile::DType::F32,
            is_input: false,
        }),
    );
    let mut stmts = vec![hoisted.context("batch scratch missing")?];
    stmts.extend(branches);
    let ir = TileIR {
        kernel_name: "resident_small_accumulate_norm_rows".into(),
        params,
        body: Block { stmts },
        shared_mem_bytes: norm_partials::ROW_THREADS * 4,
    };
    ir.validate_layouts()?;
    Ok(ir)
}
