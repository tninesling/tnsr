//! Share pure index expressions and bind them in the earliest lexical scope that
//! contains all dependencies. No loads, floating-point operations, or barriers
//! move. Wrapping arithmetic is total after divisor/shift validation, so empty
//! loops and predicated edge loads are safe even when bindings move outward.
use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;

use super::{Block, Expr, Stmt, TileIR};

type Scope = Vec<usize>;

fn roots(stmt: &mut Stmt) -> Vec<&mut Expr> {
    match stmt {
        Stmt::LetIndex { value, .. } => vec![value],
        Stmt::Load {
            row_offset,
            col_offset,
            ..
        }
        | Stmt::Store {
            row_offset,
            col_offset,
            ..
        } => vec![row_offset, col_offset],
        Stmt::LoadGlobalToSharedPredicated {
            element_index,
            row,
            col,
            tile_row,
            tile_col,
            ..
        } => vec![element_index, row, col, tile_row, tile_col],
        Stmt::LoadGlobalPredicated {
            element_index,
            row,
            col,
            ..
        }
        | Stmt::StoreGlobalPredicated {
            element_index,
            row,
            col,
            ..
        } => vec![element_index, row, col],
        Stmt::ConvertLayout {
            coordinates: Some((row, col)),
            ..
        } => vec![row, col],
        Stmt::ForLoop { start, end, .. } => vec![start, end],
        _ => Vec::new(),
    }
}

fn children(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => vec![a, b],
        Expr::FloorDiv(a, _) | Expr::Mod(a, _) | Expr::ShiftRight(a, _) | Expr::BitAnd(a, _) => {
            vec![a]
        }
        _ => Vec::new(),
    }
}

fn normalize(block: &mut Block, cache: &mut HashMap<Expr, Expr>) -> Result<()> {
    for stmt in &mut block.stmts {
        for root in roots(stmt) {
            let value = if let Some(value) = cache.get(root) {
                value.clone()
            } else {
                let value = super::index_egraph::normalize_address(root)?;
                cache.insert(root.clone(), value.clone());
                value
            };
            *root = value;
        }
        if let Stmt::ForLoop { body, .. } = stmt {
            normalize(body, cache)?;
        }
    }
    Ok(())
}

#[derive(Default)]
struct Usage {
    scopes: Vec<Scope>,
}

fn census(
    expr: &Expr,
    scope: &Scope,
    usage: &mut BTreeMap<Expr, Usage>,
    names: &mut BTreeSet<String>,
) {
    if let Expr::Var(name) = expr {
        names.insert(name.clone());
    }
    usage
        .entry(expr.clone())
        .or_default()
        .scopes
        .push(scope.clone());
    for child in children(expr) {
        census(child, scope, usage, names);
    }
}

fn collect(
    block: &mut Block,
    scope: &Scope,
    usage: &mut BTreeMap<Expr, Usage>,
    variables: &mut HashMap<String, Option<Scope>>,
    names: &mut BTreeSet<String>,
) {
    for (index, stmt) in block.stmts.iter_mut().enumerate() {
        for root in roots(stmt) {
            census(root, scope, usage, names);
        }
        if let Stmt::LetIndex { name, .. } = stmt {
            names.insert(name.clone());
        }
        if let Stmt::ForLoop { loop_var, body, .. } = stmt {
            names.insert(loop_var.clone());
            let mut inner = scope.clone();
            inner.push(index);
            // Repeated/shadowed names are conservatively excluded from hoisting.
            variables
                .entry(loop_var.clone())
                .and_modify(|value| *value = None)
                .or_insert(Some(inner.clone()));
            collect(body, &inner, usage, variables, names);
        }
    }
}

fn dependency_scope(expr: &Expr, variables: &HashMap<String, Option<Scope>>) -> Option<Scope> {
    let mut scope = match expr {
        Expr::Var(name) => variables.get(name)?.clone()?,
        _ => Vec::new(),
    };
    for child in children(expr) {
        let child_scope = dependency_scope(child, variables)?;
        if scope.starts_with(&child_scope) {
        } else if child_scope.starts_with(&scope) {
            scope = child_scope;
        } else {
            return None;
        }
    }
    Some(scope)
}

fn height(expr: &Expr) -> usize {
    1 + children(expr).into_iter().map(height).max().unwrap_or(0)
}

struct Binding {
    name: String,
    scope: Scope,
}

fn substitute(expr: &mut Expr, bindings: &BTreeMap<Expr, Binding>) {
    if let Some(binding) = bindings.get(expr) {
        *expr = Expr::Var(binding.name.clone());
        return;
    }
    substitute_children(expr, bindings);
}
fn substitute_children(expr: &mut Expr, bindings: &BTreeMap<Expr, Binding>) {
    match expr {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            substitute(a, bindings);
            substitute(b, bindings);
        }
        Expr::FloorDiv(a, _) | Expr::Mod(a, _) | Expr::ShiftRight(a, _) | Expr::BitAnd(a, _) => {
            substitute(a, bindings)
        }
        _ => {}
    }
}

fn place(block: &mut Block, scope: &Scope, bindings: &BTreeMap<Expr, Binding>) {
    let mut definitions: Vec<_> = bindings
        .iter()
        .filter(|(_, binding)| binding.scope == *scope)
        .collect();
    // Same-scope children are defined first; parent-scope bindings already dominate.
    definitions.sort_by_key(|(expr, _)| (height(expr), *expr));
    let mut result = Vec::new();
    for (expr, binding) in definitions {
        let mut value = expr.clone();
        substitute_children(&mut value, bindings);
        result.push(Stmt::LetIndex {
            name: binding.name.clone(),
            value,
        });
    }
    for (index, mut stmt) in std::mem::take(&mut block.stmts).into_iter().enumerate() {
        for root in roots(&mut stmt) {
            substitute(root, bindings);
        }
        if let Stmt::ForLoop { body, .. } = &mut stmt {
            let mut inner = scope.clone();
            inner.push(index);
            place(body, &inner, bindings);
        }
        result.push(stmt);
    }
    block.stmts = result;
}

impl TileIR {
    /// Normalize and share integer index expressions before backend lowering.
    pub fn optimize_indices(&mut self) -> Result<()> {
        let mut cache = HashMap::new();
        normalize(&mut self.body, &mut cache)?;
        let (mut usage, mut variables, mut names) =
            (BTreeMap::new(), HashMap::new(), BTreeSet::new());
        collect(
            &mut self.body,
            &Vec::new(),
            &mut usage,
            &mut variables,
            &mut names,
        );
        let mut bindings = BTreeMap::new();
        let mut serial = 0;
        for (expr, usage) in usage {
            // Primitive index reads are cheap and may be rematerialized by the
            // backend. Binding them can extend register lifetimes without sharing
            // any arithmetic; share composite expressions instead.
            if matches!(
                expr,
                Expr::Const(_)
                    | Expr::Var(_)
                    | Expr::BlockIdx(_)
                    | Expr::ThreadIdx(_)
                    | Expr::BlockDim(_)
            ) {
                continue;
            }
            let Some(scope) = dependency_scope(&expr, &variables) else {
                continue;
            };
            if !usage
                .scopes
                .iter()
                .all(|use_scope| use_scope.starts_with(&scope))
            {
                continue;
            }
            if usage.scopes.len() < 2
                && !usage
                    .scopes
                    .iter()
                    .any(|use_scope| use_scope.len() > scope.len())
            {
                continue;
            }
            let name = loop {
                let name = format!("__tnsr_index_{serial}");
                serial += 1;
                if names.insert(name.clone()) {
                    break name;
                }
            };
            bindings.insert(expr, Binding { name, scope });
        }
        place(&mut self.body, &Vec::new(), &bindings);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::Dim;
    fn load(expr: Expr) -> Stmt {
        Stmt::Load {
            dest: crate::tile::TileVar(0),
            src_param: "x".into(),
            row_offset: expr,
            col_offset: Expr::Const(0),
        }
    }
    fn ir(stmts: Vec<Stmt>) -> TileIR {
        TileIR {
            kernel_name: "indices".into(),
            params: Vec::new(),
            shared_mem_bytes: 0,
            body: Block { stmts },
        }
    }
    fn evaluate(expr: &Expr, env: &HashMap<String, u64>) -> u64 {
        match expr {
            Expr::Const(v) => *v as u64,
            Expr::Var(name) => env[name],
            Expr::BlockIdx(_) => 7,
            Expr::ThreadIdx(_) => 1,
            Expr::BlockDim(_) => 16,
            Expr::Add(a, b) => evaluate(a, env).wrapping_add(evaluate(b, env)),
            Expr::Sub(a, b) => evaluate(a, env).wrapping_sub(evaluate(b, env)),
            Expr::Mul(a, b) => evaluate(a, env).wrapping_mul(evaluate(b, env)),
            Expr::FloorDiv(a, d) => evaluate(a, env) / *d as u64,
            Expr::Mod(a, d) => evaluate(a, env) % *d as u64,
            Expr::ShiftRight(a, d) => evaluate(a, env) >> d,
            Expr::BitAnd(a, d) => evaluate(a, env) & d,
        }
    }
    fn execute(block: &Block, env: &mut HashMap<String, u64>, offsets: &mut Vec<u64>) {
        let outer = env.clone();
        for stmt in &block.stmts {
            match stmt {
                Stmt::LetIndex { name, value } => {
                    env.insert(name.clone(), evaluate(value, env));
                }
                Stmt::Load { row_offset, .. } => offsets.push(evaluate(row_offset, env)),
                Stmt::ForLoop {
                    loop_var,
                    start,
                    end,
                    body,
                } => {
                    for i in evaluate(start, env)..evaluate(end, env) {
                        env.insert(loop_var.clone(), i);
                        execute(body, env, offsets);
                    }
                }
                _ => {}
            }
        }
        *env = outer;
    }
    #[test]
    fn places_non_power_of_two_division_outside_nested_loops() {
        let batch = Expr::FloorDiv(Box::new(Expr::BlockIdx(Dim::Z)), 3);
        let outer = Expr::Var("outer".into()) * 4_usize;
        let address = batch.clone() + outer.clone() + Expr::Var("inner".into());
        let mut ir = ir(vec![Stmt::ForLoop {
            loop_var: "outer".into(),
            start: Expr::Const(0),
            end: Expr::Const(7),
            body: Block {
                stmts: vec![Stmt::ForLoop {
                    loop_var: "inner".into(),
                    start: Expr::Const(0),
                    end: Expr::Const(3),
                    body: Block {
                        stmts: vec![load(address)],
                    },
                }],
            },
        }]);
        ir.optimize_indices().unwrap();
        assert!(ir.body.stmts.iter().any(|s| matches!(
            s,
            Stmt::LetIndex {
                value: Expr::FloorDiv(_, 3),
                ..
            }
        )));
        let outer_body = ir
            .body
            .stmts
            .iter()
            .find_map(|s| {
                if let Stmt::ForLoop { body, .. } = s {
                    Some(body)
                } else {
                    None
                }
            })
            .unwrap();
        assert!(outer_body.stmts.iter().any(|s|matches!(s,Stmt::LetIndex{value: Expr::Mul(a,b),..} if **a==Expr::Const(4) && **b==Expr::Var("outer".into()))));
        let mut offsets = Vec::new();
        execute(&ir.body, &mut HashMap::new(), &mut offsets);
        let expected: Vec<_> = (0..7)
            .flat_map(|outer| (0..3).map(move |inner| 7 / 3 + outer * 4 + inner))
            .collect();
        assert_eq!(offsets, expected);
    }
    #[test]
    fn shares_commuted_addresses_without_capturing_unknown_variables() {
        let a = Expr::BlockIdx(Dim::X);
        let b = Expr::ThreadIdx(Dim::X);
        let mut ir = ir(vec![
            load(a.clone() + b.clone()),
            load(b + a),
            load(Expr::Var("external".into()) + Expr::Const(1)),
        ]);
        ir.optimize_indices().unwrap();
        let loads: Vec<_> = ir
            .body
            .stmts
            .iter()
            .filter_map(|s| {
                if let Stmt::Load { row_offset, .. } = s {
                    Some(row_offset)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(loads[0], loads[1]);
        assert!(matches!(loads[0], Expr::Var(_)));
        assert!(matches!(loads[2], Expr::Add(..)));
    }
}
