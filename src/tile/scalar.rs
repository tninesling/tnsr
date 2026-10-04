//! General scalar expressions and explicit loop-carried state for scheduled kernels.
//! These constructs describe execution, not new tensor operations.
use super::{Block, Expr, TileVar};
use crate::tensor::{BinaryOp, ReduceOp, UnaryOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarBinaryOp {
    Arithmetic(BinaryOp),
    Max,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarExpr {
    Constant(f32),
    Var(TileVar),
    Unary {
        op: UnaryOp,
        value: Box<Self>,
    },
    Binary {
        op: ScalarBinaryOp,
        lhs: Box<Self>,
        rhs: Box<Self>,
    },
    Select {
        condition: Box<ScalarPredicate>,
        then_value: Box<Self>,
        else_value: Box<Self>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarPredicate {
    IndexLt { lhs: Expr, rhs: Expr },
    Equal { lhs: ScalarExpr, rhs: ScalarExpr },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarMemory {
    Global(String),
    /// Physical element offsets in a shared allocation.
    Shared(TileVar),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopCarry {
    pub var: TileVar,
    pub initial: ScalarExpr,
}

impl From<TileVar> for ScalarExpr {
    fn from(var: TileVar) -> Self {
        Self::Var(var)
    }
}
impl ScalarExpr {
    pub fn unary(op: UnaryOp, value: impl Into<Self>) -> Self {
        Self::Unary {
            op,
            value: Box::new(value.into()),
        }
    }
    pub fn binary(op: ScalarBinaryOp, lhs: impl Into<Self>, rhs: impl Into<Self>) -> Self {
        Self::Binary {
            op,
            lhs: Box::new(lhs.into()),
            rhs: Box::new(rhs.into()),
        }
    }
    pub fn select(
        condition: ScalarPredicate,
        then_value: impl Into<Self>,
        else_value: impl Into<Self>,
    ) -> Self {
        Self::Select {
            condition: Box::new(condition),
            then_value: Box::new(then_value.into()),
            else_value: Box::new(else_value.into()),
        }
    }
    pub(crate) fn index_roots(&mut self) -> Vec<&mut Expr> {
        match self {
            Self::Constant(_) | Self::Var(_) => Vec::new(),
            Self::Unary { value, .. } => value.index_roots(),
            Self::Binary { lhs, rhs, .. } => {
                let mut roots = lhs.index_roots();
                roots.extend(rhs.index_roots());
                roots
            }
            Self::Select {
                condition,
                then_value,
                else_value,
            } => {
                let mut roots = condition.index_roots();
                roots.extend(then_value.index_roots());
                roots.extend(else_value.index_roots());
                roots
            }
        }
    }
}
impl ScalarPredicate {
    pub(crate) fn index_roots(&mut self) -> Vec<&mut Expr> {
        match self {
            Self::IndexLt { lhs, rhs } => vec![lhs, rhs],
            Self::Equal { lhs, rhs } => {
                let mut roots = lhs.index_roots();
                roots.extend(rhs.index_roots());
                roots
            }
        }
    }
}

/// A builder for ordinary scalar statements. Variable IDs remain unique across
/// nested blocks; a block only changes insertion scope, never variable identity.
#[derive(Default)]
pub struct ScalarBuilder {
    next_var: usize,
    next_loop: usize,
    pub stmts: Vec<super::Stmt>,
}
impl ScalarBuilder {
    pub fn var(&mut self) -> TileVar {
        let var = TileVar(self.next_var);
        self.next_var += 1;
        var
    }
    pub fn set(&mut self, var: TileVar, value: impl Into<ScalarExpr>) {
        self.stmts.push(super::Stmt::SetScalar {
            dest: var,
            value: value.into(),
        });
    }
    pub fn value(&mut self, value: ScalarExpr) -> TileVar {
        let var = self.var();
        self.set(var, value);
        var
    }
    pub fn load(&mut self, source: ScalarMemory, index: Expr) -> TileVar {
        let dest = self.var();
        self.stmts.push(super::Stmt::LoadScalar {
            dest,
            source,
            index,
        });
        dest
    }
    pub fn store(&mut self, target: ScalarMemory, index: Expr, value: TileVar) {
        self.stmts.push(super::Stmt::StoreScalar {
            target,
            index,
            value,
        });
    }
    pub fn block(
        &mut self,
        build: impl FnOnce(&mut Self) -> anyhow::Result<()>,
    ) -> anyhow::Result<Block> {
        let outer = std::mem::take(&mut self.stmts);
        let result = build(self);
        let body = Block {
            stmts: std::mem::replace(&mut self.stmts, outer),
        };
        result?;
        Ok(body)
    }
    pub fn when(
        &mut self,
        condition: ScalarPredicate,
        build: impl FnOnce(&mut Self) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let body = self.block(build)?;
        self.stmts.push(super::Stmt::If { condition, body });
        Ok(())
    }
    pub fn loop_name(&mut self, prefix: &str) -> String {
        let id = self.next_loop;
        self.next_loop += 1;
        format!("{prefix}_{id}")
    }
    pub fn fold(
        &mut self,
        loop_var: String,
        extent: Expr,
        carries: Vec<LoopCarry>,
        build: impl FnOnce(&mut Self, Expr) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let body = self.block(|builder| build(builder, Expr::Var(loop_var.clone())))?;
        self.stmts.push(super::Stmt::ForLoop {
            loop_var,
            start: Expr::Const(0),
            end: extent,
            carries,
            body,
        });
        Ok(())
    }
    pub fn warp_reduce(&mut self, src: TileVar, op: ReduceOp) -> TileVar {
        let dest = self.var();
        self.stmts.push(super::Stmt::WarpReduce { dest, src, op });
        dest
    }
    pub fn barrier(&mut self) {
        self.stmts.push(super::Stmt::Barrier);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{DType, KernelParam, TileIR};

    fn ir(builder: ScalarBuilder) -> TileIR {
        TileIR {
            kernel_name: "scalar_scope".into(),
            params: vec![KernelParam {
                name: "output".into(),
                dtype: DType::F32,
                is_input: false,
            }],
            body: Block {
                stmts: builder.stmts,
            },
            shared_mem_bytes: 0,
        }
    }
    #[test]
    fn carries_escape_empty_loops_but_body_definitions_do_not() {
        let mut builder = ScalarBuilder::default();
        let carry = builder.var();
        let local = builder.var();
        builder
            .fold(
                "empty".into(),
                Expr::Const(0),
                vec![LoopCarry {
                    var: carry,
                    initial: ScalarExpr::Constant(7.0),
                }],
                |builder, _| {
                    builder.set(local, ScalarExpr::Constant(9.0));
                    Ok(())
                },
            )
            .unwrap();
        builder.store(ScalarMemory::Global("output".into()), Expr::Const(0), carry);
        let mut ir = ir(builder);
        ir.validate_layouts().unwrap();
        ir.body.stmts.push(crate::tile::Stmt::StoreScalar {
            target: ScalarMemory::Global("output".into()),
            index: Expr::Const(0),
            value: local,
        });
        assert!(
            ir.validate_layouts()
                .unwrap_err()
                .to_string()
                .contains("invalid layouts")
        );
    }
    #[test]
    fn rejects_use_before_definition_and_duplicate_carries() {
        let mut builder = ScalarBuilder::default();
        let missing = builder.var();
        let dest = builder.var();
        builder.set(dest, missing);
        assert!(ir(builder).validate_layouts().is_err());
        let mut builder = ScalarBuilder::default();
        let carry = builder.var();
        builder
            .fold(
                "loop".into(),
                Expr::Const(1),
                vec![
                    LoopCarry {
                        var: carry,
                        initial: ScalarExpr::Constant(0.0),
                    },
                    LoopCarry {
                        var: carry,
                        initial: ScalarExpr::Constant(1.0),
                    },
                ],
                |_, _| Ok(()),
            )
            .unwrap();
        assert!(ir(builder).validate_layouts().is_err());
    }
}
