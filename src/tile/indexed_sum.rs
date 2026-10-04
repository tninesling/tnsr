//! Indexed multiply/sum analysis shared by producers and fused consumers.
//! This description is eliminated into ordinary scalar loads, arithmetic, and loops.
use super::region::{input_coordinate_index, lower_index_expr};
use super::*;
use crate::tensor::BinaryOp;
use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexedSource {
    Input(usize),
    /// A scalar supplied by an inlined producer at the mapped iteration point.
    Argument,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedOperand {
    pub source: IndexedSource,
    pub access: IndexMap,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedSum {
    pub extent: usize,
    /// Maps consume output coordinates followed by the reduction coordinate.
    pub lhs: IndexedOperand,
    pub rhs: IndexedOperand,
}

/// An ordinary sum accumulator. Any rule changing a summand's scale can apply
/// the same scale to accumulated contributions, independently of normalization.
pub(crate) struct SumAccumulator(pub TileVar);
impl SumAccumulator {
    pub fn new(builder: &mut ScalarBuilder) -> Self {
        Self(builder.var())
    }
    pub fn carry(&self) -> LoopCarry {
        LoopCarry {
            var: self.0,
            initial: ScalarExpr::Constant(0.0),
        }
    }
    pub fn rescale(&self, builder: &mut ScalarBuilder, scale: TileVar) {
        builder.set(
            self.0,
            ScalarExpr::binary(ScalarBinaryOp::Arithmetic(BinaryOp::Mul), self.0, scale),
        );
    }
    fn add_product(&self, builder: &mut ScalarBuilder, lhs: TileVar, rhs: TileVar) {
        let product = ScalarExpr::binary(ScalarBinaryOp::Arithmetic(BinaryOp::Mul), lhs, rhs);
        builder.set(
            self.0,
            ScalarExpr::binary(ScalarBinaryOp::Arithmetic(BinaryOp::Add), self.0, product),
        );
    }
}
impl IndexedSum {
    pub(crate) fn lower(
        &self,
        builder: &mut ScalarBuilder,
        inputs: &[RegionInput],
        coordinates: &[Expr],
    ) -> Result<TileVar> {
        let accumulator = SumAccumulator::new(builder);
        let name = builder.loop_name("sum");
        let extent =
            Expr::Const(i64::try_from(self.extent).context("indexed sum extent exceeds i64")?);
        builder.fold(
            name,
            extent,
            vec![accumulator.carry()],
            |builder, reduction| {
                let mut point = coordinates.to_vec();
                point.push(reduction);
                self.accumulate(builder, inputs, &accumulator, &point, None)
            },
        )?;
        Ok(accumulator.0)
    }
    pub(crate) fn accumulate(
        &self,
        builder: &mut ScalarBuilder,
        inputs: &[RegionInput],
        accumulator: &SumAccumulator,
        point: &[Expr],
        argument: Option<TileVar>,
    ) -> Result<()> {
        let lhs = self.operand(builder, inputs, &self.lhs, point, argument)?;
        let rhs = self.operand(builder, inputs, &self.rhs, point, argument)?;
        accumulator.add_product(builder, lhs, rhs);
        Ok(())
    }
    fn operand(
        &self,
        builder: &mut ScalarBuilder,
        inputs: &[RegionInput],
        operand: &IndexedOperand,
        point: &[Expr],
        argument: Option<TileVar>,
    ) -> Result<TileVar> {
        match operand.source {
            IndexedSource::Argument => argument.context("indexed sum argument is not bound"),
            IndexedSource::Input(input) => {
                let descriptor = inputs.get(input).context("indexed sum input is missing")?;
                let coordinates = operand
                    .access
                    .results
                    .iter()
                    .map(|e| lower_index_expr(e, point))
                    .collect::<Result<Vec<_>>>()?;
                let index = input_coordinate_index(descriptor, &coordinates)?;
                Ok(builder.load(ScalarMemory::Global(format!("input_{input}")), index))
            }
        }
    }
}
