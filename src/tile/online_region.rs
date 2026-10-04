//! Coupled online normalization and weighted contraction, represented only in the
//! scheduling IR. The tensor graph continues to use ordinary arithmetic/reductions.
use super::{Block, DType, KernelParam, RegionInput, Stmt, TileIR};
use crate::graph::NodeIndex;
use crate::tensor::{BinaryOp, Shape, UnaryOp};
use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnlineExpr {
    Input(usize),
    Unary(UnaryOp, Box<Self>),
    Binary(BinaryOp, Box<Self>, Box<Self>),
    /// Scalar dot-product producer. Operands retain their virtual access maps.
    Dot {
        lhs: usize,
        rhs: usize,
        width: usize,
    },
}

/// A normalized weighted sum over the last score axis. The running maximum,
/// normalizer, and weighted accumulator share one increasing-index traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnlineRegion {
    pub members: Vec<NodeIndex>,
    pub inputs: Vec<RegionInput>,
    pub score: OnlineExpr,
    pub weights: usize,
    pub output: NodeIndex,
    pub score_shape: Shape,
    pub output_shape: Shape,
}

impl OnlineRegion {
    /// One block per score row, with a warp producing a tile of scores and
    /// one lane per weighted output. Wider outputs retain the scalar schedule.
    pub fn block_threads(&self) -> Option<u32> {
        let columns = *self.output_shape.last()?;
        (columns <= 128)
            .then(|| u32::try_from(columns.max(32).next_power_of_two()).ok())
            .flatten()
    }

    pub fn lower_to_tile_ir(&self, region_id: usize) -> Result<TileIR> {
        anyhow::ensure!(
            self.score_shape.len() >= 2,
            "online scores require rank >= 2"
        );
        anyhow::ensure!(
            self.inputs.iter().all(|i| i.tensor.dtype == DType::F32),
            "online reduction requires f32 storage"
        );
        let extent = self.output_shape.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d).context("online output size overflow")
        })?;
        let mut params: Vec<_> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(i, _)| KernelParam {
                name: format!("input_{i}"),
                dtype: DType::F32,
                is_input: true,
            })
            .collect();
        params.push(KernelParam {
            name: "output_0".into(),
            dtype: DType::F32,
            is_input: false,
        });
        Ok(TileIR {
            kernel_name: format!("online_region_{region_id}"),
            params,
            body: Block {
                stmts: {
                    let mut stmts = Vec::new();
                    if self.block_threads().is_none() {
                        stmts.push(Stmt::BoundsCheck { extent });
                    }
                    stmts.push(Stmt::OnlineRegion {
                        region: Box::new(self.clone()),
                    });
                    stmts
                },
            },
            shared_mem_bytes: 0,
        })
    }
}
