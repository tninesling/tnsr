//! Coupled online normalization and weighted contraction, represented only in the
//! scheduling IR. The tensor graph continues to use ordinary arithmetic/reductions.
use super::{DType, KernelParam, RegionInput, TileIR};
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

/// Supported consumers of the running normalizer state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnlineConsumer {
    /// Sum of stable exponentials, without any downstream normalization.
    Normalizer,
    /// A homogeneous sum can carry a numerator under the normalizer's rescaling.
    WeightedSum { weights: usize, elementwise: bool },
}

/// Compiler analysis metadata, eliminated into ordinary scalar scheduling IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnlineRegion {
    pub members: Vec<NodeIndex>,
    pub inputs: Vec<RegionInput>,
    pub score: OnlineExpr,
    pub consumer: OnlineConsumer,
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
            !self.score_shape.is_empty(),
            "online scores require a reduction axis"
        );
        anyhow::ensure!(
            self.inputs.iter().all(|i| i.tensor.dtype == DType::F32),
            "online reduction requires f32 storage"
        );
        anyhow::ensure!(
            self.output_shape.len() == self.score_shape.len()
                && !self.score_shape.contains(&0)
                && !self.output_shape.contains(&0),
            "online reduction requires nonempty, equal-rank shapes"
        );
        self.score_shape.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d).context("online score size overflow")
        })?;
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
        let (body, shared_mem_bytes) = super::normalized_schedule::lower(self, extent)?;
        Ok(TileIR {
            kernel_name: format!("online_region_{region_id}"),
            params,
            body,
            shared_mem_bytes,
        })
    }
}
