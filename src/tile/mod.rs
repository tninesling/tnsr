pub mod access;
pub mod builder;
pub mod fusion_cost;
pub mod graph;
#[cfg(any(feature = "cuda", test))]
mod matmul_copy;
pub use fusion_cost::{AnalyticalFusionScorer, FusionCost, FusionFeatures, FusionScorer};
mod index_egraph;
mod index_optimization;
pub mod ir;
mod layout;
pub mod shared_layout;
pub use shared_layout::SharedLayout;
mod column_reduction_policy;
pub mod matmul_region;
pub mod matmul_schedule;
pub use column_reduction_policy::ColumnReductionPolicy;
#[cfg(any(feature = "cuda", test))]
pub(crate) use column_reduction_policy::{ColumnReductionLimits, ColumnReductionTile};
mod batched_attention_tile_policy;
#[cfg(feature = "cuda")]
pub(crate) use batched_attention_tile_policy::AttentionTileRole;
pub use batched_attention_tile_policy::{
    BatchedAttentionBf16TilePolicy, BatchedAttentionTileCounts,
};
pub(crate) mod large_tile_policy;
#[cfg(any(feature = "cuda", test))]
mod small_grid_split_k;
#[cfg(feature = "cuda")]
pub(crate) use small_grid_split_k::SmallGridCandidate;
#[cfg(any(feature = "cuda", test))]
pub(crate) mod ordinary_bf16_block;
pub use large_tile_policy::LargeBf16TilePolicy;
mod split_k_tile_policy;
pub use split_k_tile_policy::SplitKBf16TilePolicy;
mod medium_tile_policy;
pub use medium_tile_policy::MediumBf16TilePolicy;
pub mod online_region;
pub mod reduction_region;
pub use online_region::{OnlineConsumer, OnlineExpr, OnlineRegion};
pub mod region;
pub mod scalar;
pub use scalar::{
    LoopCarry, ScalarBinaryOp, ScalarBuilder, ScalarExpr, ScalarMemory, ScalarPredicate,
};

pub use access::{
    CompareOp, Extent, IndexExpr, IndexMap, IndexPredicate, IterDim, IterDomain, IterKind,
    ScalarValue, VirtualTensor,
};
#[allow(unused_imports)]
pub use builder::TileIRBuilder;
#[allow(unused_imports)]
pub use graph::TileGraph;
#[allow(unused_imports)]
pub use ir::{
    Block, Conv2dGeometry, DType, Dim, Expr, KernelParam, MatMulLayout, MatMulPlan,
    MaxPool2dGeometry, MemorySpace, ReduceOp, Stmt, TileDType, TileIR, TileLayout, TileVar,
};
pub use matmul_region::MatMulRegion;
pub use matmul_schedule::*;
pub use reduction_region::{
    ReductionInput, ReductionInputDomain, ReductionRegion, ReductionSchedule,
};
pub use region::{FusionRegion, RegionInput, RegionOp, RegionOpKind, RegionOutput, RegionValue};

pub(crate) mod normalized_schedule;

pub mod indexed_sum;
pub use indexed_sum::{IndexedOperand, IndexedSource, IndexedSum};

mod segments;

pub mod swiglu;
