pub mod access;
pub mod builder;
pub mod fusion_cost;
pub mod graph;
pub use fusion_cost::{AnalyticalFusionScorer, FusionCost, FusionFeatures, FusionScorer};
mod index_egraph;
mod index_optimization;
pub mod ir;
mod layout;
pub mod shared_layout;
pub use shared_layout::SharedLayout;
pub mod matmul_region;
pub mod matmul_schedule;
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

mod normalized_schedule;
