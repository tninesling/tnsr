pub mod access;
pub mod builder;
pub mod graph;
mod index_egraph;
mod index_optimization;
pub mod ir;
mod layout;
pub mod shared_layout;
pub use shared_layout::SharedLayout;
pub mod matmul_region;
pub mod matmul_schedule;
pub mod reduction_region;
pub mod region;

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
