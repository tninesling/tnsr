//! Computation graph representation and optimization.
//!
//! This module provides:
//! - Graph representation with gradient support
//! - E-graph based rewriting for optimization

mod graph_main;
pub mod rewrite;

// Re-export main graph types
pub use graph_main::{NoGrad, NodeIndex, TensorGraph, TensorGraphNode, WithGrad};
