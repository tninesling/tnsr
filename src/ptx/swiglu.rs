//! Standalone PTX packaging for the explicit SwiGLU training experiment.
use anyhow::Result;

use crate::tile::{TileGraph, swiglu::SwiGluTraining};

pub const FORWARD_KERNEL: &str = "swiglu_forward_0";
pub const BACKWARD_KERNEL: &str = "swiglu_backward_1";

/// Generate two kernels without changing graph selection or precision policy.
pub fn module_source(training: &SwiGluTraining) -> Result<String> {
    let mut graph = TileGraph {
        graph: petgraph::Graph::new(),
        region_kernels: Vec::new(),
        reduction_region_kernels: Vec::new(),
        matmul_region_kernels: Vec::new(),
        physical_nodes: None,
    };
    graph.graph.add_node(training.lower_forward()?);
    graph.graph.add_node(training.lower_backward()?);
    Ok(super::PtxGraph::from(graph).to_ptx())
}
