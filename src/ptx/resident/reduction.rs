use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::ptx::{PtxExecutor, PtxReductionMode};
use crate::tensor::{Input, TensorExpr};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(super) struct Reduction {
    pub(super) executor: PtxExecutor,
    graph: TensorGraph<f32>,
    input: NodeIndex,
    output: NodeIndex,
}
impl Reduction {
    pub(super) fn new(
        stream: Arc<CudaStream>,
        gate: Arc<Mutex<()>>,
        len: usize,
        square: bool,
    ) -> Result<Self> {
        let input: TensorExpr<f32> = Input::new("resident_buffer", vec![len / 1024, 1024]).into();
        let values = if square { input.clone() * input } else { input };
        let graph: TensorGraph<f32> = values.reduce_sum(1).reduce_sum(0).into();
        let source = graph
            .graph
            .node_indices()
            .find(|&node| matches!(graph[node], TensorGraphNode::Input { .. }))
            .context("resident reduction input absent")?;
        let output = *graph
            .toposort()
            .last()
            .context("resident reduction is empty")?;
        let executor = PtxExecutor::with_stream(stream, PtxReductionMode::DeterministicTree, gate)?;
        Ok(Self {
            executor,
            graph,
            input: source,
            output,
        })
    }
    pub(super) fn run(&mut self, values: Arc<CudaSlice<f32>>) -> Result<f32> {
        self.executor.run_device(
            &self.graph,
            HashMap::new(),
            HashMap::from([(self.input, values)]),
        )?;
        let result = self
            .executor
            .get_value(self.output)
            .context("resident reduction readback failed")?;
        ensure!(
            result.len() == 1,
            "resident reduction must return one scalar"
        );
        Ok(result[0])
    }
}
