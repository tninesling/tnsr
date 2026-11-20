//! Closed BF16 QKV Narrow-VJP packing. Primal Narrow nodes remain untouched.
use super::*;
use crate::tensor::BinaryOp;
use std::collections::HashSet;
mod kernel;
mod matcher;
pub(super) use matcher::recognize;

#[derive(Clone, Debug)]
pub(super) enum Join {
    Input(usize),
    Add(Box<Join>, Box<Join>),
}
pub(super) struct Input {
    pub source: NodeIndex,
    pub window: usize,
    pub views: Vec<TensorGraphNode<f32>>,
}
pub(super) struct QkvGradPack {
    pub output: NodeIndex,
    pub shape: Vec<usize>,
    pub inputs: Vec<Input>,
    pub tree: Join,
    pub members: HashSet<NodeIndex>,
}
fn view(value: &VirtualTensor, node: &TensorGraphNode<f32>) -> Result<VirtualTensor> {
    Ok(match node {
        TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
            value.reshape(shape.clone())?
        }
        TensorGraphNode::Permute { axes, .. } => value.permute(axes)?,
        TensorGraphNode::Transpose { .. } => {
            ensure!(value.shape.len() >= 2, "QKV gradient transpose rank");
            let mut axes: Vec<_> = (0..value.shape.len()).collect();
            let end = axes.len();
            axes.swap(end - 2, end - 1);
            value.permute(&axes)?
        }
        _ => anyhow::bail!("QKV gradient chain is not a bijective view"),
    })
}
fn logical_inputs(
    region: &QkvGradPack,
    available: &HashMap<NodeIndex, Value>,
) -> Result<Vec<Value>> {
    let mut values = Vec::new();
    for input in &region.inputs {
        let mut value = available
            .get(&input.source)
            .context("QKV gradient source unavailable")?
            .clone();
        ensure!(
            value.binding.dtype == DType::BF16
                && value.view.dtype == DType::BF16
                && value.view.shape == value.source_shape
                && value.view.access == crate::tile::IndexMap::identity(value.source_shape.len())
                && value.view.predicate.is_none()
                && value.view.out_of_bounds.is_none()
                && value.chain.is_empty(),
            "QKV gradient requires contiguous raw BF16 leaves"
        );
        for node in &input.views {
            value.view = view(&value.view, node)?;
        }
        ensure!(
            value.view.shape == vec![region.shape[0], region.shape[1], region.shape[2] / 3],
            "QKV gradient logical source shape mismatch"
        );
        values.push(value);
    }
    Ok(values)
}
impl MixedExecutor {
    pub(super) fn qkv_grad_pack(&mut self, region: &QkvGradPack) -> Result<()> {
        let values = logical_inputs(region, &self.values)?;
        let (_, output) = self.alloc(region.output, region.shape.clone(), DType::BF16)?;
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel {
            region_id: 0,
            ir: kernel::ir(region, &values)?,
        }];
        tiles.optimize_indices()?;
        let graph =
            PtxGraph::from(tiles).with_target(PtxTarget::from_context(self.stream.context())?);
        let name = graph
            .region_kernel_name(0)
            .context("QKV gradient pack kernel absent")?;
        let module = self
            .stream
            .context()
            .load_module(Ptx::from_src(graph.to_ptx()))?;
        let len = region
            .shape
            .iter()
            .try_fold(1_usize, |a, n| a.checked_mul(*n))
            .context("QKV pack size overflow")?;
        let mut pointers: Vec<_> = values.iter().map(|v| v.binding.pointer).collect();
        pointers.push(output.binding.pointer);
        self.kernels.push(Kernel {
            owner: Some(region.output),
            label: format!("QKV gradient pack fused BF16 shape={:?}", region.shape),
            function: module.load_function(name)?,
            config: LaunchConfig {
                grid_dim: (u32::try_from(len)?.div_ceil(kernel::THREADS as u32), 1, 1),
                block_dim: (kernel::THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            pointers,
            zero_before: None,
        });
        self.values.insert(region.output, output);
        self.info.qkv_grad_pack_regions += 1;
        Ok(())
    }
}
#[cfg(test)]
mod tests;

/// Test-only leaf comparison using the exact production logical input preparation.
#[cfg(test)]
pub(in crate::ptx::mixed) fn diagnostic_ir(
    region: &QkvGradPack,
    values: &HashMap<NodeIndex, Value>,
) -> Result<crate::tile::TileIR> {
    let logical = logical_inputs(region, values)?;
    kernel::ir(region, &logical)
}
