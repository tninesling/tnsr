use super::{Binding, program::Kernel};
use crate::{
    graph::TensorGraph,
    ptx::{PtxGraph, PtxTarget},
    tile::{DType, Dim, Expr, TileGraph, TileIRBuilder, ir::MatrixLayout},
};
use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{CudaStream, LaunchConfig},
    nvrtc::Ptx,
};
use std::sync::Arc;

pub(crate) fn convert(
    stream: &Arc<CudaStream>,
    source: Binding,
    target: Binding,
    accumulate: bool,
) -> Result<Kernel> {
    ensure!(
        source.len == target.len,
        "linear conversion extent mismatch"
    );
    ensure!(
        !accumulate || source.dtype == DType::F32 && target.dtype == DType::F32,
        "linear accumulation must be f32"
    );
    let len = source.len;
    let mut b = TileIRBuilder::new();
    b.start_kernel("linear_cast");
    b.add_param("input", source.dtype, true);
    b.add_param("output", target.dtype, false);
    b.bounds_check(len);
    let index = Expr::Mul(
        Box::new(Expr::BlockIdx(Dim::X)),
        Box::new(Expr::BlockDim(Dim::X)),
    ) + Expr::ThreadIdx(Dim::X);
    let layout = MatrixLayout {
        rows: 1,
        cols: len,
        row_stride: len,
    };
    let value = b.alloc_register(DType::F32, 1, 1);
    b.load_global_predicated(
        value,
        "input",
        index.clone(),
        Expr::Const(0),
        index.clone(),
        layout,
    );
    let result = if accumulate {
        let old = b.alloc_register(DType::F32, 1, 1);
        b.load_global_predicated(
            old,
            "output",
            index.clone(),
            Expr::Const(0),
            index.clone(),
            layout,
        );
        let sum = b.alloc_register(DType::F32, 1, 1);
        b.add(sum, old, value);
        sum
    } else {
        value
    };
    b.store_global_predicated("output", result, Expr::Const(0), index, layout);
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel {
        region_id: 0,
        ir: b.finish(),
    }];
    tiles.optimize_indices()?;
    let graph = PtxGraph::from(tiles).with_target(PtxTarget::from_context(stream.context())?);
    let name = graph
        .region_kernel_name(0)
        .context("linear cast kernel absent")?;
    let module = stream
        .context()
        .load_module(Ptx::from_src(graph.to_ptx()))?;
    Ok(Kernel {
        label: format!("convert {:?} to {:?} len={len}", source.dtype, target.dtype),
        function: module.load_function(name)?,
        config: LaunchConfig::for_num_elems(u32::try_from(len)?),
        pointers: vec![source.pointer, target.pointer],
        zero_before: None,
    })
}
