//! Generic TileIR lowering for resident gradient accumulation and Adam.
use super::DeviceValue;
use crate::graph::TensorGraph;
use crate::tile::ir::MatrixLayout;
use crate::tile::{
    DType, Dim, Expr, RegionInput, RegionValue, TileGraph, TileIR, TileIRBuilder, TileVar,
};
use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaModule};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub(super) struct Kernels {
    pub module: Arc<CudaModule>,
    pub names: Vec<String>,
}
impl Kernels {
    pub fn compile(device: &Arc<CudaContext>, kernels: Vec<TileIR>) -> Result<Self> {
        let count = kernels.len();
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = kernels
            .into_iter()
            .enumerate()
            .map(|(region_id, ir)| crate::tile::graph::TileRegionKernel { region_id, ir })
            .collect();
        tiles.optimize_indices()?;
        let graph = crate::ptx::PtxGraph::from(tiles)
            .with_target(crate::ptx::PtxTarget::from_context(device)?);
        let names = (0..count)
            .map(|i| {
                graph
                    .region_kernel_name(i)
                    .map(str::to_owned)
                    .context("resident kernel name absent")
            })
            .collect::<Result<Vec<_>>>()?;
        let module = device
            .load_module(Ptx::from_src(graph.to_ptx()))
            .context("compile resident TileIR module")?;
        Ok(Self { module, names })
    }
}

fn linear() -> Expr {
    Expr::Mul(
        Box::new(Expr::BlockIdx(Dim::X)),
        Box::new(Expr::BlockDim(Dim::X)),
    ) + Expr::ThreadIdx(Dim::X)
}
fn reg(b: &mut TileIRBuilder) -> TileVar {
    b.alloc_register(DType::F32, 1, 1)
}
fn load(b: &mut TileIRBuilder, name: &str, index: Expr, len: usize) -> TileVar {
    let dest = reg(b);
    b.load_global_predicated(
        dest,
        name,
        index.clone(),
        Expr::Const(0),
        index,
        MatrixLayout {
            rows: 1,
            cols: len,
            row_stride: len,
        },
    );
    dest
}
fn store(b: &mut TileIRBuilder, name: &str, value: TileVar, len: usize) {
    b.store_global_indexed(
        name,
        value,
        linear(),
        (Expr::Const(0), linear()),
        MatrixLayout {
            rows: 1,
            cols: len,
            row_stride: len,
        },
    );
}

pub(super) fn accumulate(value: &DeviceValue, len: usize) -> Result<TileIR> {
    accumulate_view(&value.view, &value.source_shape, len)
}

pub(super) fn accumulate_view(
    view: &crate::tile::VirtualTensor,
    source_shape: &[usize],
    len: usize,
) -> Result<TileIR> {
    let mut b = TileIRBuilder::new();
    b.start_kernel("resident_accumulate");
    b.add_param("gradient", view.dtype, true);
    b.add_param("accumulated", DType::F32, false);
    b.bounds_check(len);
    let descriptor = RegionInput {
        node: view.source,
        value: RegionValue(0),
        tensor: view.clone(),
        source_shape: source_shape.to_vec(),
    };
    let index = crate::tile::region::input_element_index(&descriptor, &view.shape, &linear())?;
    let incoming = reg(&mut b);
    b.load_global_predicated(
        incoming,
        "gradient",
        index,
        Expr::Const(0),
        linear(),
        MatrixLayout {
            rows: 1,
            cols: len,
            row_stride: len,
        },
    );
    let old = load(&mut b, "accumulated", linear(), len);
    let result = reg(&mut b);
    b.add(result, old, incoming);
    store(&mut b, "accumulated", result, len);
    Ok(b.finish())
}

pub(super) fn adam(len: usize) -> TileIR {
    let mut b = TileIRBuilder::new();
    b.start_kernel("resident_adam");
    for name in ["weight", "gradient", "moment", "variance", "settings"] {
        b.add_param(name, DType::F32, true);
    }
    for name in ["next_weight", "next_moment", "next_variance", "health"] {
        b.add_param(name, DType::F32, false);
    }
    b.bounds_check(len);
    let w = load(&mut b, "weight", linear(), len);
    let g = load(&mut b, "gradient", linear(), len);
    let m = load(&mut b, "moment", linear(), len);
    let v = load(&mut b, "variance", linear(), len);
    // lr, beta1, beta2, 1-beta1, 1-beta2, correction1, correction2, eps, decay, clip scale.
    let c: Vec<_> = (0..10)
        .map(|i| load(&mut b, "settings", Expr::Const(i), 10))
        .collect();
    let scaled = reg(&mut b);
    b.mul(scaled, g, c[9]);
    let decay = reg(&mut b);
    b.mul(decay, w, c[8]);
    let g = reg(&mut b);
    b.add(g, scaled, decay);
    let m_old = reg(&mut b);
    b.mul(m_old, m, c[1]);
    let m_new = reg(&mut b);
    b.mul(m_new, g, c[3]);
    let m = reg(&mut b);
    b.add(m, m_old, m_new);
    let v_old = reg(&mut b);
    b.mul(v_old, v, c[2]);
    let v_g = reg(&mut b);
    b.mul(v_g, g, c[4]);
    let v_new = reg(&mut b);
    b.mul(v_new, v_g, g);
    let v = reg(&mut b);
    b.add(v, v_old, v_new);
    let m_hat = reg(&mut b);
    b.div(m_hat, m, c[5]);
    let v_hat = reg(&mut b);
    b.div(v_hat, v, c[6]);
    let root = reg(&mut b);
    b.sqrt(root, v_hat);
    let denom = reg(&mut b);
    b.add(denom, root, c[7]);
    let step = reg(&mut b);
    b.mul(step, c[0], m_hat);
    let delta = reg(&mut b);
    b.div(delta, step, denom);
    let next = reg(&mut b);
    b.sub(next, w, delta);
    store(&mut b, "next_weight", next, len);
    store(&mut b, "next_moment", m, len);
    store(&mut b, "next_variance", v, len);
    let nw = reg(&mut b);
    b.sub(nw, next, next);
    let nm = reg(&mut b);
    b.sub(nm, m, m);
    let nv = reg(&mut b);
    b.sub(nv, v, v);
    let a = reg(&mut b);
    b.add(a, nw, nm);
    let health = reg(&mut b);
    b.add(health, a, nv);
    store(&mut b, "health", health, len);
    b.finish()
}
