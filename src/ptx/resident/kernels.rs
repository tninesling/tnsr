//! Generic TileIR lowering for resident gradient accumulation and Adam.
use super::DeviceValue;
use crate::graph::TensorGraph;
use crate::tile::ir::MatrixLayout;
use crate::tile::{
    DType, Dim, Expr, RegionInput, RegionValue, TileGraph, TileIR, TileIRBuilder, TileVar,
};
use anyhow::{Context, Result, ensure};
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
    adam_body(&mut b, "weight", "next_weight", linear(), len, len);
    b.finish()
}

/// Bounded horizontal fusion. Each grid-Y lane owns one independent parameter;
/// explicit pointer parameters retain normal allocation tracking and avoid raw
/// device-pointer tables. Parameters in a batch share their Adam settings.
pub(super) fn adam_batch(parameters: &[(usize, usize)], flat_len: usize) -> Result<TileIR> {
    ensure!(
        (2..=16).contains(&parameters.len()),
        "Adam batch requires 2 through 16 parameters"
    );
    ensure!(
        parameters
            .iter()
            .all(|&(offset, len)| len > 0
                && offset.checked_add(len).is_some_and(|end| end <= flat_len)),
        "Adam batch parameter range exceeds flat storage"
    );
    let mut b = TileIRBuilder::new();
    b.start_kernel("resident_adam_batch");
    for name in ["gradient", "moment", "variance", "settings"] {
        b.add_param(name, DType::F32, true);
    }
    for name in ["next_moment", "next_variance", "health"] {
        b.add_param(name, DType::F32, false);
    }
    for i in 0..parameters.len() {
        b.add_param(&format!("weight_{i}"), DType::F32, true);
        b.add_param(&format!("next_weight_{i}"), DType::F32, false);
    }
    for (i, &(offset, len)) in parameters.iter().enumerate() {
        let lower = i64::try_from(i)?;
        let upper = i64::try_from(i + 1)?;
        let offset = i64::try_from(offset)?;
        b.if_then(
            crate::tile::ScalarPredicate::IndexLt {
                lhs: Expr::BlockIdx(Dim::Y),
                rhs: Expr::Const(upper),
            },
            |b| {
                b.if_then(
                    crate::tile::ScalarPredicate::IndexLt {
                        lhs: Expr::Const(lower),
                        rhs: Expr::BlockIdx(Dim::Y) + Expr::Const(1),
                    },
                    |b| {
                        b.bounds_check(len);
                        adam_body(
                            b,
                            &format!("weight_{i}"),
                            &format!("next_weight_{i}"),
                            linear() + Expr::Const(offset),
                            len,
                            flat_len,
                        );
                    },
                );
            },
        );
    }
    Ok(b.finish())
}

fn store_at(b: &mut TileIRBuilder, name: &str, value: TileVar, index: Expr, len: usize) {
    b.store_global_indexed(
        name,
        value,
        index.clone(),
        (Expr::Const(0), index),
        MatrixLayout {
            rows: 1,
            cols: len,
            row_stride: len,
        },
    );
}

fn adam_body(
    b: &mut TileIRBuilder,
    weight_name: &str,
    next_name: &str,
    flat: Expr,
    len: usize,
    flat_len: usize,
) {
    let w = load(b, weight_name, linear(), len);
    let g = load(b, "gradient", flat.clone(), flat_len);
    let m = load(b, "moment", flat.clone(), flat_len);
    let v = load(b, "variance", flat.clone(), flat_len);
    // lr, beta1, beta2, 1-beta1, 1-beta2, correction1, correction2, eps, decay, clip scale.
    let c: Vec<_> = (0..10)
        .map(|i| load(b, "settings", Expr::Const(i), 10))
        .collect();
    let scaled = reg(b);
    b.mul(scaled, g, c[9]);
    let decay = reg(b);
    b.mul(decay, w, c[8]);
    let g = reg(b);
    b.add(g, scaled, decay);
    let m_old = reg(b);
    b.mul(m_old, m, c[1]);
    let m_new = reg(b);
    b.mul(m_new, g, c[3]);
    let m = reg(b);
    b.add(m, m_old, m_new);
    let v_old = reg(b);
    b.mul(v_old, v, c[2]);
    let v_g = reg(b);
    b.mul(v_g, g, c[4]);
    let v_new = reg(b);
    b.mul(v_new, v_g, g);
    let v = reg(b);
    b.add(v, v_old, v_new);
    let m_hat = reg(b);
    b.div(m_hat, m, c[5]);
    let v_hat = reg(b);
    b.div(v_hat, v, c[6]);
    let root = reg(b);
    b.sqrt(root, v_hat);
    let denom = reg(b);
    b.add(denom, root, c[7]);
    let step = reg(b);
    b.mul(step, c[0], m_hat);
    let delta = reg(b);
    b.div(delta, step, denom);
    let next = reg(b);
    b.sub(next, w, delta);
    store(b, next_name, next, len);
    store_at(b, "next_moment", m, flat.clone(), flat_len);
    store_at(b, "next_variance", v, flat.clone(), flat_len);
    let nw = reg(b);
    b.sub(nw, next, next);
    let nm = reg(b);
    b.sub(nm, m, m);
    let nv = reg(b);
    b.sub(nv, v, v);
    let a = reg(b);
    b.add(a, nw, nm);
    let health = reg(b);
    b.add(health, a, nv);
    store_at(b, "health", health, flat.clone(), flat_len);
}
