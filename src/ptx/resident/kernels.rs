//! Generic TileIR lowering for resident gradient accumulation and Adam.
use super::DeviceValue;
use crate::graph::TensorGraph;
use crate::tile::ir::MatrixLayout;
use crate::tile::{
    Block, DType, Dim, Expr, RegionInput, RegionValue, ScalarExpr, ScalarMemory, ScalarPredicate,
    Stmt, TileGraph, TileIR, TileIRBuilder, TileVar,
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

pub(super) fn accumulate(value: &DeviceValue, len: usize, overwrite: bool) -> Result<TileIR> {
    accumulate_view(&value.view, &value.source_shape, len, overwrite)
}

pub(super) fn accumulate_view(
    view: &crate::tile::VirtualTensor,
    source_shape: &[usize],
    len: usize,
    overwrite: bool,
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
    let old = if overwrite {
        let zero = reg(&mut b);
        b.zero(zero);
        zero
    } else {
        load(&mut b, "accumulated", linear(), len)
    };
    let result = reg(&mut b);
    b.add(result, old, incoming);
    store(&mut b, "accumulated", result, len);
    Ok(b.finish())
}

/// Health storage follows logical groups, independent of whether the launch
/// uses a 256-thread batch block or a 1024-thread standalone block.
pub(super) const HEALTH_GROUP_SIZE: usize = 256;

struct HealthScratch {
    shared: TileVar,
    warp_sum: TileVar,
    group_value: TileVar,
    group_sum: TileVar,
}
impl HealthScratch {
    fn new(b: &mut TileIRBuilder) -> Self {
        Self {
            // Maximum of 32 warps in the existing standalone 1024-thread block.
            // Allocate outside batched conditionals: child builders do not own
            // the enclosing kernel's static shared-memory accounting.
            shared: b.alloc_shared(DType::F32, 1, 32),
            warp_sum: reg(b),
            group_value: reg(b),
            group_sum: reg(b),
        }
    }
    fn statements(
        &self,
        per_thread: TileVar,
        offset: usize,
        groups: usize,
        total: usize,
    ) -> Result<Vec<Stmt>> {
        let lane = Expr::BitAnd(Box::new(Expr::ThreadIdx(Dim::X)), 31);
        let warp = Expr::ShiftRight(Box::new(Expr::ThreadIdx(Dim::X)), 5);
        let local_group = Expr::ShiftRight(Box::new(Expr::ThreadIdx(Dim::X)), 8);
        let global_group = Expr::ShiftRight(Box::new(linear()), 8);
        let destination = global_group.clone() + Expr::Const(i64::try_from(offset)?);
        let store = Stmt::StoreGlobalPredicated {
            dest_param: "health".into(),
            src: self.group_sum,
            element_index: destination.clone(),
            row: Expr::Const(0),
            col: destination,
            layout: MatrixLayout {
                rows: 1,
                cols: total,
                row_stride: total,
            },
        };
        Ok(vec![
            Stmt::WarpReduce {
                dest: self.warp_sum,
                src: per_thread,
                op: crate::tensor::ReduceOp::Sum,
                broadcast: false,
            },
            when(
                lane.clone(),
                Expr::Const(1),
                vec![Stmt::StoreScalar {
                    target: ScalarMemory::Shared(self.shared),
                    index: warp,
                    value: self.warp_sum,
                }],
            ),
            Stmt::Barrier,
            // All 32 lanes of the first warp in EACH 256-thread group take this
            // branch. The other seven warps skip the second full-warp reduction.
            when(
                Expr::BitAnd(Box::new(Expr::ThreadIdx(Dim::X)), 255),
                Expr::Const(32),
                vec![
                    Stmt::SetScalar {
                        dest: self.group_value,
                        value: ScalarExpr::Constant(0.0),
                    },
                    when(
                        lane.clone(),
                        Expr::Const(8),
                        vec![Stmt::LoadScalar {
                            dest: self.group_value,
                            source: ScalarMemory::Shared(self.shared),
                            index: local_group * 8usize + lane.clone(),
                        }],
                    ),
                    Stmt::WarpReduce {
                        dest: self.group_sum,
                        src: self.group_value,
                        op: crate::tensor::ReduceOp::Sum,
                        broadcast: false,
                    },
                    when(
                        lane,
                        Expr::Const(1),
                        vec![
                            // A batched grid has max(member lengths) blocks. Do NOT
                            // write a short member's excess CTAs into the next member.
                            when(
                                global_group,
                                Expr::Const(i64::try_from(groups)?),
                                vec![store],
                            ),
                        ],
                    ),
                ],
            ),
        ])
    }
}
fn when(lhs: Expr, rhs: Expr, stmts: Vec<Stmt>) -> Stmt {
    Stmt::If {
        condition: ScalarPredicate::IndexLt { lhs, rhs },
        body: Block { stmts },
    }
}

pub(super) fn adam(len: usize) -> Result<TileIR> {
    ensure!(len > 0, "Adam parameter must be nonempty");
    let mut b = TileIRBuilder::new();
    b.start_kernel("resident_adam");
    for name in ["weight", "gradient", "moment", "variance", "settings"] {
        b.add_param(name, DType::F32, true);
    }
    for name in ["next_weight", "next_moment", "next_variance", "health"] {
        b.add_param(name, DType::F32, false);
    }
    let scratch = HealthScratch::new(&mut b);
    let health = reg(&mut b);
    b.zero(health);
    let extent = i64::try_from(len)?;
    b.if_then(
        ScalarPredicate::IndexLt {
            lhs: linear(),
            rhs: Expr::Const(extent),
        },
        |b| {
            adam_body(b, "weight", "next_weight", linear(), len, len, health);
        },
    );
    let mut ir = b.finish();
    let groups = len.div_ceil(HEALTH_GROUP_SIZE);
    ir.body
        .stmts
        .extend(scratch.statements(health, 0, groups, groups)?);
    Ok(ir)
}

/// Separate standalone source variant. Build the exact legacy Adam body, then
/// reconstruct the first-accumulation checkpoint directly after its one typed
/// gradient load. The already-initialized health scalar is still positive zero
/// here and is not modified until the original final health calculation.
pub(super) fn adam_bf16_first(len: usize) -> Result<TileIR> {
    let mut ir = adam(len)?;
    ir.kernel_name = "resident_adam_bf16_first".into();
    let parameter = ir
        .params
        .iter_mut()
        .find(|p| p.name == "gradient")
        .context("legacy Adam gradient parameter absent")?;
    ensure!(
        parameter.dtype == DType::F32 && parameter.is_input,
        "unexpected legacy gradient ABI"
    );
    parameter.dtype = DType::BF16;
    let zeros = ir
        .body
        .stmts
        .iter()
        .filter_map(|s| match s {
            Stmt::Zero { tile } => Some(*tile),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [zero] = zeros.as_slice() else {
        anyhow::bail!("legacy standalone positive-zero health initializer changed");
    };
    fn patch(block: &mut Block, zero: TileVar) -> usize {
        let mut out = Vec::new();
        let mut count = 0;
        for mut statement in std::mem::take(&mut block.stmts) {
            if let Stmt::If { body, .. } = &mut statement {
                count += patch(body, zero);
            }
            let gradient = match &statement {
                Stmt::LoadGlobalPredicated {
                    dest, src_param, ..
                } if src_param == "gradient" => Some(*dest),
                _ => None,
            };
            out.push(statement);
            if let Some(g) = gradient {
                out.push(Stmt::SetScalar {
                    dest: g,
                    value: ScalarExpr::binary(
                        crate::tile::ScalarBinaryOp::AddRn,
                        ScalarExpr::Var(zero),
                        ScalarExpr::Var(g),
                    ),
                });
                count += 1;
            }
        }
        block.stmts = out;
        count
    }
    ensure!(
        patch(&mut ir.body, *zero) == 1,
        "legacy standalone gradient load count changed"
    );
    ir.validate_layouts()?;
    Ok(ir)
}

/// Bounded horizontal fusion. Each grid-Y lane owns one independent parameter;
/// descriptors carry separate element and compact-health offsets.
pub(super) fn adam_batch(
    parameters: &[(usize, usize, usize)],
    flat_len: usize,
    health_len: usize,
) -> Result<TileIR> {
    ensure!(
        (2..=16).contains(&parameters.len()),
        "Adam batch requires 2 through 16 parameters"
    );
    ensure!(
        parameters.iter().all(|&(offset, len, health)| len > 0
            && offset.checked_add(len).is_some_and(|end| end <= flat_len)
            && health
                .checked_add(len.div_ceil(HEALTH_GROUP_SIZE))
                .is_some_and(|end| end <= health_len)),
        "Adam batch parameter range exceeds flat or health storage"
    );
    let mut spans = parameters
        .iter()
        .map(|&(offset, len, health)| {
            (
                offset,
                offset + len,
                health,
                health + len.div_ceil(HEALTH_GROUP_SIZE),
            )
        })
        .collect::<Vec<_>>();
    spans.sort_unstable_by_key(|s| s.0);
    ensure!(
        spans.windows(2).all(|s| s[0].1 <= s[1].0),
        "Adam batch parameter ranges overlap"
    );
    spans.sort_unstable_by_key(|s| s.2);
    ensure!(
        spans.windows(2).all(|s| s[0].3 <= s[1].2),
        "Adam batch health ranges overlap"
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
    let scratch = HealthScratch::new(&mut b);
    let health = reg(&mut b);
    b.zero(health);
    let mut members = Vec::new();
    for (i, &(offset, len, health_offset)) in parameters.iter().enumerate() {
        let lower = i64::try_from(i)?;
        let upper = i64::try_from(i + 1)?;
        let offset = i64::try_from(offset)?;
        let extent = i64::try_from(len)?;
        b.if_then(
            ScalarPredicate::IndexLt {
                lhs: Expr::BlockIdx(Dim::Y),
                rhs: Expr::Const(upper),
            },
            |b| {
                b.if_then(
                    ScalarPredicate::IndexLt {
                        lhs: Expr::Const(lower),
                        rhs: Expr::BlockIdx(Dim::Y) + Expr::Const(1),
                    },
                    |b| {
                        b.if_then(
                            ScalarPredicate::IndexLt {
                                lhs: linear(),
                                rhs: Expr::Const(extent),
                            },
                            |b| {
                                adam_body(
                                    b,
                                    &format!("weight_{i}"),
                                    &format!("next_weight_{i}"),
                                    linear() + Expr::Const(offset),
                                    len,
                                    flat_len,
                                    health,
                                );
                            },
                        );
                    },
                );
            },
        );
        members.push(scratch.statements(
            health,
            health_offset,
            len.div_ceil(HEALTH_GROUP_SIZE),
            health_len,
        )?);
    }
    let mut ir = b.finish();
    let mut statements = members.into_iter();
    for stmt in &mut ir.body.stmts {
        if let Stmt::If { body, .. } = stmt
            && let Some(Stmt::If { body: member, .. }) = body.stmts.first_mut()
        {
            member.stmts.extend(
                statements
                    .next()
                    .context("Adam health member body missing")?,
            );
        }
    }
    ensure!(
        statements.next().is_none(),
        "Adam health member count mismatch"
    );
    Ok(ir)
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
    health: TileVar,
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
    b.add(health, a, nv);
}

#[cfg(test)]
#[path = "health_tests.rs"]
mod health_tests;

#[cfg(test)]
mod accumulation_tests {
    use super::*;

    #[test]
    fn first_accumulation_uses_positive_zero_add_without_destination_read() {
        let view = crate::tile::VirtualTensor::identity(
            crate::graph::NodeIndex::new(0),
            vec![3],
            DType::F32,
        );
        for overwrite in [false, true] {
            let ir = accumulate_view(&view, &[3], 3, overwrite).unwrap();
            let reads = ir
                .body
                .stmts
                .iter()
                .filter(|stmt| {
                    matches!(stmt,
                        Stmt::LoadGlobalPredicated { src_param, .. } if src_param == "accumulated"
                    )
                })
                .count();
            assert_eq!(reads, usize::from(!overwrite));
            if overwrite {
                let zero = ir
                    .body
                    .stmts
                    .iter()
                    .find_map(|stmt| match stmt {
                        Stmt::Zero { tile } => Some(*tile),
                        _ => None,
                    })
                    .unwrap();
                assert!(
                    ir.body
                        .stmts
                        .iter()
                        .any(|stmt| matches!(stmt, Stmt::Add { a, .. } if *a == zero))
                );
                // A direct copy of -0 would violate existing +0 + incoming semantics.
                assert_eq!((0.0_f32 + -0.0_f32).to_bits(), 0);
            }
        }
    }
}
