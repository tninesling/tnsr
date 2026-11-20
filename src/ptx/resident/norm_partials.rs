//! Opt-in first-overwrite accumulation with the original global norm row tree.
//!
//! Rows split by parameter boundaries are repaired from the stored gradient buffer;
//! arbitrary per-parameter partial sums would change floating-point association.
use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::ptx::{PtxExecutor, PtxReductionMode};
use crate::tensor::{BinaryOp, Input, TensorExpr};
use crate::tile::{
    Block, DType, Dim, Expr, KernelParam, MemorySpace, RegionInput, RegionValue, ScalarBinaryOp,
    ScalarBuilder, ScalarExpr, ScalarMemory, ScalarPredicate, SharedLayout, Stmt, TileIR,
    TileLayout, TileVar, VirtualTensor,
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::{
    collections::{BTreeSet, HashMap},
    ops::Range,
    sync::{Arc, Mutex},
};

/// Owner-side experimental norm accounting; pool peaks are reported separately.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GradientNormPartialsStatus {
    pub enabled: bool,
    pub last_step_used: bool,
    pub rows: usize,
    pub repair_rows: usize,
    pub workspace_bytes: usize,
}

pub(super) const ROW_ELEMENTS: usize = 1024;
pub(super) const ROW_THREADS: usize = 128;

#[derive(Clone, Debug)]
pub(super) struct PartialLayout {
    pub ranges: Vec<Range<usize>>,
    pub padded_len: usize,
    pub rows: usize,
    pub repair_rows: Vec<u32>,
}
impl PartialLayout {
    pub fn new(ranges: Vec<Range<usize>>, padded_len: usize) -> Result<Self> {
        ensure!(!ranges.is_empty(), "norm partials require parameter ranges");
        let mut end = 0;
        let mut repair = BTreeSet::new();
        for range in &ranges {
            ensure!(
                range.start == end && range.end > range.start,
                "norm parameter ranges must cover storage contiguously"
            );
            if range.start % ROW_ELEMENTS != 0 {
                repair.insert(u32::try_from(range.start / ROW_ELEMENTS)?);
            }
            end = range.end;
        }
        let expected = end
            .checked_add(ROW_ELEMENTS - 1)
            .context("norm padding overflow")?
            / ROW_ELEMENTS
            * ROW_ELEMENTS;
        ensure!(
            padded_len == expected && padded_len <= u32::MAX as usize,
            "norm padding must be the minimal 1024-element extent"
        );
        if end % ROW_ELEMENTS != 0 {
            repair.insert(u32::try_from(end / ROW_ELEMENTS)?);
        }
        Ok(Self {
            ranges,
            padded_len,
            rows: padded_len / ROW_ELEMENTS,
            repair_rows: repair.into_iter().collect(),
        })
    }
    pub fn accumulation_config(&self, parameter: usize) -> Result<LaunchConfig> {
        let r = self
            .ranges
            .get(parameter)
            .context("norm parameter absent")?;
        let rows = r.end.div_ceil(ROW_ELEMENTS) - r.start / ROW_ELEMENTS;
        Ok(LaunchConfig {
            grid_dim: (u32::try_from(rows)?, 1, 1),
            block_dim: (ROW_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
}

/// Only the completed first successful overwrite attempt may supply norm rows.
/// ResidentAdam remains responsible for rejecting an invalid interval entirely.
#[derive(Default, Debug)]
pub(super) struct Validity {
    attempts: usize,
    valid: bool,
}
impl Validity {
    pub fn begin_step(&mut self) {
        self.attempts = 0;
        self.valid = false;
    }
    pub fn begin_accumulation(
        &mut self,
        enabled: bool,
        first_overwrite: bool,
        pending_zero: bool,
    ) -> bool {
        let generate = enabled && first_overwrite && pending_zero && self.attempts == 0;
        self.valid = false;
        self.attempts = self.attempts.saturating_add(1);
        generate
    }
    pub fn finish_accumulation(&mut self, generated: bool, success: bool) {
        self.valid = generated && success && self.attempts == 1;
    }
    pub fn single_attempt(&self) -> bool {
        self.attempts == 1
    }
    pub fn valid(&self) -> bool {
        self.valid && self.attempts == 1
    }
}
fn arithmetic(op: BinaryOp, a: impl Into<ScalarExpr>, b: impl Into<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::binary(ScalarBinaryOp::Arithmetic(op), a, b)
}
fn shared(b: &mut ScalarBuilder) -> TileVar {
    let shared = b.var();
    b.stmts.push(Stmt::AllocTile {
        var: shared,
        space: MemorySpace::Shared,
        layout: TileLayout::Shared(SharedLayout::row_major(ROW_THREADS)),
        dtype: DType::F32,
        rows: 1,
        cols: ROW_THREADS,
    });
    shared
}
fn finish_row(b: &mut ScalarBuilder, scratch: TileVar, dot: TileVar, row: Expr) -> Result<()> {
    let lane = Expr::ThreadIdx(Dim::X);
    b.store(ScalarMemory::Shared(scratch), lane.clone(), dot);
    b.barrier();
    for stride in [64, 32, 16, 8, 4, 2, 1] {
        b.when(
            ScalarPredicate::IndexLt {
                lhs: lane.clone(),
                rhs: Expr::Const(stride),
            },
            |b| {
                let peer = b.load(
                    ScalarMemory::Shared(scratch),
                    lane.clone() + Expr::Const(stride),
                );
                b.set(dot, arithmetic(BinaryOp::Add, dot, peer));
                b.store(ScalarMemory::Shared(scratch), lane.clone(), dot);
                Ok(())
            },
        )?;
        b.barrier();
    }
    b.when(
        ScalarPredicate::IndexLt {
            lhs: lane,
            rhs: Expr::Const(1),
        },
        |b| {
            let value = b.load(ScalarMemory::Shared(scratch), Expr::Const(0));
            b.store(ScalarMemory::Global("partials".into()), row, value);
            Ok(())
        },
    )?;
    Ok(())
}
/// One CTA intersects one GLOBAL row. Parameter-local output pointers still
/// address only the checked range; the source view is evaluated at local indices.
pub(super) fn accumulate_ir(
    view: &VirtualTensor,
    source_shape: &[usize],
    range: Range<usize>,
) -> Result<TileIR> {
    let len = range
        .end
        .checked_sub(range.start)
        .context("invalid norm accumulation range")?;
    ensure!(
        len > 0 && view.shape.iter().product::<usize>() == len,
        "norm gradient extent mismatch"
    );
    ensure!(
        matches!(view.dtype, DType::F32 | DType::BF16),
        "norm gradient storage unsupported"
    );
    let mut b = ScalarBuilder::default();
    let scratch = shared(&mut b);
    let row = Expr::BlockIdx(Dim::X) + Expr::Const(i64::try_from(range.start / ROW_ELEMENTS)?);
    let local_start = Expr::Sub(
        Box::new(row.clone() * ROW_ELEMENTS),
        Box::new(Expr::Const(i64::try_from(range.start)?)),
    );
    let descriptor = RegionInput {
        node: view.source,
        value: RegionValue(0),
        tensor: view.clone(),
        source_shape: source_shape.to_vec(),
    };
    let mut actual = Vec::new();
    for part in 0..8 {
        let local = local_start.clone()
            + Expr::ThreadIdx(Dim::X)
            + Expr::Const((part * ROW_THREADS) as i64);
        let value = b.value(ScalarExpr::Constant(0.));
        b.when(
            ScalarPredicate::IndexLt {
                lhs: local.clone(),
                rhs: Expr::Const(i64::try_from(len)?),
            },
            |b| {
                let index =
                    crate::tile::region::input_element_index(&descriptor, &view.shape, &local)?;
                let incoming = b.load(ScalarMemory::Global("gradient".into()), index);
                // Preserve +0 + incoming, including the stored signed-zero result.
                b.set(
                    value,
                    arithmetic(BinaryOp::Add, ScalarExpr::Constant(0.), incoming),
                );
                b.store(ScalarMemory::Global("accumulated".into()), local, value);
                Ok(())
            },
        )?;
        actual.push(value);
    }
    // Both predicates depend on blockIdx/constants only. Every lane follows the
    // same branch through all barriers. The first predicate excludes wrapped
    // negative local_start before the second adds 1023.
    b.when(
        ScalarPredicate::IndexLt {
            lhs: local_start.clone(),
            rhs: Expr::Const(i64::try_from(len)?),
        },
        |b| {
            b.when(
                ScalarPredicate::IndexLt {
                    lhs: local_start + Expr::Const((ROW_ELEMENTS - 1) as i64),
                    rhs: Expr::Const(i64::try_from(len)?),
                },
                |b| {
                    let dot = b.value(ScalarExpr::Constant(0.));
                    for value in actual {
                        b.set(
                            dot,
                            arithmetic(BinaryOp::Add, dot, arithmetic(BinaryOp::Mul, value, value)),
                        );
                    }
                    finish_row(b, scratch, dot, row)
                },
            )
        },
    )?;
    Ok(TileIR {
        kernel_name: "resident_accumulate_norm_rows".into(),
        params: vec![
            KernelParam {
                name: "gradient".into(),
                dtype: view.dtype,
                is_input: true,
            },
            KernelParam {
                name: "accumulated".into(),
                dtype: DType::F32,
                is_input: false,
            },
            KernelParam {
                name: "partials".into(),
                dtype: DType::F32,
                is_input: false,
            },
        ],
        body: Block { stmts: b.stmts },
        shared_mem_bytes: ROW_THREADS * 4,
    })
}
pub(super) fn repair_ir() -> Result<TileIR> {
    let mut b = ScalarBuilder::default();
    let scratch = shared(&mut b);
    b.stmts.push(Stmt::LoadIndex {
        name: "norm_row".into(),
        source: "rows".into(),
        index: Expr::BlockIdx(Dim::X),
    });
    let row = Expr::Var("norm_row".into());
    let dot = b.value(ScalarExpr::Constant(0.));
    for part in 0..8 {
        let index = row.clone() * ROW_ELEMENTS
            + Expr::ThreadIdx(Dim::X)
            + Expr::Const((part * ROW_THREADS) as i64);
        let g = b.load(ScalarMemory::Global("gradient".into()), index);
        b.set(
            dot,
            arithmetic(BinaryOp::Add, dot, arithmetic(BinaryOp::Mul, g, g)),
        );
    }
    finish_row(&mut b, scratch, dot, row)?;
    Ok(TileIR {
        kernel_name: "resident_norm_boundary_rows".into(),
        params: vec![
            KernelParam {
                name: "gradient".into(),
                dtype: DType::F32,
                is_input: true,
            },
            KernelParam {
                name: "rows".into(),
                dtype: DType::U32,
                is_input: true,
            },
            KernelParam {
                name: "partials".into(),
                dtype: DType::F32,
                is_input: false,
            },
        ],
        body: Block { stmts: b.stmts },
        shared_mem_bytes: ROW_THREADS * 4,
    })
}

/// Literal old second stage: reduce [R,1] along axis zero. Do NOT route through
/// Reduction::new(R,false), whose extra reshape would introduce another tree.
pub(super) fn second_stage_graph(rows: usize) -> Result<(TensorGraph<f32>, NodeIndex, NodeIndex)> {
    ensure!(rows > 0, "norm rows cannot be empty");
    let input: TensorExpr<f32> = Input::new("resident_buffer", vec![rows, 1]).into();
    let graph: TensorGraph<f32> = input.reduce_sum(0).into();
    let source = graph
        .graph
        .node_indices()
        .find(|&n| matches!(graph[n], TensorGraphNode::Input { .. }))
        .context("norm partial input absent")?;
    let output = *graph
        .toposort()
        .last()
        .context("norm partial output absent")?;
    Ok((graph, source, output))
}
struct SecondStage {
    pub executor: PtxExecutor,
    graph: TensorGraph<f32>,
    input: NodeIndex,
    output: NodeIndex,
}
impl SecondStage {
    fn new(stream: Arc<CudaStream>, gate: Arc<Mutex<()>>, rows: usize) -> Result<Self> {
        let (graph, input, output) = second_stage_graph(rows)?;
        Ok(Self {
            executor: PtxExecutor::with_stream(stream, PtxReductionMode::DeterministicTree, gate)?,
            graph,
            input,
            output,
        })
    }
    fn run(&mut self, partials: Arc<CudaSlice<f32>>) -> Result<f32> {
        self.executor.run_device(
            &self.graph,
            HashMap::new(),
            HashMap::from([(self.input, partials)]),
        )?;
        let value = self
            .executor
            .get_value(self.output)
            .context("norm partial readback absent")?;
        ensure!(value.len() == 1, "norm partial result must be scalar");
        Ok(value[0])
    }
}
pub(super) struct Workspace {
    pub layout: PartialLayout,
    gate: Arc<Mutex<()>>,
    partials: Arc<CudaSlice<f32>>,
    rows: Option<CudaSlice<u32>>,
    repair: Option<super::kernels::Kernels>,
    second: SecondStage,
}
impl Workspace {
    pub fn new(
        stream: &Arc<CudaStream>,
        gate: Arc<Mutex<()>>,
        layout: PartialLayout,
    ) -> Result<Self> {
        let preparation_gate = gate.clone();
        let _guard = preparation_gate
            .lock()
            .map_err(|_| anyhow::anyhow!("norm workspace stream lock poisoned"))?;
        let partials = Arc::new(stream.alloc_zeros(layout.rows)?);
        let (rows, repair) = if layout.repair_rows.is_empty() {
            (None, None)
        } else {
            (
                Some(stream.memcpy_stod(&layout.repair_rows)?),
                Some(super::kernels::Kernels::compile(
                    stream.context(),
                    vec![repair_ir()?],
                )?),
            )
        };
        let second = SecondStage::new(stream.clone(), gate.clone(), layout.rows)?;
        Ok(Self {
            layout,
            gate,
            partials,
            rows,
            repair,
            second,
        })
    }
    pub fn partials_mut(&mut self) -> Result<&mut CudaSlice<f32>> {
        Arc::get_mut(&mut self.partials).context("resident norm partials borrowed")
    }
    pub fn run(&mut self, stream: &Arc<CudaStream>, gradients: &CudaSlice<f32>) -> Result<f32> {
        ensure!(
            gradients.len() == self.layout.padded_len,
            "resident norm gradient extent changed"
        );
        if let (Some(rows), Some(kernels)) = (&self.rows, &self.repair) {
            let gate = self.gate.clone();
            let _guard = gate
                .lock()
                .map_err(|_| anyhow::anyhow!("norm repair stream lock poisoned"))?;
            let function = kernels.module.load_function(&kernels.names[0])?;
            let mut launch = stream.launch_builder(&function);
            let target =
                Arc::get_mut(&mut self.partials).context("norm partial repair target borrowed")?;
            launch.arg(gradients).arg(rows).arg(target);
            // SAFETY: owner retains all three exact extents on its ordered
            // stream; layout contains unique in-bounds rows, each CTA writes
            // one disjoint partial. All gradient writes precede this launch.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (u32::try_from(self.layout.repair_rows.len())?, 1, 1),
                    block_dim: (ROW_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
        }
        self.second.run(self.partials.clone())
    }
    pub fn last_launches(&self) -> usize {
        usize::from(self.repair.is_some())
            + self.second.executor.execution_metrics().kernel_launches
    }
}
#[cfg(test)]
mod tests;
