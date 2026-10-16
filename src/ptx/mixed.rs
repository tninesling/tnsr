//! Fixed-shape execution of explicit f32/BF16 graphs, including graph autograd.
//!
//! Each arithmetic node retains its storage rounding boundary. View chains are
//! virtual and casts materialize directly into the destination dtype. This
//! planner fuses f32 regions while preserving every BF16 arithmetic boundary.
use super::{
    BF16, CudaDType, F32, PtxGraph, PtxTarget,
    linear::{
        Binding, Buffer, Captured,
        program::{Kernel, Program, binding},
    },
};
mod memory;
mod planner;
mod split_k;

use crate::{
    graph::{NodeIndex, TensorGraph, TensorGraphNode, WithGrad, mixed::value_dtypes},
    tensor::ValueDType,
    tile::{DType, MatMulPrecision, TileDType, TileGraph, TileIRBuilder, VirtualTensor},
};
use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig, sys},
    nvrtc::Ptx,
};
use std::{
    any::Any,
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
};

#[derive(Clone)]
struct Value {
    binding: Binding,
    view: VirtualTensor,
    source_shape: Vec<usize>,
    chain: Vec<TensorGraphNode<f32>>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct MixedInfo {
    /// Private temporary allocations replaced by nonoverlapping live ranges.
    pub reused_allocations: usize,
    /// Retained tensor payload before resident lifetime reuse (not peak process memory).
    pub unpooled_bytes: usize,
    pub exact_index_bytes: usize,
    pub immutable_source_bytes: usize,
    pub kernels: usize,
    pub f32_bytes: usize,
    pub bf16_bytes: usize,
    pub replay: bool,
    /// Programs whose emitted PTX contains native BF16 MMA instructions.
    pub bf16_tensor_core_programs: usize,
}

/// Owns all typed allocations and compiled functions for a single graph.
/// Uploads are explicit and excluded from `run` and `time_runs`.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct MixedTransfers {
    pub f32_uploads: usize,
    pub f32_bytes: usize,
    pub index_uploads: usize,
    pub index_bytes: usize,
    pub parameter_copies: usize,
    pub parameter_bytes: usize,
    pub upload_seconds: f64,
    pub parameter_refresh_seconds: f64,
}

struct Readback {
    buffer: CudaSlice<f32>,
    kernel: Kernel,
}

pub(crate) struct ResidentBindings {
    pub stream: Arc<CudaStream>,
    pub gate: Arc<Mutex<()>>,
    pub parameters: HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
}

pub struct MixedExecutor {
    captured: Option<Captured>,
    stream: Arc<CudaStream>,
    buffers: Vec<Buffer>,
    programs: Vec<Box<dyn Any>>,
    kernels: Vec<Kernel>,
    values: HashMap<NodeIndex, Value>,
    sources: HashMap<NodeIndex, usize>,
    source_nodes: HashMap<NodeIndex, TensorGraphNode<f32>>,
    bound_parameters: HashMap<NodeIndex, Weak<CudaSlice<f32>>>,
    gate: Arc<Mutex<()>>,
    index_graph: TensorGraph<f32>,
    index_plan: crate::graph::indices::IndexPlan,
    index_buffers: HashMap<NodeIndex, CudaSlice<u32>>,
    readbacks: RefCell<HashMap<NodeIndex, Readback>>,
    uploaded: bool,
    constants_initialized: bool,
    transfers: MixedTransfers,
    has_run: bool,
    info: MixedInfo,
}
impl Drop for MixedExecutor {
    fn drop(&mut self) {
        let gate = self.gate.clone();
        let _guard = match gate.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let _ = self.stream.synchronize();
        // Keep graph destruction, asynchronous frees and module release outside
        // every other owner's capture on the shared resident stream.
        self.captured = None;
        self.readbacks.get_mut().clear();
        self.programs.clear();
        self.buffers.clear();
        self.index_buffers.clear();
        self.kernels.clear();
        self.source_nodes.clear();
        self.index_graph.graph.clear();
        let _ = self.stream.synchronize();
    }
}
impl MixedExecutor {
    /// Homogeneous comparator using the existing whole-graph fusion planner.
    /// Only materialized nodes are readable; intermediate fused values may be absent.
    pub fn new_fused_f32<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
    ) -> Result<Self> {
        ensure!(
            value_dtypes(graph).is_empty(),
            "fused f32 comparator cannot contain casts"
        );
        Self::prepare(graph, precision, replay, true, false, None)
    }

    pub fn new<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
    ) -> Result<Self> {
        Self::prepare(graph, precision, replay, false, true, None)
    }

    /// Diagnostic comparator retaining one arithmetic operation per program.
    pub fn new_unfused<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
    ) -> Result<Self> {
        Self::prepare(graph, precision, replay, false, false, None)
    }

    fn prepare<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
        fused_f32: bool,
        fuse_f32_regions: bool,
        resident: Option<ResidentBindings>,
    ) -> Result<Self> {
        ensure!(!graph.is_empty(), "mixed graph is empty");
        let reuse_intermediates = resident.is_some();
        let (stream, gate, bound_parameters) = match resident {
            Some(bindings) => (bindings.stream, bindings.gate, bindings.parameters),
            None => (
                CudaContext::new(0)?.new_stream()?,
                Arc::new(Mutex::new(())),
                HashMap::new(),
            ),
        };
        let mut index_graph = TensorGraph::new();
        index_graph.graph = graph.graph.clone();
        let index_plan = crate::graph::indices::IndexPlan::new(graph)?;
        let dtypes = value_dtypes(graph);
        let mut result = Self {
            captured: None,
            stream,
            buffers: Vec::new(),
            programs: Vec::new(),
            kernels: Vec::new(),
            values: HashMap::new(),
            sources: HashMap::new(),
            source_nodes: HashMap::new(),
            bound_parameters: bound_parameters.keys().map(|n| (*n, Weak::new())).collect(),
            gate,
            index_graph,
            index_plan,
            index_buffers: HashMap::new(),
            readbacks: RefCell::new(HashMap::new()),
            uploaded: false,
            constants_initialized: false,
            transfers: MixedTransfers::default(),
            has_run: false,
            info: MixedInfo {
                reused_allocations: 0,
                unpooled_bytes: 0,
                exact_index_bytes: 0,
                immutable_source_bytes: 0,
                kernels: 0,
                f32_bytes: 0,
                bf16_bytes: 0,
                replay,
                bf16_tensor_core_programs: 0,
            },
        };
        // Declare the guard after the owner so errors unlock before its Drop.
        // Allocations and uploads also belong outside another owner's capture.
        let preparation_gate = result.gate.clone();
        let _preparation_guard = preparation_gate
            .lock()
            .map_err(|_| anyhow::anyhow!("mixed stream lock poisoned"))?;
        let mut f32_region = Vec::new();
        for node in graph.toposort() {
            if fused_f32
                && !matches!(
                    graph[node],
                    TensorGraphNode::Input { .. }
                        | TensorGraphNode::Parameter { .. }
                        | TensorGraphNode::Constant { .. }
                )
            {
                continue;
            }
            if let TensorGraphNode::IndexSource { .. } = &graph[node] {
                let shape = graph[node].shape().clone();
                let len = shape
                    .iter()
                    .try_fold(1_usize, |n, d| n.checked_mul(*d))
                    .context("integer extent overflow")?;
                let buffer = result.stream.alloc_zeros::<u32>(len)?;
                let binding = Binding {
                    pointer: buffer.device_ptr(&result.stream).0,
                    len,
                    dtype: DType::U32,
                };
                result.values.insert(
                    node,
                    Value {
                        binding,
                        view: VirtualTensor::identity(node, shape.clone(), DType::U32),
                        source_shape: shape,
                        chain: Vec::new(),
                    },
                );
                result.info.exact_index_bytes += len * 4;
                result.index_buffers.insert(node, buffer);
                continue;
            }
            let dtype = match dtypes.get(&node).copied().unwrap_or(ValueDType::F32) {
                ValueDType::F32 => DType::F32,
                ValueDType::BF16 => DType::BF16,
            };
            let shape = graph[node].shape().clone();
            let len = shape
                .iter()
                .try_fold(1_usize, |n, d| n.checked_mul(*d))
                .context("mixed extent overflow")?;
            ensure!(
                len > 0 && u32::try_from(len).is_ok(),
                "mixed extent must be positive and fit a CUDA launch"
            );
            let inputs = graph.inputs(node);
            let source = matches!(
                graph[node],
                TensorGraphNode::Input { .. }
                    | TensorGraphNode::Constant { .. }
                    | TensorGraphNode::Parameter { .. }
            );
            let promotion = matches!(
                graph[node],
                TensorGraphNode::Cast {
                    dtype: ValueDType::F32,
                    ..
                }
            ) && dtypes.get(&inputs[0]) == Some(&ValueDType::BF16);
            if fuse_f32_regions && !source && !promotion && dtype == DType::F32 {
                f32_region.push(node);
                continue;
            }
            if fuse_f32_regions && !source && !promotion {
                result.f32_region(graph, &f32_region, precision)?;
                f32_region.clear();
            }
            match &graph[node] {
                TensorGraphNode::Input { .. }
                | TensorGraphNode::Parameter { .. }
                | TensorGraphNode::Constant { .. } => {
                    let (index, value) = result.alloc(node, shape, DType::F32)?;
                    result.sources.insert(node, index);
                    result.source_nodes.insert(node, graph[node].clone());
                    result.values.insert(node, value);
                }
                TensorGraphNode::Cast { .. } => {
                    let source = result
                        .values
                        .get(&inputs[0])
                        .context("mixed cast source missing")?
                        .clone();
                    if source.binding.dtype == dtype
                        || (dtype == DType::F32 && source.binding.dtype == DType::BF16)
                    {
                        result.values.insert(node, source);
                    } else {
                        let (_, target) = result.alloc(node, shape, dtype)?;
                        result
                            .kernels
                            .push(materialize(&result.stream, &source, target.binding)?);
                        result.values.insert(node, target);
                    }
                }
                n if is_view(n) => {
                    let mut value = result
                        .values
                        .get(&inputs[0])
                        .context("mixed view source missing")?
                        .clone();
                    value.view = match n {
                        TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => {
                            value.view.reshape(shape)?
                        }
                        TensorGraphNode::Permute { axes, .. } => value.view.permute(axes)?,
                        TensorGraphNode::Transpose { .. } => {
                            let mut axes: Vec<usize> = (0..shape.len()).collect();
                            ensure!(axes.len() >= 2, "mixed transpose requires rank >= 2");
                            let end = axes.len();
                            axes.swap(end - 2, end - 1);
                            value.view.permute(&axes)?
                        }
                        TensorGraphNode::BroadcastAxis { axis, .. } => {
                            value.view.broadcast_axis(*axis, shape[*axis])?
                        }
                        _ => unreachable!(),
                    };
                    value.chain.push(n.clone());
                    result.values.insert(node, value);
                }
                n => {
                    let sources = inputs
                        .iter()
                        .map(|input| {
                            result
                                .values
                                .get(input)
                                .cloned()
                                .context("mixed operand missing")
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mut operands = Vec::new();
                    for source in sources {
                        if source.binding.dtype == dtype || source.binding.dtype == DType::U32 {
                            operands.push(source);
                        } else {
                            let (_, target) =
                                result.alloc(node, source.view.shape.clone(), dtype)?;
                            result.kernels.push(materialize(
                                &result.stream,
                                &source,
                                target.binding,
                            )?);
                            operands.push(target);
                        }
                    }
                    let output = if dtype == DType::BF16 {
                        result.operation::<BF16>(node, n, &operands, precision)?
                    } else {
                        result.operation::<F32>(node, n, &operands, precision)?
                    };
                    result.values.insert(node, output);
                }
            }
        }
        result.f32_region(graph, &f32_region, precision)?;
        if fused_f32 {
            let mut plain = TensorGraph::new();
            plain.graph = graph.graph.clone();
            let bindings = result
                .sources
                .keys()
                .map(|node| (*node, result.values[node].binding))
                .collect();
            let program =
                Program::<F32>::new_bound(&result.stream, plain, bindings, None, precision)?;
            result.info.f32_bytes += program.bytes();
            result.kernels.extend(program.kernels.clone());
            for (&node, &binding) in &program.values {
                if let Some(view) = program.views.get(&node) {
                    result.values.insert(
                        node,
                        Value {
                            binding: *program
                                .values
                                .get(&view.source)
                                .context("fused view source missing")?,
                            view: view.clone(),
                            source_shape: graph[view.source].shape().clone(),
                            chain: Vec::new(),
                        },
                    );
                } else {
                    result.values.insert(
                        node,
                        Value {
                            binding,
                            view: VirtualTensor::identity(
                                node,
                                graph[node].shape().clone(),
                                DType::F32,
                            ),
                            source_shape: graph[node].shape().clone(),
                            chain: Vec::new(),
                        },
                    );
                }
            }
            result.programs.push(Box::new(program));
        }
        result.info.immutable_source_bytes = result
            .source_nodes
            .values()
            .filter_map(|n| {
                if let TensorGraphNode::Constant { data, .. } = n {
                    Some(data.len() * 4)
                } else {
                    None
                }
            })
            .sum();
        if reuse_intermediates {
            result.reuse_intermediates(graph)?;
        } else {
            result.info.unpooled_bytes = result.info.f32_bytes + result.info.bf16_bytes;
        }
        result.info.kernels = result.kernels.len();
        result.upload_impl(HashMap::new(), HashMap::new(), true)?;
        result.refresh_parameters_impl(&bound_parameters, true)?;
        // Constructors may have required inputs. Zero-initialized sources are
        // safe for warmup; no user-visible output is available until upload/run.
        for k in &result.kernels {
            k.run(&result.stream)?;
        }
        result.synchronize()?;
        if replay && !result.kernels.is_empty() {
            result
                .stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let launched = result
                .kernels
                .iter()
                .try_for_each(|k| k.run(&result.stream));
            let ended = result.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            launched?;
            result.captured = Some(Captured {
                graph: Some(ended?.context("empty mixed capture")?),
                stream: result.stream.clone(),
            });
        } else {
            result.info.replay = false;
        }
        result.uploaded = false;
        Ok(result)
    }

    fn alloc(
        &mut self,
        node: NodeIndex,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<(usize, Value)> {
        let len = shape.iter().product::<usize>();
        let buffer = match dtype {
            DType::F32 => {
                self.info.f32_bytes += len * 4;
                Buffer::F32(self.stream.alloc_zeros(len)?)
            }
            DType::BF16 => {
                self.info.bf16_bytes += len * 2;
                Buffer::BF16(self.stream.alloc_zeros(len)?)
            }
            _ => anyhow::bail!("unsupported mixed storage"),
        };
        let value = Value {
            binding: binding(&buffer, &self.stream),
            view: VirtualTensor::identity(node, shape.clone(), dtype),
            source_shape: shape,
            chain: Vec::new(),
        };
        let index = self.buffers.len();
        self.buffers.push(buffer);
        Ok((index, value))
    }

    fn operation<D: CudaDType>(
        &mut self,
        node: NodeIndex,
        op: &TensorGraphNode<f32>,
        operands: &[Value],
        precision: MatMulPrecision,
    ) -> Result<Value> {
        if D::HostType::TILE_DTYPE == DType::BF16
            && let Some(output) = self.split_bf16_matmul(node, op, operands, precision)?
        {
            return Ok(output);
        }
        let mut graph = TensorGraph::<D::HostType>::new();
        let mut inputs = HashMap::new();
        let mut ends = Vec::new();
        for source in operands {
            let n = if source.binding.dtype == DType::U32 {
                let TensorGraphNode::IndexSource { source: index } =
                    &self.index_graph[source.view.source]
                else {
                    anyhow::bail!("integer binding source missing")
                };
                TensorGraphNode::IndexSource {
                    source: index.clone(),
                }
            } else {
                TensorGraphNode::Input {
                    name: "boundary",
                    shape: source.source_shape.clone(),
                }
            };
            let mut end = graph.graph.add_node(n);
            inputs.insert(end, source.binding);
            for view in &source.chain {
                let next = graph.graph.add_node(typed_node(view)?);
                graph.graph.add_edge(end, next, 0);
                end = next;
            }
            ends.push(end);
        }
        let root = if matches!(op, TensorGraphNode::Linear { .. }) {
            let a = crate::tensor::TensorExpr::node_ref(ends[0], operands[0].view.shape.clone());
            let b = crate::tensor::TensorExpr::node_ref(ends[1], operands[1].view.shape.clone());
            let bias = crate::tensor::TensorExpr::node_ref(ends[2], operands[2].view.shape.clone());
            (a.matmul(b)
                + bias
                    .reshape(vec![1, op.shape()[1]])
                    .broadcast(op.shape().clone()))
            .lower_to_graph(&mut graph)
        } else {
            let root = graph.graph.add_node(typed_node(op)?);
            for (slot, end) in ends.into_iter().enumerate() {
                graph.graph.add_edge(end, root, slot);
            }
            root
        };
        let _ = root;
        let mut program = Program::<D>::new_bound(&self.stream, graph, inputs, None, precision)?;
        ensure!(
            !matches!(op, TensorGraphNode::Linear { .. }) || program.root_fused_matmul,
            "linear bias must fuse before BF16 output rounding"
        );
        for kernel in &mut program.kernels {
            kernel.label = format!("node={} {} {}", node.index(), op.name(), kernel.label);
        }
        self.info.bf16_tensor_core_programs +=
            usize::from(program.ptx.contains("f32.bf16.bf16.f32"));
        match D::HostType::TILE_DTYPE {
            DType::F32 => self.info.f32_bytes += program.bytes(),
            DType::BF16 => self.info.bf16_bytes += program.bytes(),
            _ => anyhow::bail!("unsupported mixed program dtype"),
        }
        let value = Value {
            binding: program.output,
            view: VirtualTensor::identity(node, op.shape().clone(), D::HostType::TILE_DTYPE),
            source_shape: op.shape().clone(),
            chain: Vec::new(),
        };
        self.kernels.extend(program.kernels.clone());
        self.programs.push(Box::new(program));
        Ok(value)
    }

    pub fn upload_with_indices(
        &mut self,
        inputs: HashMap<String, Vec<f32>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<()> {
        self.upload_impl(inputs, indices, false)
    }

    pub(crate) fn for_resident<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
        bindings: ResidentBindings,
    ) -> Result<Self> {
        Self::prepare(graph, precision, replay, false, true, Some(bindings))
    }
    pub(crate) fn execution_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
    pub(crate) fn device_binding(
        &self,
        node: NodeIndex,
    ) -> Result<(Binding, VirtualTensor, Vec<usize>)> {
        ensure!(self.has_run, "mixed graph has not run");
        let value = self
            .values
            .get(&node)
            .context("mixed gradient not retained")?;
        ensure!(
            matches!(value.binding.dtype, DType::F32 | DType::BF16)
                && value.binding.dtype == value.view.dtype,
            "resident gradient storage/view mismatch"
        );
        Ok((
            value.binding,
            value.view.clone(),
            value.source_shape.clone(),
        ))
    }
    pub(crate) fn refresh_parameters(
        &mut self,
        parameters: &HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
    ) -> Result<()> {
        self.refresh_parameters_impl(parameters, false)
    }
    // `preparing` is used only by prepare while holding the stream gate.
    fn refresh_parameters_impl(
        &mut self,
        parameters: &HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
        preparing: bool,
    ) -> Result<()> {
        let began = std::time::Instant::now();
        ensure!(
            parameters.len() == self.bound_parameters.len(),
            "resident parameter set changed"
        );
        for (&node, source) in parameters {
            ensure!(
                self.bound_parameters.contains_key(&node) && self.sources.contains_key(&node),
                "unexpected resident parameter"
            );
            ensure!(
                Arc::ptr_eq(source.context(), self.stream.context())
                    && source.len() == self.values[&node].binding.len,
                "resident parameter context/extent mismatch"
            );
        }
        let gate = self.gate.clone();
        let _guard = if preparing {
            None
        } else {
            Some(
                gate.lock()
                    .map_err(|_| anyhow::anyhow!("mixed stream lock poisoned"))?,
            )
        };
        for (&node, source) in parameters {
            let previous = self
                .bound_parameters
                .get_mut(&node)
                .context("resident source missing")?;
            let identity = Arc::downgrade(source);
            if !Weak::ptr_eq(previous, &identity) {
                let Buffer::F32(target) = &mut self.buffers[self.sources[&node]] else {
                    anyhow::bail!("resident master not f32")
                };
                self.stream.memcpy_dtod(source.as_ref(), target)?;
                self.transfers.parameter_copies += 1;
                self.transfers.parameter_bytes += source.len() * 4;
                *previous = identity;
            }
        }
        self.transfers.parameter_refresh_seconds = began.elapsed().as_secs_f64();
        Ok(())
    }

    /// Validate every input and parameter before replacing any device source.
    pub fn upload(&mut self, inputs: HashMap<String, Vec<f32>>) -> Result<()> {
        self.upload_impl(inputs, HashMap::new(), false)
    }
    fn upload_impl(
        &mut self,
        inputs: HashMap<String, Vec<f32>>,
        mut indices: crate::tensor::IndexInputs,
        allow_missing_inputs: bool,
    ) -> Result<()> {
        let began = std::time::Instant::now();
        if allow_missing_inputs {
            for n in self.index_graph.graph.node_weights() {
                if let TensorGraphNode::IndexSource { source } = n
                    && let crate::tensor::indices::IndexSource::Input(name) = source.source()
                {
                    indices.insert((*name).into(), vec![0; source.shape().iter().product()]);
                }
            }
        }
        let index_values = self.index_plan.prepare(&self.index_graph, indices)?;
        let mut copies = Vec::new();
        for (&node, &index) in &self.sources {
            if self.constants_initialized
                && matches!(self.source_nodes[&node], TensorGraphNode::Constant { .. })
            {
                continue;
            }
            if self.bound_parameters.contains_key(&node) {
                continue;
            }
            let data = match &self.source_nodes[&node] {
                TensorGraphNode::Input { name, .. } => match inputs.get(*name) {
                    Some(data) => data.clone(),
                    None if allow_missing_inputs => continue,
                    None => anyhow::bail!("mixed input {name} missing"),
                },
                TensorGraphNode::Constant { data, .. } => data.as_ref().clone(),
                TensorGraphNode::Parameter { data, .. } => data.lock()?.to_vec(),
                _ => anyhow::bail!("mixed source graph changed"),
            };
            ensure!(
                data.len() == self.values[&node].binding.len,
                "mixed source length mismatch"
            );
            copies.push((index, data));
        }
        let gate = self.gate.clone();
        // Missing feeds are permitted only during gated construction warmup.
        let _guard = if allow_missing_inputs {
            None
        } else {
            Some(
                gate.lock()
                    .map_err(|_| anyhow::anyhow!("mixed stream lock poisoned"))?,
            )
        };
        self.synchronize()?;
        self.uploaded = false;
        self.has_run = false;
        self.transfers = MixedTransfers::default();
        for (index, data) in copies {
            if let Buffer::F32(buffer) = &mut self.buffers[index] {
                self.stream.memcpy_htod(&data, buffer)?;
                self.transfers.f32_uploads += 1;
                self.transfers.f32_bytes += data.len() * 4;
            }
        }
        for (node, data) in index_values {
            self.transfers.index_uploads += 1;
            self.transfers.index_bytes += data.len() * 4;
            self.stream.memcpy_htod(
                data.as_slice(),
                self.index_buffers
                    .get_mut(&node)
                    .context("integer allocation missing")?,
            )?;
        }
        self.uploaded = true;
        self.constants_initialized = true;
        self.transfers.upload_seconds = began.elapsed().as_secs_f64();
        Ok(())
    }
    pub fn run(&mut self) -> Result<()> {
        let gate = self.gate.clone();
        let _guard = gate
            .lock()
            .map_err(|_| anyhow::anyhow!("mixed stream lock poisoned"))?;
        ensure!(self.uploaded, "mixed inputs have not been uploaded");
        if let Some(captured) = &self.captured {
            captured
                .graph
                .as_ref()
                .context("mixed capture missing")?
                .launch()?;
        } else {
            for k in &self.kernels {
                k.run(&self.stream)?;
            }
        }
        self.has_run = true;
        Ok(())
    }
    pub fn read(&self, node: NodeIndex) -> Result<Vec<f32>> {
        let gate = self.gate.clone();
        let _guard = gate
            .lock()
            .map_err(|_| anyhow::anyhow!("mixed stream lock poisoned"))?;
        ensure!(self.has_run, "mixed graph has not run");
        let value = self
            .values
            .get(&node)
            .with_context(|| format!("mixed read node {} missing", node.index()))?;
        let mut readbacks = self.readbacks.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(entry) = readbacks.entry(node) {
            let output = self
                .stream
                .alloc_zeros::<f32>(value.view.shape.iter().product())?;
            let target = Binding {
                pointer: output.device_ptr(&self.stream).0,
                len: output.len(),
                dtype: DType::F32,
            };
            let kernel = materialize(&self.stream, value, target)?;
            entry.insert(Readback {
                buffer: output,
                kernel,
            });
        }
        let readback = readbacks
            .get(&node)
            .context("mixed readback cache missing")?;
        readback.kernel.run(&self.stream)?;
        let mut host = vec![0.; readback.buffer.len()];
        self.stream.memcpy_dtoh(&readback.buffer, &mut host)?;
        self.synchronize()?;
        Ok(host)
    }
    pub fn gradients(
        &self,
        graph: &TensorGraph<f32, WithGrad>,
    ) -> Result<HashMap<usize, Vec<f32>>> {
        graph
            .gradient_metadata()
            .param_to_grad
            .iter()
            .map(|(&id, &node)| Ok((id, self.read(node)?)))
            .collect()
    }
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(Into::into)
    }
    /// Ordered captured kernel labels for matching a CUDA graph node trace.
    pub fn kernel_labels(&self) -> Vec<&str> {
        self.kernels
            .iter()
            .map(|kernel| kernel.label.as_str())
            .collect()
    }

    pub fn transfers(&self) -> &MixedTransfers {
        &self.transfers
    }
    pub fn info(&self) -> &MixedInfo {
        &self.info
    }
    pub fn time_runs(&mut self, repeats: usize) -> Result<f64> {
        ensure!(repeats > 0, "mixed timing needs positive repeats");
        let begin = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..repeats {
            self.run()?;
        }
        let end = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        Ok(f64::from(begin.elapsed_ms(&end)?) * 1000. / repeats as f64)
    }
}

fn is_view(n: &TensorGraphNode<f32>) -> bool {
    matches!(
        n,
        TensorGraphNode::Reshape { .. }
            | TensorGraphNode::Flatten { .. }
            | TensorGraphNode::Transpose { .. }
            | TensorGraphNode::Permute { .. }
            | TensorGraphNode::BroadcastAxis { .. }
    )
}

fn typed_node<D>(n: &TensorGraphNode<f32>) -> Result<TensorGraphNode<D>> {
    Ok(match n {
        TensorGraphNode::Embedding { shape } => TensorGraphNode::Embedding {
            shape: shape.clone(),
        },
        TensorGraphNode::EmbeddingBackward { shape } => TensorGraphNode::EmbeddingBackward {
            shape: shape.clone(),
        },
        TensorGraphNode::IndexedCrossEntropy { shape } => TensorGraphNode::IndexedCrossEntropy {
            shape: shape.clone(),
        },
        TensorGraphNode::IndexedCrossEntropyBackward { shape } => {
            TensorGraphNode::IndexedCrossEntropyBackward {
                shape: shape.clone(),
            }
        }
        TensorGraphNode::Concat { axis, shape } => TensorGraphNode::Concat {
            axis: *axis,
            shape: shape.clone(),
        },
        TensorGraphNode::Unary { op, shape } => TensorGraphNode::Unary {
            op: *op,
            shape: shape.clone(),
        },
        TensorGraphNode::Binary { op, shape } => TensorGraphNode::Binary {
            op: *op,
            shape: shape.clone(),
        },
        TensorGraphNode::MatMul { shape } => TensorGraphNode::MatMul {
            shape: shape.clone(),
        },
        TensorGraphNode::ReduceAxis { op, axis, shape } => TensorGraphNode::ReduceAxis {
            op: *op,
            axis: *axis,
            shape: shape.clone(),
        },
        TensorGraphNode::Reshape { shape } => TensorGraphNode::Reshape {
            shape: shape.clone(),
        },
        TensorGraphNode::Flatten { shape } => TensorGraphNode::Flatten {
            shape: shape.clone(),
        },
        TensorGraphNode::Transpose { shape } => TensorGraphNode::Transpose {
            shape: shape.clone(),
        },
        TensorGraphNode::Permute { axes, shape } => TensorGraphNode::Permute {
            axes: axes.clone(),
            shape: shape.clone(),
        },
        TensorGraphNode::BroadcastAxis { axis, shape } => TensorGraphNode::BroadcastAxis {
            axis: *axis,
            shape: shape.clone(),
        },
        TensorGraphNode::Gt { shape } => TensorGraphNode::Gt {
            shape: shape.clone(),
        },
        TensorGraphNode::Mask { shape } => TensorGraphNode::Mask {
            shape: shape.clone(),
        },
        TensorGraphNode::Narrow { axis, start, shape } => TensorGraphNode::Narrow {
            axis: *axis,
            start: *start,
            shape: shape.clone(),
        },
        _ => anyhow::bail!("mixed operator {} is not supported yet", n.name()),
    })
}

fn materialize(stream: &Arc<CudaStream>, source: &Value, target: Binding) -> Result<Kernel> {
    ensure!(
        source.view.shape.iter().product::<usize>() == target.len,
        "mixed materialization extent mismatch"
    );
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    let node = tiles.graph.add_node(TileIRBuilder::new().finish());
    tiles.set_materialized_view_dtype(node, &source.view, &source.source_shape, target.dtype)?;
    tiles.optimize_indices()?;
    let graph = PtxGraph::from(tiles).with_target(PtxTarget::from_context(stream.context())?);
    let name = graph
        .kernel_name(node)
        .context("mixed cast kernel absent")?;
    let module = stream
        .context()
        .load_module(Ptx::from_src(graph.to_ptx()))?;
    Ok(Kernel {
        label: format!(
            "materialize {:?} to {:?} shape={:?}",
            source.binding.dtype, target.dtype, source.view.shape
        ),
        function: module.load_function(name)?,
        config: LaunchConfig::for_num_elems(u32::try_from(target.len)?),
        pointers: vec![source.binding.pointer, target.pointer],
        zero_before: None,
    })
}

#[cfg(test)]
mod tests;
