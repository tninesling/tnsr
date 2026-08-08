use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use num_traits::Float;

pub mod alloc;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod data;
pub mod graph;
pub mod nn;
pub mod optimizer;
#[cfg(feature = "cuda")]
pub mod ptx;
pub mod runtime;
pub mod tensor;
pub mod tile;

pub use runtime::{Backend, Runtime};

use crate::alloc::{AllocStats, BufferPool};
use crate::graph::{TensorGraph, TensorGraphNode, WithGrad, liveness};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::trace_span;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

#[cfg(feature = "parallel")]
pub type SliceIter<'a, T> = rayon::iter::MinLen<rayon::slice::Iter<'a, T>>;
#[cfg(not(feature = "parallel"))]
pub type SliceIter<'a, T> = std::slice::Iter<'a, T>;

#[cfg(feature = "parallel")]
pub type SliceIterMut<'a, T> = rayon::iter::MinLen<rayon::slice::IterMut<'a, T>>;
#[cfg(not(feature = "parallel"))]
pub type SliceIterMut<'a, T> = std::slice::IterMut<'a, T>;

#[cfg(feature = "parallel")]
mod parallel_config {
    /// Minimum number of elements for parallel iteration
    pub const ELEMENTWISE_THRESHOLD: usize = 8192;

    /// Minimum matrix dimension for parallel matmul
    pub const MATMUL_THRESHOLD: usize = 128;
}

/// RAII guard for Chrome tracing session.
///
/// Ensures trace data is properly flushed to disk when dropped.
/// Created by [`init_chrome_tracing`].
pub struct TracingGuard {
    _guard: tracing_chrome::FlushGuard,
}

/// Initialize Chrome tracing and write output to the specified file.
///
/// Returns a [`TracingGuard`] that must be kept alive for the duration of tracing.
/// When dropped, all trace data is flushed to the file.
pub fn init_chrome_tracing(file_path: &str) -> Result<TracingGuard, Box<dyn std::error::Error>> {
    let (chrome_layer, guard) = ChromeLayerBuilder::new().file(file_path).build();

    tracing_subscriber::registry().with(chrome_layer).init();

    Ok(TracingGuard { _guard: guard })
}

/// Executes computation graphs and provides gradient access.
///
/// Implementors provide different execution strategies (CPU, GPU, etc.)
/// while maintaining a common interface for graph evaluation.
pub trait Executor<D> {
    /// Execute a computation graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph to execute
    /// * `inputs` - Named input tensors as flat vectors
    ///
    /// # Returns
    ///
    /// The output tensor as a flat vector, or an error if execution fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Required inputs are missing
    /// - Shapes are incompatible
    /// - Memory allocation fails
    /// - Invalid operations are encountered
    fn execute<G>(
        &mut self,
        graph: &TensorGraph<D, G>,
        inputs: HashMap<String, Vec<D>>,
    ) -> Result<Vec<D>>
    where
        TensorGraph<D, G>: Clone,
        TensorGraph<f32, G>: Clone;

    /// Retrieve computed gradients for parameters.
    ///
    /// Gradient buffers remain available until the next call to
    /// [`Executor::execute`]. Concrete executors may also provide an explicit
    /// `release_gradients` method to return them to their buffer pool sooner.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph with gradient metadata
    ///
    /// # Returns
    ///
    /// A map from parameter IDs to their gradient vectors.
    ///
    /// # Panics
    ///
    /// Panics if the graph does not have gradient metadata or if gradient values
    /// are missing from the executor's computed values.
    fn get_gradients(&self, graph: &TensorGraph<D, WithGrad>) -> HashMap<usize, Vec<D>>;
}

/// CPU-based executor for tensor computation graphs.
///
/// Implements the [`Executor`] trait using CPU operations. When the `parallel`
/// feature is enabled, uses Rayon for parallelism on larger tensors.
///
/// Suitable for testing, small models, and platforms without GPU support.
///
/// # Example
///
/// ```rust
/// use tnsr::{SimpleExecutor, Executor};
/// use tnsr::tensor::TensorExpr;
/// use tnsr::graph::TensorGraph;
/// use std::collections::HashMap;
///
/// let mut executor = SimpleExecutor::new();
/// let x = TensorExpr::<f32>::input("x", vec![2, 3]);
/// let graph: TensorGraph<f32> = x.into();
///
/// let mut inputs = HashMap::new();
/// inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
///
/// let result = executor.execute(&graph, inputs).unwrap();
/// assert_eq!(result.len(), 6);
/// ```
/// A node value held by an executor.
///
/// Computed intermediates are [`Value::Owned`] buffers sourced from the
/// executor's [`BufferPool`] and returned to it once liveness analysis shows
/// they are dead. Graph- and caller-owned data is referenced without copying.
enum Value<D> {
    /// Computed intermediate; returned to the buffer pool once dead.
    Owned(Vec<D>),
    /// Constant data shared with the graph.
    Constant(Arc<Vec<D>>),
    /// Parameter data shared with the graph (and the optimizer).
    Parameter(Arc<Mutex<Vec<D>>>),
}

impl<D> Value<D> {
    /// Run `f` with the value's elements.
    fn with<R>(&self, f: impl FnOnce(&[D]) -> R) -> R {
        match self {
            Value::Owned(buf) => f(buf),
            Value::Constant(data) => f(data),
            Value::Parameter(data) => f(&data.lock().unwrap()),
        }
    }

    /// Number of elements in the value.
    fn len(&self) -> usize {
        self.with(|v| v.len())
    }

    /// Run `f` with two values without locking the same parameter twice.
    fn with_pair<R>(&self, other: &Self, f: impl FnOnce(&[D], &[D]) -> R) -> R {
        match (self, other) {
            (Value::Parameter(a), Value::Parameter(b)) if Arc::ptr_eq(a, b) => {
                let data = a.lock().unwrap();
                f(&data, &data)
            }
            _ => self.with(|a| other.with(|b| f(a, b))),
        }
    }
}

pub struct SimpleExecutor<D = f32> {
    values: HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: BufferPool<D>,
    stats: AllocStats,
}

impl<D> SimpleExecutor<D> {
    /// Create a new CPU executor.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            pool: BufferPool::default(),
            stats: AllocStats::default(),
        }
    }

    /// Retrieve the computed value for a specific graph node.
    ///
    /// Only pinned values (sinks such as the execution output and gradient
    /// nodes) are guaranteed to remain available after execution;
    /// intermediate values are released once their last consumer has
    /// executed. Useful for debugging.
    pub fn get_value(&self, node_idx: petgraph::graph::NodeIndex) -> Option<Vec<D>>
    where
        D: Clone,
    {
        self.values.get(&node_idx).map(|v| v.with(|s| s.to_vec()))
    }

    /// Memory allocation statistics gathered during execution.
    ///
    /// Counters are cumulative across executions; call
    /// [`SimpleExecutor::reset_stats`] to start fresh.
    pub fn stats(&self) -> &AllocStats {
        &self.stats
    }

    /// Reset allocation counters without discarding pooled buffers.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.pool.reset_stats();
    }

    /// Discard cached buffers to release memory held by the executor.
    pub fn clear_pool(&mut self) {
        self.pool.clear();
    }

    /// Release pinned gradient buffers before the next execution.
    pub fn release_gradients(&mut self, graph: &TensorGraph<D, WithGrad>)
    where
        D: Clone,
    {
        for grad_node in graph.gradient_metadata().param_to_grad.values() {
            if let Some(Value::Owned(buf)) = self.values.remove(grad_node) {
                self.pool.give(buf);
            }
        }
    }
}

impl<D> Default for SimpleExecutor<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Float + Send + Sync> SimpleExecutor<D> {
    fn apply_unary(op: &tensor::UnaryOp, v: D) -> D {
        match op {
            tensor::UnaryOp::Neg => -v,
            tensor::UnaryOp::Exp => v.exp(),
            tensor::UnaryOp::Log => v.ln(),
            tensor::UnaryOp::Relu => v.max(D::zero()),
        }
    }

    fn apply_binary(op: &tensor::BinaryOp, x: D, y: D) -> D {
        match op {
            tensor::BinaryOp::Add => x + y,
            tensor::BinaryOp::Sub => x - y,
            tensor::BinaryOp::Mul => x * y,
            tensor::BinaryOp::Div => x / y,
        }
    }

    /// Apply a unary op elementwise, writing into `out`.
    fn unary(&self, op: &tensor::UnaryOp, x: &[D], out: &mut [D]) {
        let _span = trace_span!("unary", op = ?op).entered();
        get_iter_mut(out)
            .zip(get_iter(x))
            .for_each(|(o, &v)| *o = Self::apply_unary(op, v));
    }

    /// Apply a unary op elementwise in place (used for fused op sequences).
    #[cfg(feature = "fusion")]
    fn unary_inplace(&self, op: &tensor::UnaryOp, buf: &mut [D]) {
        let _span = trace_span!("unary_inplace", op = ?op).entered();
        get_iter_mut(buf).for_each(|v| *v = Self::apply_unary(op, *v));
    }

    /// Apply a binary op elementwise, writing into `out`.
    fn binary(&self, op: &tensor::BinaryOp, a: &[D], b: &[D], out: &mut [D]) {
        let _span = trace_span!("binary", op = ?op).entered();
        get_iter_mut(out)
            .zip(get_iter(a))
            .zip(get_iter(b))
            .for_each(|((o, &x), &y)| *o = Self::apply_binary(op, x, y));
    }

    fn gt(&self, a: &[D], b: &[D], out: &mut [D]) {
        let _span = trace_span!("gt").entered();
        get_iter_mut(out)
            .zip(get_iter(a))
            .zip(get_iter(b))
            .for_each(|((o, &x), &y)| *o = if x > y { D::one() } else { D::zero() });
    }

    fn mask(&self, values: &[D], condition: &[D], out: &mut [D]) {
        let _span = trace_span!("mask").entered();
        get_iter_mut(out)
            .zip(get_iter(values))
            .zip(get_iter(condition))
            .for_each(|((o, &v), &c)| *o = if c != D::zero() { v } else { D::zero() });
    }
}

impl<D> Executor<D> for SimpleExecutor<D>
where
    D: Float + Send + Sync,
{
    fn execute<G>(
        &mut self,
        graph: &TensorGraph<D, G>,
        mut inputs: HashMap<String, Vec<D>>,
    ) -> Result<Vec<D>> {
        let order = graph.toposort();
        let liveness = liveness::analyze(graph, &order);

        // Return buffers held by prior executions to the pool.
        for (_, value) in self.values.drain() {
            if let Value::Owned(buf) = value {
                self.pool.give(buf);
            }
        }
        let mut live_bytes = 0usize;

        for (pos, node_idx) in order.iter().enumerate() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data, .. } => {
                    let _span = trace_span!("constant", node = node_idx.index()).entered();
                    Value::Constant(data.clone())
                }
                TensorGraphNode::Input { name, .. } => {
                    let _span =
                        trace_span!("input", node = node_idx.index(), name = name).entered();
                    // Input nodes are deduplicated by name at lowering time,
                    // so each input is moved (not copied) exactly once.
                    let data = inputs
                        .remove(*name)
                        .with_context(|| format!("Input '{}' not found", name))?;
                    Value::Owned(data)
                }
                TensorGraphNode::Parameter { data, .. } => {
                    let _span = trace_span!("parameter", node = node_idx.index()).entered();
                    Value::Parameter(data.clone())
                }
                TensorGraphNode::Unary { op, .. } => {
                    let inputs = graph.inputs(*node_idx);
                    let x = self.values.get(&inputs[0]).with_context(|| {
                        format!("Value for node {} not computed", inputs[0].index())
                    })?;
                    let mut out = self.pool.take(x.len());
                    x.with(|x| self.unary(op, x, &mut out));
                    Value::Owned(out)
                }
                #[cfg(feature = "fusion")]
                TensorGraphNode::FusedUnary { ops, .. } => {
                    let inputs = graph.inputs(*node_idx);
                    let x = self.values.get(&inputs[0]).with_context(|| {
                        format!("Value for node {} not computed", inputs[0].index())
                    })?;
                    // Apply each unary operation in sequence, in place.
                    let mut out = self.pool.take(x.len());
                    x.with(|x| out.copy_from_slice(x));
                    for op in ops {
                        self.unary_inplace(op, &mut out);
                    }
                    Value::Owned(out)
                }
                TensorGraphNode::Binary { op, .. } => {
                    let ins = graph.inputs(*node_idx);
                    let a = self.values.get(&ins[0]).with_context(|| {
                        format!("Value for node {} not computed", ins[0].index())
                    })?;
                    let b = self.values.get(&ins[1]).with_context(|| {
                        format!("Value for node {} not computed", ins[1].index())
                    })?;
                    anyhow::ensure!(
                        a.len() == b.len(),
                        "Binary op shape mismatch: {} vs {}",
                        a.len(),
                        b.len()
                    );
                    let mut out = self.pool.take(a.len());
                    a.with_pair(b, |a, b| self.binary(op, a, b, &mut out));
                    Value::Owned(out)
                }
                TensorGraphNode::MatMul { .. } => {
                    let _span = trace_span!("matmul", node = node_idx.index()).entered();
                    Value::Owned(matmul_forward(
                        graph,
                        &self.values,
                        &mut self.pool,
                        *node_idx,
                    )?)
                }
                TensorGraphNode::Transpose { .. } => {
                    let _span = trace_span!("transpose", node = node_idx.index()).entered();
                    Value::Owned(transpose_forward(
                        graph,
                        &self.values,
                        &mut self.pool,
                        *node_idx,
                    )?)
                }
                TensorGraphNode::Reshape { .. } => {
                    let _span = trace_span!("reshape", node = node_idx.index()).entered();
                    let ins = graph.inputs(*node_idx);
                    let x = self.values.get(&ins[0]).with_context(|| {
                        format!("Value for node {} not computed", ins[0].index())
                    })?;
                    let mut out = self.pool.take(x.len());
                    x.with(|v| out.copy_from_slice(v));
                    Value::Owned(out)
                }
                TensorGraphNode::Permute { axes, .. } => {
                    let _span = trace_span!("permute", node = node_idx.index()).entered();
                    Value::Owned(permute_forward(
                        graph,
                        &self.values,
                        &mut self.pool,
                        *node_idx,
                        axes,
                    )?)
                }
                TensorGraphNode::BroadcastAxis { axis, .. } => {
                    let _span = trace_span!("broadcast_axis", node = node_idx.index()).entered();
                    Value::Owned(broadcast_axis_forward(
                        graph,
                        &self.values,
                        &mut self.pool,
                        *node_idx,
                        *axis,
                    )?)
                }
                TensorGraphNode::ReduceAxis { op, axis, .. } => {
                    let _span = trace_span!(
                        "reduce_axis",
                        op = node.name(),
                        axis = *axis,
                        node = node_idx.index()
                    )
                    .entered();
                    Value::Owned(reduce_axis_forward(
                        graph,
                        &self.values,
                        &mut self.pool,
                        *node_idx,
                        op,
                        *axis,
                    )?)
                }
                TensorGraphNode::Gt { .. } => {
                    let _span = trace_span!("gt", node = node_idx.index()).entered();
                    let ins = graph.inputs(*node_idx);
                    let a = self.values.get(&ins[0]).with_context(|| {
                        format!("Value for node {} not computed", ins[0].index())
                    })?;
                    let b = self.values.get(&ins[1]).with_context(|| {
                        format!("Value for node {} not computed", ins[1].index())
                    })?;
                    anyhow::ensure!(
                        a.len() == b.len(),
                        "Gt op shape mismatch: {} vs {}",
                        a.len(),
                        b.len()
                    );
                    let mut out = self.pool.take(a.len());
                    a.with_pair(b, |a, b| self.gt(a, b, &mut out));
                    Value::Owned(out)
                }
                TensorGraphNode::Mask { .. } => {
                    let _span = trace_span!("mask", node = node_idx.index()).entered();
                    let ins = graph.inputs(*node_idx);
                    let values = self.values.get(&ins[0]).with_context(|| {
                        format!("Value for node {} not computed", ins[0].index())
                    })?;
                    let condition = self.values.get(&ins[1]).with_context(|| {
                        format!("Condition for node {} not computed", ins[1].index())
                    })?;
                    anyhow::ensure!(
                        values.len() == condition.len(),
                        "Mask op shape mismatch: {} vs {}",
                        values.len(),
                        condition.len()
                    );
                    let mut out = self.pool.take(values.len());
                    values.with_pair(condition, |v, c| self.mask(v, c, &mut out));
                    Value::Owned(out)
                }
                TensorGraphNode::Conv2d {
                    stride, padding, ..
                } => {
                    let _span = trace_span!("conv2d", node = node_idx.index()).entered();
                    Value::Owned(conv2d_forward(
                        graph,
                        &self.values,
                        &mut self.pool,
                        *node_idx,
                        *stride,
                        *padding,
                    )?)
                }
                TensorGraphNode::ConvTranspose2d {
                    stride, padding, ..
                } => {
                    let _span = trace_span!("conv_transpose_2d", node = node_idx.index()).entered();
                    Value::Owned(conv_transpose_2d_forward(
                        graph,
                        &self.values,
                        &mut self.pool,
                        *node_idx,
                        *stride,
                        *padding,
                    )?)
                }
                TensorGraphNode::Conv2dBackwardWeight {
                    stride, padding, ..
                } => {
                    let _span =
                        trace_span!("conv2d_backward_weight", node = node_idx.index()).entered();
                    Value::Owned(conv2d_backward_weight_forward(
                        graph,
                        &self.values,
                        &mut self.pool,
                        *node_idx,
                        *stride,
                        *padding,
                    )?)
                }
                TensorGraphNode::MaxPool2d {
                    kernel_size,
                    stride,
                    ..
                } => {
                    let _span = trace_span!("max_pool2d", node = node_idx.index()).entered();
                    Value::Owned(max_pool2d_forward(
                        graph,
                        &self.values,
                        &mut self.pool,
                        *node_idx,
                        *kernel_size,
                        *stride,
                    )?)
                }
                TensorGraphNode::MaxPool2dBackward {
                    kernel_size,
                    stride,
                    ..
                } => {
                    let _span =
                        trace_span!("max_pool2d_backward", node = node_idx.index()).entered();
                    Value::Owned(max_pool2d_backward_forward(
                        graph,
                        &self.values,
                        &mut self.pool,
                        *node_idx,
                        *kernel_size,
                        *stride,
                    )?)
                }
                TensorGraphNode::Flatten { .. } => {
                    let _span = trace_span!("flatten", node = node_idx.index()).entered();
                    let ins = graph.inputs(*node_idx);
                    let x = self.values.get(&ins[0]).with_context(|| {
                        format!("Value for node {} not computed", ins[0].index())
                    })?;
                    let mut out = self.pool.take(x.len());
                    x.with(|v| out.copy_from_slice(v));
                    Value::Owned(out)
                }
            };
            let bytes = result.len() * std::mem::size_of::<D>();
            self.values.insert(*node_idx, result);
            live_bytes += bytes;
            self.stats.record_live(live_bytes);

            // Release values whose last use was this node.
            for &dead in liveness.free_after(pos) {
                if let Some(value) = self.values.remove(&dead) {
                    live_bytes -= value.len() * std::mem::size_of::<D>();
                    if let Value::Owned(buf) = value {
                        self.pool.give(buf);
                    }
                }
            }
        }

        // Sync cumulative pool counters into the public stats.
        self.stats = self.stats.with_pool_stats(self.pool.stats());
        tracing::debug!(
            bytes_allocated = self.stats.bytes_allocated,
            buffers_allocated = self.stats.buffers_allocated,
            peak_live_bytes = self.stats.peak_live_bytes,
            pool_hits = self.stats.pool_hits,
            pool_misses = self.stats.pool_misses,
            "execute memory stats"
        );

        let last_node = order
            .last()
            .context("Graph is empty, no nodes to execute")?;
        let out = self
            .values
            .get(last_node)
            .context("Output value not computed")?;
        Ok(out.with(|v| v.to_vec()))
    }

    fn get_gradients(&self, graph: &TensorGraph<D, WithGrad>) -> HashMap<usize, Vec<D>> {
        let mut result = HashMap::new();

        // Iterate through all parameters in the gradient metadata
        for (param_id, grad_node_idx) in &graph.gradient_metadata().param_to_grad {
            // Look up the gradient value from our computed values
            if let Some(grad_value) = self.values.get(grad_node_idx) {
                result.insert(*param_id, grad_value.with(|v| v.to_vec()));
            }
        }

        result
    }
}

fn rowmajor_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut s = vec![0usize; shape.len()];
    s[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn matmul_forward<D, G>(
    graph: &TensorGraph<D, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: &mut BufferPool<D>,
    node_idx: petgraph::graph::NodeIndex,
) -> Result<Vec<D>>
where
    D: Float + Send + Sync,
{
    let inputs_idx = graph.inputs(node_idx);
    let a_idx = inputs_idx[0];
    let b_idx = inputs_idx[1];
    let a_val = values
        .get(&a_idx)
        .context("Left operand value not computed for matmul")?;
    let b_val = values
        .get(&b_idx)
        .context("Right operand value not computed for matmul")?;
    let a_shape = graph.graph[a_idx].shape();
    let b_shape = graph.graph[b_idx].shape();
    anyhow::ensure!(
        a_shape.len() >= 2 && b_shape.len() >= 2,
        "Matmul operands must have rank >= 2, got {}D and {}D",
        a_shape.len(),
        b_shape.len()
    );
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 1];
    anyhow::ensure!(
        k == b_shape[b_shape.len() - 2],
        "Matmul inner dimensions must match: {a_shape:?} and {b_shape:?}"
    );
    let out_shape = graph.graph[node_idx].shape();
    let out_batch_shape = &out_shape[..out_shape.len() - 2];
    let batch_count: usize = out_batch_shape.iter().product();

    let mut out = pool.take(batch_count * m * n);
    a_val.with_pair(b_val, |a, b| {
        #[cfg(feature = "parallel")]
        if batch_count == 1 && m >= parallel_config::MATMUL_THRESHOLD {
            out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    let mut sum = D::zero();
                    for p in 0..k {
                        sum = sum + a[i * k + p] * b[p * n + j];
                    }
                    row[j] = sum;
                }
            });
            return;
        }

        let a_batch_shape = &a_shape[..a_shape.len() - 2];
        let b_batch_shape = &b_shape[..b_shape.len() - 2];
        let out_batch_strides = rowmajor_strides(out_batch_shape);
        let a_batch_strides = rowmajor_strides(a_batch_shape);
        let b_batch_strides = rowmajor_strides(b_batch_shape);
        for batch in 0..batch_count {
            let a_batch = broadcast_batch_index(
                batch,
                out_batch_shape,
                &out_batch_strides,
                a_batch_shape,
                &a_batch_strides,
            );
            let b_batch = broadcast_batch_index(
                batch,
                out_batch_shape,
                &out_batch_strides,
                b_batch_shape,
                &b_batch_strides,
            );
            let a_offset = a_batch * m * k;
            let b_offset = b_batch * k * n;
            let out_offset = batch * m * n;
            for i in 0..m {
                for j in 0..n {
                    let mut sum = D::zero();
                    for p in 0..k {
                        sum = sum + a[a_offset + i * k + p] * b[b_offset + p * n + j];
                    }
                    out[out_offset + i * n + j] = sum;
                }
            }
        }
    });
    Ok(out)
}

fn broadcast_batch_index(
    output_index: usize,
    output_shape: &[usize],
    output_strides: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
) -> usize {
    let rank_difference = output_shape.len() - input_shape.len();
    let mut input_index = 0;
    let mut remainder = output_index;
    for (axis, &stride) in output_strides.iter().enumerate() {
        let coordinate = remainder / stride;
        remainder %= stride;
        if axis >= rank_difference {
            let input_axis = axis - rank_difference;
            if input_shape[input_axis] != 1 {
                input_index += coordinate * input_strides[input_axis];
            }
        }
    }
    input_index
}

fn conv2d_forward<D, G>(
    graph: &TensorGraph<D, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: &mut BufferPool<D>,
    node_idx: petgraph::graph::NodeIndex,
    stride: usize,
    padding: usize,
) -> Result<Vec<D>>
where
    D: Float + Send + Sync,
{
    let inputs_idx = graph.inputs(node_idx);
    let input_idx = inputs_idx[0];
    let weight_idx = inputs_idx[1];
    let input_val = values
        .get(&input_idx)
        .context("Input value not computed for conv2d")?;
    let weight_val = values
        .get(&weight_idx)
        .context("Weight value not computed for conv2d")?;
    let in_shape = graph.graph[input_idx].shape();
    let w_shape = graph.graph[weight_idx].shape();
    let out_shape = graph.graph[node_idx].shape();

    let n = in_shape[0];
    let c_in = in_shape[1];
    let h = in_shape[2];
    let w = in_shape[3];
    let c_out = w_shape[0];
    let kh = w_shape[2];
    let kw = w_shape[3];
    let h_out = out_shape[2];
    let w_out = out_shape[3];

    let mut out = pool.take(out_shape.iter().product());
    out.fill(D::zero());

    input_val.with(|input| {
        weight_val.with(|weight| {
            for on in 0..n {
                for oc in 0..c_out {
                    for ohi in 0..h_out {
                        for owi in 0..w_out {
                            let mut sum = D::zero();
                            for ic in 0..c_in {
                                for khi in 0..kh {
                                    let ih_raw = ohi * stride + khi;
                                    if ih_raw >= padding && ih_raw - padding < h {
                                        let ih = ih_raw - padding;
                                        for kwi in 0..kw {
                                            let iw_raw = owi * stride + kwi;
                                            if iw_raw >= padding && iw_raw - padding < w {
                                                let iw = iw_raw - padding;
                                                let in_val =
                                                    input[((on * c_in + ic) * h + ih) * w + iw];
                                                let w_val = weight
                                                    [((oc * c_in + ic) * kh + khi) * kw + kwi];
                                                sum = sum + in_val * w_val;
                                            }
                                        }
                                    }
                                }
                            }
                            out[((on * c_out + oc) * h_out + ohi) * w_out + owi] = sum;
                        }
                    }
                }
            }
        })
    });
    Ok(out)
}

fn conv_transpose_2d_forward<D, G>(
    graph: &TensorGraph<D, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: &mut BufferPool<D>,
    node_idx: petgraph::graph::NodeIndex,
    stride: usize,
    padding: usize,
) -> Result<Vec<D>>
where
    D: Float + Send + Sync,
{
    let inputs_idx = graph.inputs(node_idx);
    let grad_idx = inputs_idx[0];
    let weight_idx = inputs_idx[1];
    let grad_val = values
        .get(&grad_idx)
        .context("Grad output value not computed for conv_transpose_2d")?;
    let weight_val = values
        .get(&weight_idx)
        .context("Weight value not computed for conv_transpose_2d")?;
    let grad_shape = graph.graph[grad_idx].shape();
    let w_shape = graph.graph[weight_idx].shape();
    let out_shape = graph.graph[node_idx].shape();

    let n = out_shape[0];
    let c_in = out_shape[1];
    let h = out_shape[2];
    let w = out_shape[3];
    let c_out = w_shape[0];
    let kh = w_shape[2];
    let kw = w_shape[3];
    let h_out = grad_shape[2];
    let w_out = grad_shape[3];

    let mut out = pool.take(out_shape.iter().product());
    out.fill(D::zero());

    grad_val.with(|grad| {
        weight_val.with(|weight| {
            for on in 0..n {
                for oc in 0..c_out {
                    for ohi in 0..h_out {
                        for owi in 0..w_out {
                            let g = grad[((on * c_out + oc) * h_out + ohi) * w_out + owi];
                            for ic in 0..c_in {
                                for khi in 0..kh {
                                    let ih_raw = ohi * stride + khi;
                                    if ih_raw >= padding && ih_raw - padding < h {
                                        let ih = ih_raw - padding;
                                        for kwi in 0..kw {
                                            let iw_raw = owi * stride + kwi;
                                            if iw_raw >= padding && iw_raw - padding < w {
                                                let iw = iw_raw - padding;
                                                let w_val = weight
                                                    [((oc * c_in + ic) * kh + khi) * kw + kwi];
                                                out[((on * c_in + ic) * h + ih) * w + iw] = out
                                                    [((on * c_in + ic) * h + ih) * w + iw]
                                                    + g * w_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        })
    });
    Ok(out)
}

fn conv2d_backward_weight_forward<D, G>(
    graph: &TensorGraph<D, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: &mut BufferPool<D>,
    node_idx: petgraph::graph::NodeIndex,
    stride: usize,
    padding: usize,
) -> Result<Vec<D>>
where
    D: Float + Send + Sync,
{
    let inputs_idx = graph.inputs(node_idx);
    let input_idx = inputs_idx[0];
    let grad_idx = inputs_idx[1];
    let input_val = values
        .get(&input_idx)
        .context("Input value not computed for conv2d_backward_weight")?;
    let grad_val = values
        .get(&grad_idx)
        .context("Grad output value not computed for conv2d_backward_weight")?;
    let in_shape = graph.graph[input_idx].shape();
    let grad_shape = graph.graph[grad_idx].shape();
    let out_shape = graph.graph[node_idx].shape();

    let n = in_shape[0];
    let c_in = in_shape[1];
    let h = in_shape[2];
    let w = in_shape[3];
    let c_out = out_shape[0];
    let kh = out_shape[2];
    let kw = out_shape[3];
    let h_out = grad_shape[2];
    let w_out = grad_shape[3];

    let mut out = pool.take(out_shape.iter().product());
    out.fill(D::zero());

    input_val.with(|input| {
        grad_val.with(|grad| {
            for on in 0..n {
                for oc in 0..c_out {
                    for ohi in 0..h_out {
                        for owi in 0..w_out {
                            let g = grad[((on * c_out + oc) * h_out + ohi) * w_out + owi];
                            for ic in 0..c_in {
                                for khi in 0..kh {
                                    let ih_raw = ohi * stride + khi;
                                    if ih_raw >= padding && ih_raw - padding < h {
                                        let ih = ih_raw - padding;
                                        for kwi in 0..kw {
                                            let iw_raw = owi * stride + kwi;
                                            if iw_raw >= padding && iw_raw - padding < w {
                                                let iw = iw_raw - padding;
                                                let in_val =
                                                    input[((on * c_in + ic) * h + ih) * w + iw];
                                                out[((oc * c_in + ic) * kh + khi) * kw + kwi] = out
                                                    [((oc * c_in + ic) * kh + khi) * kw + kwi]
                                                    + g * in_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        })
    });
    Ok(out)
}

fn max_pool2d_forward<D, G>(
    graph: &TensorGraph<D, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: &mut BufferPool<D>,
    node_idx: petgraph::graph::NodeIndex,
    kernel_size: usize,
    stride: usize,
) -> Result<Vec<D>>
where
    D: Float + Send + Sync,
{
    let inputs_idx = graph.inputs(node_idx);
    let x_idx = inputs_idx[0];
    let x_val = values
        .get(&x_idx)
        .context("Input value not computed for max_pool2d")?;
    let in_shape = graph.graph[x_idx].shape();
    let out_shape = graph.graph[node_idx].shape();

    let n = in_shape[0];
    let c = in_shape[1];
    let h = in_shape[2];
    let w = in_shape[3];
    let h_out = out_shape[2];
    let w_out = out_shape[3];

    let mut out = pool.take(out_shape.iter().product());

    x_val.with(|x| {
        for on in 0..n {
            for oc in 0..c {
                for ohi in 0..h_out {
                    for owi in 0..w_out {
                        let mut max_val = D::neg_infinity();
                        for khi in 0..kernel_size {
                            let ih = ohi * stride + khi;
                            if ih < h {
                                for kwi in 0..kernel_size {
                                    let iw = owi * stride + kwi;
                                    if iw < w {
                                        let val = x[((on * c + oc) * h + ih) * w + iw];
                                        max_val = max_val.max(val);
                                    }
                                }
                            }
                        }
                        out[((on * c + oc) * h_out + ohi) * w_out + owi] = max_val;
                    }
                }
            }
        }
    });
    Ok(out)
}

fn max_pool2d_backward_forward<D, G>(
    graph: &TensorGraph<D, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: &mut BufferPool<D>,
    node_idx: petgraph::graph::NodeIndex,
    kernel_size: usize,
    stride: usize,
) -> Result<Vec<D>>
where
    D: Float + Send + Sync,
{
    let inputs_idx = graph.inputs(node_idx);
    let input_idx = inputs_idx[0];
    let output_idx = inputs_idx[1];
    let grad_idx = inputs_idx[2];
    let input_val = values
        .get(&input_idx)
        .context("Input value not computed for max_pool2d_backward")?;
    let output_val = values
        .get(&output_idx)
        .context("Output value not computed for max_pool2d_backward")?;
    let grad_val = values
        .get(&grad_idx)
        .context("Grad output value not computed for max_pool2d_backward")?;
    let in_shape = graph.graph[input_idx].shape();
    let out_shape = graph.graph[output_idx].shape();
    let grad_shape = graph.graph[grad_idx].shape();

    let n = in_shape[0];
    let c = in_shape[1];
    let h = in_shape[2];
    let w = in_shape[3];
    let h_out = out_shape[2];
    let w_out = out_shape[3];

    assert_eq!(out_shape, grad_shape);

    let mut out = pool.take(in_shape.iter().product());
    out.fill(D::zero());

    input_val.with(|input| {
        output_val.with(|output| {
            grad_val.with(|grad| {
                for on in 0..n {
                    for oc in 0..c {
                        for ohi in 0..h_out {
                            for owi in 0..w_out {
                                let o_val = output[((on * c + oc) * h_out + ohi) * w_out + owi];
                                let g = grad[((on * c + oc) * h_out + ohi) * w_out + owi];
                                for khi in 0..kernel_size {
                                    let ih = ohi * stride + khi;
                                    if ih < h {
                                        for kwi in 0..kernel_size {
                                            let iw = owi * stride + kwi;
                                            if iw < w {
                                                let in_val =
                                                    input[((on * c + oc) * h + ih) * w + iw];
                                                if in_val == o_val {
                                                    out[((on * c + oc) * h + ih) * w + iw] =
                                                        out[((on * c + oc) * h + ih) * w + iw] + g;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            })
        })
    });
    Ok(out)
}

fn transpose_forward<D, G>(
    graph: &TensorGraph<D, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: &mut BufferPool<D>,
    node_idx: petgraph::graph::NodeIndex,
) -> Result<Vec<D>>
where
    D: Float,
{
    let inputs_idx = graph.inputs(node_idx);
    let a_idx = inputs_idx[0];
    let rank = graph.graph[a_idx].shape().len();
    anyhow::ensure!(rank >= 2, "Transpose requires rank >= 2, got {rank}D");
    let mut axes: Vec<usize> = (0..rank).collect();
    axes.swap(rank - 2, rank - 1);
    permute_forward(graph, values, pool, node_idx, &axes)
}

fn permute_forward<D, G>(
    graph: &TensorGraph<D, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: &mut BufferPool<D>,
    node_idx: petgraph::graph::NodeIndex,
    axes: &[usize],
) -> Result<Vec<D>>
where
    D: Float,
{
    let input_idx = graph.inputs(node_idx)[0];
    let input = values
        .get(&input_idx)
        .context("Input value not computed for permute")?;
    let input_shape = graph.graph[input_idx].shape();
    let output_shape = graph.graph[node_idx].shape();
    let input_strides = rowmajor_strides(input_shape);
    let output_strides = rowmajor_strides(output_shape);
    let output_size: usize = output_shape.iter().product();
    let mut out = pool.take(output_size);
    input.with(|input| {
        for (output_index, output) in out.iter_mut().enumerate() {
            let mut remainder = output_index;
            let mut input_index = 0;
            for (output_axis, &stride) in output_strides.iter().enumerate() {
                let coordinate = remainder / stride;
                remainder %= stride;
                input_index += coordinate * input_strides[axes[output_axis]];
            }
            *output = input[input_index];
        }
    });
    Ok(out)
}

fn broadcast_axis_forward<D, G>(
    graph: &TensorGraph<D, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: &mut BufferPool<D>,
    node_idx: petgraph::graph::NodeIndex,
    axis: usize,
) -> Result<Vec<D>>
where
    D: Float,
{
    let in_idx = graph.inputs(node_idx)[0];
    let in_val = values
        .get(&in_idx)
        .context("Input value not computed for broadcast_axis")?;
    let in_shape = graph.graph[in_idx].shape();
    let out_shape = graph.graph[node_idx].shape();
    let out_size: usize = out_shape.iter().product();
    let in_strides = rowmajor_strides(in_shape);
    let out_strides = rowmajor_strides(out_shape);

    let mut out = pool.take(out_size);
    in_val.with(|in_val| {
        for (out_idx, out_elem) in out.iter_mut().enumerate() {
            let mut in_linear = 0usize;
            let mut rem = out_idx;
            for (dim, &stride) in out_strides.iter().enumerate() {
                let coord = rem / stride;
                rem %= stride;
                let in_coord = if dim == axis { 0 } else { coord };
                in_linear += in_coord * in_strides[dim];
            }
            *out_elem = in_val[in_linear];
        }
    });
    Ok(out)
}

fn reduce_axis_forward<D, G>(
    graph: &TensorGraph<D, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Value<D>>,
    pool: &mut BufferPool<D>,
    node_idx: petgraph::graph::NodeIndex,
    op: &tensor::ReduceOp,
    axis: usize,
) -> Result<Vec<D>>
where
    D: Float,
{
    let in_idx = graph.inputs(node_idx)[0];
    let x = values
        .get(&in_idx)
        .context("Input value not computed for reduce_axis")?;
    let in_shape = graph.graph[in_idx].shape();
    let out_shape = graph.graph[node_idx].shape();
    let out_size: usize = out_shape.iter().product();
    let axis_size = in_shape[axis];
    let in_strides = rowmajor_strides(in_shape);
    let out_strides = rowmajor_strides(out_shape);

    // Pooled buffers hold stale data, so initialize every element.
    let mut out = pool.take(out_size);
    match op {
        tensor::ReduceOp::Max => out.fill(D::neg_infinity()),
        _ => out.fill(D::zero()),
    }

    x.with(|x| {
        let total: usize = in_shape.iter().product();
        for (idx, &val) in x.iter().enumerate().take(total) {
            let mut rem = idx;
            let mut out_linear = 0usize;
            for (dim, &stride) in in_strides.iter().enumerate() {
                let coord = rem / stride;
                rem %= stride;
                let out_coord = if dim == axis { 0 } else { coord };
                out_linear += out_coord * out_strides[dim];
            }
            match op {
                tensor::ReduceOp::Sum => {
                    out[out_linear] = out[out_linear] + val;
                }
                tensor::ReduceOp::Mean => {
                    out[out_linear] = out[out_linear] + val / D::from(axis_size).unwrap();
                }
                tensor::ReduceOp::Max => {
                    out[out_linear] = out[out_linear].max(val);
                }
            }
        }
    });
    Ok(out)
}

/// Get an iterator over a slice, parallel if the `parallel` feature is enabled.
///
/// Automatically uses parallel iteration for large slices when compiled with
/// the `parallel` feature, falling back to sequential iteration otherwise.
#[cfg(feature = "parallel")]
pub fn get_iter<T: Sync>(slice: &[T]) -> SliceIter<'_, T> {
    slice
        .par_iter()
        .with_min_len(parallel_config::ELEMENTWISE_THRESHOLD)
}

/// Get an iterator over a slice, parallel if the `parallel` feature is enabled.
///
/// Automatically uses parallel iteration for large slices when compiled with
/// the `parallel` feature, falling back to sequential iteration otherwise.
#[cfg(not(feature = "parallel"))]
pub fn get_iter<T>(slice: &[T]) -> SliceIter<'_, T> {
    slice.iter()
}

/// Get a mutable iterator over a slice, parallel if the `parallel` feature is
/// enabled.
///
/// Automatically uses parallel iteration for large slices when compiled with
/// the `parallel` feature, falling back to sequential iteration otherwise.
#[cfg(feature = "parallel")]
pub fn get_iter_mut<T: Send>(slice: &mut [T]) -> SliceIterMut<'_, T> {
    slice
        .par_iter_mut()
        .with_min_len(parallel_config::ELEMENTWISE_THRESHOLD)
}

/// Get a mutable iterator over a slice, parallel if the `parallel` feature is
/// enabled.
///
/// Automatically uses parallel iteration for large slices when compiled with
/// the `parallel` feature, falling back to sequential iteration otherwise.
#[cfg(not(feature = "parallel"))]
pub fn get_iter_mut<T>(slice: &mut [T]) -> SliceIterMut<'_, T> {
    slice.iter_mut()
}
