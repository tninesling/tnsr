//! Runtime execution engine for tensor computation graphs.
//!
//! This crate provides executors that execute computation graphs
//! lowered from the `tensor` crate. Gradients are computed during execution
//! when using graphs with gradient metadata (`TensorGraph<D, WithGrad>`).
//!
//! # Architecture
//!
//! - **Runtime**: Unified interface with automatic backend selection. Recommended for most users.
//!   Automatically chooses between CPU and GPU at runtime based on availability.
//!
//! - **Executor Trait**: Common interface for graph execution and gradient retrieval,
//!   returning `Result<Vec<D>>` for proper error handling.
//!
//! - **SimpleExecutor**: CPU-based executor with optional parallelism (via `parallel` feature).
//!   Uses naive algorithms suitable for testing and small models.
//!
//! - **CudaExecutor** (optional): GPU-accelerated executor using CUDA kernels (via `cuda` feature).
//!   Automatically manages device memory and kernel launches.
//!
//! # Getting Started
//!
//! Most users should use [`Runtime`] which automatically selects the best available backend:
//!
//! ```rust
//! use runtime::Runtime;
//! use tensor::{TensorExpr, graph::TensorGraph};
//! use std::collections::HashMap;
//!
//! let mut runtime = Runtime::new();
//! let x = TensorExpr::<f32>::input("x", vec![2, 3]);
//! let graph: TensorGraph<f32> = x.into();
//!
//! let mut inputs = HashMap::new();
//! inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! let result = runtime.execute(&graph, inputs).unwrap();
//! assert_eq!(result.len(), 6);
//! ```
//!
//! # Gradient Computation
//!
//! Gradients are computed during execution using graphs created with
//! `TensorGraph::with_gradients()`. After executing the graph, use
//! `get_gradients()` to retrieve parameter gradients for optimization.
//!
//! # Error Handling
//!
//! Execution failures return `Result` types with `anyhow::Error` for:
//! - Missing inputs or computed values
//! - Shape mismatches
//! - Memory allocation failures
//! - Invalid operations (e.g., unsupported dimensions)
//! - CUDA-specific errors (kernel launch failures, device errors)
//!
//! Use `.unwrap()` in tests or the `?` operator in production code to handle errors.
//!
//! # Features
//!
//! - `cuda`: Enable GPU acceleration via CUDA (requires CUDA toolkit)
//! - `parallel`: Enable CPU parallelism via Rayon
//!
//! # Advanced: Direct Executor Usage
//!
//! For direct control over the backend, you can use [`SimpleExecutor`] or
//! [`CudaExecutor`] directly:
//!
//! ```rust
//! use runtime::{Executor, SimpleExecutor};
//! use tensor::{TensorExpr, graph::TensorGraph};
//! use std::collections::HashMap;
//!
//! let mut executor = SimpleExecutor::new();
//! let x = TensorExpr::<f32>::input("x", vec![2, 3]);
//! let graph: TensorGraph<f32> = x.into();
//!
//! let mut inputs = HashMap::new();
//! inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! let result = executor.forward(&graph, inputs).unwrap();
//! assert_eq!(result.len(), 6);
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result};

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod optimizer;
mod ptx;
mod runtime;
mod tile;

pub use runtime::{Backend, Runtime};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;
use tensor::graph::WithGrad;
use tracing::trace_span;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

/// Iterator type for slice traversal, conditionally parallel based on features.
///
/// With the `parallel` feature enabled, uses Rayon for parallel iteration.
/// Otherwise, uses standard sequential iteration.
#[cfg(feature = "parallel")]
pub type SliceIter<'a, T> = rayon::iter::MinLen<rayon::slice::Iter<'a, T>>;

/// Iterator type for slice traversal, conditionally parallel based on features.
///
/// With the `parallel` feature enabled, uses Rayon for parallel iteration.
/// Otherwise, uses standard sequential iteration.
#[cfg(not(feature = "parallel"))]
pub type SliceIter<'a, T> = std::slice::Iter<'a, T>;

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
///
/// # Example
///
/// ```no_run
/// use runtime::init_chrome_tracing;
///
/// let _guard = init_chrome_tracing("trace.json").unwrap();
/// // ... perform traced operations ...
/// // Trace is flushed when _guard is dropped
/// ```
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
    ) -> Result<Vec<D>>;

    /// Retrieve computed gradients for parameters.
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
/// use runtime::Runtime;
/// use tensor::{TensorExpr, graph::TensorGraph};
/// use std::collections::HashMap;
///
/// let mut runtime = Runtime::new();
/// let x = TensorExpr::<f32>::input("x", vec![2, 3]);
/// let graph: TensorGraph<f32> = x.into();
///
/// let mut inputs = HashMap::new();
/// inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
///
/// let result = runtime.execute(&graph, inputs).unwrap();
/// assert_eq!(result.len(), 6);
/// ```
#[derive(Default)]
pub struct SimpleExecutor {
    values: HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
}

impl SimpleExecutor {
    /// Create a new CPU executor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieve the computed value for a specific graph node.
    ///
    /// Only available after execution has completed.
    /// Useful for debugging intermediate values.
    pub fn get_value(&self, node_idx: petgraph::graph::NodeIndex) -> Option<&Vec<f32>> {
        self.values.get(&node_idx)
    }

    fn neg(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("neg").entered();
        get_iter(x).map(|v| -v).collect()
    }

    fn exp(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("exp").entered();

        get_iter(x).copied().map(f32::exp).collect()
    }

    fn log(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("log").entered();

        get_iter(x).copied().map(f32::ln).collect()
    }

    fn relu(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("relu").entered();

        get_iter(x).map(|v| v.max(0.0)).collect()
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("add").entered();

        get_iter(a).zip(get_iter(b)).map(|(x, y)| x + y).collect()
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("sub").entered();

        get_iter(a).zip(get_iter(b)).map(|(x, y)| x - y).collect()
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("mul").entered();
        get_iter(a).zip(get_iter(b)).map(|(x, y)| x * y).collect()
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("div").entered();
        get_iter(a).zip(get_iter(b)).map(|(x, y)| x / y).collect()
    }

    fn gt(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("gt").entered();
        get_iter(a)
            .zip(get_iter(b))
            .map(|(x, y)| if x > y { 1.0 } else { 0.0 })
            .collect()
    }

    fn mask(&self, values: &[f32], condition: &[f32]) -> Vec<f32> {
        let _span = trace_span!("mask").entered();
        eprintln!(
            "mask function: values={:?}, condition={:?}",
            values, condition
        );
        let result: Vec<f32> = get_iter(values)
            .zip(get_iter(condition))
            .map(|(v, c)| if *c != 0.0 { *v } else { 0.0 })
            .collect();
        eprintln!("mask result: {:?}", result);
        result
    }
}

impl Executor<f32> for SimpleExecutor {
    fn execute<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let order = graph.toposort();
        for node_idx in order.iter() {
            let node = &graph[*node_idx];
            println!(
                "DEBUG: Executing node {}: {}",
                node_idx.index(),
                node.name()
            );
            let result = match node {
                TensorGraphNode::Constant { data } => {
                    let _span = trace_span!("constant", node = node_idx.index()).entered();
                    data.as_ref().clone()
                }
                TensorGraphNode::Input { name } => {
                    let _span =
                        trace_span!("input", node = node_idx.index(), name = name).entered();
                    inputs
                        .get::<str>(name)
                        .with_context(|| format!("Input '{}' not found", name))?
                        .clone()
                }
                TensorGraphNode::Parameter { data, .. } => {
                    let _span = trace_span!("parameter", node = node_idx.index()).entered();
                    data.lock().unwrap().clone()
                }
                TensorGraphNode::Unary { op } => {
                    let inputs = graph.inputs(*node_idx);
                    let x = self.values.get(&inputs[0]).with_context(|| {
                        format!("Value for node {} not computed", inputs[0].index())
                    })?;
                    match op {
                        tensor::UnaryOp::Neg => self.neg(x),
                        tensor::UnaryOp::Exp => self.exp(x),
                        tensor::UnaryOp::Log => self.log(x),
                        tensor::UnaryOp::Relu => self.relu(x),
                    }
                }
                TensorGraphNode::Binary { op } => {
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
                    match op {
                        tensor::BinaryOp::Add => self.add(a, b),
                        tensor::BinaryOp::Sub => self.sub(a, b),
                        tensor::BinaryOp::Mul => self.mul(a, b),
                        tensor::BinaryOp::Div => self.div(a, b),
                    }
                }
                TensorGraphNode::MatMul => {
                    let _span = trace_span!("matmul", node = node_idx.index()).entered();
                    matmul_forward(graph, &self.values, *node_idx)?
                }
                TensorGraphNode::Transpose => {
                    let _span = trace_span!("transpose", node = node_idx.index()).entered();
                    transpose_forward(graph, &self.values, *node_idx)?
                }
                TensorGraphNode::BroadcastAxis { axis } => {
                    let _span = trace_span!("broadcast_axis", node = node_idx.index()).entered();
                    broadcast_axis_forward(graph, &self.values, *node_idx, *axis)?
                }
                TensorGraphNode::ReduceAxis { op, axis } => {
                    let _span = trace_span!(
                        "reduce_axis",
                        op = node.name(),
                        axis = *axis,
                        node = node_idx.index()
                    )
                    .entered();
                    reduce_axis_forward(graph, &self.values, *node_idx, op, *axis)?
                }
                TensorGraphNode::Gt => {
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
                    self.gt(a, b)
                }
                TensorGraphNode::Mask => {
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
                    self.mask(values, condition)
                }
            };
            self.values.insert(*node_idx, result);
        }
        let last_node = order
            .last()
            .context("Graph is empty, no nodes to execute")?;
        let out = self
            .values
            .get(last_node)
            .context("Output value not computed")?;
        Ok(out.clone())
    }

    fn get_gradients(&self, graph: &TensorGraph<f32, WithGrad>) -> HashMap<usize, Vec<f32>> {
        let mut result = HashMap::new();

        // Iterate through all parameters in the gradient metadata
        for (param_id, grad_node_idx) in &graph.gradient_metadata().param_to_grad {
            // Look up the gradient value from our computed values
            if let Some(grad_value) = self.values.get(grad_node_idx) {
                result.insert(*param_id, grad_value.clone());
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

fn matmul_forward<G>(
    graph: &TensorGraph<f32, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
) -> Result<Vec<f32>> {
    let inputs_idx = graph.inputs(node_idx);
    let a_idx = inputs_idx[0];
    let b_idx = inputs_idx[1];
    let a = values
        .get(&a_idx)
        .context("Left operand value not computed for matmul")?;
    let b = values
        .get(&b_idx)
        .context("Right operand value not computed for matmul")?;
    let a_shape = graph
        .shapes
        .get(&a_idx)
        .context("Shape missing for matmul left operand")?;
    let b_shape = graph
        .shapes
        .get(&b_idx)
        .context("Shape missing for matmul right operand")?;
    anyhow::ensure!(
        a_shape.len() == 2,
        "Matmul left operand must be 2D, got {}D",
        a_shape.len()
    );
    anyhow::ensure!(
        b_shape.len() == 2,
        "Matmul right operand must be 2D, got {}D",
        b_shape.len()
    );
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        if m >= parallel_config::MATMUL_THRESHOLD {
            let mut out = vec![0.0f32; m * n];
            out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    row[j] = sum;
                }
            });
            return Ok(out);
        }
    }

    // Sequential fallback
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    Ok(out)
}

fn transpose_forward<G>(
    graph: &TensorGraph<f32, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
) -> Result<Vec<f32>> {
    let inputs_idx = graph.inputs(node_idx);
    let a_idx = inputs_idx[0];
    let a = values
        .get(&a_idx)
        .context("Input value not computed for transpose")?;
    let a_shape = graph
        .shapes
        .get(&a_idx)
        .context("Shape missing for transpose input")?;

    anyhow::ensure!(
        a_shape.len() == 2,
        "Transpose currently only supports 2D matrices, got {}D",
        a_shape.len()
    );

    let (m, n) = (a_shape[0], a_shape[1]);
    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }

    Ok(out)
}

fn broadcast_axis_forward<G>(
    graph: &TensorGraph<f32, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
    axis: usize,
) -> Result<Vec<f32>> {
    let in_idx = graph.inputs(node_idx)[0];
    let in_val = values
        .get(&in_idx)
        .context("Input value not computed for broadcast_axis")?;
    let in_shape = graph
        .shapes
        .get(&in_idx)
        .context("Input shape missing for broadcast_axis")?;
    let out_shape = graph
        .shapes
        .get(&node_idx)
        .context("Output shape missing for broadcast_axis")?;
    let out_size: usize = out_shape.iter().product();
    let in_strides = rowmajor_strides(in_shape);
    let out_strides = rowmajor_strides(out_shape);

    let mut out = vec![0.0f32; out_size];
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
    Ok(out)
}

fn reduce_axis_forward<G>(
    graph: &TensorGraph<f32, G>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
    op: &tensor::ReduceOp,
    axis: usize,
) -> Result<Vec<f32>> {
    let in_idx = graph.inputs(node_idx)[0];
    let x = values
        .get(&in_idx)
        .context("Input value not computed for reduce_axis")?;
    let in_shape = graph
        .shapes
        .get(&in_idx)
        .context("Input shape missing for reduce_axis")?;
    let out_shape = graph
        .shapes
        .get(&node_idx)
        .context("Output shape missing for reduce_axis")?;
    let out_size: usize = out_shape.iter().product();
    let axis_size = in_shape[axis];
    let in_strides = rowmajor_strides(in_shape);
    let out_strides = rowmajor_strides(out_shape);

    let mut out = match op {
        tensor::ReduceOp::Max => vec![f32::NEG_INFINITY; out_size],
        _ => vec![0.0f32; out_size],
    };

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
                out[out_linear] += val;
            }
            tensor::ReduceOp::Mean => {
                out[out_linear] += val / axis_size as f32;
            }
            tensor::ReduceOp::Max => {
                out[out_linear] = out[out_linear].max(val);
            }
        }
    }
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
