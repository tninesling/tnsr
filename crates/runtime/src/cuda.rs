use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use tensor::graph::{TensorGraph, TensorGraphNode};

use crate::{BackwardResult, Executor};
use tracing::trace_span;

static PTX: &str = include_str!("cuda-kernels.ptx");

pub struct CudaExecutor {
    device: Arc<CudaContext>,
    module: Arc<CudaModule>,
    values: HashMap<petgraph::graph::NodeIndex, CudaSlice<f32>>,
    grads: HashMap<petgraph::graph::NodeIndex, CudaSlice<f32>>,
}

impl Default for CudaExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaExecutor {
    pub fn new() -> Self {
        let device = CudaContext::new(0).expect("Failed to initialize CUDA device 0");
        let module = device
            .load_module(Ptx::from_src(PTX))
            .expect("Failed to load PTX module with cudarc");
        CudaExecutor {
            device,
            module,
            values: HashMap::new(),
            grads: HashMap::new(),
        }
    }

    fn neg(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("neg").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("neg").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA neg failed");
        out
    }

    fn exp(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("exp").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("exp").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA exp failed");
        out
    }

    fn log(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("log").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("log").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA log failed");
        out
    }

    fn relu(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("relu").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("relu").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA relu failed");
        out
    }

    fn add(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("add").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("add").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA add failed");
        out
    }

    fn sub(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("sub").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("sub").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA sub failed");
        out
    }

    fn mul(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("mul").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("mul").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA mul failed");
        out
    }

    fn div(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("div").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("div").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA div failed");
        out
    }

    fn gt(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("gt").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("gt").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA gt failed");
        out
    }

    fn mask(&self, values: &CudaSlice<f32>, condition: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("mask").entered();
        assert_eq!(
            values.len(),
            condition.len(),
            "binary op input length mismatch"
        );
        let len = values.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("mask").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(values);
        launcher.arg(&len_u64);
        launcher.arg(condition);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA mask failed");
        out
    }

    fn matmul(
        &self,
        lhs: &CudaSlice<f32>,
        rhs: &CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> CudaSlice<f32> {
        let _span = trace_span!("matmul").entered();
        let stream = self.device.default_stream();

        let out_len = m * n;
        let mut out = stream.alloc_zeros::<f32>(m * n).unwrap();

        let f = self
            .module
            .load_function("matmul")
            .expect("Failed to load matmul function");
        let cfg = LaunchConfig::for_num_elems(out_len as u32);
        let lhs_len_u64 = lhs.len() as u64;
        let rhs_len_u64 = rhs.len() as u64;
        let m_u64 = m as u64;
        let n_u64 = n as u64;
        let k_u64 = k as u64;
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&lhs_len_u64);
        launcher.arg(rhs);
        launcher.arg(&rhs_len_u64);
        launcher.arg(&mut out);
        launcher.arg(&m_u64);
        launcher.arg(&n_u64);
        launcher.arg(&k_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA matmul failed");

        out
    }

    fn neg_grad(&self, dy: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("neg_grad").entered();
        self.neg(dy)
    }

    fn exp_grad(&self, dy: &CudaSlice<f32>, y: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("exp_grad").entered();
        self.mul(dy, y)
    }

    fn log_grad(&self, dy: &CudaSlice<f32>, x: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("log_grad").entered();
        self.div(dy, x)
    }

    fn relu_grad(&self, dy: &CudaSlice<f32>, x: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("relu_grad").entered();
        let len = dy.len();
        let stream = self.device.default_stream();

        // Create a zeros buffer for comparison: x > 0
        let zeros = stream.alloc_zeros::<f32>(len).unwrap();

        // Compute condition: x > 0 (returns 1.0 where true, 0.0 where false)
        let condition = self.gt(x, &zeros);

        // Apply mask: dy where x > 0, else 0
        // Note: mask kernel has swapped semantics - first arg is checked for zero, second is returned
        // TODO: Update mask kernel with flipped arg order
        self.mask(&condition, dy)
    }

    fn add_grad(&self, dy: &CudaSlice<f32>) -> (CudaSlice<f32>, CudaSlice<f32>) {
        let _span = trace_span!("add_grad").entered();
        (dy.clone(), dy.clone())
    }

    fn sub_grad(&self, dy: &CudaSlice<f32>) -> (CudaSlice<f32>, CudaSlice<f32>) {
        let _span = trace_span!("sub_grad").entered();
        let db = self.neg(dy);
        (dy.clone(), db)
    }

    fn mul_grad(
        &self,
        dy: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
    ) -> (CudaSlice<f32>, CudaSlice<f32>) {
        let _span = trace_span!("mul_grad").entered();
        let da = self.mul(dy, b);
        let db = self.mul(dy, a);
        (da, db)
    }

    fn div_grad(
        &self,
        dy: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
    ) -> (CudaSlice<f32>, CudaSlice<f32>) {
        let _span = trace_span!("div_grad").entered();
        let len = dy.len();
        let stream = self.device.default_stream();

        let mut dy_host = vec![0.0f32; len];
        let mut a_host = vec![0.0f32; len];
        let mut b_host = vec![0.0f32; len];
        stream.memcpy_dtoh(dy, &mut dy_host).unwrap();
        stream.memcpy_dtoh(a, &mut a_host).unwrap();
        stream.memcpy_dtoh(b, &mut b_host).unwrap();

        let tmp_a_host: Vec<f32> = dy_host
            .iter()
            .zip(b_host.iter())
            .map(|(g, b)| g / b)
            .collect();
        let tmp_b_host: Vec<f32> = dy_host
            .iter()
            .zip(a_host.iter())
            .map(|(g, a)| -g * a)
            .collect();
        let b_sq_host: Vec<f32> = b_host.iter().map(|b| b * b).collect();
        let tmp_b_final: Vec<f32> = tmp_b_host
            .iter()
            .zip(b_sq_host.iter())
            .map(|(t, bsq)| t / bsq)
            .collect();

        let mut da = stream.alloc_zeros::<f32>(len).unwrap();
        let mut db = stream.alloc_zeros::<f32>(len).unwrap();
        stream.memcpy_htod(&tmp_a_host, &mut da).unwrap();
        stream.memcpy_htod(&tmp_b_final, &mut db).unwrap();
        (da, db)
    }

    /// For mamtul A x B = C, where A is [m, k], B is [k, n], C is [m, n],
    /// we have gradient dC of C, which is shape [m, n], along with B.
    ///
    /// The gradient w.r.t. A is: dA = dC x B^T
    fn matmul_grad_left(
        &self,
        dc: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> CudaSlice<f32> {
        let _span = trace_span!("matmul_grad_left").entered();
        let b_t = self.transpose(b, k, n);
        self.matmul(dc, &b_t, m, k, n)
    }

    /// For mamtul A x B = C, where A is [m, k], B is [k, n], C is [m, n],
    /// we have gradient dC of C, which is shape [m, n], along with A.
    ///
    /// The gradient w.r.t. A is: dA = dC x B^T
    fn matmul_grad_right(
        &self,
        a: &CudaSlice<f32>,
        dc: &CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> CudaSlice<f32> {
        let _span = trace_span!("matmul_grad_right").entered();
        let a_t = self.transpose(a, m, k);
        self.matmul(&a_t, dc, k, n, m)
    }

    fn transpose(&self, input: &CudaSlice<f32>, rows: usize, cols: usize) -> CudaSlice<f32> {
        let _span = trace_span!("transpose").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("transpose_2d").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let rows_u64 = rows as u64;
        let cols_u64 = cols as u64;
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&rows_u64);
        launcher.arg(&cols_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA transpose failed");
        out
    }

    /// Get the value of a specific node from the executor's cache after a forward pass
    pub fn get_value(&self, node_idx: petgraph::graph::NodeIndex) -> Option<Vec<f32>> {
        self.values.get(&node_idx).map(|cuda_slice| {
            let mut host_vec = vec![0.0f32; cuda_slice.len()];
            self.device
                .default_stream()
                .memcpy_dtoh(cuda_slice, &mut host_vec)
                .unwrap();
            host_vec
        })
    }

    /// Accumulate gradient on GPU
    fn accumulate_grad_gpu(
        &mut self,
        node_idx: petgraph::graph::NodeIndex,
        new_grad: CudaSlice<f32>,
    ) {
        let _span = trace_span!("accumulate_grad_gpu", node = node_idx.index()).entered();
        if let Some(existing_grad) = self.grads.get(&node_idx) {
            // Add the new gradient to the existing one
            let len = existing_grad.len();
            let len_u64 = len as u64;
            let stream = self.device.default_stream();
            let mut accumulated = stream.alloc_zeros::<f32>(len).unwrap();

            let f = self.module.load_function("add").unwrap();
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut launcher = stream.launch_builder(&f);
            launcher.arg(existing_grad);
            launcher.arg(&len_u64);
            launcher.arg(&new_grad);
            launcher.arg(&len_u64);
            launcher.arg(&mut accumulated);
            launcher.arg(&len_u64);
            unsafe { launcher.launch(cfg) }.expect("CUDA grad accumulation failed");

            self.grads.insert(node_idx, accumulated);
        } else {
            self.grads.insert(node_idx, new_grad);
        }
    }
}

impl Executor<f32> for CudaExecutor {
    fn forward(&mut self, graph: &TensorGraph<f32>, inputs: HashMap<String, Vec<f32>>) -> Vec<f32> {
        let order = graph.toposort();
        let _fwd_span = trace_span!("forward", nodes = order.len()).entered();

        for node_idx in order.iter() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data } => {
                    let _span = trace_span!("constant", node = node_idx.index(), size = data.len())
                        .entered();
                    let stream = self.device.default_stream();
                    let mut device_data = stream.alloc_zeros::<f32>(data.len()).unwrap();
                    stream
                        .memcpy_htod(data.as_slice(), &mut device_data)
                        .unwrap();
                    device_data
                }
                TensorGraphNode::Input { name } => {
                    let _span =
                        trace_span!("input", node = node_idx.index(), name = name).entered();
                    let val = inputs.get::<str>(name).expect("Input not found").clone();
                    let stream = self.device.default_stream();
                    let mut device_data = stream.alloc_zeros::<f32>(val.len()).unwrap();
                    stream.memcpy_htod(&val, &mut device_data).unwrap();
                    device_data
                }
                TensorGraphNode::Unary { op } => {
                    let inputs = graph.inputs(*node_idx);
                    let input = self.values.get(&inputs[0]).unwrap();
                    match op {
                        tensor::UnaryOp::Neg => self.neg(input),
                        tensor::UnaryOp::Exp => self.exp(input),
                        tensor::UnaryOp::Log => self.log(input),
                        tensor::UnaryOp::Relu => self.relu(input),
                    }
                }
                TensorGraphNode::Binary { op } => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = self.values.get(&ins[0]).unwrap();
                    let rhs = self.values.get(&ins[1]).unwrap();
                    match op {
                        tensor::BinaryOp::Add => self.add(lhs, rhs),
                        tensor::BinaryOp::Sub => self.sub(lhs, rhs),
                        tensor::BinaryOp::Mul => self.mul(lhs, rhs),
                        tensor::BinaryOp::Div => self.div(lhs, rhs),
                    }
                }
                TensorGraphNode::MatMul => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = self.values.get(&ins[0]).unwrap();
                    let rhs = self.values.get(&ins[1]).unwrap();
                    let lhs_shape = graph.shapes.get(&ins[0]).expect("shape missing for lhs");
                    let rhs_shape = graph.shapes.get(&ins[1]).expect("shape missing for rhs");
                    assert_eq!(lhs_shape.len(), 2, "MatMul expects 2D lhs");
                    assert_eq!(rhs_shape.len(), 2, "MatMul expects 2D rhs");
                    let m = lhs_shape[0];
                    let k = lhs_shape[1];
                    let n = rhs_shape[1];
                    assert_eq!(k, rhs_shape[0], "MatMul inner dim mismatch");

                    self.matmul(lhs, rhs, m, n, k)
                }
                TensorGraphNode::Parameter { data, .. } => {
                    let v = data.lock().unwrap();
                    let _span =
                        trace_span!("parameter", node = node_idx.index(), size = v.len()).entered();
                    let stream = self.device.default_stream();
                    let mut device_data = stream.alloc_zeros::<f32>(v.len()).unwrap();
                    stream.memcpy_htod(v.as_slice(), &mut device_data).unwrap();
                    device_data
                }
                TensorGraphNode::BroadcastAxis { axis } => {
                    let stream = self.device.default_stream();
                    let in_idx = graph.inputs(*node_idx)[0];
                    let in_val = self.values.get(&in_idx).unwrap();
                    let in_shape = graph.shapes.get(&in_idx).unwrap();
                    let out_shape = graph.shapes.get(node_idx).unwrap();
                    let out_size: usize = out_shape.iter().product();

                    assert_eq!(
                        in_shape.len(),
                        out_shape.len(),
                        "BroadcastAxis requires matching rank"
                    );
                    assert_eq!(
                        in_shape[*axis], 1,
                        "BroadcastAxis requires input axis size to be 1"
                    );

                    if out_shape.len() == 2 {
                        let m = out_shape[0];
                        let n = out_shape[1];
                        let m_u64 = m as u64;
                        let n_u64 = n as u64;
                        let mut out = stream.alloc_zeros::<f32>(out_size).unwrap();

                        if *axis == 0 {
                            let f = self.module.load_function("broadcast_row").unwrap();
                            let cfg = LaunchConfig::for_num_elems(n as u32);
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(in_val);
                            launcher.arg(&n_u64);
                            launcher.arg(&mut out);
                            launcher.arg(&m_u64);
                            launcher.arg(&n_u64);
                            unsafe { launcher.launch(cfg) }.expect("CUDA broadcast_row failed");
                        } else if *axis == 1 {
                            let f = self.module.load_function("broadcast_col").unwrap();
                            let cfg = LaunchConfig::for_num_elems(m as u32);
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(in_val);
                            launcher.arg(&m_u64);
                            launcher.arg(&mut out);
                            launcher.arg(&m_u64);
                            launcher.arg(&n_u64);
                            unsafe { launcher.launch(cfg) }.expect("CUDA broadcast_col failed");
                        } else {
                            panic!("BroadcastAxis axis out of bounds");
                        }
                        out
                    } else {
                        panic!("BroadcastAxis currently only supports 2D tensors on GPU");
                    }
                }
                TensorGraphNode::ReduceAxis { op, axis } => {
                    let in_idx = graph.inputs(*node_idx)[0];
                    let x_device = self.values.get(&in_idx).unwrap();
                    let in_shape = graph.shapes.get(&in_idx).unwrap();
                    let out_shape = graph.shapes.get(node_idx).unwrap();
                    let out_len: usize = out_shape.iter().product();

                    assert_eq!(in_shape.len(), out_shape.len(), "ReduceAxis preserves rank");
                    assert_eq!(out_shape[*axis], 1, "ReduceAxis output axis must be 1");

                    let stream = self.device.default_stream();

                    if in_shape.len() == 2 {
                        let m = in_shape[0];
                        let n = in_shape[1];
                        let m_u64 = m as u64;
                        let n_u64 = n as u64;
                        let in_len_u64 = x_device.len() as u64;
                        let mut out_device = stream.alloc_zeros::<f32>(out_len).unwrap();

                        if *axis == 1 {
                            let kernel_name = match op {
                                tensor::ReduceOp::Sum => "reduce_sum_rows",
                                tensor::ReduceOp::Max => "reduce_max_rows",
                                tensor::ReduceOp::Mean => "reduce_mean_rows",
                            };
                            let f = self.module.load_function(kernel_name).unwrap();
                            let cfg = LaunchConfig::for_num_elems(m as u32);
                            let out_len_u64 = m_u64;
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(x_device);
                            launcher.arg(&in_len_u64);
                            launcher.arg(&mut out_device);
                            launcher.arg(&out_len_u64);
                            launcher.arg(&n_u64);
                            unsafe { launcher.launch(cfg) }.expect("CUDA reduce rows failed");
                        } else if *axis == 0 {
                            let kernel_name = match op {
                                tensor::ReduceOp::Sum => "reduce_sum_cols",
                                tensor::ReduceOp::Max => "reduce_max_cols",
                                tensor::ReduceOp::Mean => "reduce_mean_cols",
                            };
                            let f = self.module.load_function(kernel_name).unwrap();
                            let cfg = LaunchConfig::for_num_elems(n as u32);
                            let out_len_u64 = n_u64;
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(x_device);
                            launcher.arg(&in_len_u64);
                            launcher.arg(&mut out_device);
                            launcher.arg(&m_u64);
                            launcher.arg(&out_len_u64);
                            unsafe { launcher.launch(cfg) }.expect("CUDA reduce cols failed");
                        } else {
                            panic!("ReduceAxis axis out of bounds");
                        }

                        out_device
                    } else {
                        panic!("ReduceAxis currently only supports 2D tensors on GPU");
                    }
                }
            };
            self.values.insert(*node_idx, result);
        }

        let last_node_idx = order.last().expect("Graph is empty");
        let out_device = self.values.get(last_node_idx).expect("Output not found");
        let mut out_host = vec![0.0f32; out_device.len()];
        self.device
            .default_stream()
            .memcpy_dtoh(out_device, &mut out_host)
            .unwrap();
        out_host
    }

    fn backward(
        &mut self,
        graph: &TensorGraph<f32>,
        loss_node: petgraph::graph::NodeIndex,
        seed_grad: Option<Vec<f32>>,
    ) -> BackwardResult<f32> {
        let order = graph.toposort();
        let _bwd_span = trace_span!("backward", nodes = order.len()).entered();

        // Backward pass - gradients computation
        let mut param_grads: HashMap<usize, Vec<f32>> = HashMap::new();

        // Initialize gradient for loss node
        let loss_shape = graph
            .shapes
            .get(&loss_node)
            .expect("Loss node shape missing");
        let seed = seed_grad.unwrap_or_else(|| vec![1.0f32; loss_shape.iter().product()]);
        let stream = self.device.default_stream();
        let mut seed_gpu = stream.alloc_zeros::<f32>(seed.len()).unwrap();
        stream.memcpy_htod(&seed, &mut seed_gpu).unwrap();
        self.grads.insert(loss_node, seed_gpu);

        // Backward pass in reverse topological order
        for &node_idx in order.iter().rev() {
            let Some(dy) = self.grads.get(&node_idx) else {
                continue;
            };
            let node = &graph[node_idx];
            match node {
                TensorGraphNode::Constant { .. } => {}
                TensorGraphNode::Input { .. } => {}
                TensorGraphNode::Parameter { id, .. } => {
                    let _span =
                        trace_span!("parameter_grad", node = node_idx.index(), size = dy.len())
                            .entered();
                    // Copy gradient to host for accumulation
                    let mut grad_host = vec![0.0f32; dy.len()];
                    self.device
                        .default_stream()
                        .memcpy_dtoh(dy, &mut grad_host)
                        .unwrap();
                    param_grads
                        .entry(*id)
                        .and_modify(|g| {
                            for (dst, src) in g.iter_mut().zip(grad_host.iter()) {
                                *dst += *src;
                            }
                        })
                        .or_insert(grad_host);
                }
                TensorGraphNode::Unary { op } => {
                    let x_idx = graph.inputs(node_idx)[0];
                    let dx = match op {
                        tensor::UnaryOp::Neg => self.neg_grad(dy),
                        tensor::UnaryOp::Exp => {
                            let y = self.values.get(&node_idx).unwrap();
                            self.exp_grad(dy, y)
                        }
                        tensor::UnaryOp::Log => {
                            let x = self.values.get(&x_idx).unwrap();
                            self.log_grad(dy, x)
                        }
                        tensor::UnaryOp::Relu => {
                            let x = self.values.get(&x_idx).unwrap();
                            self.relu_grad(dy, x)
                        }
                    };
                    self.accumulate_grad_gpu(x_idx, dx);
                }
                TensorGraphNode::Binary { op } => {
                    let ins = graph.inputs(node_idx);
                    let a_idx = ins[0];
                    let b_idx = ins[1];

                    let (da, db) = match op {
                        tensor::BinaryOp::Add => self.add_grad(dy),
                        tensor::BinaryOp::Sub => self.sub_grad(dy),
                        tensor::BinaryOp::Mul => {
                            let a_val = self.values.get(&a_idx).unwrap();
                            let b_val = self.values.get(&b_idx).unwrap();
                            self.mul_grad(dy, a_val, b_val)
                        }
                        tensor::BinaryOp::Div => {
                            let a_val = self.values.get(&a_idx).unwrap();
                            let b_val = self.values.get(&b_idx).unwrap();
                            self.div_grad(dy, a_val, b_val)
                        }
                    };

                    self.accumulate_grad_gpu(a_idx, da);
                    self.accumulate_grad_gpu(b_idx, db);
                }
                TensorGraphNode::MatMul => {
                    let _span = trace_span!("matmul", node = node_idx.index()).entered();
                    let ins = graph.inputs(node_idx);
                    let a_idx = ins[0];
                    let b_idx = ins[1];
                    let a_val = self.values.get(&a_idx).unwrap();
                    let b_val = self.values.get(&b_idx).unwrap();
                    let a_shape = graph.shapes.get(&a_idx).unwrap();
                    let b_shape = graph.shapes.get(&b_idx).unwrap();

                    let m = a_shape[0];
                    let n = b_shape[1];
                    let k = a_shape[1];

                    let da = self.matmul_grad_left(dy, b_val, m, n, k);
                    let db = self.matmul_grad_right(a_val, dy, m, n, k);
                    self.accumulate_grad_gpu(a_idx, da);
                    self.accumulate_grad_gpu(b_idx, db);
                }
                TensorGraphNode::BroadcastAxis { axis } => {
                    let x_idx = graph.inputs(node_idx)[0];
                    let x_shape = graph.shapes.get(&x_idx).unwrap();
                    let y_shape = graph.shapes.get(&node_idx).unwrap();

                    if x_shape.len() == 2 {
                        let m = y_shape[0];
                        let n = y_shape[1];
                        let m_u64 = m as u64;
                        let n_u64 = n as u64;
                        let in_len_u64 = dy.len() as u64;

                        let stream = self.device.default_stream();
                        let out_len = x_shape.iter().product();
                        let mut dx = stream.alloc_zeros::<f32>(out_len).unwrap();

                        if *axis == 1 {
                            let f = self.module.load_function("reduce_sum_rows").unwrap();
                            let cfg = LaunchConfig::for_num_elems(m as u32);
                            let out_len_u64 = m_u64;
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(dy);
                            launcher.arg(&in_len_u64);
                            launcher.arg(&mut dx);
                            launcher.arg(&out_len_u64);
                            launcher.arg(&n_u64);
                            unsafe { launcher.launch(cfg) }.expect("CUDA reduce rows failed");
                        } else if *axis == 0 {
                            let f = self.module.load_function("reduce_sum_cols").unwrap();
                            let cfg = LaunchConfig::for_num_elems(n as u32);
                            let out_len_u64 = n_u64;
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(dy);
                            launcher.arg(&in_len_u64);
                            launcher.arg(&mut dx);
                            launcher.arg(&m_u64);
                            launcher.arg(&out_len_u64);
                            unsafe { launcher.launch(cfg) }.expect("CUDA reduce cols failed");
                        } else {
                            panic!("BroadcastAxis backward: axis out of bounds");
                        }

                        self.accumulate_grad_gpu(x_idx, dx);
                    } else {
                        panic!("BroadcastAxis backward currently only supports 2D tensors on GPU");
                    }
                }
                TensorGraphNode::ReduceAxis { op, axis } => {
                    let x_idx = graph.inputs(node_idx)[0];
                    let x_shape = graph.shapes.get(&x_idx).unwrap();

                    match op {
                        tensor::ReduceOp::Sum => {
                            if x_shape.len() == 2 {
                                let m = x_shape[0];
                                let n = x_shape[1];
                                let m_u64 = m as u64;
                                let n_u64 = n as u64;
                                let stream = self.device.default_stream();
                                let out_size = x_shape.iter().product();
                                let mut dx = stream.alloc_zeros::<f32>(out_size).unwrap();

                                if *axis == 0 {
                                    let f = self.module.load_function("broadcast_row").unwrap();
                                    let cfg = LaunchConfig::for_num_elems(n as u32);
                                    let mut launcher = stream.launch_builder(&f);
                                    launcher.arg(dy);
                                    launcher.arg(&n_u64);
                                    launcher.arg(&mut dx);
                                    launcher.arg(&m_u64);
                                    launcher.arg(&n_u64);
                                    unsafe { launcher.launch(cfg) }
                                        .expect("CUDA broadcast_row failed");
                                } else if *axis == 1 {
                                    let f = self.module.load_function("broadcast_col").unwrap();
                                    let cfg = LaunchConfig::for_num_elems(m as u32);
                                    let mut launcher = stream.launch_builder(&f);
                                    launcher.arg(dy);
                                    launcher.arg(&m_u64);
                                    launcher.arg(&mut dx);
                                    launcher.arg(&m_u64);
                                    launcher.arg(&n_u64);
                                    unsafe { launcher.launch(cfg) }
                                        .expect("CUDA broadcast_col failed");
                                } else {
                                    panic!("ReduceAxis backward: axis out of bounds");
                                }

                                self.accumulate_grad_gpu(x_idx, dx);
                            } else {
                                panic!(
                                    "ReduceAxis backward currently only supports 2D tensors on GPU"
                                );
                            }
                        }
                        tensor::ReduceOp::Mean => {
                            let mut dy_host = vec![0.0f32; dy.len()];
                            self.device
                                .default_stream()
                                .memcpy_dtoh(dy, &mut dy_host)
                                .unwrap();
                            let mut y_aligned_shape = x_shape.clone();
                            y_aligned_shape[*axis] = 1;
                            let mut dx_host = crate::expand_to(&dy_host, &y_aligned_shape, x_shape);
                            let axis_size = x_shape[*axis];
                            for v in dx_host.iter_mut() {
                                *v /= axis_size as f32;
                            }
                            let stream = self.device.default_stream();
                            let mut dx = stream.alloc_zeros::<f32>(dx_host.len()).unwrap();
                            stream.memcpy_htod(&dx_host, &mut dx).unwrap();
                            self.accumulate_grad_gpu(x_idx, dx);
                        }
                        tensor::ReduceOp::Max => {
                            let zeros = vec![0.0f32; x_shape.iter().product()];
                            let stream = self.device.default_stream();
                            let mut dx = stream.alloc_zeros::<f32>(zeros.len()).unwrap();
                            stream.memcpy_htod(&zeros, &mut dx).unwrap();
                            self.accumulate_grad_gpu(x_idx, dx);
                        }
                    }
                }
            }
        }

        // Get final loss value from GPU
        let loss_gpu = self.values.get(&loss_node).unwrap();
        let mut loss_value = vec![0.0f32; loss_gpu.len()];
        self.device
            .default_stream()
            .memcpy_dtoh(loss_gpu, &mut loss_value)
            .unwrap();

        // Convert GPU gradients to host
        let mut grads_by_node = HashMap::new();
        for (node_idx, grad_gpu) in self.grads.drain() {
            let mut grad_host = vec![0.0f32; grad_gpu.len()];
            self.device
                .default_stream()
                .memcpy_dtoh(&grad_gpu, &mut grad_host)
                .unwrap();
            grads_by_node.insert(node_idx, grad_host);
        }

        BackwardResult {
            grads_by_node,
            grads_by_param: param_grads,
            loss_value,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use std::sync::{Arc, Mutex};
    use tensor::Constant;
    use tensor::Parameter;
    use tensor::TensorExpr;

    use super::*;
    use crate::SimpleExecutor;

    const EPSILON: f32 = 1e-5;

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr) => {
            if $a
                .iter()
                .zip($b.iter())
                .any(|(a, b)| (a - b).abs() > EPSILON)
            {
                panic!(
                    "assertion failed: `(left ~= right)` (left: `{:?}`, right: `{:?}`)",
                    $a, $b
                );
            }
        };
    }

    type UnaryApply = fn(Constant<f32>) -> TensorExpr<f32>;
    fn neg_node(a: Constant<f32>) -> TensorExpr<f32> {
        -TensorExpr::from(a)
    }
    fn exp_node(a: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a).exp()
    }
    fn log_node(a: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a).log()
    }
    fn relu_node(a: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a).relu()
    }

    #[rstest]
    #[case::neg(
          vec![1.0f32, -2.0, 3.0, -4.0],
          vec![-1.0, 2.0, -3.0, 4.0],
          neg_node as UnaryApply
      )]
    #[case::exp(
          vec![0.0f32, 1.0, 2.0, 3.0],
          vec![1.0, std::f32::consts::E, std::f32::consts::E.powi(2), std::f32::consts::E.powi(3)],
          exp_node as UnaryApply
      )]
    #[case::log(
          vec![1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2), std::f32::consts::E.powi(3)],
          vec![0.0, 1.0, 2.0, 3.0],
          log_node as UnaryApply
      )]
    #[case::relu(
          vec![-1.0f32, 2.0, -3.0, 4.0],
          vec![0.0, 2.0, 0.0, 4.0],
          relu_node as UnaryApply
      )]
    fn elementwise_unary_f32(
        #[case] input: Vec<f32>,
        #[case] expected: Vec<f32>,
        #[case] apply: UnaryApply,
    ) {
        let a = Constant::new(input, vec![2, 2]);
        let node = apply(a);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let mut exec = CudaExecutor::new();
        let result = exec.forward(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    type BinaryApply = fn(Constant<f32>, Constant<f32>) -> TensorExpr<f32>;
    fn add_node(a: Constant<f32>, b: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a) + TensorExpr::from(b)
    }
    fn sub_node(a: Constant<f32>, b: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a) - TensorExpr::from(b)
    }
    fn mul_node(a: Constant<f32>, b: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a) * TensorExpr::from(b)
    }
    fn div_node(a: Constant<f32>, b: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a) / TensorExpr::from(b)
    }

    #[rstest]
    #[case::add(
          vec![1.0f32, 2.0, 3.0, 4.0],
          vec![5.0f32, 6.0, 7.0, 8.0],
          vec![6.0, 8.0, 10.0, 12.0],
          add_node as BinaryApply
      )]
    #[case::sub(
          vec![5.0f32, 6.0, 7.0, 8.0],
          vec![1.0f32, 2.0, 3.0, 4.0],
          vec![4.0, 4.0, 4.0, 4.0],
          sub_node as BinaryApply
      )]
    #[case::mul(
          vec![1.0f32, 2.0, 3.0, 4.0],
          vec![5.0f32, 6.0, 7.0, 8.0],
          vec![5.0, 12.0, 21.0, 32.0],
          mul_node as BinaryApply
      )]
    #[case::div(
          vec![5.0f32, 12.0, 21.0, 32.0],
          vec![1.0f32, 2.0, 3.0, 4.0],
          vec![5.0, 6.0, 7.0, 8.0],
          div_node as BinaryApply
      )]
    fn elementwise_binary_f32(
        #[case] a_in: Vec<f32>,
        #[case] b_in: Vec<f32>,
        #[case] expected: Vec<f32>,
        #[case] apply: BinaryApply,
    ) {
        let a = Constant::new(a_in, vec![2, 2]);
        let b = Constant::new(b_in, vec![2, 2]);
        let node = apply(a, b);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let mut exec = CudaExecutor::new();
        let result = exec.forward(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test]
    fn matmul_f32() {
        let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Constant::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let node = TensorExpr::from(a).matmul(b);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let mut cpu = SimpleExecutor::new();
        let expected = cpu.forward(&graph, Default::default());

        let mut cuda = CudaExecutor::new();
        let result = cuda.forward(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test]
    fn reduce_sum_rows_axis1_2x3() {
        let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let node = TensorExpr::from(a).reduce_sum(1);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let mut cpu = SimpleExecutor::new();
        let expected = cpu.forward(&graph, Default::default());

        let mut cuda = CudaExecutor::new();
        let result = cuda.forward(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test]
    fn reduce_sum_cols_axis0_2x3() {
        let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let node = TensorExpr::from(a).reduce_sum(0);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let mut cpu = SimpleExecutor::new();
        let expected = cpu.forward(&graph, Default::default());

        let mut cuda = CudaExecutor::new();
        let result = cuda.forward(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test]
    fn reduce_sum_all_scalar() {
        let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let node = TensorExpr::from(a).reduce_sum(1).reduce_sum(0);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let mut cpu = SimpleExecutor::new();
        let expected = cpu.forward(&graph, Default::default());

        let mut cuda = CudaExecutor::new();
        let result = cuda.forward(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test_log::test]
    fn reduce_max_and_mean_match_cpu() {
        let a = Constant::new(vec![1.0, -2.0, 3.5, 0.5, 10.0, -1.0], vec![2, 3]);

        // max axis 1
        let node_max_r = TensorExpr::from(a.clone()).reduce_max(1);
        let mut g1 = TensorGraph::new();
        node_max_r.lower_to_graph(&mut g1);
        let mut cpu = SimpleExecutor::new();
        let exp1 = cpu.forward(&g1, Default::default());
        let mut cuda = CudaExecutor::new();
        let res1 = cuda.forward(&g1, Default::default());
        assert_approx_eq!(res1, exp1);

        // max axis 0
        let node_max_c = TensorExpr::from(a.clone()).reduce_max(0);
        let mut g2 = TensorGraph::new();
        node_max_c.lower_to_graph(&mut g2);
        let exp2 = cpu.forward(&g2, Default::default());
        let res2 = cuda.forward(&g2, Default::default());
        assert_approx_eq!(res2, exp2);

        // mean axis 1
        let a3 = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let node_mean_r = TensorExpr::from(a3.clone()).reduce_mean(1);
        let mut g3 = TensorGraph::new();
        node_mean_r.lower_to_graph(&mut g3);
        let exp3 = cpu.forward(&g3, Default::default());
        let res3 = cuda.forward(&g3, Default::default());
        assert_approx_eq!(res3, exp3);

        // mean all via mean_all()
        let a4 = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let node_mean_all = TensorExpr::from(a4).mean_all();
        let mut g4 = TensorGraph::new();
        node_mean_all.lower_to_graph(&mut g4);
        let exp4 = cpu.forward(&g4, Default::default());
        let res4 = cuda.forward(&g4, Default::default());
        assert_approx_eq!(res4, exp4);
    }

    #[test]
    fn parameter_passthrough_matches_cpu() {
        let p = TensorExpr::parameter(vec![1.0f32, -2.0, 3.5, 0.5], vec![2, 2]);
        let mut g = TensorGraph::new();
        p.lower_to_graph(&mut g);

        let mut cpu = SimpleExecutor::new();
        let exp = cpu.forward(&g, Default::default());

        let mut cuda = CudaExecutor::new();
        let res = cuda.forward(&g, Default::default());

        assert_approx_eq!(res, exp);
    }

    #[test_log::test]
    fn broadcast_col_vector_to_matrix_matches_cpu() {
        // [2,1] -> [2,3]
        let a = Constant::new(vec![1.0f32, 2.0], vec![2, 1]);
        let node = TensorExpr::from(a).broadcast(vec![2, 3]);
        let mut g = TensorGraph::new();
        node.lower_to_graph(&mut g);

        let mut cpu = SimpleExecutor::new();
        let exp = cpu.forward(&g, Default::default());

        let mut cuda = CudaExecutor::new();
        let res = cuda.forward(&g, Default::default());

        assert_approx_eq!(res, exp);
    }

    #[test]
    fn broadcast_row_vector_to_matrix_matches_cpu() {
        // [1,3] -> [2,3]
        let a = Constant::new(vec![1.0f32, 2.0, 3.0], vec![1, 3]);
        let node = TensorExpr::from(a).broadcast(vec![2, 3]);
        let mut g = TensorGraph::new();
        node.lower_to_graph(&mut g);

        let mut cpu = SimpleExecutor::new();
        let exp = cpu.forward(&g, Default::default());

        let mut cuda = CudaExecutor::new();
        let res = cuda.forward(&g, Default::default());

        assert_approx_eq!(res, exp);
    }

    #[test_log::test]
    fn broadcast_scalar_to_matrix_matches_cpu() {
        // [] -> [2,3]
        let a = Constant::new(vec![3.0f32], vec![]);
        let node = TensorExpr::from(a).broadcast(vec![2, 3]);
        let mut g = TensorGraph::new();
        node.lower_to_graph(&mut g);

        let mut cpu = SimpleExecutor::new();
        let exp = cpu.forward(&g, Default::default());

        let mut cuda = CudaExecutor::new();
        let res = cuda.forward(&g, Default::default());

        assert_approx_eq!(res, exp);
    }

    #[test]
    fn test_cuda_backward_simple_mul() {
        // Test CUDA backward pass with simple x^2 computation
        // This verifies GPU gradients match CPU gradients

        let mut graph = tensor::graph::TensorGraph::new();

        // Add parameter node with data = [2.0]
        let param_id = 0;
        let param_data = Arc::new(Mutex::new(vec![2.0f32]));
        let param_node = graph
            .graph
            .add_node(tensor::graph::TensorGraphNode::Parameter {
                id: param_id,
                data: param_data.clone(),
            });
        graph.shapes.insert(param_node, vec![1]);

        // Add multiplication node: x * x
        let mul_node = graph
            .graph
            .add_node(tensor::graph::TensorGraphNode::Binary {
                op: tensor::BinaryOp::Mul,
            });
        graph.shapes.insert(mul_node, vec![1]);

        // Connect parameter to both inputs of multiplication
        graph.graph.add_edge(param_node, mul_node, 0);
        graph.graph.add_edge(param_node, mul_node, 1);

        let inputs = std::collections::HashMap::new();

        // Test with CPU first
        let mut cpu = SimpleExecutor::new();
        cpu.forward(&graph, inputs.clone());
        let cpu_result = cpu.backward(&graph, mul_node, Some(vec![1.0f32]));

        // Test with CUDA
        let mut cuda = CudaExecutor::new();
        cuda.forward(&graph, inputs.clone());
        let cuda_result = cuda.backward(&graph, mul_node, Some(vec![1.0f32]));

        // Check that gradients match
        assert_eq!(
            cpu_result.grads_by_param.len(),
            cuda_result.grads_by_param.len()
        );
        assert_eq!(cpu_result.grads_by_param.len(), 1);

        let cpu_grad = cpu_result.grads_by_param.get(&param_id).unwrap();
        let cuda_grad = cuda_result.grads_by_param.get(&param_id).unwrap();

        // Gradient of x^2 at x=2 should be 2*x = 4
        assert_approx_eq!(*cpu_grad, vec![4.0f32]);
        assert_approx_eq!(*cuda_grad, vec![4.0f32]);

        // Check loss values match
        assert_approx_eq!(cpu_result.loss_value, cuda_result.loss_value);
    }

    #[test]
    fn test_cuda_backward_binary_add() {
        // Test CUDA backward pass with binary addition
        // This should use GPU kernels for both forward and backward

        let a = Constant::new(vec![1.0f32, 2.0], vec![2]);
        let b = Constant::new(vec![3.0f32, 4.0], vec![2]);
        let node = TensorExpr::from(a) + TensorExpr::from(b); // Should produce [4.0, 6.0]

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);

        // Test with CPU first
        let mut cpu = SimpleExecutor::new();
        cpu.forward(&graph, Default::default());
        let cpu_result = cpu.backward(&graph, loss_node, Some(vec![1.0f32, 1.0]));

        // Test with CUDA
        let mut cuda = CudaExecutor::new();
        cuda.forward(&graph, Default::default());
        let cuda_result = cuda.backward(&graph, loss_node, Some(vec![1.0f32, 1.0]));

        // Check that results match
        assert_approx_eq!(cpu_result.loss_value, cuda_result.loss_value);
        assert_eq!(
            cpu_result.grads_by_param.len(),
            cuda_result.grads_by_param.len()
        );
    }

    #[test]
    fn test_cuda_matmul_grad_small() {
        // Test matmul backward pass with small matrices
        // Forward: A[2,3] @ B[3,2] = C[2,2]
        // Backward: dA = dC @ B^T, dB = A^T @ dC

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]

        let a = Parameter::new(a_data, vec![2, 3]);
        let a_id = a.id();
        let b = Parameter::new(b_data, vec![3, 2]);
        let b_id = b.id();
        let node = TensorExpr::from(a).matmul(b);

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);

        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32, 1.0, 1.0, 1.0]; // [2, 2]

        // Test with CPU first
        let mut cpu = SimpleExecutor::new();
        cpu.forward(&graph, inputs.clone());
        let cpu_result = cpu.backward(&graph, loss_node, Some(seed_grad.clone()));

        // Test with CUDA
        let mut cuda = CudaExecutor::new();
        cuda.forward(&graph, inputs.clone());
        let cuda_result = cuda.backward(&graph, loss_node, Some(seed_grad));

        // Check that gradients match
        assert_eq!(
            cpu_result.grads_by_param.len(),
            cuda_result.grads_by_param.len()
        );
        assert_eq!(cpu_result.grads_by_param.len(), 2);

        let cpu_grad_a = cpu_result.grads_by_param.get(&a_id).unwrap();
        let cuda_grad_a = cuda_result.grads_by_param.get(&a_id).unwrap();
        assert_eq!(cpu_grad_a.len(), 6); // [2, 3]
        assert_approx_eq!(*cpu_grad_a, *cuda_grad_a);

        let cpu_grad_b = cpu_result.grads_by_param.get(&b_id).unwrap();
        let cuda_grad_b = cuda_result.grads_by_param.get(&b_id).unwrap();
        assert_eq!(cpu_grad_b.len(), 6); // [3, 2]
        assert_approx_eq!(*cpu_grad_b, *cuda_grad_b);

        // Check loss values match
        assert_approx_eq!(cpu_result.loss_value, cuda_result.loss_value);
    }

    #[test]
    fn test_cuda_matmul_grad_mnist_size() {
        // Test matmul backward with MNIST-like dimensions
        // Forward: images[128, 784] @ W[784, 128] = hidden[128, 128]

        let batch_size = 128;
        let input_dim = 784;
        let hidden_dim = 128;

        // Initialize with small values to avoid numerical issues
        let x_data = vec![0.01f32; batch_size * input_dim];
        let w_data = vec![0.01f32; input_dim * hidden_dim];

        let x = Parameter::new(x_data, vec![batch_size, input_dim]);
        let x_id = x.id();
        let w = Parameter::new(w_data, vec![input_dim, hidden_dim]);
        let w_id = w.id();
        let node = TensorExpr::from(x).matmul(w);

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);

        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32; batch_size * hidden_dim]; // [128, 128]

        // Test with CPU first
        let mut cpu = SimpleExecutor::new();
        cpu.forward(&graph, inputs.clone());
        let cpu_result = cpu.backward(&graph, loss_node, Some(seed_grad.clone()));

        // Test with CUDA
        let mut cuda = CudaExecutor::new();
        cuda.forward(&graph, inputs.clone());
        let cuda_result = cuda.backward(&graph, loss_node, Some(seed_grad));

        // Check that gradients match
        assert_eq!(
            cpu_result.grads_by_param.len(),
            cuda_result.grads_by_param.len()
        );
        assert_eq!(cpu_result.grads_by_param.len(), 2);

        let cpu_grad_x = cpu_result.grads_by_param.get(&x_id).unwrap();
        let cuda_grad_x = cuda_result.grads_by_param.get(&x_id).unwrap();
        assert_eq!(cpu_grad_x.len(), batch_size * input_dim); // [128, 784]
        assert_approx_eq!(*cpu_grad_x, *cuda_grad_x);

        let cpu_grad_w = cpu_result.grads_by_param.get(&w_id).unwrap();
        let cuda_grad_w = cuda_result.grads_by_param.get(&w_id).unwrap();
        assert_eq!(cpu_grad_w.len(), input_dim * hidden_dim); // [784, 128]
        assert_approx_eq!(*cpu_grad_w, *cuda_grad_w);

        // Check loss values match
        assert_approx_eq!(cpu_result.loss_value, cuda_result.loss_value);
    }

    #[test]
    fn test_cuda_matmul_grad_asymmetric() {
        // Test with very asymmetric matrices to catch dimension errors
        // Forward: A[5, 100] @ B[100, 3] = C[5, 3]

        let m = 5;
        let k = 100;
        let n = 3;

        let a_data = vec![0.1f32; m * k];
        let b_data = vec![0.1f32; k * n];

        let a = Parameter::new(a_data, vec![m, k]);
        let a_id = a.id();
        let b = Parameter::new(b_data, vec![k, n]);
        let b_id = b.id();
        let node = TensorExpr::from(a).matmul(b);

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);

        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32; m * n]; // [5, 3]

        // Test with CPU first
        let mut cpu = SimpleExecutor::new();
        cpu.forward(&graph, inputs.clone());
        let cpu_result = cpu.backward(&graph, loss_node, Some(seed_grad.clone()));

        // Test with CUDA
        let mut cuda = CudaExecutor::new();
        cuda.forward(&graph, inputs.clone());
        let cuda_result = cuda.backward(&graph, loss_node, Some(seed_grad));

        // Check that gradients match
        assert_eq!(
            cpu_result.grads_by_param.len(),
            cuda_result.grads_by_param.len()
        );
        assert_eq!(cpu_result.grads_by_param.len(), 2);

        let cpu_grad_a = cpu_result.grads_by_param.get(&a_id).unwrap();
        let cuda_grad_a = cuda_result.grads_by_param.get(&a_id).unwrap();
        assert_eq!(cpu_grad_a.len(), m * k); // [5, 100]
        assert_approx_eq!(*cpu_grad_a, *cuda_grad_a);

        let cpu_grad_b = cpu_result.grads_by_param.get(&b_id).unwrap();
        let cuda_grad_b = cuda_result.grads_by_param.get(&b_id).unwrap();
        assert_eq!(cpu_grad_b.len(), k * n); // [100, 3]
        assert_approx_eq!(*cpu_grad_b, *cuda_grad_b);

        // Check loss values match
        assert_approx_eq!(cpu_result.loss_value, cuda_result.loss_value);
    }

    #[test]
    fn test_cuda_transpose_correctness() {
        // Test that transpose works correctly for different sizes
        let cuda = CudaExecutor::new();
        let stream = cuda.device.default_stream();

        // Test 1: 2x3 matrix
        let data_2x3 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut input_2x3 = stream.alloc_zeros::<f32>(6).unwrap();
        stream.memcpy_htod(&data_2x3, &mut input_2x3).unwrap();

        let output_3x2 = cuda.transpose(&input_2x3, 2, 3);
        let mut result_3x2 = vec![0.0f32; 6];
        stream.memcpy_dtoh(&output_3x2, &mut result_3x2).unwrap();

        // Expected: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let expected_3x2 = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        assert_approx_eq!(result_3x2, expected_3x2);

        // Test 2: 3x2 matrix (reverse of above)
        let data_3x2 = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mut input_3x2 = stream.alloc_zeros::<f32>(6).unwrap();
        stream.memcpy_htod(&data_3x2, &mut input_3x2).unwrap();

        let output_2x3 = cuda.transpose(&input_3x2, 3, 2);
        let mut result_2x3 = vec![0.0f32; 6];
        stream.memcpy_dtoh(&output_2x3, &mut result_2x3).unwrap();

        let expected_2x3 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_approx_eq!(result_2x3, expected_2x3);
    }
}
