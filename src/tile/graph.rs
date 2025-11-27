use super::builder::TileIRBuilder;
use super::ir::{DType, Dim, Expr, MatMulLayout, ReduceOp, TileIR};
use crate::graph::{TensorGraph, TensorGraphNode};
use crate::tensor;
use petgraph::{Direction, Graph, graph::NodeIndex};
use std::collections::HashMap;

#[allow(dead_code)]
pub struct TileGraph {
    pub graph: Graph<TileIR, usize>,
}

impl<G> From<TensorGraph<f32, G>> for TileGraph {
    fn from(tensor_graph: TensorGraph<f32, G>) -> Self {
        // Pre-compute input shapes for MatMul, BroadcastAxis, ReduceAxis, and Transpose nodes
        let mut op_input_shapes: HashMap<NodeIndex, Vec<Vec<usize>>> = HashMap::new();
        for idx in tensor_graph.graph.node_indices() {
            if matches!(
                tensor_graph.graph[idx],
                TensorGraphNode::MatMul
                    | TensorGraphNode::BroadcastAxis { .. }
                    | TensorGraphNode::ReduceAxis { .. }
                    | TensorGraphNode::Transpose
            ) {
                let inputs: Vec<Vec<usize>> = tensor_graph
                    .graph
                    .neighbors_directed(idx, Direction::Incoming)
                    .filter_map(|pred_idx| tensor_graph.shapes.get(&pred_idx).cloned())
                    .collect();
                op_input_shapes.insert(idx, inputs);
            }
        }

        Self {
            graph: tensor_graph.graph.map_owned(
                |idx, node| {
                    let shape = tensor_graph.shapes.get(&idx).unwrap();
                    // Get input shapes for operations that need them
                    let input_shapes = op_input_shapes
                        .get(&idx)
                        .map(|v| v.iter().map(|s| s.as_slice()).collect::<Vec<_>>())
                        .unwrap_or_default();
                    Self::lower_node(node, shape, &input_shapes)
                },
                |_, e| e,
            ),
        }
    }
}

impl TileGraph {
    #[allow(dead_code)]
    fn lower_node(
        node: TensorGraphNode<f32>,
        shape: &[usize],
        input_shapes: &[&[usize]],
    ) -> TileIR {
        match node {
            TensorGraphNode::Constant { data } => Self::lower_constant(data, shape),
            TensorGraphNode::Input { name } => Self::lower_input(name, shape),
            TensorGraphNode::Parameter { id, data } => Self::lower_parameter(id, data, shape),
            TensorGraphNode::Unary { op } => Self::lower_unary(op, shape),
            TensorGraphNode::Binary { op } => Self::lower_binary(op, shape),
            TensorGraphNode::MatMul => {
                let _m = shape[0];
                let n = shape[1];
                let k = input_shapes[0][1];
                // Pad dimensions to multiples of TILE_SIZE (16)
                const TILE_SIZE: usize = 16;
                let m_padded = _m.div_ceil(TILE_SIZE) * TILE_SIZE;
                let n_padded = n.div_ceil(TILE_SIZE) * TILE_SIZE;
                let k_padded = k.div_ceil(TILE_SIZE) * TILE_SIZE;
                Self::lower_matmul(m_padded, n_padded, k_padded, false, false)
            }
            TensorGraphNode::Transpose => {
                let input_shape = input_shapes.first().map(|s| s.to_vec()).unwrap_or_default();
                Self::lower_transpose(input_shape, shape)
            }
            TensorGraphNode::BroadcastAxis { axis } => {
                let input_shape = input_shapes.first().map(|s| s.to_vec()).unwrap_or_default();
                Self::lower_broadcast_axis(axis, input_shape, shape)
            }
            TensorGraphNode::ReduceAxis { op, axis } => {
                let input_shape = input_shapes.first().map(|s| s.to_vec()).unwrap_or_default();
                Self::lower_reduce_axis(op, axis, input_shape, shape)
            }
            TensorGraphNode::Gt => Self::lower_gt(shape),
            TensorGraphNode::Mask => Self::lower_mask(shape),
        }
    }

    #[allow(dead_code)]
    fn lower_matmul(_m: usize, n: usize, k: usize, transpose_a: bool, transpose_b: bool) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("matmul");

        // Add parameters for input and output matrices
        builder.add_param("A", DType::F32, true);
        builder.add_param("B", DType::F32, true);
        builder.add_param("C", DType::F32, false);

        // Tile sizes for shared memory
        let tile_m = 16;
        let tile_n = 16;
        let tile_k = 16;

        // Allocate shared memory for input tiles
        let a_smem = builder.alloc_shared(DType::F32, tile_m, tile_k);
        let b_smem = builder.alloc_shared(DType::F32, tile_k, tile_n);

        // Allocate register tiles for computation
        let a_reg = builder.alloc_register(DType::F32, tile_m, tile_k);
        let b_reg = builder.alloc_register(DType::F32, tile_k, tile_n);
        let c_reg = builder.alloc_register(DType::F32, tile_m, tile_n);

        // Initialize accumulator
        builder.zero(c_reg);

        // Calculate number of K tiles needed
        let k_tiles = k.div_ceil(tile_k);

        // Main tiling loop over K dimension
        builder.for_loop("k_tile", 0, k_tiles as i64, |builder, k_var| {
            // Load A tile to shared memory
            // Each thread loads one element: A[base_row + threadIdx.y][base_col + threadIdx.x]
            // For A: base_row = blockIdx.y * tile_m, base_col = k_tile * tile_k
            let a_row = Expr::Add(
                Box::new(Expr::BlockIdx(Dim::Y) * tile_m),
                Box::new(Expr::ThreadIdx(Dim::Y)),
            );
            let a_col = Expr::Add(
                Box::new(Expr::Var(k_var.clone()) * tile_k),
                Box::new(Expr::ThreadIdx(Dim::X)),
            );
            // Linear offset for A (row-major): row * K + col
            let a_offset = Expr::Add(Box::new(a_row * k), Box::new(a_col));
            builder.load_global_to_shared(a_smem, "A", a_offset, Expr::Const(0));

            // Load B tile to shared memory
            // Each thread loads one element: B[base_row + threadIdx.y][base_col + threadIdx.x]
            // For B: base_row = k_tile * tile_k, base_col = blockIdx.x * tile_n
            let b_row = Expr::Add(
                Box::new(Expr::Var(k_var) * tile_k),
                Box::new(Expr::ThreadIdx(Dim::Y)),
            );
            let b_col = Expr::Add(
                Box::new(Expr::BlockIdx(Dim::X) * tile_n),
                Box::new(Expr::ThreadIdx(Dim::X)),
            );
            // Linear offset for B (row-major): row * N + col
            let b_offset = Expr::Add(Box::new(b_row * n), Box::new(b_col));
            builder.load_global_to_shared(b_smem, "B", b_offset, Expr::Const(0));

            builder.barrier();

            // Load from shared memory to registers
            builder.load_shared_to_register(a_reg, a_smem);
            builder.load_shared_to_register(b_reg, b_smem);

            // Compute: C_reg += A_reg @ B_reg
            let layout = match (transpose_a, transpose_b) {
                (false, false) => MatMulLayout::NN,
                (false, true) => MatMulLayout::NT,
                (true, false) => MatMulLayout::TN,
                (true, true) => MatMulLayout::TT,
            };

            builder.matmul(c_reg, a_reg, b_reg, layout);

            builder.barrier();
        });

        // Store result to global memory
        // Each thread stores one element of the output tile
        // Global row = blockIdx.y * tile_m + threadIdx.y
        // Global col = blockIdx.x * tile_n + threadIdx.x
        // Linear offset = (global_row) * N + (global_col)
        let global_row = Expr::Add(
            Box::new(Expr::BlockIdx(Dim::Y) * tile_m),
            Box::new(Expr::ThreadIdx(Dim::Y)),
        );
        let global_col = Expr::Add(
            Box::new(Expr::BlockIdx(Dim::X) * tile_n),
            Box::new(Expr::ThreadIdx(Dim::X)),
        );
        let linear_offset = Expr::Add(Box::new(global_row * n), Box::new(global_col));
        builder.store("C", c_reg, linear_offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_constant(_data: std::sync::Arc<Vec<f32>>, shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("constant");
        builder.add_param("constant_data", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = shape.iter().product();
        let tile_const = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_out = builder.alloc_register(DType::F32, total_elements, 1);

        builder.load_global_to_shared(tile_const, "constant_data", Expr::Const(0), Expr::Const(0));
        builder.add(tile_out, tile_const, tile_const);
        builder.store("output", tile_out, Expr::Const(0), Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_input(name: &'static str, shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel(&format!("input_{}", name));
        builder.add_param(name, DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = shape.iter().product();
        let tile_in = builder.alloc_register(DType::F32, total_elements, 1);

        builder.load_global_to_shared(tile_in, name, Expr::Const(0), Expr::Const(0));
        builder.store("output", tile_in, Expr::Const(0), Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_parameter(
        _id: usize,
        _data: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
        shape: &[usize],
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("parameter");
        builder.add_param("param", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = shape.iter().product();
        let tile_param = builder.alloc_register(DType::F32, total_elements, 1);

        builder.load_global_to_shared(tile_param, "param", Expr::Const(0), Expr::Const(0));
        builder.store("output", tile_param, Expr::Const(0), Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_unary(op: tensor::UnaryOp, shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        let kernel_name = match op {
            tensor::UnaryOp::Neg => "neg",
            tensor::UnaryOp::Exp => "exp",
            tensor::UnaryOp::Log => "log",
            tensor::UnaryOp::Relu => "relu",
        };
        builder.start_kernel(kernel_name);
        builder.add_param("input", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = shape.iter().product();
        let tile_in = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_out = builder.alloc_register(DType::F32, total_elements, 1);

        // Each thread processes one element using thread index
        let offset: Expr = Expr::ThreadIdx(Dim::X);
        builder.load_global_to_shared(tile_in, "input", offset.clone(), Expr::Const(0));

        match op {
            tensor::UnaryOp::Neg => builder.neg(tile_out, tile_in),
            tensor::UnaryOp::Exp => builder.exp(tile_out, tile_in),
            tensor::UnaryOp::Log => builder.log(tile_out, tile_in),
            tensor::UnaryOp::Relu => builder.relu(tile_out, tile_in),
        }

        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_binary(op: tensor::BinaryOp, shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        let kernel_name = match op {
            tensor::BinaryOp::Add => "add",
            tensor::BinaryOp::Sub => "sub",
            tensor::BinaryOp::Mul => "mul",
            tensor::BinaryOp::Div => "div",
        };
        builder.start_kernel(kernel_name);
        builder.add_param("a", DType::F32, true);
        builder.add_param("b", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = shape.iter().product();
        let tile_a = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_b = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_out = builder.alloc_register(DType::F32, total_elements, 1);

        // Each thread processes one element using thread index
        let offset: Expr = Expr::ThreadIdx(Dim::X);
        builder.load_global_to_shared(tile_a, "a", offset.clone(), Expr::Const(0));
        builder.load_global_to_shared(tile_b, "b", offset.clone(), Expr::Const(0));

        match op {
            tensor::BinaryOp::Add => builder.add(tile_out, tile_a, tile_b),
            tensor::BinaryOp::Sub => builder.sub(tile_out, tile_a, tile_b),
            tensor::BinaryOp::Mul => builder.mul(tile_out, tile_a, tile_b),
            tensor::BinaryOp::Div => builder.div(tile_out, tile_a, tile_b),
        }

        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_transpose(input_shape: Vec<usize>, output_shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("transpose");
        builder.add_param("input", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        // For now, just handle the simple case of 2D transpose
        // TODO: generalize to arbitrary dimensional transpose
        assert_eq!(
            input_shape.len(),
            2,
            "Transpose only supports 2D tensors for now"
        );
        assert_eq!(
            output_shape.len(),
            2,
            "Transpose only supports 2D tensors for now"
        );
        assert_eq!(
            input_shape[0], output_shape[1],
            "Input rows should equal output cols"
        );
        assert_eq!(
            input_shape[1], output_shape[0],
            "Input cols should equal output rows"
        );

        let total_elements: usize = output_shape.iter().product();
        let tile_in = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_out = builder.alloc_register(DType::F32, total_elements, 1);

        // Don't load here - Transpose will load directly from global memory with transposed addressing
        // Each thread handles one output element
        builder.transpose(
            tile_out,
            tile_in,
            input_shape.clone(),
            output_shape.to_vec(),
        );

        // Each thread writes its result using its thread index
        let offset = Expr::ThreadIdx(Dim::X);
        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_broadcast_axis(
        axis: usize,
        input_shape: Vec<usize>,
        output_shape: &[usize],
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("broadcast_axis");
        builder.add_param("input", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = output_shape.iter().product();
        let tile_in = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_out = builder.alloc_register(DType::F32, total_elements, 1);

        // Don't load here - BroadcastAxis will load directly from global memory
        // Each thread handles one output element
        builder.broadcast_axis(tile_out, tile_in, axis, input_shape, output_shape.to_vec());

        // Each thread writes its result using its thread index
        let offset = Expr::ThreadIdx(Dim::X);
        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_reduce_axis(
        op: tensor::ReduceOp,
        axis: usize,
        input_shape: Vec<usize>,
        output_shape: &[usize],
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("reduce_axis");
        builder.add_param("input", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = output_shape.iter().product();
        let tile_in = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_out = builder.alloc_register(DType::F32, total_elements, 1);

        // Don't load here - ReduceAxis will load directly from global memory
        // Each thread handles one output element

        let tile_op = match op {
            tensor::ReduceOp::Sum => ReduceOp::Sum,
            tensor::ReduceOp::Max => ReduceOp::Max,
            tensor::ReduceOp::Mean => ReduceOp::Mean,
        };
        builder.reduce_axis(
            tile_out,
            tile_in,
            tile_op,
            axis,
            input_shape,
            output_shape.to_vec(),
        );

        // Each thread writes its result using its thread index
        let offset = Expr::ThreadIdx(Dim::X);
        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_gt(shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("gt");
        builder.add_param("a", DType::F32, true);
        builder.add_param("b", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = shape.iter().product();
        let tile_a = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_b = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_out = builder.alloc_register(DType::F32, total_elements, 1);

        // Each thread processes one element using thread index
        let offset: Expr = Expr::ThreadIdx(Dim::X);
        builder.load_global_to_shared(tile_a, "a", offset.clone(), Expr::Const(0));
        builder.load_global_to_shared(tile_b, "b", offset.clone(), Expr::Const(0));
        builder.gt(tile_out, tile_a, tile_b);
        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_mask(shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("mask");
        builder.add_param("values", DType::F32, true);
        builder.add_param("condition", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = shape.iter().product();
        let tile_values = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_cond = builder.alloc_register(DType::F32, total_elements, 1);
        let tile_out = builder.alloc_register(DType::F32, total_elements, 1);

        // Each thread processes one element using thread index
        let offset: Expr = Expr::ThreadIdx(Dim::X);
        builder.load_global_to_shared(tile_values, "values", offset.clone(), Expr::Const(0));
        builder.load_global_to_shared(tile_cond, "condition", offset.clone(), Expr::Const(0));
        builder.mask(tile_out, tile_values, tile_cond);
        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }
}
