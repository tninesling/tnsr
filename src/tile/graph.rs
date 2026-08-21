use super::builder::TileIRBuilder;
use super::ir::{
    Conv2dGeometry, DType, Dim, Expr, MatMulLayout, MatMulPlan, MatrixLayout, MaxPool2dGeometry,
    ReduceOp, TileDType, TileIR,
};
use crate::graph::{TensorGraph, TensorGraphNode};
use crate::tensor;
use petgraph::{Graph, graph::NodeIndex};
use std::collections::HashMap;

#[allow(dead_code)]
pub struct TileGraph {
    pub graph: Graph<TileIR, usize>,
}

fn conv2d_geometry(
    input: &[usize],
    weight: &[usize],
    output: &[usize],
    stride: usize,
    padding: usize,
) -> Conv2dGeometry {
    assert_eq!(input.len(), 4, "convolution input must be NCHW");
    assert_eq!(weight.len(), 4, "convolution weight must be OIHW");
    assert_eq!(output.len(), 4, "convolution output must be NCHW");
    Conv2dGeometry {
        batch: input[0],
        input_channels: input[1],
        input_height: input[2],
        input_width: input[3],
        output_channels: weight[0],
        output_height: output[2],
        output_width: output[3],
        kernel_height: weight[2],
        kernel_width: weight[3],
        stride,
        padding,
    }
}

fn max_pool2d_geometry(
    input: &[usize],
    output: &[usize],
    kernel_size: usize,
    stride: usize,
) -> MaxPool2dGeometry {
    assert_eq!(input.len(), 4, "max-pool input must be NCHW");
    assert_eq!(output.len(), 4, "max-pool output must be NCHW");
    MaxPool2dGeometry {
        batch: input[0],
        channels: input[1],
        input_height: input[2],
        input_width: input[3],
        output_height: output[2],
        output_width: output[3],
        kernel_size,
        stride,
    }
}

impl<D: TileDType, G> From<TensorGraph<D, G>> for TileGraph {
    fn from(tensor_graph: TensorGraph<D, G>) -> Self {
        Self::from_graph(&tensor_graph)
    }
}

impl TileGraph {
    pub fn from_graph<D: TileDType, G>(tensor_graph: &TensorGraph<D, G>) -> Self {
        // Pre-compute input shapes for MatMul, BroadcastAxis, ReduceAxis, and Transpose nodes
        let mut op_input_shapes: HashMap<NodeIndex, Vec<Vec<usize>>> = HashMap::new();
        for idx in tensor_graph.graph.node_indices() {
            if matches!(
                tensor_graph.graph[idx],
                TensorGraphNode::MatMul { .. }
                    | TensorGraphNode::BroadcastAxis { .. }
                    | TensorGraphNode::ReduceAxis { .. }
                    | TensorGraphNode::Transpose { .. }
                    | TensorGraphNode::Permute { .. }
                    | TensorGraphNode::Embedding { .. }
                    | TensorGraphNode::EmbeddingBackward { .. }
                    | TensorGraphNode::IndexedCrossEntropy { .. }
                    | TensorGraphNode::IndexedCrossEntropyBackward { .. }
                    | TensorGraphNode::Conv2d { .. }
                    | TensorGraphNode::ConvTranspose2d { .. }
                    | TensorGraphNode::Conv2dBackwardWeight { .. }
                    | TensorGraphNode::MaxPool2d { .. }
                    | TensorGraphNode::MaxPool2dBackward { .. }
            ) {
                let inputs: Vec<Vec<usize>> = tensor_graph
                    .inputs(idx)
                    .into_iter()
                    .map(|pred_idx| tensor_graph.graph[pred_idx].shape().clone())
                    .collect();
                op_input_shapes.insert(idx, inputs);
            }
        }

        Self {
            graph: tensor_graph.graph.map(
                |idx, node| {
                    let shape = node.shape().clone();
                    // Get input shapes for operations that need them
                    let input_shapes = op_input_shapes
                        .get(&idx)
                        .map(|v| v.iter().map(|s| s.as_slice()).collect::<Vec<_>>())
                        .unwrap_or_default();
                    Self::lower_node(node.clone(), D::TILE_DTYPE, &shape, &input_shapes)
                },
                |_, e| *e,
            ),
        }
    }

    #[allow(dead_code)]
    fn lower_node<D: TileDType>(
        node: TensorGraphNode<D>,
        dtype: DType,
        shape: &[usize],
        input_shapes: &[&[usize]],
    ) -> TileIR {
        match node {
            TensorGraphNode::Constant { data, .. } => Self::lower_constant(dtype, data, shape),
            TensorGraphNode::Input { name, .. } => Self::lower_input(dtype, name, shape),
            TensorGraphNode::Parameter { id, data, .. } => {
                Self::lower_parameter(dtype, id, data, shape)
            }
            TensorGraphNode::Unary { op, .. } => Self::lower_unary(dtype, op, shape),
            #[cfg(feature = "fusion")]
            TensorGraphNode::FusedUnary { ops, .. } => Self::lower_fused_unary(dtype, &ops, shape),
            TensorGraphNode::Binary { op, .. } => Self::lower_binary(dtype, op, shape),
            TensorGraphNode::MatMul { .. } => {
                assert!(
                    shape.len() >= 2 && input_shapes.iter().all(|input| input.len() >= 2),
                    "matmul requires rank >= 2"
                );
                let m = shape[shape.len() - 2];
                let n = shape[shape.len() - 1];
                let k = input_shapes[0][input_shapes[0].len() - 1];
                Self::lower_matmul(dtype, m, n, k, false, false)
            }
            TensorGraphNode::Embedding { .. } => {
                let weight_shape = input_shapes[0];
                assert_eq!(weight_shape.len(), 2, "embedding weight must have rank 2");
                Self::lower_embedding(dtype, weight_shape[0], weight_shape[1], input_shapes[1])
            }
            TensorGraphNode::EmbeddingBackward { .. } => {
                assert_eq!(shape.len(), 2, "embedding gradient must have rank 2");
                Self::lower_embedding_backward(dtype, shape[0], shape[1], input_shapes[0])
            }
            TensorGraphNode::IndexedCrossEntropy { .. } => {
                let logits_shape = input_shapes[0];
                let vocabulary = *logits_shape
                    .last()
                    .expect("indexed cross entropy logits must have rank >= 1");
                Self::lower_indexed_cross_entropy(dtype, vocabulary, shape.iter().product())
            }
            TensorGraphNode::IndexedCrossEntropyBackward { .. } => {
                let vocabulary = *shape
                    .last()
                    .expect("indexed cross entropy gradient must have rank >= 1");
                Self::lower_indexed_cross_entropy_backward(
                    dtype,
                    vocabulary,
                    input_shapes[1].iter().product(),
                )
            }
            TensorGraphNode::Transpose { .. } => {
                let input_shape = input_shapes.first().map(|s| s.to_vec()).unwrap_or_default();
                let mut axes: Vec<usize> = (0..input_shape.len()).collect();
                let rank = axes.len();
                assert!(rank >= 2, "transpose requires rank >= 2");
                axes.swap(rank - 2, rank - 1);
                Self::lower_reindex(dtype, input_shape, shape, axes)
            }
            TensorGraphNode::Permute { axes, .. } => {
                let input_shape = input_shapes.first().map(|s| s.to_vec()).unwrap_or_default();
                Self::lower_reindex(dtype, input_shape, shape, axes)
            }
            TensorGraphNode::BroadcastAxis { axis, .. } => {
                let input_shape = input_shapes.first().map(|s| s.to_vec()).unwrap_or_default();
                Self::lower_broadcast_axis(dtype, axis, input_shape, shape)
            }
            TensorGraphNode::ReduceAxis { op, axis, .. } => {
                let input_shape = input_shapes.first().map(|s| s.to_vec()).unwrap_or_default();
                Self::lower_reduce_axis(dtype, op, axis, input_shape, shape)
            }
            TensorGraphNode::Gt { .. } => Self::lower_gt(dtype, shape),
            TensorGraphNode::Mask { .. } => Self::lower_mask(dtype, shape),
            TensorGraphNode::Conv2d {
                stride, padding, ..
            } => Self::lower_conv2d(
                dtype,
                input_shapes[0],
                input_shapes[1],
                shape,
                stride,
                padding,
            ),
            TensorGraphNode::ConvTranspose2d {
                stride, padding, ..
            } => Self::lower_conv_transpose2d(
                dtype,
                input_shapes[0],
                input_shapes[1],
                shape,
                stride,
                padding,
            ),
            TensorGraphNode::Conv2dBackwardWeight {
                stride, padding, ..
            } => Self::lower_conv2d_backward_weight(
                dtype,
                input_shapes[0],
                input_shapes[1],
                shape,
                stride,
                padding,
            ),
            TensorGraphNode::MaxPool2d {
                kernel_size,
                stride,
                ..
            } => Self::lower_max_pool2d(dtype, input_shapes[0], shape, kernel_size, stride),
            TensorGraphNode::MaxPool2dBackward {
                kernel_size,
                stride,
                ..
            } => Self::lower_max_pool2d_backward(
                dtype,
                input_shapes[0],
                input_shapes[1],
                kernel_size,
                stride,
            ),
            TensorGraphNode::Flatten { .. } | TensorGraphNode::Reshape { .. } => Self::lower_view(),
        }
    }

    fn lower_embedding(
        dtype: DType,
        vocabulary: usize,
        width: usize,
        indices_shape: &[usize],
    ) -> TileIR {
        let index_count = indices_shape.iter().product();
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("embedding");
        builder.add_param("weight", dtype, true);
        builder.add_param("indices", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(index_count * width);
        builder.embedding(vocabulary, width, index_count);
        builder.finish()
    }

    fn lower_embedding_backward(
        dtype: DType,
        vocabulary: usize,
        width: usize,
        indices_shape: &[usize],
    ) -> TileIR {
        let index_count = indices_shape.iter().product();
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("embedding_backward");
        builder.add_param("indices", dtype, true);
        builder.add_param("grad_output", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(index_count * width);
        builder.embedding_backward(vocabulary, width, index_count);
        builder.finish()
    }

    fn lower_indexed_cross_entropy(dtype: DType, vocabulary: usize, row_count: usize) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("indexed_cross_entropy");
        builder.add_param("logits", dtype, true);
        builder.add_param("targets", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(row_count);
        builder.indexed_cross_entropy(vocabulary, row_count);
        builder.finish()
    }

    fn lower_indexed_cross_entropy_backward(
        dtype: DType,
        vocabulary: usize,
        row_count: usize,
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("indexed_cross_entropy_backward");
        builder.add_param("logits", dtype, true);
        builder.add_param("targets", dtype, true);
        builder.add_param("grad_output", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(row_count * vocabulary);
        builder.indexed_cross_entropy_backward(vocabulary, row_count);
        builder.finish()
    }

    fn lower_conv2d(
        dtype: DType,
        input: &[usize],
        weight: &[usize],
        output: &[usize],
        stride: usize,
        padding: usize,
    ) -> TileIR {
        let geometry = conv2d_geometry(input, weight, output, stride, padding);
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("conv2d");
        builder.add_param("input", dtype, true);
        builder.add_param("weight", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(output.iter().product());
        builder.conv2d(geometry);
        builder.finish()
    }

    fn lower_conv_transpose2d(
        dtype: DType,
        grad_output: &[usize],
        weight: &[usize],
        output: &[usize],
        stride: usize,
        padding: usize,
    ) -> TileIR {
        let geometry = conv2d_geometry(output, weight, grad_output, stride, padding);
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("conv_transpose2d");
        builder.add_param("grad_output", dtype, true);
        builder.add_param("weight", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(output.iter().product());
        builder.conv_transpose2d(geometry);
        builder.finish()
    }

    fn lower_conv2d_backward_weight(
        dtype: DType,
        input: &[usize],
        grad_output: &[usize],
        output: &[usize],
        stride: usize,
        padding: usize,
    ) -> TileIR {
        let geometry = conv2d_geometry(input, output, grad_output, stride, padding);
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("conv2d_backward_weight");
        builder.add_param("input", dtype, true);
        builder.add_param("grad_output", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(output.iter().product());
        builder.conv2d_backward_weight(geometry);
        builder.finish()
    }

    fn lower_max_pool2d(
        dtype: DType,
        input: &[usize],
        output: &[usize],
        kernel_size: usize,
        stride: usize,
    ) -> TileIR {
        let geometry = max_pool2d_geometry(input, output, kernel_size, stride);
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("max_pool2d");
        builder.add_param("input", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(output.iter().product());
        builder.max_pool2d(geometry);
        builder.finish()
    }

    fn lower_max_pool2d_backward(
        dtype: DType,
        input: &[usize],
        pooled: &[usize],
        kernel_size: usize,
        stride: usize,
    ) -> TileIR {
        let geometry = max_pool2d_geometry(input, pooled, kernel_size, stride);
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("max_pool2d_backward");
        builder.add_param("input", dtype, true);
        builder.add_param("pooled", dtype, true);
        builder.add_param("grad_output", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(input.iter().product());
        builder.max_pool2d_backward(geometry);
        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_matmul(
        dtype: DType,
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool,
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("matmul");

        // Add parameters for input and output matrices
        builder.add_param("A", dtype, true);
        builder.add_param("B", dtype, true);
        builder.add_param("C", dtype, false);

        // Tile sizes for shared memory
        let tile_m = 16;
        let tile_n = 16;
        let tile_k = 16;
        let plan = if dtype == DType::F32 {
            MatMulPlan::for_shape(m, n, k)
        } else {
            MatMulPlan::ScalarF32
        };

        // Allocate shared memory for input tiles
        let operand_dtype = match plan {
            MatMulPlan::ScalarF32 => dtype,
            MatMulPlan::TensorCoreTf32 => DType::TF32,
        };
        let a_smem = builder.alloc_shared(operand_dtype, tile_m, tile_k);
        let b_smem = builder.alloc_shared(operand_dtype, tile_k, tile_n);
        let c_smem = match plan {
            MatMulPlan::ScalarF32 => None,
            MatMulPlan::TensorCoreTf32 => Some(builder.alloc_shared(DType::F32, tile_m, tile_n)),
        };

        // Allocate register tiles for computation
        let a_reg = builder.alloc_register(DType::F32, tile_m, tile_k);
        let b_reg = builder.alloc_register(DType::F32, tile_k, tile_n);
        let c_reg = match plan {
            MatMulPlan::ScalarF32 => builder.alloc_register(DType::F32, tile_m, tile_n),
            MatMulPlan::TensorCoreTf32 => builder.alloc_fragment(DType::F32, tile_m, tile_n),
        };
        if let Some(c_smem) = c_smem {
            builder.load_shared_to_register(c_reg, c_smem);
        }

        // Initialize accumulator
        builder.zero(c_reg);

        // Calculate number of K tiles needed
        let k_tiles = k.div_ceil(tile_k);

        // Main tiling loop over K dimension
        builder.for_loop("k_tile", 0, k_tiles as i64, |builder, k_var| {
            let a_row = Expr::Add(
                Box::new(Expr::BlockIdx(Dim::Y) * tile_m),
                Box::new(Expr::ThreadIdx(Dim::Y)),
            );
            let a_col = Expr::Add(
                Box::new(Expr::Var(k_var.clone()) * tile_k),
                Box::new(Expr::ThreadIdx(Dim::X)),
            );
            builder.load_global_to_shared_predicated(
                a_smem,
                "A",
                a_row,
                a_col,
                MatrixLayout {
                    rows: m,
                    cols: k,
                    row_stride: k,
                },
            );

            let b_row = Expr::Add(
                Box::new(Expr::Var(k_var) * tile_k),
                Box::new(Expr::ThreadIdx(Dim::Y)),
            );
            let b_col = Expr::Add(
                Box::new(Expr::BlockIdx(Dim::X) * tile_n),
                Box::new(Expr::ThreadIdx(Dim::X)),
            );
            builder.load_global_to_shared_predicated(
                b_smem,
                "B",
                b_row,
                b_col,
                MatrixLayout {
                    rows: k,
                    cols: n,
                    row_stride: n,
                },
            );

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

            builder.matmul(c_reg, a_reg, b_reg, layout, plan);

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
        builder.store_global_predicated(
            "C",
            c_reg,
            global_row,
            global_col,
            MatrixLayout {
                rows: m,
                cols: n,
                row_stride: n,
            },
        );

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_constant<D: TileDType>(
        dtype: DType,
        _data: std::sync::Arc<Vec<D>>,
        shape: &[usize],
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("constant");
        builder.add_param("constant_data", dtype, true);
        builder.add_param("output", dtype, false);

        let total_elements: usize = shape.iter().product();
        let tile_const = builder.alloc_register(dtype, total_elements, 1);
        let tile_out = builder.alloc_register(dtype, total_elements, 1);

        builder.load_global_to_shared(tile_const, "constant_data", Expr::Const(0), Expr::Const(0));
        builder.add(tile_out, tile_const, tile_const);
        builder.store("output", tile_out, Expr::Const(0), Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_input(dtype: DType, name: &'static str, shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel(&format!("input_{}", name));
        builder.add_param(name, dtype, true);
        builder.add_param("output", dtype, false);

        let total_elements: usize = shape.iter().product();
        let tile_in = builder.alloc_register(dtype, total_elements, 1);

        builder.load_global_to_shared(tile_in, name, Expr::Const(0), Expr::Const(0));
        builder.store("output", tile_in, Expr::Const(0), Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_parameter<D: TileDType>(
        dtype: DType,
        _id: usize,
        _data: std::sync::Arc<std::sync::Mutex<Vec<D>>>,
        shape: &[usize],
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("parameter");
        builder.add_param("param", dtype, true);
        builder.add_param("output", dtype, false);

        let total_elements: usize = shape.iter().product();
        let tile_param = builder.alloc_register(dtype, total_elements, 1);

        builder.load_global_to_shared(tile_param, "param", Expr::Const(0), Expr::Const(0));
        builder.store("output", tile_param, Expr::Const(0), Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_unary(dtype: DType, op: tensor::UnaryOp, _shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        let kernel_name = match op {
            tensor::UnaryOp::Neg => "neg",
            tensor::UnaryOp::Exp => "exp",
            tensor::UnaryOp::Log => "log",
            tensor::UnaryOp::Relu => "relu",
        };
        builder.start_kernel(kernel_name);
        builder.add_param("input", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(_shape.iter().product());

        // Each thread processes one scalar element
        let tile_in = builder.alloc_register(dtype, 1, 1);
        let tile_out = builder.alloc_register(dtype, 1, 1);

        let global_tid = Self::global_tid();

        builder.load_global_to_shared(tile_in, "input", global_tid.clone(), Expr::Const(0));

        match op {
            tensor::UnaryOp::Neg => builder.neg(tile_out, tile_in),
            tensor::UnaryOp::Exp => builder.exp(tile_out, tile_in),
            tensor::UnaryOp::Log => builder.log(tile_out, tile_in),
            tensor::UnaryOp::Relu => builder.relu(tile_out, tile_in),
        }

        builder.store("output", tile_out, global_tid, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    #[cfg(feature = "fusion")]
    fn lower_fused_unary(dtype: DType, ops: &[tensor::UnaryOp], _shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("fused_unary");
        builder.add_param("input", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(_shape.iter().product());

        // Each thread processes one scalar element
        // Allocate scalar registers (1 element per thread)
        let tile_in = builder.alloc_register(dtype, 1, 1);
        let mut current_tile = tile_in;

        let global_tid = Self::global_tid();

        builder.load_global_to_shared(tile_in, "input", global_tid.clone(), Expr::Const(0));

        // Apply each operation in sequence
        for op in ops.iter() {
            let next_tile = builder.alloc_register(dtype, 1, 1);

            match op {
                tensor::UnaryOp::Neg => builder.neg(next_tile, current_tile),
                tensor::UnaryOp::Exp => builder.exp(next_tile, current_tile),
                tensor::UnaryOp::Log => builder.log(next_tile, current_tile),
                tensor::UnaryOp::Relu => builder.relu(next_tile, current_tile),
            }

            current_tile = next_tile;
        }

        builder.store("output", current_tile, global_tid, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_binary(dtype: DType, op: tensor::BinaryOp, _shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        let kernel_name = match op {
            tensor::BinaryOp::Add => "add",
            tensor::BinaryOp::Sub => "sub",
            tensor::BinaryOp::Mul => "mul",
            tensor::BinaryOp::Div => "div",
        };
        builder.start_kernel(kernel_name);
        builder.add_param("a", dtype, true);
        builder.add_param("b", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(_shape.iter().product());

        // Each thread processes one scalar element
        let tile_a = builder.alloc_register(dtype, 1, 1);
        let tile_b = builder.alloc_register(dtype, 1, 1);
        let tile_out = builder.alloc_register(dtype, 1, 1);

        let global_tid = Self::global_tid();

        builder.load_global_to_shared(tile_a, "a", global_tid.clone(), Expr::Const(0));
        builder.load_global_to_shared(tile_b, "b", global_tid.clone(), Expr::Const(0));

        match op {
            tensor::BinaryOp::Add => builder.add(tile_out, tile_a, tile_b),
            tensor::BinaryOp::Sub => builder.sub(tile_out, tile_a, tile_b),
            tensor::BinaryOp::Mul => builder.mul(tile_out, tile_a, tile_b),
            tensor::BinaryOp::Div => builder.div(tile_out, tile_a, tile_b),
        }

        builder.store("output", tile_out, global_tid, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_reindex(
        dtype: DType,
        input_shape: Vec<usize>,
        output_shape: &[usize],
        axes: Vec<usize>,
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("reindex");
        builder.add_param("input", dtype, true);
        builder.add_param("output", dtype, false);

        let total_elements: usize = output_shape.iter().product();
        builder.bounds_check(total_elements);
        let tile_in = builder.alloc_register(dtype, 1, 1);
        let tile_out = builder.alloc_register(dtype, 1, 1);

        // Don't load here - Transpose will load directly from global memory with transposed addressing
        // Each thread handles one output element
        builder.reindex(tile_out, tile_in, input_shape, output_shape.to_vec(), axes);

        // Each thread writes its result using its thread index
        let offset = Self::global_tid();
        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_broadcast_axis(
        dtype: DType,
        axis: usize,
        input_shape: Vec<usize>,
        output_shape: &[usize],
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("broadcast_axis");
        builder.add_param("input", dtype, true);
        builder.add_param("output", dtype, false);

        let total_elements: usize = output_shape.iter().product();
        builder.bounds_check(total_elements);
        let tile_in = builder.alloc_register(dtype, 1, 1);
        let tile_out = builder.alloc_register(dtype, 1, 1);

        // Don't load here - BroadcastAxis will load directly from global memory
        // Each thread handles one output element
        builder.broadcast_axis(tile_out, tile_in, axis, input_shape, output_shape.to_vec());

        // Each thread writes its result using its thread index
        let offset = Self::global_tid();
        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_reduce_axis(
        dtype: DType,
        op: tensor::ReduceOp,
        axis: usize,
        input_shape: Vec<usize>,
        output_shape: &[usize],
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("reduce_axis");
        builder.add_param("input", dtype, true);
        builder.add_param("output", dtype, false);

        let total_elements: usize = output_shape.iter().product();
        builder.bounds_check(total_elements);
        let tile_in = builder.alloc_register(dtype, 1, 1);
        // Reductions use an f32 accumulator and narrow only at the output boundary.
        let tile_out = builder.alloc_register(DType::F32, 1, 1);

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
        let offset = Self::global_tid();
        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_gt(dtype: DType, shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("gt");
        builder.add_param("a", dtype, true);
        builder.add_param("b", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(shape.iter().product());

        let tile_a = builder.alloc_register(dtype, 1, 1);
        let tile_b = builder.alloc_register(dtype, 1, 1);
        let tile_out = builder.alloc_register(dtype, 1, 1);

        // Each thread processes one element using thread index
        let offset = Self::global_tid();
        builder.load_global_to_shared(tile_a, "a", offset.clone(), Expr::Const(0));
        builder.load_global_to_shared(tile_b, "b", offset.clone(), Expr::Const(0));
        builder.gt(tile_out, tile_a, tile_b);
        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_mask(dtype: DType, shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("mask");
        builder.add_param("values", dtype, true);
        builder.add_param("condition", dtype, true);
        builder.add_param("output", dtype, false);
        builder.bounds_check(shape.iter().product());

        let tile_values = builder.alloc_register(dtype, 1, 1);
        let tile_cond = builder.alloc_register(dtype, 1, 1);
        let tile_out = builder.alloc_register(dtype, 1, 1);

        // Each thread processes one element using thread index
        let offset = Self::global_tid();
        builder.load_global_to_shared(tile_values, "values", offset.clone(), Expr::Const(0));
        builder.load_global_to_shared(tile_cond, "condition", offset.clone(), Expr::Const(0));
        builder.mask(tile_out, tile_values, tile_cond);
        builder.store("output", tile_out, offset, Expr::Const(0));

        builder.finish()
    }

    fn lower_view() -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("contiguous_view");
        builder.finish()
    }

    fn global_tid() -> Expr {
        Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::BlockIdx(Dim::X)),
                Box::new(Expr::BlockDim(Dim::X)),
            )),
            Box::new(Expr::ThreadIdx(Dim::X)),
        )
    }
}
