use super::builder::TileIRBuilder;
use super::ir::{
    Conv2dGeometry, DType, Dim, Expr, MatMulLayout, MatMulPlan, MatrixLayout, MaxPool2dGeometry,
    ReduceOp, TileIR,
};
#[cfg(feature = "cuda")]
use super::{FusionRegion, MatMulRegion, ReductionRegion};
use crate::graph::{TensorGraph, TensorGraphNode};
use crate::tensor;
use petgraph::{Graph, graph::NodeIndex};
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::collections::HashSet;

#[allow(dead_code)]
pub struct TileGraph {
    pub graph: Graph<TileIR, usize>,
    #[cfg(feature = "cuda")]
    pub(crate) region_kernels: Vec<TileRegionKernel>,
    #[cfg(feature = "cuda")]
    pub(crate) reduction_region_kernels: Vec<TileRegionKernel>,
    #[cfg(feature = "cuda")]
    pub(crate) matmul_region_kernels: Vec<TileRegionKernel>,
    #[cfg(feature = "cuda")]
    pub(crate) physical_nodes: Option<HashSet<NodeIndex>>,
}

#[cfg(feature = "cuda")]
pub(crate) struct TileRegionKernel {
    pub region_id: usize,
    pub ir: TileIR,
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

impl<G> From<TensorGraph<f32, G>> for TileGraph {
    fn from(tensor_graph: TensorGraph<f32, G>) -> Self {
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
            graph: tensor_graph.graph.map_owned(
                |idx, node| {
                    let shape = node.shape().clone();
                    // Get input shapes for operations that need them
                    let input_shapes = op_input_shapes
                        .get(&idx)
                        .map(|v| v.iter().map(|s| s.as_slice()).collect::<Vec<_>>())
                        .unwrap_or_default();
                    Self::lower_node(node, &shape, &input_shapes)
                },
                |_, e| e,
            ),
            #[cfg(feature = "cuda")]
            region_kernels: Vec::new(),
            #[cfg(feature = "cuda")]
            reduction_region_kernels: Vec::new(),
            #[cfg(feature = "cuda")]
            matmul_region_kernels: Vec::new(),
            #[cfg(feature = "cuda")]
            physical_nodes: None,
        }
    }
}

impl TileGraph {
    #[cfg(feature = "cuda")]
    pub(crate) fn add_fusion_regions(&mut self, regions: &[FusionRegion]) -> anyhow::Result<()> {
        self.region_kernels = regions
            .iter()
            .enumerate()
            .map(|(region_id, region)| {
                Ok(TileRegionKernel {
                    region_id,
                    ir: region.lower_to_tile_ir(region_id)?,
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn add_reduction_regions(
        &mut self,
        regions: &[ReductionRegion],
    ) -> anyhow::Result<()> {
        self.reduction_region_kernels = regions
            .iter()
            .enumerate()
            .map(|(region_id, region)| {
                Ok(TileRegionKernel {
                    region_id,
                    ir: region.lower_to_tile_ir(region_id)?,
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn add_matmul_regions(&mut self, regions: &[MatMulRegion]) -> anyhow::Result<()> {
        self.matmul_region_kernels = regions
            .iter()
            .enumerate()
            .map(|(region_id, region)| TileRegionKernel {
                region_id,
                ir: Self::lower_matmul_region(region, region_id),
            })
            .collect();
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn set_physical_nodes(&mut self, nodes: HashSet<NodeIndex>) {
        self.physical_nodes = Some(nodes);
    }

    #[allow(dead_code)]
    fn lower_node(
        node: TensorGraphNode<f32>,
        shape: &[usize],
        input_shapes: &[&[usize]],
    ) -> TileIR {
        match node {
            TensorGraphNode::Constant { data, .. } => Self::lower_constant(data, shape),
            TensorGraphNode::Input { name, .. } => Self::lower_input(name, shape),
            TensorGraphNode::Parameter { id, data, .. } => Self::lower_parameter(id, data, shape),
            TensorGraphNode::Unary { op, .. } => Self::lower_unary(op, shape),
            #[cfg(feature = "fusion")]
            TensorGraphNode::FusedUnary { ops, .. } => Self::lower_fused_unary(&ops, shape),
            TensorGraphNode::Binary { op, .. } => Self::lower_binary(op, shape),
            TensorGraphNode::MatMul { .. } => {
                assert!(
                    shape.len() >= 2 && input_shapes.iter().all(|input| input.len() >= 2),
                    "matmul requires rank >= 2"
                );
                let m = shape[shape.len() - 2];
                let n = shape[shape.len() - 1];
                let k = input_shapes[0][input_shapes[0].len() - 1];
                Self::lower_matmul(m, n, k, false, false)
            }
            TensorGraphNode::Embedding { .. } => {
                let weight_shape = input_shapes[0];
                assert_eq!(weight_shape.len(), 2, "embedding weight must have rank 2");
                Self::lower_embedding(weight_shape[0], weight_shape[1], input_shapes[1])
            }
            TensorGraphNode::EmbeddingBackward { .. } => {
                assert_eq!(shape.len(), 2, "embedding gradient must have rank 2");
                Self::lower_embedding_backward(shape[0], shape[1], input_shapes[0])
            }
            TensorGraphNode::IndexedCrossEntropy { .. } => {
                let logits_shape = input_shapes[0];
                let vocabulary = *logits_shape
                    .last()
                    .expect("indexed cross entropy logits must have rank >= 1");
                Self::lower_indexed_cross_entropy(vocabulary, shape.iter().product())
            }
            TensorGraphNode::IndexedCrossEntropyBackward { .. } => {
                let vocabulary = *shape
                    .last()
                    .expect("indexed cross entropy gradient must have rank >= 1");
                Self::lower_indexed_cross_entropy_backward(
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
                Self::lower_reindex(input_shape, shape, axes)
            }
            TensorGraphNode::Permute { axes, .. } => {
                let input_shape = input_shapes.first().map(|s| s.to_vec()).unwrap_or_default();
                Self::lower_reindex(input_shape, shape, axes)
            }
            TensorGraphNode::BroadcastAxis { axis, .. } => {
                let input_shape = input_shapes.first().map(|s| s.to_vec()).unwrap_or_default();
                Self::lower_broadcast_axis(axis, input_shape, shape)
            }
            TensorGraphNode::ReduceAxis { op, axis, .. } => {
                let input_shape = input_shapes.first().map(|s| s.to_vec()).unwrap_or_default();
                Self::lower_reduce_axis(op, axis, input_shape, shape)
            }
            TensorGraphNode::Gt { .. } => Self::lower_gt(shape),
            TensorGraphNode::Mask { .. } => Self::lower_mask(shape),
            TensorGraphNode::Conv2d {
                stride, padding, ..
            } => Self::lower_conv2d(input_shapes[0], input_shapes[1], shape, stride, padding),
            TensorGraphNode::ConvTranspose2d {
                stride, padding, ..
            } => Self::lower_conv_transpose2d(
                input_shapes[0],
                input_shapes[1],
                shape,
                stride,
                padding,
            ),
            TensorGraphNode::Conv2dBackwardWeight {
                stride, padding, ..
            } => Self::lower_conv2d_backward_weight(
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
            } => Self::lower_max_pool2d(input_shapes[0], shape, kernel_size, stride),
            TensorGraphNode::MaxPool2dBackward {
                kernel_size,
                stride,
                ..
            } => Self::lower_max_pool2d_backward(
                input_shapes[0],
                input_shapes[1],
                kernel_size,
                stride,
            ),
            TensorGraphNode::Flatten { .. } | TensorGraphNode::Reshape { .. } => Self::lower_view(),
        }
    }

    fn lower_embedding(vocabulary: usize, width: usize, indices_shape: &[usize]) -> TileIR {
        let index_count = indices_shape.iter().product();
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("embedding");
        builder.add_param("weight", DType::F32, true);
        builder.add_param("indices", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(index_count * width);
        builder.embedding(vocabulary, width, index_count);
        builder.finish()
    }

    fn lower_embedding_backward(
        vocabulary: usize,
        width: usize,
        indices_shape: &[usize],
    ) -> TileIR {
        let index_count = indices_shape.iter().product();
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("embedding_backward");
        builder.add_param("indices", DType::F32, true);
        builder.add_param("grad_output", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(index_count * width);
        builder.embedding_backward(vocabulary, width, index_count);
        builder.finish()
    }

    fn lower_indexed_cross_entropy(vocabulary: usize, row_count: usize) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("indexed_cross_entropy");
        builder.add_param("logits", DType::F32, true);
        builder.add_param("targets", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(row_count);
        builder.indexed_cross_entropy(vocabulary, row_count);
        builder.finish()
    }

    fn lower_indexed_cross_entropy_backward(vocabulary: usize, row_count: usize) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("indexed_cross_entropy_backward");
        builder.add_param("logits", DType::F32, true);
        builder.add_param("targets", DType::F32, true);
        builder.add_param("grad_output", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(row_count * vocabulary);
        builder.indexed_cross_entropy_backward(vocabulary, row_count);
        builder.finish()
    }

    fn lower_conv2d(
        input: &[usize],
        weight: &[usize],
        output: &[usize],
        stride: usize,
        padding: usize,
    ) -> TileIR {
        let geometry = conv2d_geometry(input, weight, output, stride, padding);
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("conv2d");
        builder.add_param("input", DType::F32, true);
        builder.add_param("weight", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(output.iter().product());
        builder.conv2d(geometry);
        builder.finish()
    }

    fn lower_conv_transpose2d(
        grad_output: &[usize],
        weight: &[usize],
        output: &[usize],
        stride: usize,
        padding: usize,
    ) -> TileIR {
        let geometry = conv2d_geometry(output, weight, grad_output, stride, padding);
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("conv_transpose2d");
        builder.add_param("grad_output", DType::F32, true);
        builder.add_param("weight", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(output.iter().product());
        builder.conv_transpose2d(geometry);
        builder.finish()
    }

    fn lower_conv2d_backward_weight(
        input: &[usize],
        grad_output: &[usize],
        output: &[usize],
        stride: usize,
        padding: usize,
    ) -> TileIR {
        let geometry = conv2d_geometry(input, output, grad_output, stride, padding);
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("conv2d_backward_weight");
        builder.add_param("input", DType::F32, true);
        builder.add_param("grad_output", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(output.iter().product());
        builder.conv2d_backward_weight(geometry);
        builder.finish()
    }

    fn lower_max_pool2d(
        input: &[usize],
        output: &[usize],
        kernel_size: usize,
        stride: usize,
    ) -> TileIR {
        let geometry = max_pool2d_geometry(input, output, kernel_size, stride);
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("max_pool2d");
        builder.add_param("input", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(output.iter().product());
        builder.max_pool2d(geometry);
        builder.finish()
    }

    fn lower_max_pool2d_backward(
        input: &[usize],
        pooled: &[usize],
        kernel_size: usize,
        stride: usize,
    ) -> TileIR {
        let geometry = max_pool2d_geometry(input, pooled, kernel_size, stride);
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("max_pool2d_backward");
        builder.add_param("input", DType::F32, true);
        builder.add_param("pooled", DType::F32, true);
        builder.add_param("grad_output", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(input.iter().product());
        builder.max_pool2d_backward(geometry);
        builder.finish()
    }

    #[allow(dead_code)]
    fn lower_matmul(m: usize, n: usize, k: usize, transpose_a: bool, transpose_b: bool) -> TileIR {
        Self::lower_matmul_impl(m, n, k, transpose_a, transpose_b, None)
    }

    #[cfg(feature = "cuda")]
    fn lower_matmul_region(region: &super::MatMulRegion, region_id: usize) -> TileIR {
        Self::lower_matmul_impl(
            region.m,
            region.n,
            region.k,
            false,
            false,
            Some((region, region_id)),
        )
    }

    fn lower_matmul_impl(
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool,
        region: Option<(&super::MatMulRegion, usize)>,
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        if let Some((_, region_id)) = region {
            builder.start_kernel(&format!("matmul_region_{region_id}"));
        } else {
            builder.start_kernel("matmul");
        }

        let (a_param, b_param) = if let Some((region, _)) = region {
            for index in 0..region.inputs.len() {
                builder.add_param(&format!("input_{index}"), DType::F32, true);
            }
            for index in 0..region.outputs.len() {
                builder.add_param(&format!("output_{index}"), DType::F32, false);
            }
            ("input_0", "input_1")
        } else {
            builder.add_param("A", DType::F32, true);
            builder.add_param("B", DType::F32, true);
            builder.add_param("C", DType::F32, false);
            ("A", "B")
        };

        // Tile sizes for shared memory
        let tile_m = 16;
        let tile_n = 16;
        let tile_k = 16;
        let plan = MatMulPlan::for_shape(m, n, k);

        // Allocate shared memory for input tiles
        let operand_dtype = match plan {
            MatMulPlan::ScalarF32 => DType::F32,
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
                a_param,
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
                b_param,
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
        let output_layout = MatrixLayout {
            rows: m,
            cols: n,
            row_stride: n,
        };
        if let Some((region, _)) = region {
            use super::RegionOpKind;

            let matmul_element = builder.alloc_register(DType::F32, 1, 1);
            builder.load_tile_element(matmul_element, c_reg);
            let mut registers = HashMap::new();
            registers.insert(region.matmul_value, matmul_element);
            let linear_element = Expr::Add(
                Box::new(Expr::Mul(
                    Box::new(global_row.clone()),
                    Box::new(Expr::Const(n as i64)),
                )),
                Box::new(global_col.clone()),
            );
            for (index, input) in region.inputs.iter().enumerate().skip(2) {
                let value = builder.alloc_register(DType::F32, 1, 1);
                let element_index = super::region::input_element_index(
                    input,
                    &region.output_shape,
                    &linear_element,
                )
                .expect("matmul epilogue input access is invalid");
                builder.load_global_predicated(
                    value,
                    &format!("input_{index}"),
                    element_index,
                    global_row.clone(),
                    global_col.clone(),
                    output_layout,
                );
                registers.insert(input.value, value);
            }
            for operation in &region.epilogue_operations {
                let output = builder.alloc_register(DType::F32, 1, 1);
                let register = |value| {
                    *registers
                        .get(&value)
                        .expect("matmul epilogue SSA value is unavailable")
                };
                match operation.kind {
                    RegionOpKind::Unary { op, input } => match op {
                        tensor::UnaryOp::Neg => builder.neg(output, register(input)),
                        tensor::UnaryOp::Exp => builder.exp(output, register(input)),
                        tensor::UnaryOp::Log => builder.log(output, register(input)),
                        tensor::UnaryOp::Relu => builder.relu(output, register(input)),
                    },
                    RegionOpKind::Binary { op, lhs, rhs } => match op {
                        tensor::BinaryOp::Add => builder.add(output, register(lhs), register(rhs)),
                        tensor::BinaryOp::Sub => builder.sub(output, register(lhs), register(rhs)),
                        tensor::BinaryOp::Mul => builder.mul(output, register(lhs), register(rhs)),
                        tensor::BinaryOp::Div => builder.div(output, register(lhs), register(rhs)),
                    },
                    RegionOpKind::Gt { lhs, rhs } => {
                        builder.gt(output, register(lhs), register(rhs))
                    }
                    RegionOpKind::Mask { values, condition } => {
                        builder.mask(output, register(values), register(condition))
                    }
                }
                registers.insert(operation.output, output);
            }
            for (index, output) in region.outputs.iter().enumerate() {
                builder.store_global_predicated(
                    &format!("output_{index}"),
                    registers[&output.value],
                    global_row.clone(),
                    global_col.clone(),
                    output_layout,
                );
            }
        } else {
            builder.store_global_predicated("C", c_reg, global_row, global_col, output_layout);
        }

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
    fn lower_unary(op: tensor::UnaryOp, _shape: &[usize]) -> TileIR {
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
        builder.bounds_check(_shape.iter().product());

        // Each thread processes one scalar element
        let tile_in = builder.alloc_register(DType::F32, 1, 1);
        let tile_out = builder.alloc_register(DType::F32, 1, 1);

        // Calculate global thread ID: blockIdx.x * blockDim.x + threadIdx.x
        let global_tid = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::BlockIdx(Dim::X)),
                Box::new(Expr::BlockDim(Dim::X)),
            )),
            Box::new(Expr::ThreadIdx(Dim::X)),
        );

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
    fn lower_fused_unary(ops: &[tensor::UnaryOp], _shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("fused_unary");
        builder.add_param("input", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(_shape.iter().product());

        // Each thread processes one scalar element
        // Allocate scalar registers (1 element per thread)
        let tile_in = builder.alloc_register(DType::F32, 1, 1);
        let mut current_tile = tile_in;

        // Calculate global thread ID: blockIdx.x * blockDim.x + threadIdx.x
        let global_tid = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::BlockIdx(Dim::X)),
                Box::new(Expr::BlockDim(Dim::X)),
            )),
            Box::new(Expr::ThreadIdx(Dim::X)),
        );

        builder.load_global_to_shared(tile_in, "input", global_tid.clone(), Expr::Const(0));

        // Apply each operation in sequence
        for op in ops.iter() {
            let next_tile = builder.alloc_register(DType::F32, 1, 1);

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
    fn lower_binary(op: tensor::BinaryOp, _shape: &[usize]) -> TileIR {
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
        builder.bounds_check(_shape.iter().product());

        // Each thread processes one scalar element
        let tile_a = builder.alloc_register(DType::F32, 1, 1);
        let tile_b = builder.alloc_register(DType::F32, 1, 1);
        let tile_out = builder.alloc_register(DType::F32, 1, 1);

        // Calculate global thread ID: blockIdx.x * blockDim.x + threadIdx.x
        let global_tid = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::BlockIdx(Dim::X)),
                Box::new(Expr::BlockDim(Dim::X)),
            )),
            Box::new(Expr::ThreadIdx(Dim::X)),
        );

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
    fn lower_reindex(input_shape: Vec<usize>, output_shape: &[usize], axes: Vec<usize>) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("reindex");
        builder.add_param("input", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = output_shape.iter().product();
        builder.bounds_check(total_elements);
        let tile_in = builder.alloc_register(DType::F32, 1, 1);
        let tile_out = builder.alloc_register(DType::F32, 1, 1);

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
        axis: usize,
        input_shape: Vec<usize>,
        output_shape: &[usize],
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("broadcast_axis");
        builder.add_param("input", DType::F32, true);
        builder.add_param("output", DType::F32, false);

        let total_elements: usize = output_shape.iter().product();
        builder.bounds_check(total_elements);
        let tile_in = builder.alloc_register(DType::F32, 1, 1);
        let tile_out = builder.alloc_register(DType::F32, 1, 1);

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
        builder.bounds_check(total_elements);
        let tile_in = builder.alloc_register(DType::F32, 1, 1);
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
    fn lower_gt(shape: &[usize]) -> TileIR {
        let mut builder = TileIRBuilder::new();
        builder.start_kernel("gt");
        builder.add_param("a", DType::F32, true);
        builder.add_param("b", DType::F32, true);
        builder.add_param("output", DType::F32, false);
        builder.bounds_check(shape.iter().product());

        let tile_a = builder.alloc_register(DType::F32, 1, 1);
        let tile_b = builder.alloc_register(DType::F32, 1, 1);
        let tile_out = builder.alloc_register(DType::F32, 1, 1);

        // Each thread processes one element using thread index
        let offset = Self::global_tid();
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
        builder.bounds_check(shape.iter().product());

        let tile_values = builder.alloc_register(DType::F32, 1, 1);
        let tile_cond = builder.alloc_register(DType::F32, 1, 1);
        let tile_out = builder.alloc_register(DType::F32, 1, 1);

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
