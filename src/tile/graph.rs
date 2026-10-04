use anyhow::Result;
#[cfg(feature = "cuda")]
use anyhow::anyhow;

use super::builder::TileIRBuilder;
use super::ir::{
    Conv2dGeometry, DType, Dim, Expr, MatMulLayout, MatMulPlan, MatrixLayout, MaxPool2dGeometry,
    ReduceOp, TileDType, TileIR,
};
use super::matmul_schedule::{MatMulCapabilities, MatMulPrecision, MatMulSchedule};
#[cfg(feature = "cuda")]
use super::region::input_coordinate_index;
use super::region::input_element_index;
#[cfg(feature = "cuda")]
use super::{FusionRegion, ReductionRegion, RegionInput};
use super::{MatMulRegion, RegionOpKind};
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

impl<D: TileDType, G> From<TensorGraph<D, G>> for TileGraph {
    fn from(tensor_graph: TensorGraph<D, G>) -> Self {
        Self::from_with_tf32_support(tensor_graph, true)
    }
}

impl TileGraph {
    /// Validate layouts and optimize indices in physical and diagnostic tile kernels.
    pub fn optimize_indices(&mut self) -> Result<()> {
        for ir in self.graph.node_weights_mut() {
            ir.validate_layouts()?;
            ir.optimize_indices()?;
        }
        #[cfg(feature = "cuda")]
        for kernel in self
            .region_kernels
            .iter_mut()
            .chain(&mut self.reduction_region_kernels)
            .chain(&mut self.matmul_region_kernels)
        {
            kernel.ir.validate_layouts()?;
            kernel.ir.optimize_indices()?;
        }
        Ok(())
    }

    /// Lower a graph while preserving its host storage dtype.
    pub fn from_graph<D: TileDType, G>(graph: &TensorGraph<D, G>) -> Self {
        Self::from_with_matmul_schedules(graph, true, &HashMap::new())
    }

    pub(crate) fn from_with_tf32_support<D: TileDType, G>(
        tensor_graph: TensorGraph<D, G>,
        supports_tf32: bool,
    ) -> Self {
        Self::from_with_matmul_schedules(&tensor_graph, supports_tf32, &HashMap::new())
    }

    pub(crate) fn from_with_matmul_schedules<D: TileDType, G>(
        tensor_graph: &TensorGraph<D, G>,
        supports_tf32: bool,
        schedules: &HashMap<NodeIndex, MatMulSchedule>,
    ) -> Self {
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
                    Self::lower_node(
                        node.clone(),
                        D::TILE_DTYPE,
                        &shape,
                        &input_shapes,
                        supports_tf32,
                        schedules.get(&idx).copied(),
                    )
                },
                |_, e| *e,
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

    #[cfg(feature = "cuda")]
    pub(crate) fn add_fusion_regions(&mut self, regions: &[FusionRegion]) -> Result<()> {
        self.region_kernels = regions
            .iter()
            .enumerate()
            .map(|(region_id, region)| {
                Ok(TileRegionKernel {
                    region_id,
                    ir: region.lower_to_tile_ir(region_id)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn add_reduction_regions(&mut self, regions: &[ReductionRegion]) -> Result<()> {
        self.reduction_region_kernels = regions
            .iter()
            .enumerate()
            .map(|(region_id, region)| {
                Ok(TileRegionKernel {
                    region_id,
                    ir: region.lower_to_tile_ir(region_id)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn add_matmul_regions(&mut self, regions: &[MatMulRegion]) -> Result<()> {
        self.matmul_region_kernels = regions
            .iter()
            .enumerate()
            .map(|(region_id, region)| {
                Ok(TileRegionKernel {
                    region_id,
                    ir: Self::lower_matmul_region(region, region_id)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn set_physical_nodes(&mut self, nodes: HashSet<NodeIndex>) {
        self.physical_nodes = Some(nodes);
    }

    #[allow(dead_code)]
    fn lower_node<D: TileDType>(
        node: TensorGraphNode<D>,
        dtype: DType,
        shape: &[usize],
        input_shapes: &[&[usize]],
        supports_tf32: bool,
        schedule: Option<MatMulSchedule>,
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
                Self::lower_matmul_impl(
                    m,
                    n,
                    k,
                    schedule.unwrap_or_else(|| {
                        MatMulSchedule::select_for_dtype(
                            m,
                            n,
                            k,
                            dtype,
                            MatMulCapabilities {
                                tf32: supports_tf32,
                                f16: supports_tf32,
                                bf16: supports_tf32,
                            },
                            MatMulPrecision::AllowTf32,
                        )
                    }),
                    None,
                    None,
                    (false, false),
                )
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
        builder.bounds_check(if dtype == DType::F32 {
            index_count * width
        } else {
            vocabulary * width
        });
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
        builder.bounds_check(if dtype == DType::F32 {
            index_count * width
        } else {
            vocabulary * width
        });
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
    #[cfg(feature = "cuda")]
    fn lower_matmul_region(region: &MatMulRegion, region_id: usize) -> Result<TileIR> {
        let operand = |value| {
            region
                .inputs
                .iter()
                .find(|input| input.value == value)
                .ok_or_else(|| anyhow!("matmul operand binding is unavailable"))
        };
        let staging = (
            stage_rows_contiguously(operand(region.lhs_value)?)?,
            stage_rows_contiguously(operand(region.rhs_value)?)?,
        );
        let address = |value, shape: &[usize], row: Expr, col: Expr| {
            let input = region
                .inputs
                .iter()
                .find(|input| input.value == value)
                .ok_or_else(|| anyhow!("matmul operand binding is unavailable"))?;
            let batch = &shape[..shape.len() - 2];
            let mut stride = 1;
            let mut coordinates = Vec::new();
            for (&extent, &output_extent) in batch.iter().rev().zip(region.batch_shape.iter().rev())
            {
                coordinates.push(if extent == 1 {
                    Expr::Const(0)
                } else {
                    Expr::Mod(
                        Box::new(Expr::FloorDiv(Box::new(Expr::BlockIdx(Dim::Z)), stride)),
                        extent,
                    )
                });
                stride *= output_extent;
            }
            coordinates.reverse();
            coordinates.extend([row, col]);
            input_coordinate_index(input, &coordinates)
        };
        let tile = region.schedule.block_tile;
        let staging_rounds = tile.k * tile.m.max(tile.n) / region.schedule.thread_count();
        let indices = (0..staging_rounds)
            .map(|round| {
                let ((a_row, a_col), (b_row, b_col)) =
                    matmul_operand_coordinates(region.schedule, staging, round);
                Ok((
                    address(region.lhs_value, &region.lhs_shape, a_row, a_col)?,
                    address(region.rhs_value, &region.rhs_shape, b_row, b_col)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::lower_matmul_impl(
            region.m,
            region.n,
            region.k,
            region.schedule,
            Some((region, region_id)),
            Some(indices),
            staging,
        ))
    }

    fn lower_matmul_impl(
        m: usize,
        n: usize,
        k: usize,
        schedule: MatMulSchedule,
        region: Option<(&MatMulRegion, usize)>,
        operand_indices: Option<Vec<(Expr, Expr)>>,
        staging: (bool, bool),
    ) -> TileIR {
        let dtype = schedule.storage_dtype;
        let mut builder = TileIRBuilder::new();
        if let Some((_, region_id)) = region {
            builder.start_kernel(&format!("matmul_region_{region_id}"));
        } else {
            builder.start_kernel("matmul");
        }

        let (a_param, b_param) = if let Some((region, _)) = region {
            for index in 0..region.inputs.len() {
                builder.add_param(&format!("input_{index}"), dtype, true);
            }
            for index in 0..region.outputs.len() {
                builder.add_param(&format!("output_{index}"), dtype, false);
            }
            // Region construction guarantees that both operand SSA values are
            // represented in the input table, including when they alias.
            let names: HashMap<_, _> = region
                .inputs
                .iter()
                .enumerate()
                .map(|(index, input)| (input.value, format!("input_{index}")))
                .collect();
            (
                names[&region.lhs_value].clone(),
                names[&region.rhs_value].clone(),
            )
        } else {
            builder.add_param("A", dtype, true);
            builder.add_param("B", dtype, true);
            builder.add_param("C", dtype, false);
            ("A".to_string(), "B".to_string())
        };

        let (tile_m, tile_n, tile_k) = (
            schedule.block_tile.m,
            schedule.block_tile.n,
            schedule.block_tile.k,
        );
        let threads = schedule.thread_count();
        let plan = schedule.plan;
        let operand_dtype = if schedule.plan == MatMulPlan::ScalarF32 {
            dtype
        } else {
            schedule.operand_dtype
        };
        let a_batch = region
            .map(|(region, _)| batch_element_offset(&region.batch_shape, &region.lhs_shape))
            .unwrap_or(Expr::Const(0));
        let b_batch = region
            .map(|(region, _)| batch_element_offset(&region.batch_shape, &region.rhs_shape))
            .unwrap_or(Expr::Const(0));
        let output_batch = if region.is_some() {
            Expr::BlockIdx(Dim::Z) * (m * n)
        } else {
            Expr::Const(0)
        };

        let a_smem =
            builder.alloc_shared_layout(operand_dtype, tile_m, tile_k, schedule.operand_layouts.0);
        let b_smem =
            builder.alloc_shared_layout(operand_dtype, tile_k, tile_n, schedule.operand_layouts.1);
        let c_smem = match plan {
            MatMulPlan::ScalarF32 => None,
            MatMulPlan::TensorCoreTf32 | MatMulPlan::TensorCoreF16 | MatMulPlan::TensorCoreBF16 => {
                Some(builder.alloc_shared(DType::F32, tile_m, tile_n))
            }
        };

        // Allocate register tiles for computation
        let c_reg = match plan {
            MatMulPlan::ScalarF32 => builder.alloc_register(DType::F32, tile_m, tile_n),
            MatMulPlan::TensorCoreTf32 | MatMulPlan::TensorCoreF16 | MatMulPlan::TensorCoreBF16 => {
                builder.alloc_fragment(DType::F32, tile_m, tile_n, schedule)
            }
        };

        // Initialize accumulator
        builder.zero(c_reg);

        // Calculate number of K tiles needed
        let k_tiles = k.div_ceil(tile_k);

        // Main tiling loop over K dimension
        builder.for_loop("k_tile", 0, k_tiles as i64, |builder, _| {
            for round in 0..tile_m * tile_k / threads {
                let ((a_row, a_col), _) = matmul_operand_coordinates(schedule, staging, round);
                builder.load_global_to_shared_indexed(
                    a_smem,
                    &a_param,
                    operand_indices
                        .as_ref()
                        .map(|indices| indices[round].0.clone())
                        .unwrap_or_else(|| a_batch.clone() + a_row.clone() * k + a_col.clone()),
                    (a_row, a_col),
                    operand_staging_coordinates(schedule, true, staging.0, round),
                    MatrixLayout {
                        rows: m,
                        cols: k,
                        row_stride: k,
                    },
                );
            }
            for round in 0..tile_k * tile_n / threads {
                let (_, (b_row, b_col)) = matmul_operand_coordinates(schedule, staging, round);
                builder.load_global_to_shared_indexed(
                    b_smem,
                    &b_param,
                    operand_indices
                        .as_ref()
                        .map(|indices| indices[round].1.clone())
                        .unwrap_or_else(|| b_batch.clone() + b_row.clone() * n + b_col.clone()),
                    (b_row, b_col),
                    operand_staging_coordinates(schedule, false, staging.1, round),
                    MatrixLayout {
                        rows: k,
                        cols: n,
                        row_stride: n,
                    },
                );
            }

            builder.barrier();

            // Compute: C_reg += A_reg @ B_reg
            builder.matmul(c_reg, a_smem, b_smem, MatMulLayout::NN, schedule);

            builder.barrier();
        });

        if let Some(c_smem) = c_smem {
            builder.convert_layout(c_smem, c_reg);
            builder.barrier();
        }
        let output = |builder: &mut TileIRBuilder, round: usize| {
            let (local_row, local_col) =
                tile_thread_coordinates(schedule, tile_m, tile_n, false, round);
            // Pointwise consumers require one scalar per thread. Scalar accumulators
            // already match; fragment accumulators need an explicit shared exchange.
            let matmul_element = if let Some(c_smem) = c_smem {
                let scalar = builder.alloc_register(DType::F32, 1, 1);
                builder.extract_scalar(scalar, c_smem, (local_row.clone(), local_col.clone()));
                scalar
            } else {
                c_reg
            };

            // Each round extracts and stores one output per thread.
            let global_row = Expr::Add(
                Box::new(Expr::BlockIdx(Dim::Y) * tile_m),
                Box::new(local_row),
            );
            let global_col = Expr::Add(
                Box::new(Expr::BlockIdx(Dim::X) * tile_n),
                Box::new(local_col),
            );
            let output_layout = MatrixLayout {
                rows: m,
                cols: n,
                row_stride: n,
            };
            if let Some((region, _)) = region {
                let mut registers = HashMap::new();
                registers.insert(region.matmul_value, matmul_element);
                let linear_element = output_batch.clone()
                    + Expr::Add(
                        Box::new(Expr::Mul(
                            Box::new(global_row.clone()),
                            Box::new(Expr::Const(n as i64)),
                        )),
                        Box::new(global_col.clone()),
                    );
                for (index, input) in region.inputs.iter().enumerate().filter(|(_, input)| {
                    region
                        .epilogue_operations
                        .iter()
                        .any(|operation| match operation.kind {
                            RegionOpKind::Unary { input: value, .. } => value == input.value,
                            RegionOpKind::Binary { lhs, rhs, .. }
                            | RegionOpKind::Gt { lhs, rhs } => {
                                lhs == input.value || rhs == input.value
                            }
                            RegionOpKind::Mask { values, condition } => {
                                values == input.value || condition == input.value
                            }
                        })
                }) {
                    let value = builder.alloc_register(DType::F32, 1, 1);
                    let element_index =
                        input_element_index(input, &region.output_shape, &linear_element)
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
                            tensor::BinaryOp::Add => {
                                builder.add(output, register(lhs), register(rhs))
                            }
                            tensor::BinaryOp::Sub => {
                                builder.sub(output, register(lhs), register(rhs))
                            }
                            tensor::BinaryOp::Mul => {
                                builder.mul(output, register(lhs), register(rhs))
                            }
                            tensor::BinaryOp::Div => {
                                builder.div(output, register(lhs), register(rhs))
                            }
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
                    builder.store_global_indexed(
                        &format!("output_{index}"),
                        registers[&output.value],
                        linear_element.clone(),
                        (global_row.clone(), global_col.clone()),
                        output_layout,
                    );
                }
            } else {
                builder.store_global_predicated(
                    "C",
                    matmul_element,
                    global_row,
                    global_col,
                    output_layout,
                );
            }
        };
        for round in 0..tile_m * tile_n / threads {
            output(&mut builder, round);
        }

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

fn staging_coordinates(transposed: bool) -> (Expr, Expr) {
    if transposed {
        (Expr::ThreadIdx(Dim::X), Expr::ThreadIdx(Dim::Y))
    } else {
        (Expr::ThreadIdx(Dim::Y), Expr::ThreadIdx(Dim::X))
    }
}

// Expanded schedules traverse each shared tile in complete, disjoint rounds.
// Swapping the decoding order keeps logical transposes coalesced even for rectangular tiles.
fn operand_staging_coordinates(
    schedule: MatMulSchedule,
    lhs: bool,
    transposed: bool,
    round: usize,
) -> (Expr, Expr) {
    if schedule.block_threads == (16, 16, 1) {
        return staging_coordinates(transposed);
    }
    let (rows, cols) = if lhs {
        (schedule.block_tile.m, schedule.block_tile.k)
    } else {
        (schedule.block_tile.k, schedule.block_tile.n)
    };
    tile_thread_coordinates(schedule, rows, cols, transposed, round)
}

fn tile_thread_coordinates(
    schedule: MatMulSchedule,
    rows: usize,
    cols: usize,
    transposed: bool,
    round: usize,
) -> (Expr, Expr) {
    if schedule.block_threads == (16, 16, 1) {
        return staging_coordinates(transposed);
    }
    let width = if transposed { rows } else { cols };
    if width == schedule.block_threads.0 as usize {
        let row = Expr::ThreadIdx(Dim::Y)
            + Expr::Const((round * schedule.block_threads.1 as usize) as i64);
        let col = Expr::ThreadIdx(Dim::X);
        return if transposed { (col, row) } else { (row, col) };
    }
    let index = Expr::Const((round * schedule.thread_count()) as i64)
        + Expr::ThreadIdx(Dim::Y) * schedule.block_threads.0 as usize
        + Expr::ThreadIdx(Dim::X);
    if transposed {
        (
            Expr::Mod(Box::new(index.clone()), rows),
            Expr::FloorDiv(Box::new(index), rows),
        )
    } else {
        (
            Expr::FloorDiv(Box::new(index.clone()), cols),
            Expr::Mod(Box::new(index), cols),
        )
    }
}

// Choose which matrix coordinate varies across adjacent staging threads. This
// changes only the tile traversal, not its contents or MMA layout.
// Unit steps also handle a transpose composed with reshape/permutation maps.
#[cfg(feature = "cuda")]
fn stage_rows_contiguously(input: &RegionInput) -> Result<bool> {
    let rank = input.tensor.shape.len();
    if rank < 2 || input.tensor.shape[rank - 2] < 2 || input.tensor.shape[rank - 1] < 2 {
        return Ok(false);
    }
    let offset = |iteration: &[usize]| -> Result<usize> {
        input
            .tensor
            .source_index(iteration)?
            .iter()
            .zip(&input.source_shape)
            .try_fold(0usize, |offset, (&coordinate, &extent)| {
                offset
                    .checked_mul(extent)
                    .and_then(|offset| offset.checked_add(coordinate))
                    .ok_or_else(|| anyhow!("operand storage offset overflows usize"))
            })
    };
    let mut coordinate = vec![0; rank];
    let base = offset(&coordinate)?;
    coordinate[rank - 2] = 1;
    let row = offset(&coordinate)?.checked_sub(base);
    coordinate[rank - 2] = 0;
    coordinate[rank - 1] = 1;
    let col = offset(&coordinate)?.checked_sub(base);
    Ok(row == Some(1) && col.is_some_and(|stride| stride > 1))
}

fn matmul_operand_coordinates(
    schedule: MatMulSchedule,
    staging: (bool, bool),
    round: usize,
) -> ((Expr, Expr), (Expr, Expr)) {
    let (a_local_row, a_local_col) = operand_staging_coordinates(schedule, true, staging.0, round);
    let (b_local_row, b_local_col) = operand_staging_coordinates(schedule, false, staging.1, round);
    let a_row = Expr::BlockIdx(Dim::Y) * schedule.block_tile.m + a_local_row;
    let a_col = Expr::Var("k_tile".into()) * schedule.block_tile.k + a_local_col;
    let b_row = Expr::Var("k_tile".into()) * schedule.block_tile.k + b_local_row;
    let b_col = Expr::BlockIdx(Dim::X) * schedule.block_tile.n + b_local_col;
    ((a_row, a_col), (b_row, b_col))
}

/// Decode the output batch coordinate and drop broadcast dimensions before
/// computing an operand's contiguous matrix offset.
fn batch_element_offset(output_batch: &[usize], input_shape: &[usize]) -> Expr {
    // General index normalization and invariant binding happen after scheduling.
    let input_batch = &input_shape[..input_shape.len() - 2];
    let padding = output_batch.len() - input_batch.len();
    let mut offset = Expr::Const(0);
    let mut output_stride = 1;
    let mut input_stride = input_shape[input_shape.len() - 2] * input_shape[input_shape.len() - 1];
    for dimension in (0..output_batch.len()).rev() {
        if dimension >= padding {
            let extent = input_batch[dimension - padding];
            if extent != 1 {
                let coordinate = Expr::Mod(
                    Box::new(Expr::FloorDiv(
                        Box::new(Expr::BlockIdx(Dim::Z)),
                        output_stride,
                    )),
                    output_batch[dimension],
                );
                offset = offset + coordinate * input_stride;
            }
            input_stride *= extent;
        }
        output_stride *= output_batch[dimension];
    }
    offset
}
