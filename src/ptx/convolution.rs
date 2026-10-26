//! Bounded schedules for small convolution weight tensors with long reductions.
use crate::tile::Conv2dGeometry;

pub(crate) const WEIGHT_GRADIENT_THREADS: u32 = 256;

pub(crate) fn weight_gradient_threads(geometry: Conv2dGeometry) -> u32 {
    let weights = geometry
        .output_channels
        .saturating_mul(geometry.input_channels)
        .saturating_mul(geometry.kernel_height)
        .saturating_mul(geometry.kernel_width);
    let reduction = geometry
        .batch
        .saturating_mul(geometry.output_height)
        .saturating_mul(geometry.output_width);
    if weights != 0 && weights <= 4096 && reduction >= 256 {
        WEIGHT_GRADIENT_THREADS
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::{TensorGraph, TensorGraphNode},
        ptx::PtxGraph,
        tensor::TensorExpr,
        tile::TileGraph,
    };

    fn geometry(weights: usize, reduction: usize) -> Conv2dGeometry {
        Conv2dGeometry {
            batch: 1,
            input_channels: 1,
            input_height: 1,
            input_width: reduction,
            output_channels: weights,
            output_height: 1,
            output_width: reduction,
            kernel_height: 1,
            kernel_width: 1,
            stride: 1,
            padding: 0,
        }
    }

    #[test]
    fn weight_tree_dispatch_is_bounded_by_output_and_reduction_parallelism() {
        for (weights, reduction, expected) in [
            (150, 50176, WEIGHT_GRADIENT_THREADS),
            (2400, 6400, WEIGHT_GRADIENT_THREADS),
            (4096, 256, WEIGHT_GRADIENT_THREADS),
            (4097, 256, 1),
            (150, 255, 1),
            (0, 256, 1),
            (589824, 169, 1),
        ] {
            assert_eq!(
                weight_gradient_threads(geometry(weights, reduction)),
                expected
            );
        }
    }

    #[test]
    fn tree_lowering_keeps_all_block_lanes_alive_and_records_its_launch() {
        let mut graph = TensorGraph::<f32>::new();
        let x = graph.add_expr(&TensorExpr::input("x", vec![1, 1, 17, 17]), false);
        let dy = graph.add_expr(&TensorExpr::input("dy", vec![1, 3, 17, 17]), false);
        let dw = graph.graph.add_node(TensorGraphNode::Conv2dBackwardWeight {
            shape: vec![3, 1, 1, 1],
            stride: 1,
            padding: 0,
        });
        graph.graph.add_edge(x, dw, 0);
        graph.graph.add_edge(dy, dw, 1);
        let serial = PtxGraph::from(TileGraph::from_graph(&graph));
        assert_eq!(serial.conv_weight_threads(dw), 1);
        assert!(!serial.to_ptx().contains("bar.sync"));
        let mut tiles = TileGraph::from_graph(&graph);
        tiles.schedule_conv_weight_reductions();
        assert!(
            !tiles.graph[dw]
                .body
                .stmts
                .iter()
                .any(|s| matches!(s, crate::tile::Stmt::BoundsCheck { .. }))
        );
        let tree = PtxGraph::from(tiles);
        assert_eq!(tree.conv_weight_threads(dw), WEIGHT_GRADIENT_THREADS);
        assert!(tree.to_ptx().contains("bar.sync"));
        assert!(tree.to_ptx().contains("reduction_partials"));
        assert!(!tree.to_ptx().contains("atom."));
    }
    #[test]
    fn bound_half_storage_keeps_serial_weight_order_and_recompile_restores_tree() {
        use crate::ptx::{PtxExecutor, PtxReductionMode};
        use std::collections::HashMap;
        let mut executor = match PtxExecutor::<crate::ptx::F32>::try_new_with_reduction_mode(
            PtxReductionMode::DeterministicTree,
        ) {
            Ok(value) => value,
            Err(error)
                if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                    && error
                        .to_string()
                        .starts_with("Failed to initialize CUDA device 0") =>
            {
                return;
            }
            Err(error) => panic!("{error:#}"),
        };
        let mut graph = TensorGraph::<f32>::new();
        let x = graph.add_expr(&TensorExpr::input("x", vec![1, 1, 17, 17]), false);
        let dy = graph.add_expr(&TensorExpr::input("dy", vec![1, 3, 17, 17]), false);
        let dw = graph.graph.add_node(TensorGraphNode::Conv2dBackwardWeight {
            shape: vec![3, 1, 1, 1],
            stride: 1,
            padding: 0,
        });
        graph.graph.add_edge(x, dw, 0);
        graph.graph.add_edge(dy, dw, 1);
        for node in [x, dy, dw] {
            executor
                .compile_bound_storage(&graph, &HashMap::from([(node, crate::tile::DType::BF16)]))
                .unwrap();
            assert!(!executor.module_source().unwrap().contains("bar.sync"));
            assert!(
                executor
                    .conv_weight_tree_launch_config(dw, 3)
                    .unwrap()
                    .is_none()
            );
        }
        executor.compile(&graph).unwrap();
        assert!(executor.module_source().unwrap().contains("bar.sync"));
        let launch = executor
            .conv_weight_tree_launch_config(dw, 3)
            .unwrap()
            .unwrap();
        assert_eq!(launch.grid_dim, (3, 1, 1));
        assert_eq!(launch.block_dim, (WEIGHT_GRADIENT_THREADS, 1, 1));
    }
}
