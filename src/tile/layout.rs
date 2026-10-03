//! Check physical producer/consumer representations before backend lowering.
//! Logical tensor views remain index maps; these layouts describe local tiles only.
use std::collections::HashMap;

use anyhow::{Context, Result};

use super::{
    Block, DType, MatMulLayout, MatMulPlan, MemorySpace, Stmt, TileIR, TileLayout, TileVar,
};

#[derive(Clone, Copy)]
struct Tile {
    layout: TileLayout,
    dtype: DType,
    rows: usize,
    cols: usize,
}

impl TileIR {
    /// Reject incompatible physical layouts instead of inferring conversions in PTX.
    pub fn validate_layouts(&self) -> Result<()> {
        validate(&self.body, &mut HashMap::new())
            .with_context(|| format!("invalid layouts in {}", self.kernel_name))
    }
}

fn validate(block: &Block, tiles: &mut HashMap<TileVar, Tile>) -> Result<()> {
    let get = |tiles: &HashMap<TileVar, Tile>, var: &TileVar| {
        tiles
            .get(var)
            .copied()
            .with_context(|| format!("tile {} has no layout", var.0))
    };
    for stmt in &block.stmts {
        match stmt {
            Stmt::AllocTile {
                var,
                space,
                layout,
                dtype,
                rows,
                cols,
            } => {
                let valid = match (space, layout) {
                    (MemorySpace::Register, TileLayout::ThreadScalar) => true,
                    (MemorySpace::Shared, TileLayout::SharedRowMajor { row_stride }) => {
                        *row_stride == *cols
                    }
                    (
                        MemorySpace::Fragment,
                        TileLayout::WarpAccumulator {
                            operand_dtype,
                            block_width,
                        },
                    ) => {
                        *rows == 16
                            && *cols == 16
                            && *dtype == DType::F32
                            && matches!(operand_dtype, DType::TF32 | DType::F16 | DType::BF16)
                            && *block_width > 0
                            && *block_width <= 32
                            && 32 % block_width == 0
                    }
                    _ => false,
                };
                anyhow::ensure!(valid, "tile {} allocation does not match its layout", var.0);
                tiles.insert(
                    *var,
                    Tile {
                        layout: *layout,
                        dtype: *dtype,
                        rows: *rows,
                        cols: *cols,
                    },
                );
            }
            Stmt::LoadSharedToReg { dest, src } => {
                let source = get(tiles, src)?;
                anyhow::ensure!(
                    matches!(source.layout, TileLayout::SharedRowMajor { .. }),
                    "operand binding requires a shared tile"
                );
                get(tiles, dest)?;
                tiles.insert(*dest, source);
            }
            Stmt::MatMul {
                dest,
                a,
                b,
                layout,
                schedule,
            } => {
                anyhow::ensure!(
                    matches!(layout, MatMulLayout::NN),
                    "matmul transposes must be expressed by logical access maps"
                );
                for (var, rows, cols) in [
                    (a, schedule.block_tile.m, schedule.block_tile.k),
                    (b, schedule.block_tile.k, schedule.block_tile.n),
                ] {
                    let tile = get(tiles, var)?;
                    let dtype = if schedule.plan == MatMulPlan::ScalarF32 {
                        schedule.storage_dtype
                    } else {
                        schedule.operand_dtype
                    };
                    anyhow::ensure!(
                        tile.layout == TileLayout::SharedRowMajor { row_stride: cols }
                            && tile.rows == rows
                            && tile.cols == cols
                            && tile.dtype == dtype,
                        "matmul operand {} layout does not match its schedule",
                        var.0
                    );
                }
                let dest = get(tiles, dest)?;
                let expected = schedule.accumulator_tile_layout();
                anyhow::ensure!(
                    (schedule.plan == MatMulPlan::ScalarF32)
                        == (expected == TileLayout::ThreadScalar),
                    "matmul arithmetic and accumulator layout are incompatible"
                );
                anyhow::ensure!(
                    dest.layout == expected
                        && dest.dtype == DType::F32
                        && dest.rows == schedule.block_tile.m
                        && dest.cols == schedule.block_tile.n,
                    "matmul accumulator layout does not match its schedule"
                );
            }
            Stmt::ConvertLayout { dest, src } => {
                let (dest, src) = (get(tiles, dest)?, get(tiles, src)?);
                let valid = match (src.layout, dest.layout) {
                    (TileLayout::WarpAccumulator { .. }, TileLayout::SharedRowMajor { .. }) => {
                        src.rows == dest.rows && src.cols == dest.cols && dest.dtype == DType::F32
                    }
                    (TileLayout::SharedRowMajor { .. }, TileLayout::ThreadScalar) => {
                        dest.rows == 1 && dest.cols == 1 && dest.dtype == DType::F32
                    }
                    (TileLayout::ThreadScalar, TileLayout::ThreadScalar) => src.dtype == dest.dtype,
                    _ => false,
                };
                anyhow::ensure!(valid, "unsupported tile layout conversion");
            }
            Stmt::Add { dest, a, b }
            | Stmt::Sub { dest, a, b }
            | Stmt::Mul { dest, a, b }
            | Stmt::Div { dest, a, b }
            | Stmt::Gt { dest, a, b } => {
                for var in [dest, a, b] {
                    anyhow::ensure!(
                        get(tiles, var)?.layout == TileLayout::ThreadScalar,
                        "pointwise operation requires thread scalars"
                    );
                }
            }
            Stmt::Neg { dest, src }
            | Stmt::Exp { dest, src }
            | Stmt::Log { dest, src }
            | Stmt::Relu { dest, src } => {
                for var in [dest, src] {
                    anyhow::ensure!(
                        get(tiles, var)?.layout == TileLayout::ThreadScalar,
                        "pointwise operation requires thread scalars"
                    );
                }
            }
            Stmt::Mask {
                dest,
                values,
                condition,
            } => {
                for var in [dest, values, condition] {
                    anyhow::ensure!(
                        get(tiles, var)?.layout == TileLayout::ThreadScalar,
                        "pointwise operation requires thread scalars"
                    );
                }
            }
            Stmt::LoadGlobalToSharedPredicated { dest, .. } => {
                anyhow::ensure!(
                    matches!(get(tiles, dest)?.layout, TileLayout::SharedRowMajor { .. }),
                    "staged operand load requires a shared tile"
                );
            }
            Stmt::LoadGlobalPredicated { dest, .. } => {
                anyhow::ensure!(
                    get(tiles, dest)?.layout == TileLayout::ThreadScalar,
                    "scalar load requires a thread scalar tile"
                );
            }
            Stmt::StoreGlobalPredicated { src, .. } | Stmt::Store { src, .. } => {
                anyhow::ensure!(
                    get(tiles, src)?.layout == TileLayout::ThreadScalar,
                    "global store requires an explicit conversion to thread scalars"
                );
            }
            Stmt::ForLoop { body, .. } => validate(body, &mut tiles.clone())?,
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{graph::TensorGraph, tensor::TensorExpr, tile::TileGraph};

    #[test]
    fn scalar_layout_flows_directly_and_fragment_exchange_is_explicit() {
        for size in [4, 32] {
            let input = TensorExpr::constant(vec![0.5; size * size], vec![size, size]);
            let graph: TensorGraph<f32> = input.clone().matmul(input).into();
            let tile_graph = TileGraph::from(graph);
            let ir = tile_graph
                .graph
                .node_weights()
                .find(|ir| ir.kernel_name == "matmul")
                .unwrap();
            ir.validate_layouts().unwrap();
            let conversions = ir
                .body
                .stmts
                .iter()
                .filter(|stmt| matches!(stmt, Stmt::ConvertLayout { .. }))
                .count();
            assert_eq!(conversions, if size == 4 { 0 } else { 2 });
            for stmt in &ir.body.stmts {
                if let Stmt::ForLoop { body, .. } = stmt {
                    assert!(
                        !body
                            .stmts
                            .iter()
                            .any(|stmt| matches!(stmt, Stmt::ConvertLayout { .. }))
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_implicit_fragment_store_and_incompatible_conversion() {
        let input = TensorExpr::constant(vec![0.5; 32 * 32], vec![32, 32]);
        let graph: TensorGraph<f32> = input.clone().matmul(input).into();
        let mut tile_graph = TileGraph::from(graph);
        let ir = tile_graph
            .graph
            .node_weights_mut()
            .find(|ir| ir.kernel_name == "matmul")
            .unwrap();
        let fragment = ir
            .body
            .stmts
            .iter()
            .find_map(|stmt| match stmt {
                Stmt::AllocTile {
                    var,
                    layout: TileLayout::WarpAccumulator { .. },
                    ..
                } => Some(*var),
                _ => None,
            })
            .unwrap();
        let store = ir
            .body
            .stmts
            .iter_mut()
            .find(|stmt| matches!(stmt, Stmt::StoreGlobalPredicated { .. }))
            .unwrap();
        if let Stmt::StoreGlobalPredicated { src, .. } = store {
            *src = fragment;
        }
        assert!(
            ir.validate_layouts()
                .unwrap_err()
                .root_cause()
                .to_string()
                .contains("explicit conversion")
        );
        ir.body.stmts.pop();
        let conversion = ir
            .body
            .stmts
            .iter_mut()
            .find(|stmt| matches!(stmt, Stmt::ConvertLayout { .. }))
            .unwrap();
        if let Stmt::ConvertLayout { dest, src } = conversion {
            *src = *dest;
        }
        assert!(
            ir.validate_layouts()
                .unwrap_err()
                .root_cause()
                .to_string()
                .contains("unsupported tile layout conversion")
        );
    }
}
