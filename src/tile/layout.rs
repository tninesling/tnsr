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
        validate(&self.body, &mut HashMap::new(), &self.params)
            .with_context(|| format!("invalid layouts in {}", self.kernel_name))
    }
}

fn validate(
    block: &Block,
    tiles: &mut HashMap<TileVar, Tile>,
    params: &[super::KernelParam],
) -> Result<()> {
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
                    (MemorySpace::Shared, TileLayout::Shared(layout)) => {
                        layout.validate(*rows, *cols, *dtype)?;
                        true
                    }
                    (
                        MemorySpace::Fragment,
                        TileLayout::WarpAccumulator {
                            operand_dtype,
                            block_width,
                            warp_topology,
                        },
                    ) => {
                        matches!(warp_topology.0, 1 | 2 | 4 | 8)
                            && matches!(warp_topology.1, 1 | 2 | 4 | 8)
                            && warp_topology.0 * warp_topology.1 <= 8
                            && *rows == 16 * warp_topology.0
                            && *cols == 16 * warp_topology.1
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
            Stmt::SelectSharedStage {
                dest,
                first,
                second,
                ..
            } => {
                let (first, second) = (get(tiles, first)?, get(tiles, second)?);
                anyhow::ensure!(
                    matches!(first.layout, TileLayout::Shared(_))
                        && first.layout == second.layout
                        && first.dtype == second.dtype
                        && first.rows == second.rows
                        && first.cols == second.cols,
                    "pipeline stages must have identical shared layouts"
                );
                tiles.insert(*dest, first);
            }
            Stmt::LoadSharedToReg { dest, src } => {
                let source = get(tiles, src)?;
                anyhow::ensure!(
                    matches!(source.layout, TileLayout::Shared(_)),
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
                    matches!(schedule.pipeline_stages, 1 | 2),
                    "matmul pipeline must have one or two stages"
                );
                anyhow::ensure!(
                    matches!(layout, MatMulLayout::NN),
                    "matmul transposes must be expressed by logical access maps"
                );
                for (var, rows, cols, physical) in [
                    (
                        a,
                        schedule.block_tile.m,
                        schedule.block_tile.k,
                        schedule.operand_layouts.0,
                    ),
                    (
                        b,
                        schedule.block_tile.k,
                        schedule.block_tile.n,
                        schedule.operand_layouts.1,
                    ),
                ] {
                    let tile = get(tiles, var)?;
                    let dtype = if schedule.plan == MatMulPlan::ScalarF32
                        || schedule.pipeline_stages == 2
                    {
                        schedule.storage_dtype
                    } else {
                        schedule.operand_dtype
                    };
                    anyhow::ensure!(
                        tile.layout == TileLayout::Shared(physical)
                            && tile.rows == rows
                            && tile.cols == cols
                            && tile.dtype == dtype,
                        "matmul operand {} layout does not match its schedule",
                        var.0
                    );
                }
                if schedule.plan != MatMulPlan::ScalarF32 {
                    anyhow::ensure!(
                        schedule
                            .operand_layouts
                            .0
                            .supports_wmma(schedule.operand_dtype)
                            && schedule
                                .operand_layouts
                                .1
                                .supports_wmma(schedule.operand_dtype),
                        "matmul layout is incompatible with WMMA fragments"
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
            Stmt::ConvertLayout {
                dest,
                src,
                coordinates,
            } => {
                let (dest, src) = (get(tiles, dest)?, get(tiles, src)?);
                let valid = match (src.layout, dest.layout) {
                    (TileLayout::WarpAccumulator { .. }, TileLayout::Shared(layout)) => {
                        layout.supports_wmma(DType::F32)
                            && coordinates.is_none()
                            && src.rows == dest.rows
                            && src.cols == dest.cols
                            && dest.dtype == DType::F32
                    }
                    (TileLayout::Shared(_), TileLayout::ThreadScalar) => {
                        dest.rows == 1 && dest.cols == 1 && dest.dtype == DType::F32
                    }
                    (TileLayout::ThreadScalar, TileLayout::ThreadScalar) => {
                        coordinates.is_none() && src.dtype == dest.dtype
                    }
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
            Stmt::AsyncCopy {
                dest,
                layout,
                copy_bytes,
                ..
            } => {
                let tile = get(tiles, dest)?;
                anyhow::ensure!(
                    matches!(copy_bytes, 4 | 16),
                    "unsupported asynchronous copy width"
                );
                let group = copy_bytes / tile.dtype.size_bytes();
                let TileLayout::Shared(physical) = tile.layout else {
                    anyhow::bail!("asynchronous copy requires shared memory");
                };
                anyhow::ensure!(
                    physical.row_stride.is_multiple_of(group)
                        && physical.xor_mask & (group - 1) == 0
                        && layout.cols.is_multiple_of(group),
                    "asynchronous copies require aligned contiguous groups"
                );
            }
            Stmt::LoadGlobalToSharedPredicated { dest, .. } => {
                anyhow::ensure!(
                    matches!(get(tiles, dest)?.layout, TileLayout::Shared(_)),
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
            Stmt::SetScalar { dest, value } => {
                validate_scalar(value, tiles)?;
                define_scalar(*dest, tiles)?;
            }
            Stmt::LoadScalar { dest, source, .. } => {
                validate_memory(source, tiles, params, false)?;
                define_scalar(*dest, tiles)?;
            }
            Stmt::StoreScalar { target, value, .. } => {
                validate_scalar(&super::ScalarExpr::Var(*value), tiles)?;
                validate_memory(target, tiles, params, true)?;
            }
            Stmt::WarpReduce { dest, src, .. } => {
                validate_scalar(&super::ScalarExpr::Var(*src), tiles)?;
                define_scalar(*dest, tiles)?;
            }
            Stmt::If { condition, body } => {
                validate_predicate(condition, tiles)?;
                validate(body, &mut tiles.clone(), params)?;
            }
            Stmt::ForLoop { carries, body, .. } => {
                let mut carries_seen = std::collections::HashSet::new();
                for carry in carries {
                    anyhow::ensure!(
                        carries_seen.insert(carry.var),
                        "loop carry {} appears twice",
                        carry.var.0
                    );
                    validate_scalar(&carry.initial, tiles)?;
                    define_scalar(carry.var, tiles)?;
                }
                validate(body, &mut tiles.clone(), params)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn define_scalar(var: TileVar, tiles: &mut HashMap<TileVar, Tile>) -> Result<()> {
    if tiles.contains_key(&var) {
        validate_scalar(&super::ScalarExpr::Var(var), tiles)?;
    }
    tiles.insert(var, scalar_tile());
    Ok(())
}

fn scalar_tile() -> Tile {
    Tile {
        layout: TileLayout::ThreadScalar,
        dtype: DType::F32,
        rows: 1,
        cols: 1,
    }
}
fn validate_scalar(value: &super::ScalarExpr, tiles: &HashMap<TileVar, Tile>) -> Result<()> {
    use super::ScalarExpr;
    match value {
        ScalarExpr::Constant(_) => Ok(()),
        ScalarExpr::Var(var) => {
            let tile = tiles
                .get(var)
                .with_context(|| format!("scalar {} is not defined", var.0))?;
            anyhow::ensure!(
                tile.layout == TileLayout::ThreadScalar && tile.dtype == DType::F32,
                "scalar {} has incompatible layout",
                var.0
            );
            Ok(())
        }
        ScalarExpr::Unary { value, .. } => validate_scalar(value, tiles),
        ScalarExpr::Binary { lhs, rhs, .. } => {
            validate_scalar(lhs, tiles)?;
            validate_scalar(rhs, tiles)
        }
        ScalarExpr::Select {
            condition,
            then_value,
            else_value,
        } => {
            validate_predicate(condition, tiles)?;
            validate_scalar(then_value, tiles)?;
            validate_scalar(else_value, tiles)
        }
    }
}
fn validate_predicate(
    condition: &super::ScalarPredicate,
    tiles: &HashMap<TileVar, Tile>,
) -> Result<()> {
    match condition {
        super::ScalarPredicate::IndexLt { .. } => Ok(()),
        super::ScalarPredicate::Equal { lhs, rhs } => {
            validate_scalar(lhs, tiles)?;
            validate_scalar(rhs, tiles)
        }
    }
}
fn validate_memory(
    memory: &super::ScalarMemory,
    tiles: &HashMap<TileVar, Tile>,
    params: &[super::KernelParam],
    store: bool,
) -> Result<()> {
    match memory {
        super::ScalarMemory::Global(name) => {
            let param = params
                .iter()
                .find(|p| p.name == *name)
                .with_context(|| format!("scalar memory parameter {name} is missing"))?;
            anyhow::ensure!(
                matches!(param.dtype, DType::F32 | DType::F16 | DType::BF16),
                "scalar memory parameter {name} has unsupported dtype"
            );
            anyhow::ensure!(
                !store || !param.is_input,
                "scalar store targets input parameter {name}"
            );
        }
        super::ScalarMemory::Shared(var) => {
            let tile = tiles
                .get(var)
                .context("scalar shared allocation is missing")?;
            anyhow::ensure!(
                matches!(tile.layout, TileLayout::Shared(_)) && tile.dtype == DType::F32,
                "scalar shared memory must have f32 storage"
            );
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
        if let Stmt::ConvertLayout { dest, src, .. } = conversion {
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
