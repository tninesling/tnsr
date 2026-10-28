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
        for param in self.params.iter().filter(|param| param.dtype == DType::U32) {
            let indexed_slot = self.body.stmts.iter().any(|stmt| match stmt {
                Stmt::Embedding { .. } | Stmt::EmbeddingBackward { .. } => param.name == "indices",
                Stmt::IndexedCrossEntropy { .. } | Stmt::IndexedCrossEntropyBackward { .. } => {
                    param.name == "targets"
                }
                Stmt::LoadIndex { source, .. } => source == &param.name,
                _ => false,
            });
            anyhow::ensure!(
                param.is_input && indexed_slot,
                "integer storage is only supported for indexed arguments"
            );
        }
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
                anyhow::ensure!(
                    *dtype != DType::U32,
                    "integer indices cannot be floating computation tiles"
                );
                let valid = match (space, layout) {
                    (MemorySpace::Register, TileLayout::ThreadScalar) => true,
                    (MemorySpace::Register, TileLayout::ThreadArray { elements }) => {
                        *elements == 4 && *dtype == DType::F32 && *rows == 32 && *cols == 32
                    }
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
                            warp_fragments,
                        },
                    ) => {
                        matches!(warp_topology.0, 1 | 2 | 4 | 8)
                            && matches!(warp_topology.1, 1 | 2 | 4 | 8)
                            && warp_topology.0 * warp_topology.1 <= 8
                            && matches!(warp_fragments.0, 1 | 2 | 4)
                            && matches!(warp_fragments.1, 1 | 2 | 4)
                            && *rows == 16 * warp_topology.0 * warp_fragments.0
                            && *cols == 16 * warp_topology.1 * warp_fragments.1
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
            Stmt::AliasShared {
                dest,
                src,
                layout,
                rows,
                cols,
            } => {
                let source = get(tiles, src)?;
                let TileLayout::Shared(source_layout) = source.layout else {
                    anyhow::bail!("shared alias source must be shared memory");
                };
                layout.validate(*rows, *cols, source.dtype)?;
                anyhow::ensure!(
                    layout.allocation_bytes(*rows, *cols, source.dtype)
                        <= source_layout.allocation_bytes(source.rows, source.cols, source.dtype),
                    "shared alias exceeds its source allocation"
                );
                tiles.insert(
                    *dest,
                    Tile {
                        layout: TileLayout::Shared(*layout),
                        dtype: source.dtype,
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
                    (schedule.pipeline_stages == 2)
                        == (schedule.async_operands.0 || schedule.async_operands.1),
                    "matmul async operands must match its pipeline stages"
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
                        || schedule.plan == MatMulPlan::TensorCoreTf32x3
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
                        schedule.warp_topology.0 * schedule.warp_topology.1
                            <= schedule.thread_count() / 32,
                        "matmul computing warps exceed the block thread count"
                    );
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
                if let TileLayout::ThreadArray { elements } = expected {
                    anyhow::ensure!(
                        elements == 4
                            && schedule.storage_dtype == DType::F32
                            && schedule.block_tile.m == 32
                            && schedule.block_tile.n == 32
                            && schedule.block_tile.k == 16
                            && schedule.block_threads == (16, 16, 1),
                        "unsupported thread accumulator mapping"
                    );
                }
                anyhow::ensure!(
                    (schedule.plan == MatMulPlan::ScalarF32)
                        == matches!(
                            expected,
                            TileLayout::ThreadScalar | TileLayout::ThreadArray { .. }
                        ),
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
            Stmt::ExtractRegister { dest, src, index } => {
                let (dest, src) = (get(tiles, dest)?, get(tiles, src)?);
                let TileLayout::ThreadArray { elements } = src.layout else {
                    anyhow::bail!("register extraction requires a thread array");
                };
                anyhow::ensure!(
                    *index < elements
                        && dest.layout == TileLayout::ThreadScalar
                        && dest.dtype == DType::F32
                        && dest.rows == 1
                        && dest.cols == 1,
                    "invalid register array extraction"
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
                    (TileLayout::Shared(layout), TileLayout::WarpAccumulator { .. }) => {
                        !layout.column_major
                            && layout.supports_wmma(DType::F32)
                            && coordinates.is_none()
                            && src.rows == dest.rows
                            && src.cols == dest.cols
                            && src.dtype == DType::F32
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
            | Stmt::Sqrt { dest, src }
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
                        && (if physical.column_major {
                            layout.rows
                        } else {
                            layout.cols
                        })
                        .is_multiple_of(group),
                    "asynchronous copies require aligned contiguous groups"
                );
            }
            Stmt::Load { src_param, .. } => {
                reject_integer_memory(params, src_param)?;
            }
            Stmt::LoadGlobalToSharedPredicated {
                dest, src_param, ..
            } => {
                reject_integer_memory(params, src_param)?;
                anyhow::ensure!(
                    matches!(get(tiles, dest)?.layout, TileLayout::Shared(_)),
                    "staged operand load requires a shared tile"
                );
            }
            Stmt::LoadGlobalPredicated {
                dest, src_param, ..
            } => {
                reject_integer_memory(params, src_param)?;
                anyhow::ensure!(
                    get(tiles, dest)?.layout == TileLayout::ThreadScalar,
                    "scalar load requires a thread scalar tile"
                );
            }
            Stmt::StoreGlobalPredicated {
                src, dest_param, ..
            }
            | Stmt::Store {
                src, dest_param, ..
            } => {
                reject_integer_memory(params, dest_param)?;
                anyhow::ensure!(
                    get(tiles, src)?.layout == TileLayout::ThreadScalar,
                    "global store requires an explicit conversion to thread scalars"
                );
            }
            Stmt::LoadIndex { source, .. } => {
                anyhow::ensure!(
                    params
                        .iter()
                        .any(|p| p.name == *source && p.is_input && p.dtype == DType::U32),
                    "index load requires a u32 input"
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

fn reject_integer_memory(params: &[super::KernelParam], name: &str) -> Result<()> {
    anyhow::ensure!(
        params
            .iter()
            .all(|param| param.name != name || param.dtype != DType::U32),
        "integer indices require an indexed load, not floating memory access"
    );
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
                matches!(tile.layout, TileLayout::Shared(_))
                    && matches!(tile.dtype, DType::F32 | DType::F16 | DType::BF16),
                "scalar shared memory must have floating-point storage"
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
    fn exact_integer_storage_rejects_floating_loads_and_tiles() {
        let mut ir = TileIR {
            kernel_name: "embedding".into(),
            params: vec![super::super::KernelParam {
                name: "indices".into(),
                dtype: DType::U32,
                is_input: true,
            }],
            body: Block {
                stmts: vec![Stmt::Embedding {
                    vocabulary: 683,
                    width: 1,
                    index_count: 1,
                }],
            },
            shared_mem_bytes: 0,
        };
        ir.validate_layouts().unwrap();
        ir.body.stmts.push(Stmt::Load {
            dest: TileVar(0),
            src_param: "indices".into(),
            row_offset: super::super::Expr::Const(0),
            col_offset: super::super::Expr::Const(0),
        });
        assert!(
            ir.validate_layouts()
                .unwrap_err()
                .to_string()
                .contains("invalid layouts")
        );
        ir.body.stmts.pop();
        ir.body.stmts.push(Stmt::AllocTile {
            var: TileVar(0),
            space: MemorySpace::Register,
            layout: TileLayout::ThreadScalar,
            dtype: DType::U32,
            rows: 1,
            cols: 1,
        });
        assert!(ir.validate_layouts().is_err());
        ir.body.stmts.pop();
        ir.params[0].is_input = false;
        assert!(ir.validate_layouts().is_err());
    }

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
    #[test]
    fn shared_alias_checks_physical_extent_including_padding() {
        use crate::tile::SharedLayout;
        let mut ir = TileIR {
            kernel_name: "shared_alias".into(),
            params: vec![],
            shared_mem_bytes: 4096,
            body: Block {
                stmts: vec![
                    Stmt::AllocTile {
                        var: TileVar(0),
                        space: MemorySpace::Shared,
                        layout: TileLayout::Shared(SharedLayout {
                            row_stride: 64,
                            column_major: true,
                            xor_mask: 0,
                        }),
                        dtype: DType::BF16,
                        rows: 64,
                        cols: 32,
                    },
                    Stmt::AliasShared {
                        dest: TileVar(1),
                        src: TileVar(0),
                        layout: SharedLayout::row_major(64),
                        rows: 32,
                        cols: 64,
                    },
                ],
            },
        };
        ir.validate_layouts().unwrap();
        if let Stmt::AliasShared { rows, .. } = &mut ir.body.stmts[1] {
            *rows = 33;
        }
        assert!(ir.validate_layouts().is_err());
        if let Stmt::AliasShared { rows, layout, .. } = &mut ir.body.stmts[1] {
            *rows = 32;
            *layout = SharedLayout::row_major(72);
        }
        assert!(ir.validate_layouts().is_err());
        ir.body.stmts[0] = Stmt::AllocTile {
            var: TileVar(0),
            space: MemorySpace::Register,
            layout: TileLayout::ThreadScalar,
            dtype: DType::F32,
            rows: 1,
            cols: 1,
        };
        assert!(ir.validate_layouts().is_err());
    }
}
