//! Disconnected optional transport proofs. Default scalar arithmetic is frozen.
use super::{
    backward,
    common::*,
    forward,
    leaf::{Descriptor, ReadLayout, Shape, Transport, UncompiledPair},
};
use crate::{
    graph::{NodeIndex, TensorGraph},
    ptx::{PtxGraph, PtxTarget},
    tile::*,
};
use anyhow::{Result, ensure};
use std::collections::HashMap;

mod gpu;

fn ptx(ir: TileIR) -> Result<String> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles
        .region_kernels
        .push(crate::tile::graph::TileRegionKernel { region_id: 0, ir });
    tiles.optimize_indices()?;
    Ok(PtxGraph::from(tiles)
        .with_target(PtxTarget {
            compute_capability: (8, 9),
            ..PtxTarget::sm80()
        })
        .to_ptx())
}

#[test]
fn online_async16_scalar_default_matches_revision4_ptx_goldens_first() -> Result<()> {
    let shape = Shape::new(8, 8)?;
    let defaults = [
        forward::ir(shape, ReadLayout::SequenceMajor)?,
        backward::d_query_ir(shape, ReadLayout::SequenceMajor, ReadLayout::SequenceMajor)?,
        backward::d_key_value_ir(shape, ReadLayout::SequenceMajor, ReadLayout::SequenceMajor)?,
    ];
    let explicit = [
        forward::ir_with_transport(shape, ReadLayout::SequenceMajor, Transport::Scalar)?,
        backward::d_query_ir_with_transport(
            shape,
            ReadLayout::SequenceMajor,
            ReadLayout::SequenceMajor,
            Transport::Scalar,
        )?,
        backward::d_key_value_ir_with_transport(
            shape,
            ReadLayout::SequenceMajor,
            ReadLayout::SequenceMajor,
            Transport::Scalar,
        )?,
    ];
    for ((old, new), filename) in defaults.into_iter().zip(explicit).zip([
        "online-forward.ptx",
        "online-dquery.ptx",
        "online-dkeyvalue.ptx",
    ]) {
        let original = ptx(old)?;
        let current = ptx(new)?;
        assert_eq!(
            original, current,
            "default/explicit Scalar source drift {filename}"
        );
        if let Some(directory) = std::env::var_os("TNSR_ONLINE_SCALAR_GOLDENS") {
            assert_eq!(
                current,
                std::fs::read_to_string(std::path::Path::new(&directory).join(filename))?,
                "revision4 PTX source drift {filename}"
            );
        }
        assert!(!current.contains("cp.async"));
    }
    Ok(())
}

fn eval(
    expr: &Expr,
    env: &HashMap<String, usize>,
    x: usize,
    y: usize,
    bx: usize,
    by: usize,
) -> usize {
    match expr {
        Expr::Const(value) => usize::try_from(*value).unwrap(),
        Expr::ThreadIdx(Dim::X) => x,
        Expr::ThreadIdx(Dim::Y) => y,
        Expr::BlockIdx(Dim::X) => bx,
        Expr::BlockIdx(Dim::Y) => by,
        Expr::Var(name) => env[name],
        Expr::Add(a, b) => eval(a, env, x, y, bx, by) + eval(b, env, x, y, bx, by),
        Expr::Sub(a, b) => eval(a, env, x, y, bx, by) - eval(b, env, x, y, bx, by),
        Expr::Mul(a, b) => eval(a, env, x, y, bx, by) * eval(b, env, x, y, bx, by),
        Expr::FloorDiv(a, n) => eval(a, env, x, y, bx, by) / n,
        Expr::Mod(a, n) => eval(a, env, x, y, bx, by) % n,
        other => panic!("unexpected transport index expression {other:?}"),
    }
}

#[test]
fn online_async16_full_copy_coverage_alignment_and_byte_bounds() -> Result<()> {
    for shape in [
        Shape::new(8, 8)?,
        Shape::new(16, 8)?,
        Shape::new(32, 8)?,
        Shape::new(4, 16)?,
    ] {
        for layout in [ReadLayout::HeadMajor, ReadLayout::SequenceMajor] {
            for column_major in [false, true] {
                let mut b = ScalarBuilder::default();
                let target = b.var();
                stage_with_transport(
                    &mut b,
                    shape,
                    layout,
                    Stage {
                        prefix: "proof",
                        source: "query",
                        row_base: Expr::BlockIdx(Dim::X) * 64usize,
                        dest: target,
                        column_major,
                    },
                    Transport::Async16,
                )?;
                let [
                    Stmt::ForLoop {
                        loop_var,
                        end,
                        body,
                        ..
                    },
                ] = b.stmts.as_slice()
                else {
                    panic!("expected exact two-round transport loop")
                };
                assert!(matches!(end, Expr::Const(2)));
                let [
                    Stmt::AsyncCopy {
                        element_index,
                        row,
                        col,
                        tile_row,
                        tile_col,
                        layout: bounds,
                        copy_bytes,
                        ..
                    },
                ] = body.stmts.as_slice()
                else {
                    panic!("expected one raw-copy instruction per round")
                };
                assert_eq!(*copy_bytes, 16);
                assert_eq!((bounds.rows, bounds.cols), (512, 64));
                assert_eq!(
                    bounds.row_stride,
                    if layout == ReadLayout::HeadMajor {
                        64
                    } else {
                        shape.heads * 64
                    }
                );
                let mut coverage = [0_u8; 4096];
                let mut environment = HashMap::new();
                for by in 0..shape.bh() {
                    for bx in 0..8 {
                        for round in 0..2 {
                            environment.insert(loop_var.clone(), round);
                            for y in 0..8 {
                                for x in 0..32 {
                                    let source = eval(element_index, &environment, x, y, bx, by);
                                    let r = eval(row, &environment, x, y, bx, by);
                                    let c = eval(col, &environment, x, y, bx, by);
                                    let tr = eval(tile_row, &environment, x, y, bx, by);
                                    let tc = eval(tile_col, &environment, x, y, bx, by);
                                    let physical = if column_major {
                                        tc * 64 + tr
                                    } else {
                                        tr * 64 + tc
                                    };
                                    assert!(r < 512 && c + 7 < 64 && tr < 64 && tc < 64);
                                    assert!(source.is_multiple_of(8) && physical.is_multiple_of(8));
                                    assert!(
                                        source + 7 < shape.elements() && physical * 2 + 16 <= 8192
                                    );
                                    let local = (round * 256 + y * 32 + x) * 8;
                                    assert_eq!(
                                        physical, local,
                                        "column-major coordinates must preserve physical row-key order"
                                    );
                                    for offset in 0..8 {
                                        let expected = match layout {
                                            ReadLayout::HeadMajor => {
                                                (by * 512 + r) * 64 + c + offset
                                            }
                                            ReadLayout::SequenceMajor => {
                                                ((by / shape.heads * 512 + r) * shape.heads
                                                    + by % shape.heads)
                                                    * 64
                                                    + c
                                                    + offset
                                            }
                                        };
                                        assert_eq!(source + offset, expected);
                                        if by == 0 && bx == 0 {
                                            coverage[physical + offset] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                assert!(coverage.iter().all(|&count| count == 1));
            }
        }
    }
    Ok(())
}

fn copy_count(block: &Block) -> usize {
    block
        .stmts
        .iter()
        .map(|stmt| match stmt {
            Stmt::AsyncCopy { .. } => 1,
            Stmt::ForLoop { body, .. } => copy_count(body),
            _ => 0,
        })
        .sum()
}
fn waits(block: &Block) -> usize {
    let mut result = 0;
    for (index, stmt) in block.stmts.iter().enumerate() {
        if matches!(stmt, Stmt::AsyncWait) {
            assert!(index > 0 && matches!(block.stmts[index - 1], Stmt::AsyncCommit));
            assert!(
                matches!(block.stmts.get(index + 1), Some(Stmt::Barrier)),
                "wait must immediately precede original publication barrier"
            );
            result += 1;
        }
        if let Stmt::ForLoop { body, .. } = stmt {
            result += waits(body);
        }
    }
    result
}

#[test]
fn online_async16_group_waits_and_alias_reader_retirement_are_explicit() -> Result<()> {
    let shape = Shape::new(8, 8)?;
    for layout in [ReadLayout::HeadMajor, ReadLayout::SequenceMajor] {
        let kernels = [
            forward::ir_with_transport(shape, layout, Transport::Async16)?,
            backward::d_query_ir_with_transport(shape, layout, layout, Transport::Async16)?,
            backward::d_key_value_ir_with_transport(shape, layout, layout, Transport::Async16)?,
        ];
        for (index, ir) in kernels.iter().enumerate() {
            ir.validate_layouts()?;
            assert_eq!(ir.shared_mem_bytes, 49152);
            assert!(ir.kernel_name.ends_with("_Async16"));
            for statement in &ir.body.stmts {
                if let Stmt::ViewSharedArena {
                    byte_offset,
                    dtype,
                    rows,
                    cols,
                    layout,
                    ..
                } = statement
                    && *dtype == DType::BF16
                    && layout.row_stride == 64
                {
                    assert!(byte_offset.is_multiple_of(16));
                    assert_eq!((*rows, *cols), (64, 64));
                    assert_eq!(layout.xor_mask, 0);
                    assert!(*byte_offset + 8192 <= 49152);
                }
            }
            let source = ptx(ir.clone())?;
            assert!(
                source.contains("cp.async")
                    && source.contains("cp.async.commit_group")
                    && source.contains("cp.async.wait_group 0")
            );
            assert!(
                source.contains(".shared .align 32"),
                "existing32B arena alignment satisfies the16B copy contract"
            );
            assert_eq!(copy_count(&ir.body), if index == 0 { 3 } else { 4 });
            assert_eq!(waits(&ir.body), if index == 0 { 3 } else { 1 });
            let outer = ir
                .body
                .stmts
                .iter()
                .find_map(|s| match s {
                    Stmt::ForLoop { loop_var, body, .. } if loop_var.starts_with("causal_") => {
                        Some(body)
                    }
                    _ => None,
                })
                .unwrap();
            assert!(
                matches!(outer.stmts.last(), Some(Stmt::Barrier)),
                "next outer iteration cannot overwrite old readers"
            );
            if index > 0 {
                assert!(
                    outer.stmts[..4]
                        .iter()
                        .all(|s| matches!(s,Stmt::ForLoop {body,..} if copy_count(body)==1))
                );
                assert!(
                    matches!(outer.stmts[4], Stmt::AsyncCommit)
                        && matches!(outer.stmts[5], Stmt::AsyncWait)
                        && matches!(outer.stmts[6], Stmt::Barrier)
                );
                let second = outer
                    .stmts
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| matches!(s, Stmt::MatMul { .. }))
                    .nth(1)
                    .unwrap()
                    .0;
                assert!(matches!(
                    outer.stmts[second + 1],
                    Stmt::ConvertLayout { .. }
                ));
                assert!(matches!(outer.stmts[second + 2], Stmt::Barrier));
                assert!(matches!(
                    outer.stmts[second + 3],
                    Stmt::ConvertLayout { .. }
                ));
                assert!(matches!(outer.stmts[second + 4], Stmt::Barrier));
            } else {
                let first = outer
                    .stmts
                    .iter()
                    .position(|s| matches!(s, Stmt::MatMul { .. }))
                    .unwrap();
                assert!(matches!(outer.stmts[first + 1], Stmt::ConvertLayout { .. }));
                assert!(
                    matches!(outer.stmts[first + 2], Stmt::Barrier),
                    "QK readers must retire before K-to-V alias overwrite"
                );
                let value=outer.stmts.iter().position(|s|matches!(s,Stmt::ForLoop {loop_var,..} if loop_var.starts_with("stage_value"))).unwrap();
                assert!(
                    outer.stmts[..value]
                        .iter()
                        .any(|s| matches!(s, Stmt::MatMul { .. }))
                );
                assert!(
                    outer.stmts[..value]
                        .iter()
                        .any(|s| matches!(s, Stmt::Barrier))
                );
            }
        }
    }
    Ok(())
}

fn descriptor(shape: Shape, index: usize) -> Descriptor {
    let source_shape = vec![shape.batch, 512, shape.heads * 64];
    let mut view =
        VirtualTensor::identity(NodeIndex::new(index), source_shape.clone(), DType::BF16)
            .reshape(vec![shape.batch, 512, shape.heads, 64])
            .unwrap()
            .permute(&[0, 2, 1, 3])
            .unwrap();
    if index == 1 {
        view = view.permute(&[0, 1, 3, 2]).unwrap();
    }
    Descriptor {
        pointer: 0x1000 + (index as u64) * 0x1000000,
        storage_len: shape.elements(),
        storage_dtype: DType::BF16,
        source_shape,
        view,
    }
}

#[test]
fn online_async16_alignment_admission_is_whole_pair_or_whole_scalar_fallback() -> Result<()> {
    let shape = Shape::new(8, 8)?;
    let inputs = std::array::from_fn(|index| descriptor(shape, index));
    assert_eq!(
        UncompiledPair::with_transport(shape, inputs.clone(), Transport::Async16)?.transport,
        Transport::Async16
    );
    assert_eq!(
        UncompiledPair::new(shape, inputs.clone())?.transport,
        Transport::Scalar
    );
    for index in 0..4 {
        let mut bad = inputs.clone();
        bad[index].pointer += 2;
        assert_eq!(
            UncompiledPair::with_transport(shape, bad, Transport::Async16)?.transport,
            Transport::Scalar
        );
        let mut bad = inputs.clone();
        bad[index].storage_len -= 8;
        ensure!(
            UncompiledPair::with_transport(shape, bad, Transport::Async16).is_err(),
            "partial views must reject rather than transport individually"
        );
    }
    Ok(())
}
