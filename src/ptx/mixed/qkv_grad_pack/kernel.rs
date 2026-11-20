use super::*;
use crate::tile::{
    Block, Dim, Expr, KernelParam, MemorySpace, RegionInput, RegionValue, ScalarBinaryOp,
    ScalarBuilder, ScalarExpr, ScalarMemory, ScalarPredicate, SharedLayout, Stmt, TileIR,
    TileLayout, TileVar,
};
pub(super) const THREADS: usize = 256;
fn round(b: &mut ScalarBuilder, scratch: TileVar, slot: Expr, value: TileVar) -> TileVar {
    b.store(ScalarMemory::Shared(scratch), slot.clone(), value);
    b.load(ScalarMemory::Shared(scratch), slot)
}
fn join(
    b: &mut ScalarBuilder,
    tree: &Join,
    values: &[TileVar],
    scratch: TileVar,
    slot: &Expr,
    root: bool,
) -> TileVar {
    match tree {
        Join::Input(i) => values[*i],
        Join::Add(a, c) => {
            let a = join(b, a, values, scratch, slot, false);
            let c = join(b, c, values, scratch, slot, false);
            let sum = b.value(ScalarExpr::binary(
                ScalarBinaryOp::Arithmetic(BinaryOp::Add),
                a,
                c,
            ));
            if root {
                sum
            } else {
                round(b, scratch, slot.clone(), sum)
            }
        }
    }
}
pub(super) fn ir(region: &QkvGradPack, values: &[Value]) -> Result<TileIR> {
    ensure!(
        values.len() == 3 && region.inputs.len() == 3,
        "QKV pack input count mismatch"
    );
    let len = region
        .shape
        .iter()
        .try_fold(1_usize, |a, n| a.checked_mul(*n))
        .context("QKV pack extent overflow")?;
    let width = region.shape[2] / 3;
    let mut b = ScalarBuilder::default();
    let scratch = b.var();
    // Each thread owns one 32-bit-aligned BF16 slot. No other thread accesses
    // it, so the typed store/load round-trip requires no block barrier.
    b.stmts.push(Stmt::AllocTile {
        var: scratch,
        space: MemorySpace::Shared,
        layout: TileLayout::Shared(SharedLayout::row_major(THREADS * 2)),
        dtype: DType::BF16,
        rows: 1,
        cols: THREADS * 2,
    });
    b.stmts.push(Stmt::BoundsCheck { extent: len });
    let thread = Expr::ThreadIdx(Dim::X);
    let linear = Expr::Mul(
        Box::new(Expr::BlockIdx(Dim::X)),
        Box::new(Expr::BlockDim(Dim::X)),
    ) + thread.clone();
    let column = Expr::Mod(Box::new(linear.clone()), region.shape[2]);
    let row = Expr::FloorDiv(Box::new(linear.clone()), region.shape[2]);
    let logical = row * width + Expr::Mod(Box::new(column.clone()), width);
    let slot = thread * 2usize;
    let registers: Vec<_> = (0..3).map(|_| b.value(ScalarExpr::Constant(0.0))).collect();
    for (index, (input, value)) in region.inputs.iter().zip(values).enumerate() {
        let descriptor = RegionInput {
            node: input.source,
            value: RegionValue(index),
            tensor: value.view.clone(),
            source_shape: value.source_shape.clone(),
        };
        let address =
            crate::tile::region::input_element_index(&descriptor, &value.view.shape, &logical)?;
        let body = b.block(|b| {
            let loaded = b.load(ScalarMemory::Global(format!("input_{index}")), address);
            // Reproduce the original pad's CastBF16 before either Add, even
            // for raw BF16 signaling NaNs. Other branches stay exact +0.
            let padded = round(b, scratch, slot.clone(), loaded);
            b.set(registers[index], padded);
            Ok(())
        })?;
        let body = Block {
            stmts: vec![Stmt::If {
                condition: ScalarPredicate::IndexLt {
                    lhs: column.clone(),
                    rhs: Expr::Const(i64::try_from((input.window + 1) * width)?),
                },
                body,
            }],
        };
        b.stmts.push(Stmt::If {
            condition: ScalarPredicate::IndexLt {
                lhs: Expr::Const(i64::try_from(input.window * width)?),
                rhs: column.clone() + Expr::Const(1),
            },
            body,
        });
    }
    let result = join(&mut b, &region.tree, &registers, scratch, &slot, true);
    b.store(ScalarMemory::Global("output".into()), linear, result);
    let mut params: Vec<_> = (0..3)
        .map(|i| KernelParam {
            name: format!("input_{i}"),
            dtype: DType::BF16,
            is_input: true,
        })
        .collect();
    params.push(KernelParam {
        name: "output".into(),
        dtype: DType::BF16,
        is_input: false,
    });
    Ok(TileIR {
        kernel_name: "qkv_grad_pack".into(),
        params,
        body: Block { stmts: b.stmts },
        shared_mem_bytes: THREADS * 4,
    })
}
