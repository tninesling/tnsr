//! Contiguous axis slices and concatenations lower to guarded scalar copies.
use super::{
    Block, DType, Dim, Expr, KernelParam, ScalarBuilder, ScalarMemory, ScalarPredicate, Stmt,
    TileIR,
};

fn index(n: usize) -> Expr {
    assert!(n <= i64::MAX as usize, "segment extent exceeds index range");
    Expr::Const(n as i64)
}
fn tid() -> Expr {
    Expr::Mul(
        Box::new(Expr::BlockIdx(Dim::X)),
        Box::new(Expr::BlockDim(Dim::X)),
    ) + Expr::ThreadIdx(Dim::X)
}
fn kernel(dtype: DType, inputs: usize, extent: usize, name: &str) -> (TileIR, ScalarBuilder) {
    let mut params: Vec<_> = (0..inputs)
        .map(|i| KernelParam {
            name: format!("input_{i}"),
            dtype,
            is_input: true,
        })
        .collect();
    params.push(KernelParam {
        name: "output".into(),
        dtype,
        is_input: false,
    });
    let mut builder = ScalarBuilder::default();
    builder.stmts.push(Stmt::BoundsCheck { extent });
    (
        TileIR {
            kernel_name: name.into(),
            shared_mem_bytes: 0,
            params,
            body: Block { stmts: Vec::new() },
        },
        builder,
    )
}
pub(super) fn narrow(
    dtype: DType,
    input: &[usize],
    output: &[usize],
    axis: usize,
    start: usize,
) -> TileIR {
    let extent = output.iter().product();
    let (mut ir, mut builder) = kernel(dtype, 1, extent, "narrow");
    if extent > 0 {
        let inner: usize = output[axis + 1..].iter().product();
        let width = output[axis] * inner;
        let source = Expr::FloorDiv(Box::new(tid()), width) * (input[axis] * inner)
            + index(start * inner)
            + Expr::Mod(Box::new(tid()), width);
        let value = builder.load(ScalarMemory::Global("input_0".into()), source);
        builder.store(ScalarMemory::Global("output".into()), tid(), value);
    }
    ir.body.stmts = builder.stmts;
    ir
}
pub(super) fn concat(dtype: DType, inputs: &[&[usize]], output: &[usize], axis: usize) -> TileIR {
    let extent = output.iter().product();
    let (mut ir, mut builder) = kernel(dtype, inputs.len(), extent, "concat");
    if extent > 0 {
        let inner: usize = output[axis + 1..].iter().product();
        let coordinate = Expr::Mod(
            Box::new(Expr::FloorDiv(Box::new(tid()), inner)),
            output[axis],
        );
        let outer = Expr::FloorDiv(Box::new(tid()), output[axis] * inner);
        let remainder = Expr::Mod(Box::new(tid()), inner);
        let mut prefix = 0;
        for (i, shape) in inputs.iter().enumerate() {
            // The unsigned subtraction wraps before this segment. Its comparison
            // then fails, so those threads never evaluate an out-of-range load.
            let delta = Expr::Sub(Box::new(coordinate.clone()), Box::new(index(prefix)));
            let source =
                outer.clone() * (shape[axis] * inner) + delta.clone() * inner + remainder.clone();
            let condition = ScalarPredicate::IndexLt {
                lhs: delta,
                rhs: index(shape[axis]),
            };
            let outer_stmts = std::mem::take(&mut builder.stmts);
            let value = builder.load(ScalarMemory::Global(format!("input_{i}")), source);
            builder.store(ScalarMemory::Global("output".into()), tid(), value);
            let body = Block {
                stmts: std::mem::replace(&mut builder.stmts, outer_stmts),
            };
            builder.stmts.push(Stmt::If { condition, body });
            prefix += shape[axis];
        }
    }
    ir.body.stmts = builder.stmts;
    ir
}
