//! Exact row IR, admitted only by selected-original-plan edge evidence.
//! Backward's mean-term contraction policy comes from that plan; the graph's
//! Add shape alone is insufficient to select a numerical contract.
use super::*;
use crate::tile::{
    Block, Dim, Expr, KernelParam, MemorySpace, ScalarBinaryOp, ScalarBuilder, ScalarExpr,
    ScalarMemory, ScalarPredicate, SharedLayout, Stmt, TileIR, TileLayout, TileVar,
};
const THREADS: usize = 128;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TailBoundary {
    Unresolved,
    /// Original mean-term Mul stored F32 before a separate Add kernel.
    SeparateMeanTerm,
    /// Original pointwise region could contract the mean-term Mul and Add.
    OriginalPointwiseOpportunity,
}
fn value(b: &mut ScalarBuilder, op: BinaryOp, a: TileVar, c: TileVar) -> TileVar {
    b.value(ScalarExpr::binary(ScalarBinaryOp::Arithmetic(op), a, c))
}
fn rounded(b: &mut ScalarBuilder, op: BinaryOp, a: TileVar, c: TileVar) -> TileVar {
    let op = match op {
        BinaryOp::Add => ScalarBinaryOp::AddRn,
        BinaryOp::Sub => ScalarBinaryOp::SubRn,
        BinaryOp::Mul => ScalarBinaryOp::MulRn,
        BinaryOp::Div => ScalarBinaryOp::Arithmetic(BinaryOp::Div),
    };
    b.value(ScalarExpr::binary(op, a, c))
}
fn shared(b: &mut ScalarBuilder) -> TileVar {
    let var = b.var();
    b.stmts.push(Stmt::AllocTile {
        var,
        space: MemorySpace::Shared,
        layout: TileLayout::Shared(SharedLayout::row_major(THREADS)),
        dtype: DType::F32,
        rows: 1,
        cols: THREADS,
    });
    var
}
fn checkpoint(b: &mut ScalarBuilder, scratch: TileVar, v: TileVar) -> TileVar {
    let lane = Expr::ThreadIdx(Dim::X);
    b.store(ScalarMemory::Shared(scratch), lane.clone(), v);
    b.load(ScalarMemory::Shared(scratch), lane)
}
fn column(k: usize) -> Expr {
    Expr::ThreadIdx(Dim::X) + Expr::Const((k * THREADS) as i64)
}
fn element(k: usize, width: usize) -> Expr {
    Expr::BlockIdx(Dim::X) * width + column(k)
}
fn load(b: &mut ScalarBuilder, name: &str, index: Expr) -> TileVar {
    b.load(ScalarMemory::Global(name.into()), index)
}
fn store(b: &mut ScalarBuilder, name: &str, index: Expr, v: TileVar) {
    b.store(ScalarMemory::Global(name.into()), index, v);
}
fn lane_sum(b: &mut ScalarBuilder, values: &[TileVar]) -> TileVar {
    let mut sum = b.value(ScalarExpr::Constant(0.));
    for &v in values {
        sum = rounded(b, BinaryOp::Add, sum, v);
    }
    sum
}
fn reduce(b: &mut ScalarBuilder, scratch: TileVar, partial: TileVar) -> Result<TileVar> {
    let lane = Expr::ThreadIdx(Dim::X);
    b.store(ScalarMemory::Shared(scratch), lane.clone(), partial);
    b.stmts.push(Stmt::Barrier);
    for stride in [64, 32, 16, 8, 4, 2, 1] {
        let body = b.block(|b| {
            let peer = b.load(
                ScalarMemory::Shared(scratch),
                lane.clone() + Expr::Const(stride),
            );
            let sum = value(b, BinaryOp::Add, partial, peer);
            b.set(partial, sum);
            b.store(ScalarMemory::Shared(scratch), lane.clone(), partial);
            Ok(())
        })?;
        b.stmts.push(Stmt::If {
            condition: ScalarPredicate::IndexLt {
                lhs: lane.clone(),
                rhs: Expr::Const(stride),
            },
            body,
        });
        b.stmts.push(Stmt::Barrier);
    }
    // Each reduction has its OWN scratch, so a later reduction/private
    // checkpoint cannot overwrite lane0 before all warps complete this load.
    Ok(b.load(ScalarMemory::Shared(scratch), Expr::Const(0)))
}
fn finish(
    b: ScalarBuilder,
    name: &str,
    input: &[&str],
    output: &[&str],
    incoming_dtype: Option<DType>,
) -> TileIR {
    let params = input
        .iter()
        .map(|s| KernelParam {
            name: (*s).into(),
            dtype: if *s == "incoming" {
                incoming_dtype.unwrap_or(DType::F32)
            } else {
                DType::F32
            },
            is_input: true,
        })
        .chain(output.iter().map(|s| KernelParam {
            name: (*s).into(),
            dtype: DType::F32,
            is_input: false,
        }))
        .collect();
    TileIR {
        kernel_name: name.into(),
        params,
        body: Block { stmts: b.stmts },
        shared_mem_bytes: 3 * THREADS * 4,
    }
}
fn validate(pair: &LayerNormPair) -> Result<()> {
    ensure!(
        supported_geometry(&pair.shape),
        "unsupported LayerNorm row geometry"
    );
    ensure!(
        pair.shape
            .iter()
            .try_fold(1_usize, |a, &b| a.checked_mul(b))
            .is_some_and(|n| u32::try_from(n).is_ok()),
        "LayerNorm extent overflow"
    );
    Ok(())
}
pub(super) fn forward(pair: &LayerNormPair) -> Result<TileIR> {
    validate(pair)?;
    let feature_width = pair.shape[2];
    let elements = feature_width / THREADS;
    let mut b = ScalarBuilder::default();
    let first = shared(&mut b);
    let second = shared(&mut b);
    let local = shared(&mut b);
    // Exact grid is B*S CTAs of128 threads. No lane may return before a barrier;
    // W512/W1024 give each lane four/eight ascending valid coordinates, no padding.
    let x = (0..elements)
        .map(|k| load(&mut b, "z", element(k, feature_width)))
        .collect::<Vec<_>>();
    let total = lane_sum(&mut b, &x);
    let total = reduce(&mut b, first, total)?;
    let width = b.value(ScalarExpr::Constant(feature_width as f32));
    let mean = value(&mut b, BinaryOp::Div, total, width);
    let mean = checkpoint(&mut b, local, mean);
    let mut centered = Vec::new();
    let mut squares = Vec::new();
    for &x in &x {
        let c = rounded(&mut b, BinaryOp::Sub, x, mean);
        let c = checkpoint(&mut b, local, c);
        let square = rounded(&mut b, BinaryOp::Mul, c, c);
        // Original square kernel stored before the Mean kernel. Prevent a new
        // square+accumulate FMA; audit the generated sequence before activation.
        let square = checkpoint(&mut b, local, square);
        centered.push(c);
        squares.push(square);
    }
    let sq_total = lane_sum(&mut b, &squares);
    let sq_total = reduce(&mut b, second, sq_total)?;
    let variance = value(&mut b, BinaryOp::Div, sq_total, width);
    let variance = checkpoint(&mut b, local, variance);
    let epsilon = b.value(ScalarExpr::Constant(1e-5));
    let a = rounded(&mut b, BinaryOp::Add, variance, epsilon);
    let logarithm = b.value(ScalarExpr::unary(UnaryOp::Log, a));
    let half = b.value(ScalarExpr::Constant(-0.5));
    let scaled = value(&mut b, BinaryOp::Mul, logarithm, half);
    let r = b.value(ScalarExpr::unary(UnaryOp::Exp, scaled));
    for (k, c) in centered.into_iter().enumerate() {
        let n = value(&mut b, BinaryOp::Mul, c, r);
        let gamma = load(&mut b, "gamma", column(k));
        let beta = load(&mut b, "beta", column(k));
        let product = value(&mut b, BinaryOp::Mul, n, gamma);
        let y = value(&mut b, BinaryOp::Add, product, beta);
        store(&mut b, "normalized", element(k, feature_width), n);
        store(&mut b, "y", element(k, feature_width), y);
    }
    let body = b.block(|b| {
        let row = Expr::BlockIdx(Dim::X);
        store(b, "mean", row.clone(), mean);
        store(b, "a", row.clone(), a);
        store(b, "r", row, r);
        Ok(())
    })?;
    b.stmts.push(Stmt::If {
        condition: ScalarPredicate::IndexLt {
            lhs: Expr::ThreadIdx(Dim::X),
            rhs: Expr::Const(1),
        },
        body,
    });
    Ok(finish(
        b,
        "layer_norm_pair_forward",
        &["z", "gamma", "beta"],
        &["y", "normalized", "mean", "a", "r"],
        None,
    ))
}
pub(super) fn backward(
    pair: &LayerNormPair,
    tail: TailBoundary,
    incoming_dtype: DType,
) -> Result<TileIR> {
    validate(pair)?;
    ensure!(
        matches!(incoming_dtype, DType::F32 | DType::BF16),
        "LN incoming storage unsupported"
    );
    ensure!(
        tail != TailBoundary::Unresolved,
        "LayerNorm tail contraction audit required"
    );
    let feature_width = pair.shape[2];
    let elements = feature_width / THREADS;
    let mut b = ScalarBuilder::default();
    let first = shared(&mut b);
    let second = shared(&mut b);
    let local = shared(&mut b);
    let row = Expr::BlockIdx(Dim::X);
    let mean = load(&mut b, "mean", row.clone());
    let a = load(&mut b, "a", row.clone());
    let r = load(&mut b, "r", row);
    let half = b.value(ScalarExpr::Constant(-0.5));
    let scale = b.value(ScalarExpr::Constant(1. / feature_width as f32));
    let mut centered = Vec::new();
    let mut direct = Vec::new();
    let mut quotients = Vec::new();
    for k in 0..elements {
        let x = load(&mut b, "z", element(k, feature_width));
        let c = rounded(&mut b, BinaryOp::Sub, x, mean);
        let c = checkpoint(&mut b, local, c);
        let incoming = load(&mut b, "incoming", element(k, feature_width));
        let gamma = load(&mut b, "gamma", column(k));
        let g = value(&mut b, BinaryOp::Mul, incoming, gamma);
        let u = rounded(&mut b, BinaryOp::Mul, g, r);
        let u = checkpoint(&mut b, local, u);
        let e = value(&mut b, BinaryOp::Mul, g, c);
        let exp_vjp = value(&mut b, BinaryOp::Mul, e, r);
        let scaled = value(&mut b, BinaryOp::Mul, exp_vjp, half);
        let q = value(&mut b, BinaryOp::Div, scaled, a);
        // Original quotient was a standalone stored Div result before SumW.
        let q = checkpoint(&mut b, local, q);
        centered.push(c);
        direct.push(u);
        quotients.push(q);
    }
    let q_sum = lane_sum(&mut b, &quotients);
    let dv = reduce(&mut b, first, q_sum)?;
    let ds = rounded(&mut b, BinaryOp::Mul, dv, scale);
    let mut center_gradient = Vec::new();
    let mut negatives = Vec::new();
    for (&u, &c) in direct.iter().zip(&centered) {
        let p1 = rounded(&mut b, BinaryOp::Mul, ds, c);
        let p1 = checkpoint(&mut b, local, p1);
        let cc1 = rounded(&mut b, BinaryOp::Add, u, p1);
        let p2 = rounded(&mut b, BinaryOp::Mul, ds, c);
        let p2 = checkpoint(&mut b, local, p2);
        let cc2 = rounded(&mut b, BinaryOp::Add, cc1, p2);
        let negative = b.value(ScalarExpr::unary(UnaryOp::Neg, cc2));
        center_gradient.push(cc2);
        negatives.push(negative);
    }
    let neg_sum = lane_sum(&mut b, &negatives);
    let dm = reduce(&mut b, second, neg_sum)?;
    let separate = if tail == TailBoundary::SeparateMeanTerm {
        let term = rounded(&mut b, BinaryOp::Mul, dm, scale);
        Some(checkpoint(&mut b, local, term))
    } else {
        None
    };
    for (k, cc) in center_gradient.into_iter().enumerate() {
        let dz = if let Some(term) = separate {
            rounded(&mut b, BinaryOp::Add, cc, term)
        } else {
            // Keep an independent single-use Mul/Add expression for each
            // coordinate. CSE/contraction still needs actual baseline/candidate
            // SASS and finite-bit qualification; this enum is not that proof.
            b.value(ScalarExpr::binary(
                ScalarBinaryOp::Arithmetic(BinaryOp::Add),
                cc,
                ScalarExpr::binary(ScalarBinaryOp::Arithmetic(BinaryOp::Mul), dm, scale),
            ))
        };
        store(&mut b, "dz", element(k, feature_width), dz);
    }
    Ok(finish(
        b,
        "layer_norm_pair_backward",
        &["z", "gamma", "incoming", "mean", "a", "r"],
        &["dz"],
        Some(incoming_dtype),
    ))
}
