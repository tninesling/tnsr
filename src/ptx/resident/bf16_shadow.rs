//! Optional sparse resident BF16 master shadows with qualified paired stores.
//!
//! Future integration contract (not implemented by this leaf): initialize
//! current/staged BF16 shadows only for graph-declared eligible whole-matrix
//! parameters (the current decoder census finds 21, not every registered master).
//! Pair publication with existing master health/revision checks. A finite F32
//! master may round to BF16 infinity; the BF16 shadow must not add rejection.
//!
//! Capture compatibility includes BOTH physical typed addresses and layouts.
//! Storage, revision and typed Arc identity are freshly validated outside the
//! key; active captures lease both allocations, inactive captures retain no
//! strong allocation leases. A leaf fallback alone cannot replace a removed
//! cast by binding the F32 master. Preserve complete ordinary-cast fallthrough.
//!
//! This module only qualifies arithmetic-preserving leaf code. It does not
//! implement sparse owner selection, capture lifetimes, publication or reuse.
//! Payload budgets exclude padding, allocator granularity, modules, captures
//! and extra retained versions. Scalar BF16 stores (four/thread in vector Adam)
//! are a correctness starting point, not a coalescing/performance claim.
use super::{kernels, vector_adam};
use crate::tile::{
    Block, DType, Dim, Expr, Stmt, TileIR, TileIRBuilder,
    ir::{KernelParam, MatrixLayout},
};
use anyhow::{Context, Result, bail, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum AdamKind {
    F32,
    Bf16First,
    Vector4,
}
impl AdamKind {
    pub(super) fn original(self, len: usize) -> Result<TileIR> {
        match self {
            Self::F32 => kernels::adam(len),
            Self::Bf16First => kernels::adam_bf16_first(len),
            Self::Vector4 => vector_adam::ir(len),
        }
    }
}

fn linear() -> Expr {
    Expr::Mul(
        Box::new(Expr::BlockIdx(Dim::X)),
        Box::new(Expr::BlockDim(Dim::X)),
    ) + Expr::ThreadIdx(Dim::X)
}

/// Qualification-only strict transform. Debug structural equality is deliberate:
/// TileIR has no PartialEq, and admitting arbitrary caller IR would widen this
/// proof. Only the exact current canonical generator is accepted, not a pattern
/// that happens to contain a store with the right name.
pub(super) fn paired_existing_ir(mut ir: TileIR, len: usize, kind: AdamKind) -> Result<TileIR> {
    ensure!(
        len > 0 && i64::try_from(len).is_ok(),
        "paired shadow requires a representable nonempty extent"
    );
    let original = kind.original(len)?;
    ensure!(
        format!("{ir:?}") == format!("{original:?}"),
        "unsupported noncanonical Adam IR"
    );
    ensure!(
        ir.params.len() == 9 && !ir.params.iter().any(|p| p.name == "next_shadow"),
        "unsupported shadow ABI"
    );
    ir.params.push(KernelParam {
        name: "next_shadow".into(),
        dtype: DType::BF16,
        is_input: false,
    });
    let mut count = 0;
    fn patch(block: &mut Block, len: usize, vector: bool, count: &mut usize) -> Result<()> {
        let mut out = Vec::new();
        for mut stmt in std::mem::take(&mut block.stmts) {
            let mut additions = Vec::new();
            match &mut stmt {
                Stmt::If { body, .. } => patch(body, len, vector, count)?,
                Stmt::StoreGlobalPredicated {
                    dest_param,
                    src,
                    element_index,
                    row,
                    col,
                    layout,
                } if dest_param == "next_weight" => {
                    ensure!(
                        !vector
                            && *element_index == linear()
                            && *row == Expr::Const(0)
                            && *col == linear()
                            && layout.rows == 1
                            && layout.cols == len
                            && layout.row_stride == len,
                        "unsupported scalar next-weight store"
                    );
                    additions.push(Stmt::StoreGlobalPredicated {
                        dest_param: "next_shadow".into(),
                        src: *src,
                        element_index: element_index.clone(),
                        row: row.clone(),
                        col: col.clone(),
                        layout: *layout,
                    });
                    *count += 1;
                }
                Stmt::StoreGlobalF32x4 {
                    values,
                    param,
                    group_index,
                    extent,
                } if param == "next_weight" => {
                    ensure!(
                        vector && *group_index == linear() && *extent == len,
                        "unsupported vector next-weight store"
                    );
                    for (j, src) in values.iter().enumerate() {
                        let index = linear() * 4usize + Expr::Const(i64::try_from(j)?);
                        additions.push(Stmt::StoreGlobalPredicated {
                            dest_param: "next_shadow".into(),
                            src: *src,
                            element_index: index.clone(),
                            row: Expr::Const(0),
                            col: index,
                            layout: MatrixLayout {
                                rows: 1,
                                cols: len,
                                row_stride: len,
                            },
                        });
                        *count += 1;
                    }
                }
                Stmt::AllocTile { .. }
                | Stmt::Zero { .. }
                | Stmt::SetScalar { .. }
                | Stmt::LoadGlobalPredicated { .. }
                | Stmt::LoadGlobalF32x4 { .. }
                | Stmt::StoreGlobalPredicated { .. }
                | Stmt::StoreGlobalF32x4 { .. }
                | Stmt::Add { .. }
                | Stmt::Sub { .. }
                | Stmt::Mul { .. }
                | Stmt::Div { .. }
                | Stmt::Sqrt { .. }
                | Stmt::WarpReduce { .. }
                | Stmt::LoadScalar { .. }
                | Stmt::StoreScalar { .. }
                | Stmt::Barrier => {}
                _ => bail!("unsupported Adam statement in paired shadow transform"),
            }
            out.push(stmt);
            out.extend(additions);
        }
        block.stmts = out;
        Ok(())
    }
    let vector = matches!(kind, AdamKind::Vector4);
    patch(&mut ir.body, len, vector, &mut count)?;
    ensure!(
        count == if vector { 4 } else { 1 },
        "paired shadow output multiplicity changed"
    );
    ir.kernel_name.push_str("_paired_shadow");
    ir.validate_layouts()?;
    Ok(ir)
}
pub(super) fn paired_ir(len: usize, kind: AdamKind) -> Result<TileIR> {
    paired_existing_ir(kind.original(len)?, len, kind)
}

/// Initial F32-to-BF16 conversion. Its test oracle is the ordinary materialized
/// Cast lowerer, built independently rather than using this helper twice.
pub(super) fn initialize_ir(len: usize) -> Result<TileIR> {
    ensure!(
        len > 0 && i64::try_from(len).is_ok(),
        "shadow initialization requires representable nonempty extent"
    );
    let mut b = TileIRBuilder::new();
    b.start_kernel("resident_test_shadow_initialize");
    b.add_param("master", DType::F32, true);
    b.add_param("shadow", DType::BF16, false);
    let value = b.alloc_register(DType::F32, 1, 1);
    let layout = MatrixLayout {
        rows: 1,
        cols: len,
        row_stride: len,
    };
    b.load_global_predicated(value, "master", linear(), Expr::Const(0), linear(), layout);
    b.store_global_indexed(
        "shadow",
        value,
        linear(),
        (Expr::Const(0), linear()),
        layout,
    );
    let ir = b.finish();
    ir.validate_layouts()?;
    Ok(ir)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PayloadBudget {
    pub f32_master_bytes: usize,
    pub two_bf16_versions_bytes: usize,
    pub replaced_cast_payload_bytes: usize,
    pub net_additional_payload_bytes: usize,
}
pub(super) fn payload_budget(
    parameters: usize,
    replaced_cast_elements: usize,
) -> Result<PayloadBudget> {
    ensure!(
        replaced_cast_elements <= parameters,
        "shadow cast payload exceeds registered parameters"
    );
    let f32_master_bytes = parameters
        .checked_mul(4)
        .context("shadow master byte count overflow")?;
    let replaced_cast_payload_bytes = replaced_cast_elements
        .checked_mul(2)
        .context("shadow cast byte count overflow")?;
    Ok(PayloadBudget {
        f32_master_bytes,
        two_bf16_versions_bytes: f32_master_bytes,
        replaced_cast_payload_bytes,
        net_additional_payload_bytes: f32_master_bytes - replaced_cast_payload_bytes,
    })
}
mod owner;
pub use owner::Bf16MasterShadowStatus;
pub(super) use owner::{State, operand_range};
#[cfg(test)]
mod tests;
