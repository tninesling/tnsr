use super::*;
use crate::tile::{Block, ScalarBinaryOp, ScalarExpr, Stmt};
fn strip(block: &mut Block) -> usize {
    let mut count = 0;
    block.stmts.retain(|s| {
        let added = matches!(
            s,
            Stmt::SetScalar {
                value: ScalarExpr::Binary {
                    op: ScalarBinaryOp::AddRn,
                    ..
                },
                ..
            }
        );
        count += usize::from(added);
        !added
    });
    for s in &mut block.stmts {
        if let Stmt::If { body, .. } = s {
            count += strip(body);
        }
    }
    count
}
#[test]
fn borrowed_bf16_ir_only_changes_typed_load_and_positive_zero_checkpoint() -> Result<()> {
    for len in [1, 3, 255, 256, 257, 1025, 65539] {
        let old = kernels::adam(len)?;
        let mut new = kernels::adam_bf16_first(len)?;
        assert_eq!(old.shared_mem_bytes, new.shared_mem_bytes);
        assert_eq!(
            new.params
                .iter()
                .find(|p| p.name == "gradient")
                .unwrap()
                .dtype,
            crate::tile::DType::BF16
        );
        assert_eq!(strip(&mut new.body), 1);
        new.kernel_name = old.kernel_name.clone();
        new.params
            .iter_mut()
            .find(|p| p.name == "gradient")
            .unwrap()
            .dtype = crate::tile::DType::F32;
        assert_eq!(format!("{old:?}"), format!("{new:?}"));
    }
    assert_eq!((0.0f32 + -0.0f32).to_bits(), 0.0f32.to_bits());
    Ok(())
}
#[test]
fn borrowed_bf16_single_attempt_guard_counts_disabled_and_failed_attempts() {
    let mut v = norm_partials::Validity::default();
    v.begin_step();
    assert!(!v.single_attempt());
    assert!(!v.begin_accumulation(false, false, false));
    v.finish_accumulation(false, true);
    assert!(v.single_attempt());
    assert!(!v.valid()); // Norm-row validity and source single-attempt differ.
    v.begin_accumulation(false, false, false);
    v.finish_accumulation(false, false);
    assert!(!v.single_attempt());
    v.begin_step();
    assert!(!v.single_attempt());
    assert!(v.begin_accumulation(true, true, true));
    v.finish_accumulation(true, true);
    assert!(v.single_attempt() && v.valid());
    v.begin_accumulation(true, true, false);
    v.finish_accumulation(false, false);
    assert!(!v.single_attempt() && !v.valid());
}
