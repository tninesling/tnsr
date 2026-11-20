use super::*;
use crate::{
    graph::NodeIndex,
    tile::{Block, Expr, ScalarExpr, ScalarMemory, ScalarPredicate, Stmt},
};
fn ranges(lengths: &[usize]) -> Vec<Range<usize>> {
    let mut n = 0;
    lengths
        .iter()
        .map(|&l| {
            let r = n..n + l;
            n += l;
            r
        })
        .collect()
}
fn view(i: usize, len: usize, dtype: DType) -> VirtualTensor {
    VirtualTensor::identity(NodeIndex::new(i), vec![len], dtype)
}
#[test]
fn small_accumulation_norm_exact_row_spans_coverage_and_38_to_5() {
    let mut lens = (0..38)
        .map(|i| {
            [
                1, 3, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 2048, 4096,
            ][i % 13]
        })
        .collect::<Vec<_>>();
    lens.extend([4097, 8193]);
    let r = ranges(&lens);
    let chunks = groups(&r);
    assert_eq!(chunks.len(), 5);
    assert_eq!(chunks.iter().map(Vec::len).sum::<usize>(), 38);
    let mut visits = vec![0_u8; r.last().unwrap().end];
    let mut full = std::collections::HashSet::new();
    for range in &r {
        let q = span(range).unwrap();
        assert_eq!(q, range.end.div_ceil(1024) - range.start / 1024);
        for block in 0..q {
            let row = range.start / 1024 + block;
            for part in 0..8 {
                for lane in 0..128 {
                    let local = (row as u64 * 1024)
                        .wrapping_sub(range.start as u64)
                        .wrapping_add(lane + part * 128);
                    if local < range.len() as u64 {
                        let p = range.start + local as usize;
                        assert!(range.contains(&p));
                        visits[p] += 1;
                    }
                }
            }
            if row * 1024 >= range.start && row * 1024 + 1024 <= range.end {
                assert!(full.insert(row));
            }
        }
    }
    assert!(visits.iter().all(|&n| n == 1));
    for i in 0..8_u64 {
        for y in 0..8_u64 {
            assert_eq!(y.wrapping_sub(i) < 1, y == i);
        }
    }
    assert!(span(&(0..0)).is_err());
}
fn remap_expr(x: &mut ScalarExpr, delta: usize) {
    match x {
        ScalarExpr::Var(v) => {
            if v.0 != 0 {
                v.0 -= delta;
            }
        }
        ScalarExpr::Unary { value, .. } => remap_expr(value, delta),
        ScalarExpr::Binary { lhs, rhs, .. } => {
            remap_expr(lhs, delta);
            remap_expr(rhs, delta);
        }
        ScalarExpr::Select {
            condition,
            then_value,
            else_value,
        } => {
            remap_pred(condition, delta);
            remap_expr(then_value, delta);
            remap_expr(else_value, delta);
        }
        _ => {}
    }
}
fn remap_pred(p: &mut ScalarPredicate, d: usize) {
    if let ScalarPredicate::Equal { lhs, rhs } = p {
        remap_expr(lhs, d);
        remap_expr(rhs, d);
    }
}
fn restored(b: &mut Block, delta: usize, name: &str, base: usize) {
    for s in &mut b.stmts {
        match s {
            Stmt::SetScalar { dest, value } => {
                dest.0 -= delta;
                remap_expr(value, delta);
            }
            Stmt::LoadScalar { dest, source, .. } => {
                dest.0 -= delta;
                if let ScalarMemory::Global(n) = source {
                    assert_eq!(n, name);
                    *n = "gradient".into();
                }
            }
            Stmt::StoreScalar {
                target,
                index,
                value,
            } => {
                if value.0 != 0 {
                    value.0 -= delta;
                }
                if let ScalarMemory::Global(n) = target
                    && n == "accumulated"
                {
                    let Expr::Add(a, b) = index else {
                        panic!("global output translation missing")
                    };
                    assert_eq!(**a, Expr::Const(base as i64));
                    *index = (**b).clone();
                }
            }
            Stmt::If { condition, body } => {
                remap_pred(condition, delta);
                restored(body, delta, name, base);
            }
            Stmt::Barrier => {}
            _ => panic!("unknown composed grammar"),
        }
    }
}
#[test]
fn small_accumulation_norm_every_member_body_is_exact_old_ir_after_rebasing() -> Result<()> {
    let r = ranges(&[3, 1025, 511, 4096, 1, 2048, 17, 512]);
    let padded = r.last().unwrap().end.div_ceil(1024) * 1024;
    let inputs = r
        .iter()
        .enumerate()
        .map(|(i, r)| {
            (
                view(
                    i,
                    r.len(),
                    if i % 2 == 0 { DType::BF16 } else { DType::F32 },
                ),
                vec![r.len()],
                r.clone(),
            )
        })
        .collect::<Vec<_>>();
    let composed = ir::batch_ir(&inputs, padded)?;
    assert_eq!(composed.shared_mem_bytes, 512);
    assert_eq!(composed.params.len(), 10);
    for (i, (v, shape, range)) in inputs.iter().enumerate() {
        let old = norm_partials::accumulate_ir(v, shape, range.clone())?;
        let Stmt::If { body, .. } = &composed.body.stmts[i + 1] else {
            panic!()
        };
        let Stmt::If { body, .. } = &body.stmts[0] else {
            panic!()
        };
        let mut member = body.clone();
        let Stmt::SetScalar { dest, .. } = member.stmts[0] else {
            panic!()
        };
        let delta = dest.0 - 1;
        restored(&mut member, delta, &format!("gradient_{i}"), range.start);
        assert_eq!(
            format!("{:?}", member.stmts),
            format!("{:?}", &old.body.stmts[1..])
        );
    }
    assert!(ir::batch_ir(&inputs[..1], padded).is_err());
    assert!(ir::batch_ir(&inputs, 1).is_err());
    Ok(())
}
#[test]
fn small_accumulation_norm_source_alias_and_overflow_guards() -> Result<()> {
    let r = ranges(&[3, 5]);
    let v = vec![
        (
            Binding {
                pointer: 256,
                len: 3,
                dtype: DType::BF16,
            },
            view(0, 3, DType::BF16),
            vec![3],
        ),
        (
            Binding {
                pointer: 512,
                len: 5,
                dtype: DType::F32,
            },
            view(1, 5, DType::F32),
            vec![5],
        ),
    ];
    preflight(&v, &r, 1024, 100000, 200000, 1)?;
    let mut bad = v.clone();
    bad[0].0.pointer = 100000;
    assert!(preflight(&bad, &r, 1024, 100000, 200000, 1).is_err());
    assert!(preflight(&v, &r, 1024, 100000, 100004, 1).is_err());
    assert!(bounds(u64::MAX - 3, 4, DType::F32).is_err());
    source_proof(&view(0, 6, DType::BF16), &[6], 6)?;
    let t = VirtualTensor::identity(NodeIndex::new(0), vec![2, 3], DType::BF16).permute(&[1, 0])?;
    assert!(source_proof(&t, &[2, 3], 6).is_err());
    let mut builder = crate::tile::ScalarBuilder::default();
    let mut vars = HashMap::new();
    assert!(
        ir::body(
            &Block {
                stmts: vec![Stmt::BoundsCheck { extent: 3 }]
            },
            &mut builder,
            &mut vars,
            "g",
            0
        )
        .is_err()
    );
    Ok(())
}
#[cfg(test)]
mod gpu;

#[test]
fn small_accumulation_norm_singleton_identity_is_a_whole_source() -> Result<()> {
    let singleton = view(0, 1, DType::F32);
    let no_op = singleton.clone().reshape(vec![1])?;
    eprintln!(
        "direct={:?}; reshaped={:?}",
        singleton.access.normalize_in_domain(&[1])?,
        no_op.access.normalize_in_domain(&[1])?
    );
    source_proof(&singleton, &[1], 1)?;
    source_proof(&view(0, 1, DType::BF16), &[1], 1)?;
    Ok(())
}
