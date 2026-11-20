use super::*;
#[path = "gpu.rs"]
mod gpu;
fn strip(ir: &mut TileIR) {
    ir.params.retain(|p| p.name != "next_shadow");
    ir.kernel_name = ir
        .kernel_name
        .strip_suffix("_paired_shadow")
        .unwrap()
        .to_owned();
    fn walk(b: &mut Block) {
        b.stmts.retain(
            |s| !matches!(s,Stmt::StoreGlobalPredicated{dest_param,..}if dest_param=="next_shadow"),
        );
        for s in &mut b.stmts {
            if let Stmt::If { body, .. } = s {
                walk(body)
            }
        }
    }
    walk(&mut ir.body);
}
#[test]
fn shadow_pair_only_adds_root_bf16_output_and_exact_final_register_stores() -> Result<()> {
    for len in [1usize, 1023, 1024, 65536, 66560, 2818048] {
        for kind in [AdamKind::F32, AdamKind::Bf16First, AdamKind::Vector4] {
            if matches!(kind, AdamKind::Vector4) && !len.is_multiple_of(1024) {
                assert!(paired_ir(len, kind).is_err());
                continue;
            }
            let old = kind.original(len)?;
            let mut new = paired_ir(len, kind)?;
            fn inspect(block: &Block, count: &mut usize) {
                for (i, s) in block.stmts.iter().enumerate() {
                    match s {
                        Stmt::If { body, .. } => inspect(body, count),
                        Stmt::StoreGlobalPredicated {
                            dest_param,
                            src,
                            element_index,
                            row,
                            col,
                            layout,
                        } if dest_param == "next_weight" => {
                            let Stmt::StoreGlobalPredicated {
                                dest_param: shadow,
                                src: next,
                                element_index: si,
                                row: sr,
                                col: sc,
                                layout: sl,
                            } = &block.stmts[i + 1]
                            else {
                                panic!("paired scalar store absent")
                            };
                            assert_eq!(shadow, "next_shadow");
                            assert_eq!(src, next);
                            assert_eq!(element_index, si);
                            assert_eq!(row, sr);
                            assert_eq!(col, sc);
                            assert_eq!(
                                (layout.rows, layout.cols, layout.row_stride),
                                (sl.rows, sl.cols, sl.row_stride)
                            );
                            *count += 1;
                        }
                        Stmt::StoreGlobalF32x4 {
                            values,
                            param,
                            group_index,
                            extent,
                        } if param == "next_weight" => {
                            for (j, value) in values.iter().enumerate() {
                                let Stmt::StoreGlobalPredicated {
                                    dest_param,
                                    src,
                                    element_index,
                                    row,
                                    col,
                                    layout,
                                } = &block.stmts[i + 1 + j]
                                else {
                                    panic!("paired vector lane store absent")
                                };
                                assert_eq!(dest_param, "next_shadow");
                                assert_eq!(src, value);
                                assert_eq!(
                                    *element_index,
                                    group_index.clone() * 4usize + Expr::Const(j as i64)
                                );
                                assert_eq!(row, &Expr::Const(0));
                                assert_eq!(col, element_index);
                                assert_eq!(layout.cols, *extent);
                                *count += 1;
                            }
                        }
                        _ => {}
                    }
                }
            }
            let mut stores = 0;
            inspect(&new.body, &mut stores);
            assert_eq!(
                stores,
                if matches!(kind, AdamKind::Vector4) {
                    4
                } else {
                    1
                }
            );
            let param = new.params.last().unwrap();
            assert_eq!(param.name, "next_shadow");
            assert_eq!(param.dtype, DType::BF16);
            assert!(!param.is_input);
            // Exact structural inverse proves no arithmetic, memory/health or
            // barrier instruction was moved, rewritten, added or removed.
            strip(&mut new);
            assert_eq!(format!("{new:?}"), format!("{old:?}"));
            let mut changed = old.clone();
            changed.body.stmts.push(Stmt::Barrier);
            assert!(paired_existing_ir(changed, len, kind).is_err());
            let mut changed = old.clone();
            changed.params[0].dtype = DType::BF16;
            assert!(paired_existing_ir(changed, len, kind).is_err());
            let mut changed = old;
            changed.params.push(KernelParam {
                name: "next_shadow".into(),
                dtype: DType::BF16,
                is_input: false,
            });
            assert!(paired_existing_ir(changed, len, kind).is_err());
        }
    }
    assert!(initialize_ir(0).is_err());
    assert!(paired_ir(0, AdamKind::F32).is_err());
    Ok(())
}
#[test]
fn shadow_budget_reports_both_versions_and_cast_offset() -> Result<()> {
    let b = payload_budget(211147776, 210763776)?;
    assert_eq!(b.f32_master_bytes, 844591104);
    assert_eq!(b.net_additional_payload_bytes, 423063552);
    assert_eq!(
        payload_budget(54_788_096, 54_788_096)?.net_additional_payload_bytes,
        109_576_192
    );
    assert_eq!(
        payload_budget(54_980_096, 54_788_096)?.net_additional_payload_bytes,
        110_344_192
    );
    assert!(payload_budget(1, 2).is_err());
    assert!(payload_budget(usize::MAX, 0).is_err());
    assert_eq!(payload_budget(0, 0)?.net_additional_payload_bytes, 0);
    Ok(())
}
