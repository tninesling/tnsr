use super::*;
use std::collections::HashMap;
#[path = "gpu.rs"]
mod gpu;
fn spans(len: usize) -> [Span; 8] {
    let sizes = [
        len * 4,
        len * 4,
        len * 4,
        len * 2,
        len * 4,
        len * 4,
        len * 4,
        len / 64,
    ];
    std::array::from_fn(|i| Span {
        address: (i + 1) * 0x10000000,
        bytes: sizes[i],
    })
}
#[test]
fn vector_adam_full_coverage_alignment_alias_and_whole_fallback() -> Result<()> {
    for len in [1024usize, 65536, 66560, 2818048, 3145728] {
        assert!(admitted(len, &spans(len), true));
        assert!(!admitted(len, &spans(len), false));
        let mut seen = vec![0u8; len];
        for block in 0..len / 1024 {
            for tid in 0..256 {
                for j in 0..4 {
                    let element = block * 1024 + 4 * tid + j;
                    seen[element] += 1;
                    assert_eq!(element / 256, block * 4 + tid / 64);
                }
            }
        }
        assert!(seen.iter().all(|v| *v == 1));
        for i in 0..8 {
            let mut p = spans(len);
            p[i].address += 1;
            assert!(!admitted(len, &p, true));
            p = spans(len);
            p[i].bytes -= 1;
            assert!(!admitted(len, &p, true));
        }
        for i in 4..8 {
            for j in 0..8 {
                if i != j {
                    let mut p = spans(len);
                    p[i].address = p[j].address;
                    assert!(!admitted(len, &p, true));
                }
            }
        }
        ir(len)?.validate_layouts()?;
    }
    for len in [0usize, 1, 255, 256, 1023, 1025, 65539] {
        assert!(!admitted(len, &spans(len), true));
        assert!(ir(len).is_err());
    }
    let mut p = spans(1024);
    p[0].address = usize::MAX - 15;
    assert!(!admitted(1024, &p, true));
    Ok(())
}
fn subtract(a: f32, b: f32) -> f32 {
    a - b
}
fn health(w: f32, m: f32, v: f32) -> f32 {
    (subtract(w, w) + subtract(m, m)) + subtract(v, v)
}
fn same(a: f32, b: f32) {
    assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan());
}
#[test]
fn vector_adam_health_zero_or_nan_classification_and_group_ownership() {
    for bits in 0..=u16::MAX {
        let x = f32::from(half::bf16::from_bits(bits));
        for (w, m, v) in [(x, 0., -0.), (1., x, 0.), (-0., 1., x), (x, x, x)] {
            let h = health(w, m, v);
            if w.is_finite() && m.is_finite() && v.is_finite() {
                assert_eq!(h.to_bits(), 0)
            } else {
                assert!(h.is_nan());
            }
        }
    }
    for bad in 0..1024 {
        let mut elements = [0f32; 1024];
        elements[bad] = f32::NAN;
        let mut warps = [0f32; 8];
        for tid in 0..256 {
            let e = tid * 4;
            let h = (elements[e] + elements[e + 1]) + (elements[e + 2] + elements[e + 3]);
            warps[tid / 32] += h;
        }
        for group in 0..4 {
            let result = warps[group * 2] + warps[group * 2 + 1];
            same(result, if group == bad / 256 { f32::NAN } else { 0. });
        }
    }
    for warp in 0..8 {
        let flags = (0..32)
            .map(|lane| (warp * 32 + lane) & 63 < 32)
            .collect::<Vec<_>>();
        assert!(flags.iter().all(|v| *v == flags[0]));
    }
}
fn symbolic(stmts: &[Stmt], seeds: HashMap<TileVar, String>) -> Result<HashMap<TileVar, String>> {
    let mut values = seeds;
    fn expr(x: &ScalarExpr, v: &HashMap<TileVar, String>) -> String {
        match x {
            ScalarExpr::Var(x) => v[x].clone(),
            ScalarExpr::Constant(x) => format!("c{:x}", x.to_bits()),
            ScalarExpr::Unary { op, value } => format!("{op:?}({})", expr(value, v)),
            ScalarExpr::Binary { op, lhs, rhs } => {
                format!("{op:?}({},{})", expr(lhs, v), expr(rhs, v))
            }
            _ => panic!("unsupported symbolic scalar"),
        }
    }
    for s in stmts {
        match s {
            Stmt::Zero { tile } => {
                values.insert(*tile, "+0".into());
            }
            Stmt::LoadGlobalPredicated {
                dest,
                src_param,
                element_index,
                ..
            } => {
                values.insert(
                    *dest,
                    if src_param == "settings" {
                        format!("settings[{element_index:?}]")
                    } else {
                        src_param.clone()
                    },
                );
            }
            Stmt::LoadGlobalF32x4 { dests, param, .. } => {
                for v in dests {
                    values.insert(*v, param.clone());
                }
            }
            Stmt::SetScalar { dest, value } => {
                let x = expr(value, &values);
                values.insert(*dest, x);
            }
            Stmt::Add { dest, a, b }
            | Stmt::Sub { dest, a, b }
            | Stmt::Mul { dest, a, b }
            | Stmt::Div { dest, a, b } => {
                let op = match s {
                    Stmt::Add { .. } => "add",
                    Stmt::Sub { .. } => "sub",
                    Stmt::Mul { .. } => "mul",
                    _ => "div",
                };
                let x = format!("{op}({},{})", values[a], values[b]);
                values.insert(*dest, x);
            }
            Stmt::Sqrt { dest, src } => {
                let x = format!("sqrt({})", values[src]);
                values.insert(*dest, x);
            }
            Stmt::AllocTile { .. }
            | Stmt::StoreGlobalPredicated { .. }
            | Stmt::StoreGlobalF32x4 { .. } => {}
            _ => bail!("unexpected symbolic body {s:?}"),
        }
    }
    Ok(values)
}
#[test]
fn vector_adam_preserves_all_four_scalar_dags_settings_and_typed_vector_ptx() -> Result<()> {
    let old = kernels::adam_bf16_first(65536)?;
    let new = ir(65536)?;
    let body = old
        .body
        .stmts
        .iter()
        .find_map(|s| {
            if let Stmt::If {
                condition: ScalarPredicate::IndexLt { lhs, .. },
                body,
            } = s
            {
                if *lhs == linear() { Some(body) } else { None }
            } else {
                None
            }
        })
        .unwrap();
    let zero = old
        .body
        .stmts
        .iter()
        .find_map(|s| {
            if let Stmt::Zero { tile } = s {
                Some(*tile)
            } else {
                None
            }
        })
        .unwrap();
    let reference = symbolic(&body.stmts, HashMap::from([(zero, "+0".into())]))?;
    let end = new
        .body
        .stmts
        .iter()
        .position(|s| matches!(s, Stmt::StoreGlobalF32x4 { .. }))
        .unwrap()
        + 3;
    let actual = symbolic(&new.body.stmts[..end], HashMap::new())?;
    for s in &body.stmts {
        if let Stmt::StoreGlobalPredicated {
            src, dest_param, ..
        } = s
        {
            let values = new
                .body
                .stmts
                .iter()
                .find_map(|s| {
                    if let Stmt::StoreGlobalF32x4 { values, param, .. } = s {
                        if param == dest_param {
                            Some(values)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .unwrap();
            for value in values {
                assert_eq!(actual[value], reference[src], "{dest_param} scalar DAG");
            }
        }
    }
    assert_eq!(
        old.params
            .iter()
            .map(|p| (&p.name, p.dtype, p.is_input))
            .collect::<Vec<_>>(),
        new.params
            .iter()
            .map(|p| (&p.name, p.dtype, p.is_input))
            .collect::<Vec<_>>()
    );
    assert_eq!(new.shared_mem_bytes, 32);
    assert_eq!(
        new.body
            .stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Barrier))
            .count(),
        1
    );
    let mut graph = crate::tile::TileGraph::from_graph(&crate::graph::TensorGraph::<f32>::new());
    graph.region_kernels = vec![crate::tile::graph::TileRegionKernel {
        region_id: 0,
        ir: new,
    }];
    graph.optimize_indices()?;
    let ptx = crate::ptx::PtxGraph::from(graph).to_ptx();
    assert_eq!(ptx.matches("ld.global.v4.f32").count(), 3);
    assert_eq!(ptx.matches("st.global.v4.f32").count(), 3);
    assert_eq!(ptx.matches("ld.global.b16").count(), 4);
    assert!(!ptx.contains("rcp.approx"));
    Ok(())
}

#[test]
fn vector_adam_settings_are_typed_full_readonly_and_disjoint_from_all_writes() {
    let spans = spans(1024);
    let settings = Span {
        address: 0x90000000,
        bytes: 40,
    };
    assert!(admitted_with_settings(1024, &spans, settings, true));
    assert!(!admitted_with_settings(1024, &spans, settings, false));
    for bad in [
        Span {
            address: 0,
            ..settings
        },
        Span {
            address: settings.address + 1,
            ..settings
        },
        Span {
            bytes: 36,
            ..settings
        },
        Span {
            bytes: 44,
            ..settings
        },
        Span {
            address: usize::MAX - 3,
            ..settings
        },
    ] {
        assert!(!admitted_with_settings(1024, &spans, bad, true));
    }
    for output in &spans[4..] {
        for address in [
            output.address,
            output.address - 4,
            output.address + output.bytes - 4,
        ] {
            assert!(!admitted_with_settings(
                1024,
                &spans,
                Span {
                    address,
                    ..settings
                },
                true
            ));
        }
    }
}
#[test]
fn vector_adam_status_reset_discards_submission_without_inventing_commit() {
    let mut state = State::default();
    assert!(!state.enabled);
    state.enabled = true;
    state.reset_status();
    state.status.submitted_parameters = 2;
    state.status.submitted_elements = 2048;
    state.fallback("stale");
    assert_eq!(state.status.committed_parameters, 0);
    assert_eq!(state.status.committed_elements, 0);
    state.reset_status();
    assert!(state.status.enabled);
    assert_eq!(state.status.submitted_parameters, 0);
    assert_eq!(state.status.submitted_elements, 0);
    assert!(state.status.fallback_reasons.is_empty());
}
#[test]
fn vector_adam_rejects_nested_selected_arithmetic_without_panicking() {
    let expression = ScalarExpr::Select {
        condition: Box::new(ScalarPredicate::IndexLt {
            lhs: Expr::Const(0),
            rhs: Expr::Const(1),
        }),
        then_value: Box::new(ScalarExpr::Constant(1.)),
        else_value: Box::new(ScalarExpr::Constant(0.)),
    };
    assert!(remap_expr(&expression, 0).is_err());
}
