//! Single CFG(test) experiment: same group IR, scoped pointer declarations only.
use super::*;
use crate::ptx::graph::scoped_parameters::{self, Scope};
fn policy(ir: &TileIR) -> Result<Vec<Scope>> {
    let roots = ir
        .body
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(index, s)| matches!(s, Stmt::If { .. }).then_some(index))
        .collect::<Vec<_>>();
    ensure!(
        roots.len() * 10 == ir.params.len(),
        "scoped batch root/ABI count mismatch"
    );
    roots
        .into_iter()
        .enumerate()
        .map(|(member, root)| {
            let Stmt::If { body, .. } = &ir.body.stmts[root] else {
                bail!("scoped root not If");
            };
            let inner = body
                .stmts
                .iter()
                .position(|s| matches!(s, Stmt::If { .. }))
                .context("scoped batch inner guard missing")?;
            Ok(Scope {
                path: vec![root, inner],
                parameters: ir.params[member * 10..(member + 1) * 10]
                    .iter()
                    .map(|p| p.name.clone())
                    .collect(),
            })
        })
        .collect()
}
pub(super) fn optimized(len: usize) -> Result<(TileIR, TileIR, Vec<Scope>)> {
    let mut separate = paired_ir(len, AdamKind::Vector4)?;
    let mut group = ir(len, 4)?;
    separate.optimize_indices()?;
    group.optimize_indices()?;
    let scopes = policy(&group)?;
    scoped_parameters::validate(&group, &scopes)?;
    Ok((separate, group, scopes))
}
#[test]
fn scoped_parameter_loads_keep_ordinary_ptx_and_place_ten_loads_after_full_uniform_guards()
-> Result<()> {
    use crate::{
        graph::TensorGraph,
        ptx::{PtxGraph, PtxTarget},
        tile::TileGraph,
    };
    let (separate, group, scopes) = optimized(2_818_048)?;
    let before = format!("{group:?}");
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    for (region_id, ir) in [separate.clone(), group.clone()].into_iter().enumerate() {
        tiles
            .region_kernels
            .push(crate::tile::graph::TileRegionKernel { region_id, ir });
    }
    // IRs already received the exact one optimize_indices pass used by the leaf.
    let ordinary = PtxGraph::from(tiles)
        .with_target(PtxTarget::sm80())
        .module_source();
    let (source, names) = scoped_parameters::module(
        vec![separate, group.clone()],
        group.clone(),
        &scopes,
        PtxTarget::sm80(),
    )?;
    assert_eq!(before, format!("{group:?}"));
    let start = source
        .find(&format!(".visible .entry {}", names[2]))
        .unwrap();
    assert_eq!(source[..start].trim_end(), ordinary.trim_end());
    let scoped = &source[start..];
    assert_eq!(scoped.matches(".param .u64").count(), 40);
    assert_eq!(scoped.matches("ld.param.u64").count(), 40);
    let first_guard = scoped.find("scoped_if_body_").unwrap();
    assert!(!scoped[..first_guard].contains("ld.param.u64"));
    let mut current = "";
    let mut counts = std::collections::BTreeMap::new();
    for line in scoped.lines().map(str::trim) {
        if line.ends_with(':') {
            current = line;
        }
        if line.starts_with("ld.param.u64") {
            assert!(current.starts_with("scoped_if_body_"));
            *counts.entry(current).or_insert(0usize) += 1;
        }
    }
    assert_eq!(counts.len(), 4);
    assert!(counts.values().all(|&count| count == 10));
    assert_eq!(scoped.matches("st.global.v4.f32").count(), 12);
    assert_eq!(scoped.matches("cvt.rn.bf16.f32").count(), 16);
    Ok(())
}
#[test]
fn scoped_parameter_policy_rejects_cross_region_unknown_and_nested_first_use_load_scopes()
-> Result<()> {
    let (_, group, scopes) = optimized(65536)?;
    let mut bad = scopes.clone();
    bad[0].path.push(0);
    assert!(scoped_parameters::validate(&group, &bad).is_err());
    let mut bad = scopes.clone();
    bad[0].parameters[0] = bad[1].parameters[0].clone();
    assert!(scoped_parameters::validate(&group, &bad).is_err());
    let mut bad = scopes.clone();
    bad[0].parameters[0] = "not_a_parameter".into();
    assert!(scoped_parameters::validate(&group, &bad).is_err());
    let mut bad = group.clone();
    bad.body.stmts.push(Stmt::BoundsCheck { extent: 65536 });
    assert!(scoped_parameters::validate(&bad, &scopes).is_err());
    let mut bad = group.clone();
    let root = scopes[0].path[0];
    let Stmt::If { condition, .. } = &mut bad.body.stmts[root] else {
        bail!("fixture outer guard missing");
    };
    *condition = ScalarPredicate::IndexLt {
        lhs: Expr::ThreadIdx(Dim::X),
        rhs: Expr::Const(64),
    };
    assert!(scoped_parameters::validate(&bad, &scopes).is_err());
    // A new memory use outside a selected body must not pick up a pointer whose
    // only definition is in another region, even if its type/name look valid.
    let mut bad = group.clone();
    bad.body.stmts.push(Stmt::LoadGlobalF32x4 {
        dests: [TileVar(1), TileVar(2), TileVar(3), TileVar(4)],
        param: group.params[0].name.clone(),
        group_index: Expr::Const(0),
        extent: 65536,
    });
    assert!(scoped_parameters::validate(&bad, &scopes).is_err());
    Ok(())
}
