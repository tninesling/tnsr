use super::*;
use crate::tile::MatMulSchedule;
use std::cell::Cell;
thread_local! {static REJECT:Cell<bool>=const{Cell::new(false)};}
pub(super) fn forced_rejection() -> bool {
    REJECT.with(Cell::get)
}
pub(super) struct RejectGuard;
impl RejectGuard {
    pub(super) fn install() -> Self {
        REJECT.with(|v| assert!(!v.replace(true)));
        Self
    }
}
impl Drop for RejectGuard {
    fn drop(&mut self) {
        REJECT.with(|v| v.set(false));
    }
}
pub(super) fn fixture() -> Result<(MatMulRegion, PtxTarget)> {
    use crate::tensor::TensorExpr;
    let graph: TensorGraph<half::bf16> = TensorExpr::<half::bf16>::input("a", vec![32, 512, K])
        .reshape(vec![M, K])
        .matmul(TensorExpr::input("b", vec![K, N]))
        .into();
    let plan = crate::ptx::PtxExecutionPlan::build_with_reduction_mode(
        &graph,
        crate::ptx::PtxReductionMode::DeterministicTree,
    )?;
    let [r] = plan.matmul_regions() else {
        anyhow::bail!("one rebased region")
    };
    let mut r = r.clone();
    let mut target = PtxTarget::sm80();
    target.compute_capability = (8, 9);
    target.matmul_resources = crate::tile::MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..crate::tile::MatMulResources::default()
    };
    r.schedule = MatMulSchedule::select_with_resources(
        M,
        N,
        K,
        DType::BF16,
        target.matmul_capabilities(),
        crate::tile::MatMulPrecision::Tf32x3,
        target.matmul_resources,
    )
    .with_shared_memory_reuse(true);
    r.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        target.supports_async_copy(),
        target.matmul_resources,
    )?;
    Ok((r, target))
}
#[test]
fn production_nn128x64_exact_shape_typed_flatten_and_template_rejections() -> Result<()> {
    let (r, target) = fixture()?;
    ir::guard(&r)?;
    assert!(!MixedOptions::default().nn128x64_ldmatrix);
    assert!(requested(
        &TensorGraphNode::MatMul { shape: vec![M, N] },
        &[M, K]
    ));
    for (m, n, k) in [(4096, N, K), (8192, N, K), (M, N, 1536), (M, 1408, 512)] {
        assert!(!requested(
            &TensorGraphNode::MatMul { shape: vec![m, n] },
            &[m, k]
        ));
    }
    let mut wrong = r.clone();
    wrong.inputs[0].source_shape = vec![M, K];
    assert!(ir::guard(&wrong).is_err());
    let mut wrong = r.clone();
    wrong.inputs[0].tensor.dtype = DType::F32;
    assert!(ir::guard(&wrong).is_err());
    let mut wrong = r.clone();
    wrong.inputs[0].tensor.access.results[0] = crate::tile::IndexExpr::Const(0);
    assert!(ir::guard(&wrong).is_err());
    let mut wrong = r.clone();
    wrong.schedule.warp_topology = (4, 2);
    assert!(ir::guard(&wrong).is_err());
    ir::candidate(&r, target)?.validate_layouts()?;
    Ok(())
}
#[test]
fn production_nn128x64_live_ranges_alignment_alias_and_owner_collisions() -> Result<()> {
    let bindings = [
        Binding {
            pointer: 0x10000000,
            len: M * K,
            dtype: DType::BF16,
        },
        Binding {
            pointer: 0x20000000,
            len: K * N,
            dtype: DType::BF16,
        },
        Binding {
            pointer: 0x30000000,
            len: M * N,
            dtype: DType::BF16,
        },
    ];
    validate_ranges(&bindings, &bindings)?;
    let mut bad = bindings;
    bad[2].pointer += 16;
    let mut owners = bad;
    owners[2].len += 8;
    assert!(validate_ranges(&bad, &owners).is_err());
    let mut bad = bindings;
    bad[0].dtype = DType::F32;
    assert!(validate_ranges(&bad, &bindings).is_err());
    assert!(validate_ranges(&bindings, &bindings[..2]).is_err());
    let mut owners = bindings.to_vec();
    owners.push(bindings[0]);
    assert!(validate_ranges(&bindings, &owners).is_err());
    let mut bad = bindings;
    bad[2].pointer = bad[0].pointer;
    assert!(validate_ranges(&bad, &bindings).is_err());
    let owner = NodeIndex::new(42);
    assert!(same_instance(Some(owner), "function1", owner, "function1"));
    assert!(!same_instance(Some(owner), "function2", owner, "function1"));
    assert!(!same_instance(
        Some(NodeIndex::new(43)),
        "function1",
        owner,
        "function1"
    ));
    Ok(())
}
#[test]
fn production_nn128x64_counter_debits_are_exact_and_fail_closed() -> Result<()> {
    assert_eq!(debit(8, 3, 1)?, (7, 2));
    assert_eq!(debit(1, 0, 0)?, (0, 0));
    assert!(debit(0, 1, 0).is_err());
    assert!(debit(8, 0, 1).is_err());
    Ok(())
}
#[test]
fn production_nn128x64_complete_qualified_rectangular_ptx_identity() -> Result<()> {
    let (r, target) = fixture()?;
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(std::slice::from_ref(&r))?;
    tiles.matmul_region_kernels[0].ir = ir::candidate(&r, target)?;
    tiles.optimize_indices()?;
    let actual = PtxGraph::from(tiles).with_target(target).module_source();
    let path=std::env::var_os("TNSR_NN_RECTANGULAR_GOLDEN").map(std::path::PathBuf::from).unwrap_or_else(||std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/paper-kernels/nn-bf16-ldmatrix-qualification/rectangular-v1/device-v1/candidate.ptx"));
    assert_eq!(
        actual,
        std::fs::read_to_string(path)?,
        "complete qualified rectangular module changed"
    );
    Ok(())
}
pub(super) fn snapshot(e: &MixedExecutor) -> String {
    format!(
        "{:?}",
        (
            e.kernels
                .iter()
                .map(|k| (
                    k.owner,
                    &k.label,
                    nt128_bias::ldmatrix::function_identity(&k.function).unwrap(),
                    k.config.grid_dim,
                    k.config.block_dim,
                    k.config.shared_mem_bytes,
                    &k.pointers,
                    k.zero_before.map(|b| (b.pointer, b.len, b.dtype))
                ))
                .collect::<Vec<_>>(),
            e.programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                .map(|p| (
                    &p.ptx,
                    &p.plan,
                    p.metadata.as_ref().map(|m| format!("{:?}", m.plan)),
                    p.output.pointer,
                    p.output.len,
                    p.output.dtype,
                    p.storage_bindings(&e.stream)
                        .iter()
                        .map(|b| (b.pointer, b.len, b.dtype))
                        .collect::<Vec<_>>()
                ))
                .collect::<Vec<_>>(),
            e.buffers
                .iter()
                .map(|b| {
                    let b = binding(b, &e.stream);
                    (b.pointer, b.len, b.dtype)
                })
                .collect::<Vec<_>>(),
            e.info.plain_bf16_output_programs,
            e.info.medium_bf16_square_programs,
            e.info.nn128x64_ldmatrix_programs,
            &e.nn128x64_sources
        )
    )
}
