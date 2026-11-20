use super::*;
use crate::{
    ptx::resident::{AdamGroup, ResidentAdam},
    tensor::{IndexTensor, Parameter, TensorExpr},
};
type ReadTriple = (Vec<f32>, Vec<f32>, Vec<f32>);
fn bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
            "{a:?} vs {b:?}"
        );
    }
}
#[test]
fn indexed_loss_cooperative_ordered_forward_resident_replay_and_recovery() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    for half in [false, true] {
        let rows = 3;
        let len = rows * 4096;
        let mut parameters = Vec::new();
        let mut graphs = Vec::new();
        for _ in 0..2 {
            let x = Parameter::new(
                (0..len).map(|i| (i % 23) as f32 / 16. - 0.5).collect(),
                vec![rows, 4096],
            );
            let mut logits =
                TensorExpr::from(x.clone()) + TensorExpr::input("delta", vec![rows, 4096]);
            if half {
                logits = logits.cast(ValueDType::BF16).cast(ValueDType::F32);
            }
            let loss = (logits
                .indexed_cross_entropy_indices(IndexTensor::input("targets", vec![rows]))
                * TensorExpr::input("seed", vec![rows]))
            .mean_all();
            let mut g = TensorGraph::new();
            let root = g.add_expr(&loss, false);
            graphs.push((g.with_gradients(root), root));
            parameters.push(x);
        }
        let mut owners = parameters
            .iter()
            .map(|p| {
                ResidentAdam::new_with_graph_replay(
                    vec![(p.clone(), 0)],
                    vec![AdamGroup::new(1e-4)],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        for o in &mut owners {
            o.set_update_batch_size(8)?;
            o.set_experimental_first_accumulation_overwrite(true)?;
            o.set_experimental_gradient_norm_partials(true)?;
            o.begin_step()?;
        }
        let mut executors = owners
            .iter()
            .enumerate()
            .map(|(i, o)| {
                o.mixed_executor_with_options(
                    &graphs[i].0,
                    MatMulPrecision::StrictF32,
                    MixedOptions {
                        indexed_loss_cooperative_ordered_backward: i == 1,
                        indexed_loss_cooperative_ordered_forward: i == 1,
                        direct_master_bindings: true,
                        incremental_allocation_reuse: true,
                        ..MixedOptions::default()
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            executors[0]
                .info()
                .indexed_loss_cooperative_ordered_backward_programs,
            0
        );
        assert_eq!(
            executors[1]
                .info()
                .indexed_loss_cooperative_ordered_backward_programs,
            1
        );
        assert_eq!(
            executors[0]
                .info()
                .indexed_loss_cooperative_ordered_forward_programs,
            0
        );
        assert_eq!(
            executors[1]
                .info()
                .indexed_loss_cooperative_ordered_forward_programs,
            1
        );
        assert_eq!(executors[0].info().kernels, executors[1].info().kernels);
        for (phase, microbatches) in [1, 2, 1].into_iter().enumerate() {
            for i in 0..2 {
                if phase > 0 {
                    owners[i].begin_step()?;
                }
                for micro in 0..microbatches {
                    owners[i].accumulate_mixed(
                        &mut executors[i],
                        &graphs[i].0,
                        HashMap::from([
                            ("delta".into(), vec![(phase + micro) as f32 / 64.; len]),
                            ("seed".into(), vec![1., -0.25, 0.5]),
                        ]),
                        HashMap::from([(
                            "targets".into(),
                            vec![((phase + micro) * 31) as u32, 4095, 0],
                        )]),
                    )?;
                }
            }
            let a = owners[0].step(1.)?;
            let b = owners[1].step(1.)?;
            assert_eq!(a.timestep, b.timestep);
            assert_eq!(a.gradient_norm.to_bits(), b.gradient_norm.to_bits());
            assert_eq!(a.gradient_scale.to_bits(), b.gradient_scale.to_bits());
            bits(
                &executors[0].read(graphs[0].1)?,
                &executors[1].read(graphs[1].1)?,
            );
            let ga = owners[0].gradients()?;
            let gb = owners[1].gradients()?;
            bits(&ga[&parameters[0].id()], &gb[&parameters[1].id()]);
            bits(&parameters[0].values()?, &parameters[1].values()?);
            let ma = owners[0].moments()?;
            let mb = owners[1].moments()?;
            bits(
                &ma[&parameters[0].id()].first,
                &mb[&parameters[1].id()].first,
            );
            bits(
                &ma[&parameters[0].id()].second,
                &mb[&parameters[1].id()].second,
            );
            if phase == 1 {
                for i in 0..2 {
                    let weight = parameters[i].values()?;
                    let moments = owners[i].moments()?;
                    for invalid_target in [false, true] {
                        owners[i].begin_step()?;
                        let feeds = HashMap::from([
                            ("delta".into(), vec![0.; len]),
                            (
                                "seed".into(),
                                vec![if invalid_target { 1. } else { f32::NAN }; rows],
                            ),
                        ]);
                        let indices = HashMap::from([(
                            "targets".into(),
                            vec![if invalid_target { 4096 } else { 0 }; rows],
                        )]);
                        let result = owners[i].accumulate_mixed(
                            &mut executors[i],
                            &graphs[i].0,
                            feeds,
                            indices,
                        );
                        if invalid_target {
                            assert!(result.is_err());
                        } else {
                            let _ = result;
                        }
                        assert!(owners[i].step(1.).is_err());
                        bits(&weight, &parameters[i].values()?);
                        let now = owners[i].moments()?;
                        bits(
                            &moments[&parameters[i].id()].first,
                            &now[&parameters[i].id()].first,
                        );
                        bits(
                            &moments[&parameters[i].id()].second,
                            &now[&parameters[i].id()].second,
                        );
                    }
                }
            }
        }
        for e in &executors {
            assert!(e.info().replay);
            assert_eq!(e.info().direct_capture_variants, 2);
            assert_eq!(e.info().direct_capture_misses, 2);
            assert!(e.info().direct_capture_hits > 0);
            assert_eq!(e.transfers().parameter_copies, 0);
        }
    }
    Ok(())
}
#[test]
fn indexed_loss_cooperative_ordered_unsupported_vocab_keeps_whole_baseline() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    for vocabulary in [255, 257, 2048, 4095, 4097] {
        let x = Parameter::new(vec![0.125; 2 * vocabulary], vec![2, vocabulary]);
        let loss = TensorExpr::from(x)
            .indexed_cross_entropy_indices(IndexTensor::input("targets", vec![2]))
            .mean_all();
        let mut g = TensorGraph::new();
        let root = g.add_expr(&loss, false);
        let graph = g.with_gradients(root);
        let mut old = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::StrictF32,
            true,
            MixedOptions::default(),
        )?;
        let mut new = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::StrictF32,
            true,
            MixedOptions {
                indexed_loss_cooperative_ordered_backward: true,
                indexed_loss_cooperative_ordered_forward: true,
                ..MixedOptions::default()
            },
        )?;
        assert_eq!(
            new.info()
                .indexed_loss_cooperative_ordered_backward_programs,
            0
        );
        for e in [&mut old, &mut new] {
            e.upload_with_indices(
                HashMap::new(),
                HashMap::from([("targets".into(), vec![0, (vocabulary - 1) as u32])]),
            )?;
            e.run()?;
        }
        bits(&old.read(root)?, &new.read(root)?);
        assert_eq!(old.info().kernels, new.info().kernels);
    }
    Ok(())
}

#[test]
fn indexed_loss_forward_four_modes_replay_and_actual_geometry() -> Result<()> {
    if let Err(error) = cudarc::driver::CudaContext::new(0) {
        assert!(std::env::var_os("TNSR_REQUIRE_CUDA").is_none(), "{error}");
        return Ok(());
    }
    for vocabulary in [4095usize, 4096] {
        for half in [false, true] {
            let p = Parameter::new(
                (0..3 * vocabulary)
                    .map(|i| (i % 31) as f32 / 16. - 0.75)
                    .collect(),
                vec![3, vocabulary],
            );
            let mut logits =
                TensorExpr::from(p.clone()) + TensorExpr::input("delta", vec![3, vocabulary]);
            if half {
                logits = logits.cast(ValueDType::BF16).cast(ValueDType::F32);
            }
            let row_loss =
                logits.indexed_cross_entropy_indices(IndexTensor::input("targets", vec![3]));
            let mut graph = TensorGraph::new();
            let root = graph.add_expr(&row_loss.mean_all(), false);
            let forward = graph
                .toposort()
                .into_iter()
                .find(|&n| matches!(graph[n], TensorGraphNode::IndexedCrossEntropy { .. }))
                .expect("single forward node in mean loss");
            let graph = graph.with_gradients(root);
            let mut expected: Option<Vec<ReadTriple>> = None;
            for (fwd, bwd) in [(false, false), (true, false), (false, true), (true, true)] {
                let mut e = MixedExecutor::new_with_options(
                    &graph,
                    MatMulPrecision::StrictF32,
                    true,
                    MixedOptions {
                        indexed_loss_cooperative_ordered_forward: fwd,
                        indexed_loss_cooperative_ordered_backward: bwd,
                        ..MixedOptions::default()
                    },
                )?;
                assert_eq!(
                    e.info().indexed_loss_cooperative_ordered_forward_programs,
                    usize::from(fwd && vocabulary == 4096)
                );
                assert_eq!(
                    e.info().indexed_loss_cooperative_ordered_backward_programs,
                    usize::from(bwd && vocabulary == 4096)
                );
                let forward_kernel = e
                    .kernels
                    .iter()
                    .find(|k| k.label.contains("IndexedCrossEntropy Kernel"))
                    .expect("actual forward kernel");
                assert_eq!(
                    forward_kernel.config.block_dim,
                    (if fwd && vocabulary == 4096 { 256 } else { 32 }, 1, 1)
                );
                assert_eq!(forward_kernel.config.grid_dim, (3, 1, 1));
                if let Some(dir) = std::env::var_os("TNSR_CE_ORDERED_FORWARD_PTX_DIR") {
                    let dir = std::path::PathBuf::from(dir);
                    std::fs::create_dir_all(&dir)?;
                    let threads = forward_kernel.config.block_dim.0
                        * forward_kernel.config.block_dim.1
                        * forward_kernel.config.block_dim.2;
                    std::fs::write(
                        dir.join(format!(
                            "selected-half{half}-f{fwd}-b{bwd}-v{vocabulary}.json"
                        )),
                        serde_json::to_vec_pretty(
                            &serde_json::json!({"label":forward_kernel.label,"registers":forward_kernel.function.num_regs()?,"shared":forward_kernel.function.shared_size_bytes()?,"local":forward_kernel.function.local_size_bytes()?,"max_threads":forward_kernel.function.max_threads_per_block()?,"threads":threads,"capacity":forward_kernel.function.occupancy_max_active_blocks_per_multiprocessor(threads,0,None)?,"actual_forward_programs":e.info().indexed_loss_cooperative_ordered_forward_programs,"scope":"actual selected forward function; API capacity is not measured occupancy"}),
                        )?,
                    )?;
                }

                let mut outputs = Vec::new();
                for revision in 0..3 {
                    e.upload_with_indices(
                        HashMap::from([(
                            "delta".into(),
                            vec![revision as f32 / 32.; 3 * vocabulary],
                        )]),
                        HashMap::from([(
                            "targets".into(),
                            vec![0, (vocabulary - 1) as u32, revision * 17],
                        )]),
                    )?;
                    e.run()?;
                    outputs.push((
                        e.read(forward)?,
                        e.read(root)?,
                        e.gradients(&graph)?[&p.id()].clone(),
                    ));
                }
                if let Some(old) = &expected {
                    for (a, b) in outputs.iter().zip(old) {
                        bits(&a.0, &b.0);
                        bits(&a.1, &b.1);
                        bits(&a.2, &b.2);
                    }
                } else {
                    expected = Some(outputs);
                }
                assert!(e.captured.is_some());
                assert!(
                    e.upload_with_indices(
                        HashMap::from([("delta".into(), vec![0.; 3 * vocabulary])]),
                        HashMap::from([("targets".into(), vec![vocabulary as u32; 3])])
                    )
                    .is_err()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn indexed_loss_forward_actual_jit_rejection_restores_both_original_launches() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let x = Parameter::new(vec![0.; 3 * 4096], vec![3, 4096]);
    let loss = TensorExpr::from(x)
        .indexed_cross_entropy_indices(IndexTensor::input("targets", vec![3]))
        .mean_all();
    let mut g = TensorGraph::new();
    let root = g.add_expr(&loss, false);
    let g = g.with_gradients(root);
    crate::ptx::graph::with_forced_ordered_ce_resource_rejection(|| -> Result<()> {
        for (f, b) in [(true, false), (false, true), (true, true)] {
            let mut e = MixedExecutor::new_with_options(
                &g,
                MatMulPrecision::StrictF32,
                true,
                MixedOptions {
                    indexed_loss_cooperative_ordered_forward: f,
                    indexed_loss_cooperative_ordered_backward: b,
                    ..MixedOptions::default()
                },
            )?;
            assert_eq!(
                e.info().indexed_loss_cooperative_ordered_forward_programs,
                0
            );
            assert_eq!(
                e.info().indexed_loss_cooperative_ordered_backward_programs,
                0
            );
            let forward = e
                .kernels
                .iter()
                .find(|k| k.label.contains("IndexedCrossEntropy Kernel"))
                .expect("original forward launch");
            assert_eq!(forward.config.block_dim, (32, 1, 1));
            assert_eq!(forward.config.grid_dim, (3, 1, 1));
            e.upload_with_indices(
                HashMap::new(),
                HashMap::from([("targets".into(), vec![0, 1, 2])]),
            )?;
            e.run()?;
            assert!(e.read(root)?[0].is_finite());
        }
        Ok(())
    })
}

#[test]
fn indexed_loss_forward_float_targets_cache_off_on_off_validation() -> Result<()> {
    use crate::Executor;
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let g: TensorGraph<f32> = TensorExpr::input("logits", vec![17, 4096])
        .indexed_cross_entropy(TensorExpr::input("targets", vec![17]))
        .into();
    let mut e = crate::ptx::PtxExecutor::<crate::ptx::F32>::try_new()?;
    let logits = (0..17 * 4096)
        .map(|i| (i % 37) as f32 / 16. - 1.)
        .collect::<Vec<_>>();
    let targets = (0..17).map(|i| (i * 127) as f32).collect::<Vec<_>>();
    let mut expected: Option<Vec<f32>> = None;
    for enabled in [false, true, false] {
        e.set_experimental_indexed_loss_cooperative_ordered_forward(enabled);
        let values = e.execute(
            &g,
            HashMap::from([
                ("logits".into(), logits.clone()),
                ("targets".into(), targets.clone()),
            ]),
        )?;
        assert_eq!(
            e.indexed_loss_cooperative_ordered_forward_programs(),
            usize::from(enabled)
        );
        if let Some(old) = &expected {
            bits(&values, old);
        } else {
            expected = Some(values);
        }
    }
    e.set_experimental_indexed_loss_cooperative_ordered_forward(true);
    for invalid in [0.5, f32::NAN, 4096.] {
        let mut bad = targets.clone();
        bad[0] = invalid;
        assert!(
            e.execute(
                &g,
                HashMap::from([("logits".into(), logits.clone()), ("targets".into(), bad)])
            )
            .is_err()
        );
    }
    let values = e.execute(
        &g,
        HashMap::from([("logits".into(), logits), ("targets".into(), targets)]),
    )?;
    bits(&values, expected.as_ref().unwrap());
    Ok(())
}
