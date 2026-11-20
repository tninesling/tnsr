use super::*;
use crate::{
    ptx::resident::{AdamGroup, ResidentAdam},
    tensor::{IndexTensor, Parameter, TensorExpr},
};
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
fn indexed_loss_cooperative_ordered_mixed_replay_full_state_and_rejection_recovery() -> Result<()> {
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
