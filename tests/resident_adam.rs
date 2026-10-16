#![cfg(feature = "cuda")]
use std::collections::HashMap;
use tnsr::graph::{TensorGraph, WithGrad};
use tnsr::ptx::resident::{AdamGroup, ResidentAdam};
use tnsr::tensor::{Input, Parameter, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

fn owner(parameters: Vec<(Parameter<f32>, usize)>, groups: Vec<AdamGroup>) -> Option<ResidentAdam> {
    owner_with_replay(parameters, groups, false)
}
fn owner_with_replay(
    parameters: Vec<(Parameter<f32>, usize)>,
    groups: Vec<AdamGroup>,
    replay: bool,
) -> Option<ResidentAdam> {
    let constructor = if replay {
        ResidentAdam::new_with_graph_replay
    } else {
        ResidentAdam::new
    };
    match constructor(parameters, groups) {
        Ok(owner) => Some(owner),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("initialize resident training CUDA device") =>
        {
            None
        }
        Err(error) => panic!("{error:#}"),
    }
}
fn graph(
    weight: &Parameter<f32>,
    bias: &Parameter<f32>,
    n: usize,
    scale: f32,
) -> TensorGraph<f32, WithGrad> {
    let x: TensorExpr<f32> = Input::new("x", vec![n, 2]).into();
    let y = x.matmul(TensorExpr::from(weight.clone())) + TensorExpr::from(bias.clone());
    let loss = (y.clone() * y).mean_all() * TensorExpr::constant(vec![scale], vec![1, 1]);
    let graph: TensorGraph<f32> = loss.into();
    let output = *graph.toposort().last().unwrap();
    graph.with_gradients(output)
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tolerance, "{a} != {e}");
    }
}

#[test]
fn mixed_resident_adam_refreshes_published_masters_and_rejects_nonfinite_updates() {
    use tnsr::{tensor::ValueDType, tile::MatMulPrecision};
    let w = Parameter::new(vec![0.2, -0.3, 0.4, 0.1, -0.2, 0.5], vec![2, 3]);
    let b = Parameter::new(vec![0.1, 0.2, -0.1], vec![3]);
    let cw = Parameter::new(w.values().unwrap(), vec![2, 3]);
    let cb = Parameter::new(b.values().unwrap(), vec![3]);
    let groups = vec![AdamGroup::new(0.01), AdamGroup::new(0.02)];
    let Some(mut owner) =
        owner_with_replay(vec![(w.clone(), 0), (b.clone(), 1)], groups.clone(), true)
    else {
        return;
    };
    let make = |w: &Parameter<f32>, b: &Parameter<f32>, n: usize, scale: f32| {
        let y = TensorExpr::input("x", vec![n, 2])
            .linear_bf16(w.clone().into(), b.clone().into())
            .unwrap()
            .cast(ValueDType::F32);
        let loss = (y.clone() * y).mean_all() * TensorExpr::from(scale);
        let mut g = TensorGraph::new();
        let loss = g.add_expr(&loss, false);
        g.with_gradients(loss)
    };
    let graphs = [make(&w, &b, 1, 1. / 3.), make(&w, &b, 2, 2. / 3.)];
    let cpu_graphs = [make(&cw, &cb, 1, 1. / 3.), make(&cw, &cb, 2, 2. / 3.)];
    owner.begin_step().unwrap();
    let mut executors = graphs
        .iter()
        .map(|g| owner.mixed_executor(g, MatMulPrecision::Tf32x3).unwrap())
        .collect::<Vec<_>>();
    let mut moments = [vec![0_f32; 6], vec![0_f32; 3]];
    let mut variances = moments.clone();
    for step in 1_i32..=3 {
        if step == 3 {
            let values: Vec<_> = cw.values().unwrap().iter().map(|v| v + 0.01).collect();
            cw.set_values(values.clone()).unwrap();
            w.set_values(values).unwrap();
        }
        owner.begin_step().unwrap();
        let mut gradients = [vec![0_f32; 6], vec![0_f32; 3]];
        for (i, x) in [vec![0.25, -0.5], vec![0.75, 0.5, -0.25, 0.125]]
            .into_iter()
            .enumerate()
        {
            let inputs = HashMap::from([("x".into(), x)]);
            let mut cpu = SimpleExecutor::new();
            cpu.execute(&cpu_graphs[i], inputs.clone()).unwrap();
            let g = cpu.get_gradients(&cpu_graphs[i]);
            for (target, id) in gradients.iter_mut().zip([cw.id(), cb.id()]) {
                for (a, b) in target.iter_mut().zip(&g[&id]) {
                    *a += b;
                }
            }
            owner
                .accumulate_mixed(&mut executors[i], &graphs[i], inputs, HashMap::new())
                .unwrap();
        }
        let norm = gradients
            .iter()
            .flatten()
            .map(|g| f64::from(*g).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = (0.15 / (norm + 1e-6)).min(1.) as f32;
        let status = owner.step(0.15).unwrap();
        assert_eq!(status.timestep, step as usize);
        assert!((status.gradient_norm - norm).abs() < 2e-6);
        let actual = owner.gradients().unwrap();
        let state = owner.moments().unwrap();
        for (group, (cpu, gpu)) in [(cw.clone(), w.clone()), (cb.clone(), b.clone())]
            .into_iter()
            .enumerate()
        {
            let s = &groups[group];
            let mut values = cpu.values().unwrap();
            close(
                &actual[&gpu.id()],
                &gradients[group]
                    .iter()
                    .map(|g| g * scale)
                    .collect::<Vec<_>>(),
                2e-6,
            );
            for (j, value) in values.iter_mut().enumerate() {
                let g = gradients[group][j] * scale;
                moments[group][j] = s.beta1 * moments[group][j] + (1. - s.beta1) * g;
                variances[group][j] = s.beta2 * variances[group][j] + (1. - s.beta2) * g * g;
                *value -= s.learning_rate * (moments[group][j] / (1. - s.beta1.powi(step)))
                    / ((variances[group][j] / (1. - s.beta2.powi(step))).sqrt() + s.epsilon);
            }
            cpu.set_values(values.clone()).unwrap();
            close(&gpu.values().unwrap(), &values, 2e-6);
            close(&state[&gpu.id()].first, &moments[group], 2e-6);
            close(&state[&gpu.id()].second, &variances[group], 2e-6);
        }
    }
    let weights = w.values().unwrap();
    let before = owner.moments().unwrap();
    owner.begin_step().unwrap();
    owner
        .accumulate_mixed(
            &mut executors[0],
            &graphs[0],
            HashMap::from([("x".into(), vec![f32::NAN, 1.])]),
            HashMap::new(),
        )
        .unwrap();
    assert!(owner.step(0.15).is_err());
    assert_eq!(owner.moments().unwrap(), before);
    assert_eq!(w.values().unwrap(), weights);
    owner.begin_step().unwrap();
    owner
        .accumulate_mixed(
            &mut executors[0],
            &graphs[0],
            HashMap::from([("x".into(), vec![0.25, -0.5])]),
            HashMap::new(),
        )
        .unwrap();
    assert_eq!(owner.step(0.15).unwrap().timestep, 4);
}

#[test]
fn resident_multistep_weighted_microbatches_match_cpu_gradients_moments_and_groups() {
    check_multistep(false, tnsr::ptx::PtxReductionMode::Strict);
}
#[test]
fn resident_graph_replay_matches_changed_inputs_partial_batches_groups_and_host_mutations() {
    check_multistep(true, tnsr::ptx::PtxReductionMode::Strict);
}
#[test]
fn resident_graph_tree_replay_matches_cpu_state_and_constant_replacement() {
    check_multistep(true, tnsr::ptx::PtxReductionMode::DeterministicTree);
}
fn check_multistep(replay: bool, mode: tnsr::ptx::PtxReductionMode) {
    let w = Parameter::new(vec![0.2, -0.3, 0.4, 0.1, -0.2, 0.5], vec![2, 3]);
    let b = Parameter::new(vec![0.1, 0.2, -0.1], vec![1, 3]);
    let cw = Parameter::new(w.values().unwrap(), vec![2, 3]);
    let cb = Parameter::new(b.values().unwrap(), vec![1, 3]);
    let groups = vec![
        AdamGroup::new(0.01),
        AdamGroup {
            learning_rate: 0.02,
            beta1: 0.8,
            beta2: 0.95,
            epsilon: 1e-6,
            weight_decay: 0.01,
        },
    ];
    let Some(mut owner) = owner_with_replay(
        vec![(w.clone(), 0), (b.clone(), 1), (w.clone(), 0)],
        groups.clone(),
        replay,
    ) else {
        return;
    };
    let mut executors = [
        owner.executor_with_reduction_mode(mode).unwrap(),
        owner.executor_with_reduction_mode(mode).unwrap(),
    ];
    let mut graphs = [graph(&w, &b, 1, 1.0 / 3.0), graph(&w, &b, 2, 2.0 / 3.0)];
    let mut cpu_graphs = [graph(&cw, &cb, 1, 1.0 / 3.0), graph(&cw, &cb, 2, 2.0 / 3.0)];
    let mut moments = [vec![0.0f32; 6], vec![0.0f32; 3]];
    let mut variances = moments.clone();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reader = if replay {
        let stop = stop.clone();
        let parameter = w.clone();
        Some(std::thread::spawn(move || {
            let mut reads = 0;
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                let values = parameter.values().unwrap();
                assert_eq!(values.len(), 6);
                assert!(values.iter().all(|value| value.is_finite()));
                reads += 1;
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            reads
        }))
    } else {
        None
    };
    for step in 1..=if replay { 6_i32 } else { 3_i32 } {
        if step == 3 {
            owner.set_learning_rate(0, 0.003).unwrap();
            if replay {
                let values: Vec<_> = cw
                    .values()
                    .unwrap()
                    .iter()
                    .map(|value| value + 0.01)
                    .collect();
                cw.set_values(values.clone()).unwrap();
                w.set_values(values).unwrap();
            }
        }
        if step == 4 {
            // Same code signature, different immutable constant allocations/data.
            graphs = [graph(&w, &b, 1, 0.25), graph(&w, &b, 2, 0.75)];
            cpu_graphs = [graph(&cw, &cb, 1, 0.25), graph(&cw, &cb, 2, 0.75)];
        }
        owner.begin_step().unwrap();
        let mut gradients = [vec![0.0f32; 6], vec![0.0f32; 3]];
        for (i, values) in [vec![0.25, -0.5], vec![0.75, 0.5, -0.25, 0.125]]
            .into_iter()
            .enumerate()
        {
            let values = values
                .into_iter()
                .map(|value| value + step as f32 * 0.025)
                .collect();
            let inputs = HashMap::from([("x".into(), values)]);
            let mut cpu = SimpleExecutor::new();
            cpu.execute(&cpu_graphs[i], inputs.clone()).unwrap();
            let grads = cpu.get_gradients(&cpu_graphs[i]);
            for (target, id) in gradients.iter_mut().zip([cw.id(), cb.id()]) {
                for (target, value) in target.iter_mut().zip(&grads[&id]) {
                    *target += value;
                }
            }
            owner
                .accumulate(&mut executors[i], &graphs[i], inputs)
                .unwrap_or_else(|error| {
                    panic!(
                        "step {step} shape {i}: {error:#}; {:?}",
                        executors[i].replay_metrics()
                    )
                });
            if replay && step != 1 && step != 4 {
                assert!(
                    executors[i].replay_metrics().active,
                    "{:?}",
                    executors[i].replay_metrics()
                );
                assert_eq!(executors[i].execution_metrics().graph_launches, 1);
                assert_eq!(executors[i].replay_metrics().fallbacks, 0);
            }
            assert_eq!(executors[i].compilation_count(), 1);
        }
        let norm = gradients
            .iter()
            .flatten()
            .map(|&g| f64::from(g).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = (0.15 / (norm + 1e-6)).min(1.0) as f32;
        let status = owner.step(0.15).unwrap();
        assert_eq!(status.timestep, step as usize);
        assert!((status.gradient_norm - norm).abs() < 1e-6);
        let observed = owner.gradients().unwrap();
        let state = owner.moments().unwrap();
        for (group, ((parameter, device), gradient)) in
            [(cw.clone(), w.clone()), (cb.clone(), b.clone())]
                .into_iter()
                .zip(gradients)
                .enumerate()
        {
            let mut values = parameter.values().unwrap();
            let mut settings = groups[group].clone();
            if step >= 3 && group == 0 {
                settings.learning_rate = 0.003;
            }
            close(
                &observed[&device.id()],
                &gradient.iter().map(|g| g * scale).collect::<Vec<_>>(),
                1e-6,
            );
            for (j, value) in values.iter_mut().enumerate() {
                let g = gradient[j] * scale + settings.weight_decay * *value;
                moments[group][j] = settings.beta1 * moments[group][j] + (1.0 - settings.beta1) * g;
                variances[group][j] =
                    settings.beta2 * variances[group][j] + (1.0 - settings.beta2) * g * g;
                let m = moments[group][j] / (1.0 - settings.beta1.powi(step));
                let v = variances[group][j] / (1.0 - settings.beta2.powi(step));
                *value -= settings.learning_rate * m / (v.sqrt() + settings.epsilon);
            }
            parameter.set_values(values.clone()).unwrap();
            close(&device.values().unwrap(), &values, 2e-6);
            close(&state[&device.id()].first, &moments[group], 1e-6);
            close(&state[&device.id()].second, &variances[group], 1e-6);
        }
    }
    drop(owner);
    stop.store(true, std::sync::atomic::Ordering::Release);
    if let Some(reader) = reader {
        assert!(reader.join().unwrap() > 0);
    }
    close(&w.values().unwrap(), &cw.values().unwrap(), 2e-6);
}

fn indexed_graph(weight: &Parameter<f32>, computed: bool) -> TensorGraph<f32, WithGrad> {
    let input: TensorExpr<f32> = Input::new("indices", vec![2]).into();
    let indices = if computed { input - 1.0 } else { input };
    let y = TensorExpr::from(weight.clone()).embedding(indices);
    let graph: TensorGraph<f32> = (y.clone() * y).mean_all().into();
    let output = *graph.toposort().last().unwrap();
    graph.with_gradients(output)
}

#[test]
fn resident_replay_revalidates_changed_exact_indices_before_launch_and_recovers() {
    check_indexed_replay(false);
}

#[test]
fn resident_computed_indices_use_checked_uncaptured_fallback_once() {
    check_indexed_replay(true);
}

fn check_indexed_replay(computed: bool) {
    let values: Vec<_> = (0..683 * 2)
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let p = Parameter::new(values.clone(), vec![683, 2]);
    let baseline = Parameter::new(values, vec![683, 2]);
    let Some(mut owner) = owner_with_replay(vec![(p.clone(), 0)], vec![AdamGroup::new(0.01)], true)
    else {
        return;
    };
    let Some(mut reference) = owner_with_replay(
        vec![(baseline.clone(), 0)],
        vec![AdamGroup::new(0.01)],
        false,
    ) else {
        return;
    };
    let mut executor = owner.executor().unwrap();
    let mut ordinary = reference.executor().unwrap();
    let graph = indexed_graph(&p, computed);
    let reference_graph = indexed_graph(&baseline, computed);
    for step in 0..4 {
        if step == 2 && !computed {
            let previous = p.values().unwrap();
            let launches = executor.replay_metrics().graph_launches;
            owner.begin_step().unwrap();
            let error = owner
                .accumulate(
                    &mut executor,
                    &graph,
                    HashMap::from([("indices".into(), vec![683.0, 1.0])]),
                )
                .unwrap_err();
            assert!(error.to_string().contains("out of range"), "{error:#}");
            assert_eq!(executor.replay_metrics().graph_launches, launches);
            assert!(executor.replay_metrics().active);
            assert_eq!(p.values().unwrap(), previous);
        }
        let inputs = HashMap::from([("indices".into(), vec![257.0 + step as f32, 682.0])]);
        owner.begin_step().unwrap();
        reference.begin_step().unwrap();
        owner
            .accumulate(&mut executor, &graph, inputs.clone())
            .unwrap();
        reference
            .accumulate(&mut ordinary, &reference_graph, inputs)
            .unwrap();
        owner.step(0.15).unwrap();
        reference.step(0.15).unwrap();
        close(&p.values().unwrap(), &baseline.values().unwrap(), 2e-6);
        close(
            &owner.gradients().unwrap()[&p.id()],
            &reference.gradients().unwrap()[&baseline.id()],
            2e-6,
        );
        let actual = owner.moments().unwrap();
        let expected = reference.moments().unwrap();
        close(
            &actual[&p.id()].first,
            &expected[&baseline.id()].first,
            2e-6,
        );
        close(
            &actual[&p.id()].second,
            &expected[&baseline.id()].second,
            2e-6,
        );
        let stats = executor.replay_metrics();
        if computed {
            assert_eq!(stats.fallbacks, 1, "{stats:?}");
            assert!(!stats.active);
            assert!(
                stats
                    .last_fallback
                    .as_ref()
                    .unwrap()
                    .contains("computed indices")
            );
        } else {
            assert!(stats.active, "{stats:?}");
            assert_eq!(stats.captures, 1);
            assert_eq!(stats.fallbacks, 0);
        }
    }
    executor.clear_pool();
    assert!(!executor.replay_metrics().active);
    if !computed {
        let inputs = HashMap::from([("indices".into(), vec![258.0, 681.0])]);
        owner.begin_step().unwrap();
        reference.begin_step().unwrap();
        owner
            .accumulate(&mut executor, &graph, inputs.clone())
            .unwrap();
        reference
            .accumulate(&mut ordinary, &reference_graph, inputs)
            .unwrap();
        owner.step(0.15).unwrap();
        reference.step(0.15).unwrap();
        assert!(executor.replay_metrics().active);
        assert_eq!(executor.replay_metrics().captures, 2);
        close(&p.values().unwrap(), &baseline.values().unwrap(), 2e-6);
    }
}

#[test]
fn resident_source_alias_keeps_correct_uncaptured_execution() {
    let p = Parameter::new(vec![2.0], vec![1]);
    let Some(mut owner) = owner_with_replay(vec![(p.clone(), 0)], vec![AdamGroup::new(0.01)], true)
    else {
        return;
    };
    let graph: TensorGraph<f32> = TensorExpr::from(p.clone()).into();
    let output = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(output);
    let mut executor = owner.executor().unwrap();
    for step in 1..=3 {
        owner.begin_step().unwrap();
        owner
            .accumulate(&mut executor, &graph, HashMap::new())
            .unwrap();
        owner.step(0.0).unwrap();
        close(&p.values().unwrap(), &[2.0 - step as f32 * 0.01], 2e-6);
        let stats = executor.replay_metrics();
        assert_eq!(stats.fallbacks, 1, "{stats:?}");
        assert!(!stats.active);
        assert!(
            stats
                .last_fallback
                .as_ref()
                .unwrap()
                .contains("source binding")
        );
    }
}

#[test]
fn captured_nonfinite_update_preserves_state_and_recovers_without_advancing_adam() {
    let p = Parameter::new(vec![2.0], vec![1]);
    let group = AdamGroup {
        weight_decay: f32::MAX,
        ..AdamGroup::new(0.01)
    };
    let Some(mut owner) = owner_with_replay(vec![(p.clone(), 0)], vec![group], true) else {
        return;
    };
    let value = TensorExpr::from(p.clone());
    let graph: TensorGraph<f32> = (value.clone() * value).mean_all().into();
    let output = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(output);
    let mut executor = owner.executor().unwrap();
    for _ in 0..2 {
        owner.begin_step().unwrap();
        owner
            .accumulate(&mut executor, &graph, HashMap::new())
            .unwrap();
        assert!(executor.replay_metrics().active);
        assert!(owner.step(0.0).is_err());
        assert_eq!(p.values().unwrap(), [2.0]);
        let moments = owner.moments().unwrap();
        assert_eq!(moments[&p.id()].first, [0.0]);
        assert_eq!(moments[&p.id()].second, [0.0]);
    }
    p.set_values(vec![0.0]).unwrap();
    owner.begin_step().unwrap();
    owner
        .accumulate(&mut executor, &graph, HashMap::new())
        .unwrap();
    assert_eq!(owner.step(0.0).unwrap().timestep, 1);
    assert_eq!(p.values().unwrap(), [0.0]);
    assert_eq!(executor.replay_metrics().captures, 1);
}

#[test]
fn nonfinite_update_is_rejected_without_publishing_weights_or_moments() {
    let p = Parameter::new(vec![2.0], vec![1]);
    let group = AdamGroup {
        weight_decay: f32::MAX,
        ..AdamGroup::new(0.01)
    };
    let Some(mut owner) = owner(vec![(p.clone(), 0)], vec![group]) else {
        return;
    };
    let graph: TensorGraph<f32> =
        (TensorExpr::from(p.clone()) * TensorExpr::constant(vec![1.0], vec![1])).into();
    let output = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(output);
    let mut executor = owner.executor().unwrap();
    owner.begin_step().unwrap();
    owner
        .accumulate(&mut executor, &graph, HashMap::new())
        .unwrap();
    assert!(owner.step(0.0).is_err());
    assert_eq!(p.values().unwrap(), [2.0]);
    let moments = owner.moments().unwrap();
    assert_eq!(moments[&p.id()].first, [0.0]);
    assert_eq!(moments[&p.id()].second, [0.0]);
}

#[test]
fn lazy_device_weights_survive_owner_and_refresh_cached_inference_and_archives() {
    let mut named = tnsr::parameters::NamedParameters::default();
    named
        .insert("weight".into(), vec![1.0], vec![1], "policy".into())
        .unwrap();
    let p = named.get("weight", &[1]).unwrap();
    let Some(mut owner) = owner(vec![(p.clone(), 0)], vec![AdamGroup::new(0.01)]) else {
        return;
    };
    let prediction: TensorGraph<f32> = TensorExpr::from(p.clone()).into();
    let mut inference = owner.executor().unwrap();
    assert_eq!(
        inference
            .compile_and_execute(&prediction, HashMap::new())
            .unwrap(),
        [1.0]
    );
    let loss: TensorGraph<f32> = (TensorExpr::from(p.clone()) * TensorExpr::from(p.clone())).into();
    let output = *loss.toposort().last().unwrap();
    let loss = loss.with_gradients(output);
    let mut executor = owner.executor().unwrap();
    owner.begin_step().unwrap();
    owner
        .accumulate(&mut executor, &loss, HashMap::new())
        .unwrap();
    owner.step(0.0).unwrap();
    drop(owner);
    let directory =
        std::env::temp_dir().join(format!("tnsr-resident-archive-{}", std::process::id()));
    named
        .save(&directory, serde_json::json!({"resident":true}))
        .unwrap();
    let (restored, _) = tnsr::parameters::NamedParameters::load(&directory).unwrap();
    close(
        &restored.get("weight", &[1]).unwrap().values().unwrap(),
        &[0.99],
        1e-6,
    );
    close(
        &inference
            .compile_and_execute(&prediction, HashMap::new())
            .unwrap(),
        &[0.99],
        1e-6,
    );
    assert_eq!(inference.compilation_count(), 1);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn host_mutations_import_between_steps_and_conflict_during_accumulation() {
    let p = Parameter::new(vec![1.0], vec![1]);
    let Some(mut owner) = owner(vec![(p.clone(), 0)], vec![AdamGroup::new(0.1)]) else {
        return;
    };
    let graph: TensorGraph<f32> = TensorExpr::from(p.clone()).into();
    let output = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(output);
    let mut executor = owner.executor().unwrap();
    owner.begin_step().unwrap();
    owner
        .accumulate(&mut executor, &graph, HashMap::new())
        .unwrap();
    owner.step(0.0).unwrap();
    p.set_values(vec![5.0]).unwrap();
    owner.begin_step().unwrap();
    owner
        .accumulate(&mut executor, &graph, HashMap::new())
        .unwrap();
    owner.step(0.0).unwrap();
    close(&p.values().unwrap(), &[4.9], 1e-6);
    owner.begin_step().unwrap();
    p.set_values(vec![7.0]).unwrap();
    assert!(
        owner
            .accumulate(&mut executor, &graph, HashMap::new())
            .is_err()
    );
    assert_eq!(p.values().unwrap(), [7.0]);
}
