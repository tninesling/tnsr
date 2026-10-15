#![cfg(feature = "cuda")]
use std::collections::HashMap;
use tnsr::graph::{TensorGraph, WithGrad};
use tnsr::ptx::resident::{AdamGroup, ResidentAdam};
use tnsr::tensor::{Input, Parameter, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

fn owner(parameters: Vec<(Parameter<f32>, usize)>, groups: Vec<AdamGroup>) -> Option<ResidentAdam> {
    match ResidentAdam::new(parameters, groups) {
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
fn resident_multistep_weighted_microbatches_match_cpu_gradients_moments_and_groups() {
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
    let Some(mut owner) = owner(
        vec![(w.clone(), 0), (b.clone(), 1), (w.clone(), 0)],
        groups.clone(),
    ) else {
        return;
    };
    let mut executors = [owner.executor().unwrap(), owner.executor().unwrap()];
    let graphs = [graph(&w, &b, 1, 1.0 / 3.0), graph(&w, &b, 2, 2.0 / 3.0)];
    let cpu_graphs = [graph(&cw, &cb, 1, 1.0 / 3.0), graph(&cw, &cb, 2, 2.0 / 3.0)];
    let mut moments = [vec![0.0f32; 6], vec![0.0f32; 3]];
    let mut variances = moments.clone();
    for step in 1..=3_i32 {
        if step == 3 {
            owner.set_learning_rate(0, 0.003).unwrap();
        }
        owner.begin_step().unwrap();
        let mut gradients = [vec![0.0f32; 6], vec![0.0f32; 3]];
        for (i, values) in [vec![0.25, -0.5], vec![0.75, 0.5, -0.25, 0.125]]
            .into_iter()
            .enumerate()
        {
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
                .unwrap();
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
            if step == 3 && group == 0 {
                settings.learning_rate = 0.003;
            }
            close(
                &observed[&device.id()],
                &gradient.iter().map(|g| g * scale).collect::<Vec<_>>(),
                1e-6,
            );
            for j in 0..values.len() {
                let g = gradient[j] * scale + settings.weight_decay * values[j];
                moments[group][j] = settings.beta1 * moments[group][j] + (1.0 - settings.beta1) * g;
                variances[group][j] =
                    settings.beta2 * variances[group][j] + (1.0 - settings.beta2) * g * g;
                let m = moments[group][j] / (1.0 - settings.beta1.powi(step));
                let v = variances[group][j] / (1.0 - settings.beta2.powi(step));
                values[j] -= settings.learning_rate * m / (v.sqrt() + settings.epsilon);
            }
            parameter.set_values(values.clone()).unwrap();
            close(&device.values().unwrap(), &values, 2e-6);
            close(&state[&device.id()].first, &moments[group], 1e-6);
            close(&state[&device.id()].second, &variances[group], 1e-6);
        }
    }
    drop(owner);
    close(&w.values().unwrap(), &cw.values().unwrap(), 2e-6);
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
