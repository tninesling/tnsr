use super::*;
fn check(backend: &str) {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/forty_five/fixtures/tiny");
    let model = Model::load(&directory).unwrap();
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("reference.json")).unwrap()).unwrap();
    let batch: Observations = serde_json::from_value(reference["batch"].clone()).unwrap();
    let mut worker = Worker {
        model,
        backend: backend.into(),
        predictions: HashMap::new(),
        trainer: None,
    };
    let actual = worker.infer(&batch).unwrap();
    for field in ["logits", "values"] {
        let flatten = |value: &serde_json::Value| -> Vec<f32> {
            if field == "logits" {
                value
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| {
                        row.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap() as f32)
                    })
                    .collect()
            } else {
                value
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            }
        };
        for (a, b) in flatten(&actual[field])
            .iter()
            .zip(flatten(&reference[field]))
        {
            assert!((a - b).abs() < 2e-5, "{field}: {a} != {b}");
        }
    }
    assert_eq!(actual, worker.infer(&batch).unwrap());
    assert_eq!(worker.predictions.len(), 1);
}
#[test]
fn cpu_forty_five_model_parity() {
    check("cpu");
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_forty_five_model_parity() {
    if let Err(error) = runtime("cuda") {
        assert_ne!(
            std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
            Ok("1"),
            "{error:#}"
        );
        return;
    }
    check("cuda");
}

fn check_ppo(backend: &str, fixture: &str, tree_reductions: bool) {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/forty_five/fixtures")
        .join(fixture);
    let model = Model::load(&directory).unwrap();
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("ppo.json")).unwrap()).unwrap();
    let batch = serde_json::from_value(reference["batch"].clone()).unwrap();
    let mut settings: forty_five::ppo::Settings =
        serde_json::from_value(reference["settings"].clone()).unwrap();
    settings.tree_reductions = tree_reductions;
    let mut trainer = forty_five::ppo::Trainer::new(&settings, backend.into()).unwrap();
    for (step, expected) in reference["steps"].as_array().unwrap().iter().enumerate() {
        let actual = trainer.step(&model, &batch, &settings, true).unwrap();
        for (key, value) in expected["metrics"].as_object().unwrap() {
            let a = actual["metrics"][key].as_f64().unwrap();
            let b = value.as_f64().unwrap();
            assert!((a - b).abs() < 3e-5, "{key}: {a} != {b}");
        }
        for (name, entry) in model.parameters.iter() {
            for field in ["gradients", "weights"] {
                let observed: Vec<f32> = if field == "gradients" {
                    serde_json::from_value(actual[field][name].clone()).unwrap()
                } else {
                    entry.parameter.values().unwrap()
                };
                let expected: Vec<f32> =
                    serde_json::from_value(expected[field][name].clone()).unwrap();
                assert_eq!(observed.len(), expected.len());
                for (a, b) in observed.iter().zip(expected) {
                    assert!(
                        (a - b).abs() < 2e-6,
                        "step {step} {name} {field}: {a} != {b}"
                    );
                }
            }
        }
        assert_eq!(actual["compiled_shapes"], 2);
        let mut changed = settings.clone();
        changed.tree_reductions = !tree_reductions;
        assert!(trainer.step(&model, &batch, &changed, false).is_err());
        let rates = trainer.advance_schedule(step + 1).unwrap();
        for (index, key) in ["policy_learning_rate", "value_learning_rate"]
            .iter()
            .enumerate()
        {
            assert!(
                (rates[key].as_f64().unwrap()
                    - expected["learning_rates"][index].as_f64().unwrap())
                .abs()
                    < 1e-10
            );
        }
    }
}
#[test]
fn cpu_forty_five_ppo_parity() {
    check_ppo("cpu", "tiny", false);
    check_ppo("cpu", "tiny-separate", false);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_forty_five_ppo_parity() {
    if let Err(error) = runtime("cuda") {
        assert_ne!(
            std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
            Ok("1"),
            "{error:#}"
        );
        return;
    }
    check_ppo("cuda", "tiny", false);
    check_ppo("cuda", "tiny-separate", false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_forty_five_tree_ppo_parity() {
    if let Err(error) = runtime("cuda") {
        assert_ne!(
            std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
            Ok("1"),
            "{error:#}"
        );
        return;
    }
    check_ppo("cuda", "tiny", true);
    check_ppo("cuda", "tiny-separate", true);
}

#[test]
fn cpu_ppo_rejects_tree_policy() {
    let settings = forty_five::ppo::Settings {
        tree_reductions: true,
        ..Default::default()
    };
    assert!(forty_five::ppo::Trainer::new(&settings, "cpu".into()).is_err());
}

#[test]
fn separate_critic_gradient_isolation() {
    let directory =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/forty_five/fixtures/tiny-separate");
    let model = Model::load(&directory).unwrap();
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("ppo.json")).unwrap()).unwrap();
    let batch: forty_five::ppo::Batch = serde_json::from_value(reference["batch"].clone()).unwrap();
    for value_only in [false, true] {
        let (logits, values) = model.forward(batch.observations.token_ids.len()).unwrap();
        let loss = if value_only {
            values.mean_all()
        } else {
            logits.mean_all()
        };
        let graph: TensorGraph<f32> = loss.into();
        let output = *graph.toposort().last().unwrap();
        let graph = graph.with_gradients(output);
        let mut executor = SimpleExecutor::new();
        executor
            .execute(&graph, batch.observations.inputs(&model.config).unwrap())
            .unwrap();
        let gradients = executor.get_gradients(&graph);
        let mut active = false;
        for (name, entry) in model.parameters.iter() {
            if let Some(gradient) = gradients.get(&entry.parameter.id()) {
                assert!(gradient.iter().all(|x| x.is_finite()), "{name}");
                if (entry.group == "value") != value_only {
                    assert!(
                        gradient.iter().all(|x| *x == 0.),
                        "gradient leaked into {name}"
                    );
                } else {
                    active |= gradient.iter().any(|x| *x != 0.);
                }
            }
        }
        assert!(active);
    }
}
