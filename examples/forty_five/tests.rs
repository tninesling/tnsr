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
