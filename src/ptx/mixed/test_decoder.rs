//! Synthetic parameters around the actual benchmark decoder topology.
#[allow(dead_code)]
#[path = "../../../examples/transformer_training_bench/model.rs"]
mod model;
pub(crate) use model::{Config, Model};

pub(crate) fn synthetic(config: Config) -> Model {
    use crate::parameters::NamedParameters;
    let mut parameters = NamedParameters::default();
    let mut records = Vec::new();
    let mut offset = 0;
    for (name, shape) in config.specs() {
        let count = shape.iter().product();
        let values = (0..count)
            .map(|i| {
                if name.ends_with("norm.weight") {
                    1.0
                } else if name.ends_with(".bias") {
                    0.0
                } else {
                    (i % 13) as f32 / 64.0 - 0.0625
                }
            })
            .collect();
        parameters
            .insert(name.clone(), values, shape.clone(), "train".into())
            .unwrap();
        records.push(model::TensorRecord {
            name,
            shape,
            offset,
            count,
            group: "train".into(),
        });
        offset += count;
    }
    let model = Model {
        config,
        parameters,
        records,
    };
    model.validate().unwrap();
    model
}
