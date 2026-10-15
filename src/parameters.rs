//! Named f32 parameters and a portable, validated little-endian tensor archive.
use crate::tensor::{Parameter, Shape};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Component, Path};

#[derive(Clone)]
pub struct NamedParameter {
    pub parameter: Parameter<f32>,
    pub group: String,
}
#[derive(Default, Clone)]
pub struct NamedParameters {
    entries: BTreeMap<String, NamedParameter>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorRecord {
    name: String,
    shape: Shape,
    offset: usize,
    count: usize,
    group: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    dtype: String,
    data_file: String,
    config: serde_json::Value,
    tensors: Vec<TensorRecord>,
}
fn count(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim))
        .context("parameter shape overflows usize")
}
impl NamedParameters {
    pub fn insert(
        &mut self,
        name: String,
        values: Vec<f32>,
        shape: Shape,
        group: String,
    ) -> Result<()> {
        ensure!(
            !name.is_empty() && !self.entries.contains_key(&name),
            "empty or duplicate parameter name: {name}"
        );
        ensure!(
            values.len() == count(&shape)?,
            "{name}: parameter length disagrees with shape"
        );
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "{name}: nonfinite parameter value"
        );
        ensure!(!group.is_empty(), "{name}: parameter group is empty");
        self.entries.insert(
            name,
            NamedParameter {
                parameter: Parameter::new(values, shape),
                group,
            },
        );
        Ok(())
    }
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Parameter<f32>> {
        let entry = self
            .entries
            .get(name)
            .with_context(|| format!("missing parameter {name}"))?;
        ensure!(
            entry.parameter.shape() == shape,
            "{name}: expected shape {shape:?}, got {:?}",
            entry.parameter.shape()
        );
        Ok(entry.parameter.clone())
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &NamedParameter)> {
        self.entries
            .iter()
            .map(|(name, parameter)| (name.as_str(), parameter))
    }
    pub fn validate_names(&self, expected: &HashSet<String>) -> Result<()> {
        let actual: HashSet<_> = self.entries.keys().cloned().collect();
        ensure!(
            actual == *expected,
            "parameter keys mismatch; missing: {:?}; unexpected: {:?}",
            expected.difference(&actual).collect::<Vec<_>>(),
            actual.difference(expected).collect::<Vec<_>>()
        );
        Ok(())
    }
    pub fn load(directory: impl AsRef<Path>) -> Result<(Self, serde_json::Value)> {
        let directory = directory.as_ref();
        let manifest: Manifest = serde_json::from_slice(
            &fs::read(directory.join("manifest.json")).context("read parameter manifest")?,
        )
        .context("parse parameter manifest")?;
        ensure!(
            manifest.version == 1 && manifest.dtype == "f32-le",
            "unsupported parameter archive version/dtype"
        );
        let components: Vec<_> = Path::new(&manifest.data_file).components().collect();
        ensure!(
            components.len() == 1 && matches!(components[0], Component::Normal(_)),
            "archive data_file must be a filename"
        );
        let bytes = fs::read(directory.join(&manifest.data_file)).context("read parameter data")?;
        ensure!(bytes.len() % 4 == 0, "truncated f32 parameter data");
        let mut parameters = Self::default();
        let mut cursor = 0;
        for tensor in manifest.tensors {
            ensure!(
                tensor.offset == cursor,
                "{}: noncontiguous or overlapping parameter offset",
                tensor.name
            );
            ensure!(
                tensor.count == count(&tensor.shape)?,
                "{}: invalid parameter count",
                tensor.name
            );
            let end = cursor
                .checked_add(tensor.count)
                .context("parameter offset overflow")?;
            ensure!(
                end <= bytes.len() / 4,
                "{}: parameter data out of bounds",
                tensor.name
            );
            let values = bytes[cursor * 4..end * 4]
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect();
            parameters.insert(tensor.name, values, tensor.shape, tensor.group)?;
            cursor = end;
        }
        ensure!(
            cursor == bytes.len() / 4,
            "unreferenced trailing parameter data"
        );
        Ok((parameters, manifest.config))
    }
    pub fn save(&self, directory: impl AsRef<Path>, config: serde_json::Value) -> Result<()> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory).context("create parameter archive directory")?;
        let mut writer = BufWriter::new(
            File::create(directory.join("weights.f32.tmp")).context("create parameter data")?,
        );
        let mut tensors = Vec::new();
        let mut offset = 0;
        for (name, entry) in self.iter() {
            let values = entry.parameter.values()?;
            ensure!(
                values.iter().all(|x| x.is_finite()),
                "{name}: refusing to save nonfinite parameters"
            );
            for value in &values {
                writer
                    .write_all(&value.to_le_bytes())
                    .context("write parameter data")?;
            }
            tensors.push(TensorRecord {
                name: name.into(),
                shape: entry.parameter.shape().clone(),
                offset,
                count: values.len(),
                group: entry.group.clone(),
            });
            offset += values.len();
        }
        writer.flush().context("flush parameter data")?;
        fs::rename(
            directory.join("weights.f32.tmp"),
            directory.join("weights.f32"),
        )
        .context("install parameter data")?;
        let manifest = Manifest {
            version: 1,
            dtype: "f32-le".into(),
            data_file: "weights.f32".into(),
            config,
            tensors,
        };
        fs::write(
            directory.join("manifest.json.tmp"),
            serde_json::to_vec_pretty(&manifest)?,
        )
        .context("write parameter manifest")?;
        fs::rename(
            directory.join("manifest.json.tmp"),
            directory.join("manifest.json"),
        )
        .context("install parameter manifest")?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TensorGraph;
    use crate::tensor::TensorExpr;
    use crate::{Executor, SimpleExecutor};
    use std::collections::HashMap;
    #[test]
    fn named_parameter_roundtrip_and_live_updates() {
        let mut parameters = NamedParameters::default();
        parameters
            .insert("weight".into(), vec![1., 2.], vec![2], "policy".into())
            .unwrap();
        let parameter = parameters.get("weight", &[2]).unwrap();
        let graph: TensorGraph<f32> = TensorExpr::from(parameter.clone()).into();
        parameter.set_values(vec![3., 4.]).unwrap();
        assert_eq!(
            SimpleExecutor::new()
                .execute(&graph, HashMap::new())
                .unwrap(),
            vec![3., 4.]
        );
        let directory =
            std::env::temp_dir().join(format!("tnsr-parameters-{}", std::process::id()));
        parameters
            .save(&directory, serde_json::json!({"model":"test"}))
            .unwrap();
        let (loaded, config) = NamedParameters::load(&directory).unwrap();
        assert_eq!(config["model"], "test");
        assert_eq!(
            loaded.get("weight", &[2]).unwrap().values().unwrap(),
            vec![3., 4.]
        );
        assert!(loaded.get("weight", &[3]).is_err());
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap()).unwrap();
        manifest["tensors"][0]["offset"] = 1.into();
        fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(NamedParameters::load(&directory).is_err());
        fs::remove_dir_all(directory).unwrap();
        assert!(parameter.set_values(vec![1.]).is_err());
    }
}
