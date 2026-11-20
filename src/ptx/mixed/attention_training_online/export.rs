//! Optional portable leaf snapshots; outside CUDA capture and every timing ROI.
use super::leaf::Shape;
use anyhow::Result;
use half::bf16;
use std::io::{BufWriter, Write};

pub(super) fn revision(
    shape: Shape,
    revision: usize,
    inputs: &[Vec<bf16>; 4],
    outputs: &[Vec<bf16>; 4],
    statistics: &[f32],
) -> Result<()> {
    let Some(root) = std::env::var_os("TNSR_ONLINE_EXPORT") else {
        return Ok(());
    };
    std::fs::create_dir_all(&root)?;
    let directory = std::path::Path::new(&root).join(format!("revision-{revision}"));
    // Snapshots are immutable: a repeated run must select a new export root.
    std::fs::create_dir(&directory)?;
    let names = [
        "query", "key", "value", "incoming", "output", "d_query", "d_key", "d_value",
    ];
    let mut records = Vec::new();
    for (index, name) in names.iter().enumerate() {
        let source = if index < 4 {
            &inputs[index]
        } else {
            &outputs[index - 4]
        };
        let filename = format!("{name}.bf16");
        let mut file = BufWriter::new(std::fs::File::create(directory.join(&filename))?);
        // GPU dKT is dense [BH,D,S]; portable snapshots are canonical [B,H,S,D].
        for logical in 0..shape.elements() {
            let physical = if index == 6 {
                let feature = logical % 64;
                let row = logical / 64 % 512;
                let head = logical / (512 * 64);
                (head * 64 + feature) * 512 + row
            } else {
                logical
            };
            file.write_all(&source[physical].to_bits().to_le_bytes())?;
        }
        file.flush()?;
        records.push(serde_json::json!({"name":name,"file":filename,"shape":[shape.batch,shape.heads,512,64],"count":shape.elements(),"dtype":"bf16-le","layout":"canonical BHSD"}));
    }
    let mut stats = BufWriter::new(std::fs::File::create(directory.join("statistics.f32"))?);
    for value in statistics {
        stats.write_all(&value.to_le_bytes())?;
    }
    stats.flush()?;
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "version":1,"revision":revision,"shape":[shape.batch,shape.heads,512,64],"tensors":records,
            "statistics":{"file":"statistics.f32","shape":[shape.bh(),512,3],"dtype":"f32-le","fields":["maximum","denominator","delta"]},
            "native_policy":"F32 QK; unnormalized BF16 exponential PV with F32 online rescale; O16; delta=sum(O16*dO16); recomputed normalized P16/DS16 backward; div.approx.f32; explicit RN add/sub/mul",
            "scope":"disconnected paired-leaf snapshot after full replay checks; no full-model gate or timing claim"
        }))?,
    )?;
    Ok(())
}
