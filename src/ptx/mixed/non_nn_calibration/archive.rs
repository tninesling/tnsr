use super::*;
use std::io::{BufWriter, Write};
#[derive(Clone, Copy)]
pub(super) struct Case {
    pub name: &'static str,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub split: bool,
}
pub(super) const CASES: [Case; 3] = [
    Case {
        name: "ffn_forward_nt",
        m: 16384,
        n: 1408,
        k: 512,
        split: false,
    },
    Case {
        name: "qkv_forward_nt",
        m: 16384,
        n: 1536,
        k: 512,
        split: false,
    },
    Case {
        name: "ffn_weight_tn_split4",
        m: 512,
        n: 1408,
        k: 16384,
        split: true,
    },
];
pub(super) fn values(len: usize, modulus: usize, multiplier: usize) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            bf16::from_f32(
                ((i % modulus * multiplier % modulus) as f32 - (modulus / 2) as f32)
                    / modulus as f32,
            )
        })
        .collect()
}
pub(super) fn write16(path: &Path, data: &[bf16]) -> Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    for c in data.chunks(8192) {
        let bytes = c
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        f.write_all(&bytes)?;
    }
    f.flush()?;
    Ok(())
}
pub(super) fn write32(path: &Path, data: &[f32]) -> Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    for c in data.chunks(8192) {
        let bytes = c.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        f.write_all(&bytes)?;
    }
    f.flush()?;
    Ok(())
}
pub(super) fn read16(path: &Path, len: usize) -> Result<Vec<bf16>> {
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len() == len * 2,
        "archive extent mismatch {}",
        path.display()
    );
    Ok(bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect())
}
pub(super) fn json(path: &Path, value: &serde_json::Value) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
pub(super) fn samples(case: Case, a: &[bf16], b: &[bf16], bias: &[bf16]) -> serde_json::Value {
    let mut output = Vec::new();
    let mut parts = Vec::new();
    for sample in 0..32 {
        let r = sample * 7919 % case.m;
        let c = sample * 104729 % case.n;
        let mut full = 0f64;
        for part in 0..if case.split { 4 } else { 1 } {
            let lo = part * case.k / if case.split { 4 } else { 1 };
            let hi = (part + 1) * case.k / if case.split { 4 } else { 1 };
            let mut dot = 0f64;
            for j in lo..hi {
                let x = if case.split {
                    a[j * case.m + r]
                } else {
                    a[r * case.k + j]
                };
                let y = if case.split {
                    b[j * case.n + c]
                } else {
                    b[c * case.k + j]
                };
                dot += f64::from(x.to_f32()) * f64::from(y.to_f32());
            }
            if case.split {
                parts.push(serde_json::json!({"row":r,"column":c,"part":part,"k_start":lo,"k_end":hi,"reference_f64":dot}));
            }
            full += dot;
        }
        if !case.split {
            full += f64::from(bias[c].to_f32());
        }
        output.push(serde_json::json!({"row":r,"column":c,"reference_f64":full}));
    }
    serde_json::json!({"output":output,"parts":parts,"scope":"Independent widened-BF16 F64 products and bias; sampled mathematical oracle, not F32MMA-order bit oracle or a fitted fullmodel threshold"})
}
#[test]
#[ignore = "manual CPU BF16 archive export for nonNN actual-path calibration"]
fn non_nn_calibration_export_archive() -> Result<()> {
    let dir = PathBuf::from(
        std::env::var_os("TNSR_NON_NN_INPUTS").context("set immutable input archive directory")?,
    );
    std::fs::create_dir_all(&dir)?;
    ensure!(
        !dir.join("manifest.json").exists(),
        "preserve existing input archive"
    );
    let mut cases = Vec::new();
    for c in CASES {
        let d = dir.join(c.name);
        std::fs::create_dir_all(&d)?;
        let a = values(c.m * c.k, 97, 13);
        let b = values(c.k * c.n, 89, 17);
        let bias = values(if c.split { 0 } else { c.n }, 31, 7);
        write16(&d.join("a.bf16"), &a)?;
        write16(&d.join("b.bf16"), &b)?;
        write16(&d.join("bias.bf16"), &bias)?;
        cases.push(serde_json::json!({"name":c.name,"m":c.m,"n":c.n,"k":c.k,"kind":if c.split {"tn_split4"}else{"nt_fused_linear"},"a_shape":if c.split {vec![c.k,c.m]}else{vec![c.m,c.k]},"b_shape":if c.split {vec![c.k,c.n]}else{vec![c.n,c.k]},"output_shape":[c.m,c.n],"bias_shape":if c.split {vec![]}else{vec![c.n]},"sampled_oracle":samples(c,&a,&b,&bias)}));
    }
    json(
        &dir.join("manifest.json"),
        &serde_json::json!({"schema_version":1,"dtype":"bf16-le","generator":"Rust F32 periodic values then bf16RN: A97/13,B89/17,bias31/7; synthetic hardware calibration, not realmodel activations","cases":cases}),
    )
}

#[test]
fn non_nn_calibration_f64_oracle_uses_physical_nt_tn_and_exact_partition_ranges() {
    // Integer products are exactly representable. These expected values are
    // independently hand-calculated, including the four contiguous2-key dots.
    let partial_rows = [[7., 10.], [43., 50.], [111., 122.], [211., 226.]];
    for split in [false, true] {
        let c = Case {
            name: "tiny",
            m: 2,
            n: 2,
            k: 8,
            split,
        };
        let a: Vec<bf16> = if split {
            (1..=16).map(|x| bf16::from_f32(x as f32)).collect()
        } else {
            (0..2)
                .flat_map(|r| (0..8).map(move |j| bf16::from_f32((2 * j + r + 1) as f32)))
                .collect()
        };
        let b: Vec<bf16> = if split {
            (1..=8)
                .flat_map(|t| [bf16::from_f32(t as f32), bf16::from_f32(-(t as f32))])
                .collect()
        } else {
            (0..2)
                .flat_map(|r| {
                    (1..=8)
                        .map(move |t| bf16::from_f32(if r == 0 { t as f32 } else { -(t as f32) }))
                })
                .collect()
        };
        let bias = if split {
            vec![]
        } else {
            vec![bf16::from_f32(0.5), bf16::from_f32(-0.5)]
        };
        let oracle = samples(c, &a, &b, &bias);
        for sample in oracle["output"].as_array().unwrap() {
            let r = sample["row"].as_u64().unwrap() as usize;
            let col = sample["column"].as_u64().unwrap() as usize;
            let expected = if r == 0 { 372. } else { 408. } + if split { 0. } else { 0.5 };
            assert_eq!(
                sample["reference_f64"].as_f64().unwrap(),
                if col == 0 { expected } else { -expected }
            );
        }
        if split {
            assert_eq!(oracle["parts"].as_array().unwrap().len(), 128);
            for sample in oracle["parts"].as_array().unwrap() {
                let p = sample["part"].as_u64().unwrap() as usize;
                let r = sample["row"].as_u64().unwrap() as usize;
                let col = sample["column"].as_u64().unwrap() as usize;
                assert_eq!(sample["k_start"].as_u64().unwrap(), p as u64 * 2);
                assert_eq!(sample["k_end"].as_u64().unwrap(), (p as u64 + 1) * 2);
                assert_eq!(
                    sample["reference_f64"].as_f64().unwrap(),
                    if col == 0 {
                        partial_rows[p][r]
                    } else {
                        -partial_rows[p][r]
                    }
                );
            }
        }
    }
}
