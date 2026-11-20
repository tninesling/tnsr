//! Native half of a separate native-vs-standalone-Torch component experiment.
//! No Python/framework code is called from project execution.
use super::*;
use std::io::{BufWriter, Write};

fn values(count: usize, modulus: usize, multiplier: usize) -> Vec<bf16> {
    (0..count)
        .map(|i| {
            bf16::from_f32(
                ((i % modulus * multiplier % modulus) as f32 - (modulus / 2) as f32)
                    / modulus as f32,
            )
        })
        .collect()
}
fn write_bf16(path: &Path, values: &[bf16]) -> Result<()> {
    let mut writer = BufWriter::new(std::fs::File::create(path)?);
    for chunk in values.chunks(8192) {
        let bytes = chunk
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        writer.write_all(&bytes)?;
    }
    writer.flush()?;
    Ok(())
}
fn read_bf16(path: &Path, count: usize) -> Result<Vec<bf16>> {
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len() == count.checked_mul(2).context("BF16 byte extent overflow")?,
        "BF16 archive extent mismatch"
    );
    Ok(bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect())
}
fn sanity(case: Case, a: &[bf16], b: &[bf16], output: Option<&[bf16]>) -> Result<Value> {
    let mut rows = Vec::new();
    for sample in 0..32 {
        let (row, col) = (sample * 7919 % case.m, sample * 104729 % case.n);
        let reference = (0..case.k)
            .map(|j| {
                f64::from(a[row * case.k + j].to_f32()) * f64::from(b[j * case.n + col].to_f32())
            })
            .sum::<f64>();
        let tolerance = 0.01 * (1. + reference.abs());
        let actual = output.map(|v| f64::from(v[row * case.n + col].to_f32()));
        if let Some(value) = actual {
            ensure!(
                (value - reference).abs() <= tolerance,
                "F64 sanity failed at{row},{col}"
            );
        }
        rows.push(json!({"row":row,"column":col,"reference_f64":reference,"tolerance":tolerance,"actual":actual}));
    }
    Ok(json!(rows))
}
fn classes(values: &[bf16]) -> Value {
    let mut non_nan = std::collections::BTreeMap::<u16, usize>::new();
    let mut nan = 0;
    for &v in values {
        if v.is_nan() {
            nan += 1;
        } else {
            *non_nan.entry(v.to_bits()).or_default() += 1;
        }
    }
    let non_nan = non_nan
        .into_iter()
        .map(|(bits, count)| (format!("{bits:04x}"), count))
        .collect::<std::collections::BTreeMap<_, _>>();
    json!({"elements":values.len(),"nan_elements":nan,"non_nan_bit_counts":non_nan})
}

#[test]
#[ignore = "manual deterministic BF16 byte archive export for component probes"]
fn native_bf16_mm_export_inputs() -> Result<()> {
    let dir = PathBuf::from(
        std::env::var_os("TNSR_GEMM_COMPONENT_INPUTS").context("set input archive directory")?,
    );
    std::fs::create_dir_all(&dir)?;
    ensure!(
        !dir.join("manifest.json").exists(),
        "input archive already exists; preserve it"
    );
    let mut cases = Vec::new();
    for case in CASES {
        let folder = dir.join(case.name);
        std::fs::create_dir_all(&folder)?;
        let a = values(case.m * case.k, 97, 13);
        let b = values(case.k * case.n, 89, 17);
        write_bf16(&folder.join("a.bf16"), &a)?;
        write_bf16(&folder.join("b.bf16"), &b)?;
        cases.push(json!({"name":case.name,"m":case.m,"n":case.n,"k":case.k,"a_file":format!("{}/a.bf16",case.name),"b_file":format!("{}/b.bf16",case.name),"a_shape":[case.m,case.k],"b_shape":[case.k,case.n],"output_shape":[case.m,case.n],"orientation":"NN","sampled_f64_reference":sanity(case,&a,&b,None)?}));
    }
    write_json(
        &dir.join("manifest.json"),
        &json!({"version":1,"dtype":"bf16-le","generator":"bf16::from_f32(((i%modulus*multiplier%modulus)-(modulus/2))/modulus), float32 division; A(97,13),B(89,17)","cases":cases}),
    )
}

#[test]
#[ignore = "manual native64² resident BF16 GEMM component probe; no production policy change"]
fn native64_bf16_mm_component() -> Result<()> {
    let inputs = PathBuf::from(
        std::env::var_os("TNSR_GEMM_COMPONENT_INPUTS").context("set input archive directory")?,
    );
    let output = PathBuf::from(
        std::env::var_os("TNSR_GEMM_COMPONENT_OUTPUT").context("set output directory")?,
    );
    let timing = std::env::var("TNSR_GEMM_COMPONENT_TIMING").as_deref() == Ok("1");
    let reverse = std::env::var("TNSR_GEMM_COMPONENT_REVERSE").as_deref() == Ok("1");
    std::fs::create_dir_all(&output)?;
    ensure!(
        !output.join("report.json").exists(),
        "preserve existing component report"
    );
    let manifest: Value = serde_json::from_slice(&std::fs::read(inputs.join("manifest.json"))?)?;
    ensure!(manifest["dtype"] == "bf16-le", "BF16 archive required");
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let mut cases = CASES.to_vec();
    if reverse {
        cases.reverse();
    }
    let repeats = 32;
    let samples = 32;
    let mut results = Vec::new();
    for case in cases {
        let dir = output.join(case.name);
        std::fs::create_dir_all(&dir)?;
        let a = read_bf16(&inputs.join(case.name).join("a.bf16"), case.m * case.k)?;
        let b = read_bf16(&inputs.join(case.name).join("b.bf16"), case.k * case.n)?;
        let mut mode = build(case, false, &stream, &a, &b, &dir)?;
        mode.submit(&stream)?;
        let expected = mode.read(&stream)?;
        ensure!(
            expected.iter().all(|x| x.is_finite()),
            "finite output required"
        );
        let f64_checks = sanity(case, &a, &b, Some(&expected))?;
        mode.submit(&stream)?;
        compare(&expected, &mode.read(&stream)?, "native repeated finite")?;
        let mut special = Vec::new();
        for (name, left, right) in [
            ("positive_zero", 0., 1.),
            ("negative_zero", -0., 1.),
            ("positive_infinity", 1., f32::INFINITY),
            ("zero_times_infinity", 0., f32::INFINITY),
            ("nan", f32::NAN, 1.),
        ] {
            mode.upload(
                &stream,
                &vec![bf16::from_f32(left); a.len()],
                &vec![bf16::from_f32(right); b.len()],
            )?;
            mode.submit(&stream)?;
            special.push(json!({"phase":name,"classification":classes(&mode.read(&stream)?)}));
        }
        mode.upload(&stream, &a, &b)?;
        mode.submit(&stream)?;
        compare(&expected, &mode.read(&stream)?, "native finite restoration")?;
        let mut warmups = 0;
        let mut warmup_ms = 0.;
        if timing {
            for _ in 0..10 {
                mode.submit(&stream)?;
            }
            mode.capture(&stream, repeats)?;
            let began = Instant::now();
            while began.elapsed() < Duration::from_millis(500) {
                mode.capture.as_mut().unwrap().launch()?;
                stream.synchronize()?;
                warmups += 1;
            }
            warmup_ms = began.elapsed().as_secs_f64() * 1000.;
            for _ in 0..samples {
                let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                mode.capture.as_mut().unwrap().launch()?;
                let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                let us = f64::from(start.elapsed_ms(&end)?) * 1000. / repeats as f64;
                ensure!(us.is_finite() && us > 0., "invalid event interval");
                mode.samples_us.push(us);
            }
            stream.synchronize()?;
            compare(&expected, &mode.read(&stream)?, "native post-replay finite")?;
        }
        write_bf16(&dir.join("finite.bf16"), &expected)?;
        let result = json!({"case":case.name,"m":case.m,"n":case.n,"k":case.k,"dtype":"BF16","orientation":"NN","a_file":inputs.join(case.name).join("a.bf16"),"b_file":inputs.join(case.name).join("b.bf16"),"output_file":"finite.bf16","output_elements":expected.len(),"finite":true,"repeated_bits_equal":true,"post_replay_bits_equal":timing.then_some(true),"sampled_f64_sanity":f64_checks,"special_values":special,"timing_performed":timing,"samples_us":mode.samples_us,"median_us":if timing{Some(median(&mode.samples_us))}else{None},"mean_us":if timing{Some(mode.samples_us.iter().sum::<f64>()/samples as f64)}else{None},"resources":mode.resources,"prewarm_graphs":warmups,"prewarm_wall_ms":warmup_ms,"graph_gemm_repeats":repeats,"samples":if timing{samples}else{0},"scope":"Resident accepted native64² single GEMM; no Square candidate or Torch execution in project code; setup/uploads/validation/readback excluded from CUDA events"});
        write_json(&dir.join("result.json"), &result)?;
        results.push(result);
        write_json(
            &output.join("report.json"),
            &json!({"backend":"native64","case_order":if reverse{"reverse"}else{"forward"},"cases":results,"precision":"BF16 storage and operands, F32 WMMA accumulation, one BF16 RN output; no TF32 arithmetic","protocol":"32 captured GEMM repeats,>=500ms synchronized prewarm,32 CUDAevent samples percase"}),
        )?;
    }
    Ok(())
}
