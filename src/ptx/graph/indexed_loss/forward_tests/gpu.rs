//! Direct exact ordered forward qualification.
//! Future qualification must additionally exercise real policy/cache/fallback
//! plumbing and full-model gates; this helper test alone is insufficient.
use super::*;
use anyhow::Result;
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use std::sync::Arc;
fn available() -> bool {
    let present = CudaContext::new(0).is_ok();
    assert!(
        present || std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
        "CUDA required"
    );
    present
}
fn build(stream: &Arc<CudaStream>, source: &str, cooperative: bool) -> Result<CudaFunction> {
    let module = stream.context().load_module(Ptx::from_src(source))?;
    Ok(module.load_function(if cooperative {
        "ce_forward_cooperative_proposal"
    } else {
        "ce_forward_ordered_control"
    })?)
}
fn same(a: f32, b: f32) {
    assert!(
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
        "{a:?} versus {b:?}"
    );
}
#[test]
fn proposed_forward_loss_bits_nonfinites_changed_targets_and_f64_diagnostics() -> Result<()> {
    if !available() {
        return Ok(());
    }
    let device = CudaContext::new(0)?;
    let stream = device.new_stream()?;
    let target = crate::ptx::PtxTarget::from_context(&device)?;
    let rows = 16usize;
    let extent = rows * VOCABULARY;
    for half in [false, true] {
        let dtype = if half { DType::BF16 } else { DType::F32 };
        let source = [
            render(false, dtype, DType::U32, target),
            render(true, dtype, DType::U32, target),
        ];
        let functions = [
            build(&stream, &source[0], false)?,
            build(&stream, &source[1], true)?,
        ];
        let mut logits32 = stream.alloc_zeros::<f32>(extent)?;
        let mut logits16 = stream.alloc_zeros::<half::bf16>(extent)?;
        let mut targets = stream.alloc_zeros::<u32>(rows)?;
        let mut output = [
            stream.alloc_zeros::<f32>(rows)?,
            stream.alloc_zeros::<f32>(rows)?,
        ];
        let mut diagnostics = Vec::new();
        for phase in 0..4 {
            let mut values = (0..extent)
                .map(|i| ((i + phase * 17) % 29) as f32 / 16. - 0.875)
                .collect::<Vec<_>>();
            values[..4096].fill(-18.42068);
            values[if phase == 0 { 0 } else { 4095 }] = 0.;
            for col in 0..4096 {
                values[4096 + col] = if col % 2 == 0 { 0. } else { -0. };
                values[2 * 4096 + col] = f32::from_bits(1 + col as u32 % 7);
            }
            if phase == 2 {
                values[3 * 4096..4 * 4096].fill(f32::NEG_INFINITY);
                values[4 * 4096] = f32::INFINITY;
                values[5 * 4096] = f32::NAN;
                values[6 * 4096..7 * 4096].fill(f32::NAN);
            }
            if half && phase == 3 {
                values = (0..=u16::MAX)
                    .map(|x| half::bf16::from_bits(x).to_f32())
                    .collect();
            }
            let pointer = if half {
                let packed = if phase == 3 {
                    (0..=u16::MAX)
                        .map(half::bf16::from_bits)
                        .collect::<Vec<_>>()
                } else {
                    values
                        .iter()
                        .map(|&x| half::bf16::from_f32(x))
                        .collect::<Vec<_>>()
                };
                for (v, p) in values.iter_mut().zip(&packed) {
                    *v = p.to_f32();
                }
                stream.memcpy_htod(&packed, &mut logits16)?;
                logits16.device_ptr(&stream).0
            } else {
                stream.memcpy_htod(&values, &mut logits32)?;
                logits32.device_ptr(&stream).0
            };
            let target_values = (0..rows)
                .map(|row| {
                    if row == 0 {
                        if phase == 0 { 0 } else { 4095 }
                    } else {
                        ((row + phase) * 683 % 4096) as u32
                    }
                })
                .collect::<Vec<_>>();
            stream.memcpy_htod(&target_values, &mut targets)?;
            for mode in 0..2 {
                let mut launch = stream.launch_builder(&functions[mode]);
                launch.arg(&pointer).arg(&targets).arg(&mut output[mode]);
                // SAFETY: distinct exact typed buffers, fixed full rows and the
                // matching old32/new256 launch, retained until synchronization.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (u32::try_from(rows)?, 1, 1),
                        block_dim: (if mode == 0 { 32 } else { FORWARD_THREADS }, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }?;
            }
            stream.synchronize()?;
            let old = stream.memcpy_dtov(&output[0])?;
            let new = stream.memcpy_dtov(&output[1])?;
            for (&a, &b) in old.iter().zip(&new) {
                same(a, b);
            }
            if phase == 0 {
                assert_eq!(new[0].to_bits(), 0, "ordered near-zero loss must remain +0");
            }
            if phase == 2 {
                for row in [3, 4, 5, 6] {
                    assert!(new[row].is_nan());
                }
            }
            let mut max_f64_error = 0f64;
            for row in 0..rows {
                let input = &values[row * 4096..(row + 1) * 4096];
                if input.iter().any(|v| !v.is_finite()) {
                    continue;
                }
                let maximum = input
                    .iter()
                    .copied()
                    .map(f64::from)
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum = input
                    .iter()
                    .map(|&v| (f64::from(v) - maximum).exp())
                    .sum::<f64>();
                let expected = (maximum - f64::from(input[target_values[row] as usize])) + sum.ln();
                max_f64_error = max_f64_error.max((f64::from(new[row]) - expected).abs());
            }
            diagnostics.push(serde_json::json!({"phase":phase,"old_loss_bits":old.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"new_loss_bits":new.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"f64_max_absolute_error":max_f64_error,"scope":"diagnostic only; required gate is original non-NaN bits and NaN classes"}));
        }
        if let Some(dir) = std::env::var_os("TNSR_CE_ORDERED_FORWARD_PTX_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            for (mode, source) in source.iter().enumerate() {
                std::fs::write(dir.join(format!("half{half}-{mode}.ptx")), source)?;
            }
            let resources=functions.iter().enumerate().map(|(mode,f)|Ok(serde_json::json!({"mode":mode,"threads":if mode==0 {32}else{256},"registers":f.num_regs()?,"shared":f.shared_size_bytes()?,"local":f.local_size_bytes()?,"capacity":f.occupancy_max_active_blocks_per_multiprocessor(if mode==0 {32}else{256},0,None)?}))).collect::<Result<Vec<_>>>()?;
            std::fs::write(
                dir.join(format!("half{half}-audit.json")),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"resources":resources,"cases":diagnostics,"scope":"unprofiled correctness/resources only; capacity is not achieved occupancy"}),
                )?,
            )?;
        }
    }
    Ok(())
}
