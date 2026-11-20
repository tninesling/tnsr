use super::*;
use crate::ptx::{
    PtxTarget,
    linear::{
        Binding, Buffer,
        program::{Program, ProgramKernelOptions, binding},
    },
};
use anyhow::Result;
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use std::{collections::HashMap, sync::Arc};
fn available() -> bool {
    let ok = CudaContext::new(0).is_ok();
    assert!(
        ok || std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
        "CUDA required"
    );
    ok
}
fn bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
            "index{i}: {a:?}/{b:?}"
        );
    }
}
fn probe_source(cooperative: bool, dtype: DType, target: PtxTarget) -> String {
    let arena = bumpalo::Bump::new();
    let name = if cooperative {
        "ce_stats_cooperative_probe"
    } else {
        "ce_stats_ordered_probe"
    };
    let mut f = Function::new(name, &arena);
    let mut ctx = LoweringContext::new(&arena);
    for (name, dtype) in [
        ("logits", dtype),
        ("maximum", DType::F32),
        ("sum", DType::F32),
        ("exponentials", DType::F32),
    ] {
        let ptr = f.add_global_ptr_param(arena.alloc_str(name));
        ctx.param_ptrs.insert(name.into(), ptr);
        ctx.param_dtypes.insert(name.into(), dtype);
    }
    let row = lower_expr(&mut f, &ctx, &Expr::BlockIdx(Dim::X));
    let lane = lower_expr(&mut f, &ctx, &Expr::ThreadIdx(Dim::X));
    let row_start = multiply_u64(&mut f, row.clone(), 4096);
    let (maximum, sum, exps) = if cooperative {
        let s = cooperative_ordered::row_stats_ordered_sum(&mut f, &mut ctx, row_start.clone());
        (s.maximum, s.sum, Some(s.exponentials))
    } else {
        let name = arena.alloc_str("old_stats");
        f.add_shared_memory(name, 8);
        let base = f.add_u64_register();
        f.add_inst(Inst::mov_u64(base.clone(), Operand::symbol(name)));
        let sum_addr = f.add_u64_register();
        f.add_inst(Inst::add_u64(
            sum_addr.clone(),
            base.clone(),
            Operand::imm_u64(4),
        ));
        let done = next_label(&mut ctx, "old_probe_broadcast");
        let follower = f.add_predicate_register();
        f.add_inst(Inst::setp_ne_u64(
            follower.clone(),
            lane.clone(),
            Operand::imm_u64(0),
        ));
        f.add_inst(Inst::Bra {
            condition: follower,
            target: done,
        });
        let (m, s) = emit_row_softmax_stats(&mut f, &mut ctx, row_start.clone(), 4096);
        f.add_inst(Inst::StSharedF32 {
            addr: base.clone(),
            src: vec![m],
            vec: crate::ptx::instructions::VecWidth::Scalar,
        });
        f.add_inst(Inst::StSharedF32 {
            addr: sum_addr.clone(),
            src: vec![s],
            vec: crate::ptx::instructions::VecWidth::Scalar,
        });
        f.add_inst(Inst::Label(done));
        f.add_inst(Inst::BarSync { barrier_id: 0 });
        let m = f.add_f32_register();
        let s = f.add_f32_register();
        f.add_inst(Inst::LdSharedF32 {
            dst: vec![m.clone()],
            addr: base,
            vec: crate::ptx::instructions::VecWidth::Scalar,
        });
        f.add_inst(Inst::LdSharedF32 {
            dst: vec![s.clone()],
            addr: sum_addr,
            vec: crate::ptx::instructions::VecWidth::Scalar,
        });
        (m, s, None)
    };
    let columns = begin_strided_counted_loop(
        &mut f,
        &mut ctx,
        "probe_exp_output",
        4096,
        lane.clone(),
        256,
    );
    let offset = f.add_u64_register();
    f.add_inst(Inst::add_u64(
        offset.clone(),
        row_start,
        columns.counter.clone(),
    ));
    let exp = if let Some(base) = exps {
        let bytes = multiply_u64(&mut f, columns.counter.clone(), 4);
        let addr = f.add_u64_register();
        f.add_inst(Inst::add_u64(addr.clone(), base, bytes));
        let v = f.add_f32_register();
        f.add_inst(Inst::LdSharedF32 {
            dst: vec![v.clone()],
            addr,
            vec: crate::ptx::instructions::VecWidth::Scalar,
        });
        v
    } else {
        let v = load_param_f32_at(&mut f, &ctx, "logits", offset.clone());
        let shifted = f.add_f32_register();
        f.add_inst(Inst::sub_f32(shifted.clone(), v, maximum.clone()));
        emit_exp(&mut f, shifted)
    };
    store_param_f32_at(&mut f, &ctx, "exponentials", offset, exp);
    end_counted_loop(&mut f, columns);
    let follower = f.add_predicate_register();
    f.add_inst(Inst::setp_ne_u64(
        follower.clone(),
        lane,
        Operand::imm_u64(0),
    ));
    f.add_inst(Inst::Bra {
        condition: follower,
        target: ctx.return_label,
    });
    store_param_f32_at(&mut f, &ctx, "maximum", row.clone(), maximum);
    store_param_f32_at(&mut f, &ctx, "sum", row, sum);
    f.add_inst(Inst::Label(ctx.return_label));
    f.add_inst(Inst::Ret);
    let mut module = crate::ptx::Module::new_with_target(target.compute_capability);
    module.add_function(f);
    module.to_string()
}
fn build(stream: &Arc<CudaStream>, source: &str, name: &str) -> Result<CudaFunction> {
    Ok(stream
        .context()
        .load_module(Ptx::from_src(source))?
        .load_function(name)?)
}
fn feeds(rows: usize, phase: usize, half: bool) -> Vec<f32> {
    let mut values = (0..rows * 4096)
        .map(|i| ((i + phase * 7) % 29) as f32 / 16. - 0.875)
        .collect::<Vec<_>>();
    values[..4096].fill(-18.42068);
    values[if phase == 0 { 0 } else { 4095 }] = 0.;
    for col in 0..4096 {
        values[4096 + col] = if col % 2 == 0 { 0. } else { -0. };
        values[3 * 4096 + col] = f32::from_bits(1 + col as u32 % 7);
    }
    if phase == 2 {
        values[4 * 4096] = f32::INFINITY;
        values[5 * 4096..6 * 4096].fill(f32::NEG_INFINITY);
        values[6 * 4096..7 * 4096].fill(f32::NAN);
    }
    if half {
        for v in &mut values {
            *v = half::bf16::from_f32(*v).to_f32();
        }
    }
    values
}
#[test]
fn cooperative_ordered_max_exp_sum_gradient_bits_and_f64_diagnostics() -> Result<()> {
    if !available() {
        return Ok(());
    }
    let device = CudaContext::new(0)?;
    let stream = device.new_stream()?;
    let target = PtxTarget::from_context(&device)?;
    let rows = 16;
    let len = rows * 4096;
    for half in [false, true] {
        let dtype = if half { DType::BF16 } else { DType::F32 };
        let mut logits = if half {
            Buffer::BF16(stream.alloc_zeros::<half::bf16>(len)?)
        } else {
            Buffer::F32(stream.alloc_zeros::<f32>(len)?)
        };
        let mut targets = stream.memcpy_stod(&vec![0u32; rows])?;
        let mut upstream = stream.memcpy_stod(&vec![1f32; rows])?;
        let gradient = [
            stream.alloc_zeros::<f32>(len)?,
            stream.alloc_zeros::<f32>(len)?,
        ];
        let (g, out) = graph(rows, 4096, false);
        let inputs = g.inputs(out);
        let bindings = HashMap::from([
            (inputs[0], binding(&logits, &stream)),
            (
                inputs[1],
                Binding {
                    pointer: targets.device_ptr(&stream).0,
                    len: rows,
                    dtype: DType::U32,
                },
            ),
            (
                inputs[2],
                Binding {
                    pointer: upstream.device_ptr(&stream).0,
                    len: rows,
                    dtype: DType::F32,
                },
            ),
        ]);
        let old = Program::<crate::ptx::F32>::new_bound_with_kernel_options(
            &stream,
            g.clone(),
            bindings.clone(),
            Some(Binding {
                pointer: gradient[0].device_ptr(&stream).0,
                len,
                dtype: DType::F32,
            }),
            crate::tile::MatMulPrecision::StrictF32,
            ProgramKernelOptions::default(),
        )?;
        let new = Program::<crate::ptx::F32>::new_bound_with_kernel_options(
            &stream,
            g,
            bindings,
            Some(Binding {
                pointer: gradient[1].device_ptr(&stream).0,
                len,
                dtype: DType::F32,
            }),
            crate::tile::MatMulPrecision::StrictF32,
            ProgramKernelOptions {
                indexed_loss_cooperative_ordered_backward: true,
                ..ProgramKernelOptions::default()
            },
        )?;
        assert_eq!(old.indexed_loss_cooperative_ordered_backward_programs, 0);
        assert_eq!(new.indexed_loss_cooperative_ordered_backward_programs, 1);
        assert_eq!(old.kernels.len(), new.kernels.len());
        let sources = [
            probe_source(false, dtype, target),
            probe_source(true, dtype, target),
        ];
        let probes = [
            build(&stream, &sources[0], "ce_stats_ordered_probe")?,
            build(&stream, &sources[1], "ce_stats_cooperative_probe")?,
        ];
        let mut maxima = [
            stream.alloc_zeros::<f32>(rows)?,
            stream.alloc_zeros::<f32>(rows)?,
        ];
        let mut sums = [
            stream.alloc_zeros::<f32>(rows)?,
            stream.alloc_zeros::<f32>(rows)?,
        ];
        let mut exps = [
            stream.alloc_zeros::<f32>(len)?,
            stream.alloc_zeros::<f32>(len)?,
        ];
        let mut diagnostics = Vec::new();
        for phase in 0..4 {
            let values = if half && phase == 3 {
                (0..=u16::MAX)
                    .map(|x| half::bf16::from_bits(x).to_f32())
                    .collect::<Vec<_>>()
            } else {
                feeds(rows, phase, half)
            };
            match &mut logits {
                Buffer::F32(v) => stream.memcpy_htod(&values, v)?,
                Buffer::BF16(v) => {
                    let encoded = if phase == 3 {
                        (0..=u16::MAX)
                            .map(half::bf16::from_bits)
                            .collect::<Vec<_>>()
                    } else {
                        values
                            .iter()
                            .map(|&x| half::bf16::from_f32(x))
                            .collect::<Vec<_>>()
                    };
                    stream.memcpy_htod(&encoded, v)?;
                }
            };
            let target_values = (0..rows)
                .map(|r| {
                    if r == 0 && phase != 0 {
                        4095
                    } else {
                        ((r + phase) * 683 % 4096) as u32
                    }
                })
                .collect::<Vec<_>>();
            let scales = (0..rows)
                .map(|r| match r {
                    1 => -0.,
                    2 => -2.,
                    3 => 0.5,
                    4 => 0.,
                    6 => -1.,
                    _ => 1.,
                })
                .collect::<Vec<_>>();
            stream.memcpy_htod(&target_values, &mut targets)?;
            stream.memcpy_htod(&scales, &mut upstream)?;
            for i in 0..2 {
                let logit_pointer = binding(&logits, &stream).pointer;
                let mut launch = stream.launch_builder(&probes[i]);
                launch
                    .arg(&logit_pointer)
                    .arg(&mut maxima[i])
                    .arg(&mut sums[i])
                    .arg(&mut exps[i]);
                // SAFETY: exact typed buffers, sixteen full rows and 256 contributors;
                // all owners remain alive until synchronized readback.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }?;
            }
            for p in [&old, &new] {
                for k in &p.kernels {
                    k.run(&stream)?;
                }
            }
            stream.synchronize()?;
            let max = [
                stream.memcpy_dtov(&maxima[0])?,
                stream.memcpy_dtov(&maxima[1])?,
            ];
            let sum = [stream.memcpy_dtov(&sums[0])?, stream.memcpy_dtov(&sums[1])?];
            let exp = [stream.memcpy_dtov(&exps[0])?, stream.memcpy_dtov(&exps[1])?];
            let grad = [
                stream.memcpy_dtov(&gradient[0])?,
                stream.memcpy_dtov(&gradient[1])?,
            ];
            bits(&max[0], &max[1]);
            bits(&sum[0], &sum[1]);
            bits(&exp[0], &exp[1]);
            bits(&grad[0], &grad[1]);
            if !(half && phase == 3) {
                assert_eq!(
                    max[1][3].to_bits(),
                    if half { 0 } else { 7 },
                    "maximum must preserve physical-storage subnormal semantics"
                );
            }
            assert_eq!(grad[1][4096 + target_values[1] as usize].to_bits(), 0);
            assert_eq!(
                grad[1][4096].to_bits(),
                0x8000_0000,
                "negative-zero upstream keeps each output's original multiplication sign"
            );
            if phase == 2 {
                assert!(
                    grad[1][4 * 4096..5 * 4096].iter().all(|v| v.is_nan()),
                    "zero upstream must not suppress NaNs"
                );
                assert_eq!(max[1][6].to_bits(), f32::NEG_INFINITY.to_bits());
                assert!(sum[1][5].is_nan() && sum[1][6].is_nan());
            }

            if phase == 0 {
                assert_eq!(sum[1][0].to_bits(), 1f32.to_bits());
                assert_eq!(
                    grad[1][0].to_bits(),
                    0f32.to_bits(),
                    "near-zero target gradient must keep the original ordered result"
                );
            }
            let mut max_error = 0f64;
            let mut squared = 0f64;
            let mut reference_squared = 0f64;
            for row in 0..rows {
                let input = &values[row * 4096..(row + 1) * 4096];
                if input.iter().any(|x| !x.is_finite()) {
                    continue;
                }
                let m = input
                    .iter()
                    .copied()
                    .map(f64::from)
                    .fold(f64::NEG_INFINITY, f64::max);
                let den = input.iter().map(|&x| (f64::from(x) - m).exp()).sum::<f64>();
                for (col, &v) in input.iter().enumerate() {
                    let expected = (((f64::from(v) - m).exp() / den)
                        - f64::from(u8::from(col == target_values[row] as usize)))
                        * f64::from(scales[row]);
                    let error = f64::from(grad[1][row * 4096 + col]) - expected;
                    max_error = max_error.max(error.abs());
                    squared += error * error;
                    reference_squared += expected * expected;
                }
            }
            diagnostics.push(serde_json::json!({"phase":phase,"physical_logits":format!("{dtype:?}"),"gradient_f64_max_absolute_error":max_error,"gradient_f64_relative_l2":(squared/reference_squared.max(1e-24)).sqrt(),"scope":"diagnostic only; old/new non-NaN bits and NaN classes are the required equivalence gate"}));
        }
        if let Some(dir) = std::env::var_os("TNSR_CE_ORDERED_PTX_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            for (i, source) in sources.iter().enumerate() {
                std::fs::write(dir.join(format!("probe-half{half}-{i}.ptx")), source)?;
            }
            for (i, p) in [&old, &new].iter().enumerate() {
                std::fs::write(dir.join(format!("gradient-half{half}-{i}.ptx")), &p.ptx)?;
            }
            let resources=new.kernels.iter().map(|k|Ok(serde_json::json!({"label":k.label,"registers":k.function.num_regs()?,"shared":k.function.shared_size_bytes()?,"local":k.function.local_size_bytes()?,"max_threads":k.function.max_threads_per_block()?,"capacity":k.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}))).collect::<Result<Vec<_>>>()?;
            std::fs::write(
                dir.join(format!("half{half}-diagnostics.json")),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"f64_diagnostics":diagnostics,"actual_gradient_kernel_resources":resources,"scope":"capacity is not achieved occupancy"}),
                )?,
            )?;
        }
    }
    Ok(())
}

#[test]
fn cooperative_ordered_executor_off_on_off_default_source_and_invalid_float_targets() -> Result<()>
{
    if !available() {
        return Ok(());
    }
    let (mut g, out) = graph(3, 4096, false);
    let inputs = g.inputs(out);
    g.graph[inputs[1]] = TensorGraphNode::Input {
        name: "t",
        shape: vec![3],
    };
    let mut e = crate::ptx::PtxExecutor::try_new()?;
    let mut expected = Vec::new();
    let mut original = String::new();
    for (revision, enabled) in [false, true, false].into_iter().enumerate() {
        e.set_experimental_indexed_loss_cooperative_ordered_backward(enabled);
        if revision > 0 {
            assert!(
                e.execute_compiled(&g, HashMap::new())
                    .unwrap_err()
                    .to_string()
                    .contains("No compiled graph signature")
            );
        }
        e.compile(&g)?;
        assert_eq!(
            e.indexed_loss_cooperative_ordered_backward_programs(),
            usize::from(enabled)
        );
        if revision == 0 {
            original = e.module_source().unwrap().to_owned();
        }
        if revision == 2 {
            assert_eq!(e.module_source().unwrap(), original);
        }
        for phase in 0..2 {
            let x = (0..3 * 4096)
                .map(|i| ((i + phase) % 31) as f32 / 16. - 1.)
                .collect();
            let feed = HashMap::from([
                ("x".into(), x),
                ("t".into(), vec![0., 4095., (phase * 31) as f32]),
                ("dy".into(), vec![1., -0., -2.]),
            ]);
            let actual = e.execute_compiled(&g, feed)?;
            if revision == 0 {
                expected.push(actual);
            } else {
                bits(&actual, &expected[phase]);
            }
        }
        for invalid in [1.5, f32::NAN, -1., 4096.] {
            assert!(
                e.execute_compiled(
                    &g,
                    HashMap::from([
                        ("x".into(), vec![0.; 3 * 4096]),
                        ("t".into(), vec![invalid; 3]),
                        ("dy".into(), vec![1.; 3])
                    ])
                )
                .is_err()
            );
        }
    }
    Ok(())
}
