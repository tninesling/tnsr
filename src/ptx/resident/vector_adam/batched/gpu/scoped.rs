//! One untimed resource/math trial for the explicit scoped-load policy.
use super::*;
use crate::ptx::graph::scoped_parameters as loads;
use crate::ptx::resident::vector_adam::batched::scoped::optimized;
#[test]
#[ignore = "manual scoped-pointer resource/math trial; no timing or production batching"]
fn paired_vector_scoped_parameters_preserve_canonical_math_and_health() -> Result<()> {
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => {
            ensure!(
                std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            return Ok(());
        }
    };
    // SAFETY: this private test owns all modules, captures and allocations, uses
    // one ordered stream, and synchronizes at readback/capture/destruction.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream()?;
    let root = std::path::PathBuf::from(
        std::env::var_os("TNSR_PAIRED_VECTOR_SCOPED_DIR")
            .context("set fresh scoped resource/math output")?,
    );
    let qualified = std::path::PathBuf::from(
        std::env::var_os("TNSR_PAIRED_VECTOR_BATCH_QUALIFIED_PTX_DIR")
            .context("set old math-qualified PTX directory")?,
    );
    std::fs::create_dir_all(&root)?;
    ensure!(
        !root.join("qualification.json").exists(),
        "fresh scoped resource/math proof required"
    );
    let mut reports = Vec::new();
    for len in [1_048_576usize, 2_818_048, 3_145_728] {
        let (separate, eager, scopes) = optimized(len)?;
        let (source, names) = loads::module(
            vec![separate, eager.clone()],
            eager,
            &scopes,
            PtxTarget::from_context(&context)?,
        )?;
        let marker = format!(".visible .entry {}", names[2]);
        let start = source.find(&marker).context("scoped entry missing")?;
        let old_path = qualified.join(format!("len{len}-members4.ptx"));
        let old = std::fs::read_to_string(&old_path)?;
        ensure!(
            source[..start].trim_end() == old.trim_end(),
            "ordinary canonical/eager PTX changed in scoped experiment"
        );
        let path = root.join(format!("len{len}-scoped.ptx"));
        ensure!(!path.exists(), "fresh scoped loaded PTX path required");
        std::fs::write(&path, &source)?;
        let module = context.load_module(Ptx::from_src(source))?;
        let functions = names
            .iter()
            .map(|n| module.load_function(n).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        let resources=functions.iter().zip(&names).map(|(f,name)|->Result<_>{Ok(serde_json::json!({"name":name,
            "registers":f.num_regs()?,"shared_bytes":f.shared_size_bytes()?,"local_bytes":f.local_size_bytes()?,
            "max_threads":f.max_threads_per_block()?,"capacity256":f.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}))}).collect::<Result<Vec<_>>>()?;
        std::fs::write(
            root.join(format!("len{len}-resources.json")),
            serde_json::to_vec_pretty(&serde_json::json!({
                "elements_per_member":len,"members":4,"root_pointer_parameters":40,"parameter_bytes":320,
                "loads_per_executed_full_CTA_region":10,"ordinary_entries_byte_identical":true,"functions":resources,
            }))?,
        )?;
        // This is a safety/resource gate, not an improvement assertion. A
        // numerically valid result with unchanged registers/capacity parks the
        // mechanism; it cannot justify a timing or owner-dispatch claim.
        ensure!(
            functions
                .iter()
                .all(|f| f.local_size_bytes().ok() == Some(0))
                && functions[2].num_regs()? <= 128
                && functions[2].shared_size_bytes()? == 32
                && functions[2].max_threads_per_block()? >= 256
                && functions[2].occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? >= 2,
            "scoped resource safety screen failed; inspect retained resource/PTX evidence"
        );
        let mut members = (0..4)
            .map(|_| member(&stream, len))
            .collect::<Result<Vec<_>>>()?;
        let bindings = members
            .iter()
            .map(|m| spans(m, len, &stream))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            admitted(len, &bindings),
            "actual scoped batch typed spans/alias proof failed"
        );
        let mut comparisons = 0usize;
        let mut readonly_checks = 0usize;
        for phase in 0..6 {
            let feeds = (0..4)
                .map(|member| feed(len, member, phase))
                .collect::<Vec<_>>();
            for (member, feed) in members.iter_mut().zip(&feeds) {
                upload(&stream, &mut member.input, feed)?;
                poison(&stream, &mut member.old, len)?;
                poison(&stream, &mut member.new, len)?;
            }
            launch_separate(&stream, &functions[0], &mut members, len)?;
            launch_batch(&stream, &functions[2], &mut members, len)?;
            stream.synchronize()?;
            for member in &members {
                comparisons += compare(&stream, member, len)?;
            }
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let submitted = (|| -> Result<()> {
                launch_separate(&stream, &functions[0], &mut members, len)?;
                launch_batch(&stream, &functions[2], &mut members, len)
            })();
            let capture = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            submitted?;
            let capture = capture?.context("empty scoped math capture")?;
            for _ in 0..2 {
                for member in &mut members {
                    poison(&stream, &mut member.old, len)?;
                    poison(&stream, &mut member.new, len)?;
                }
                capture.launch()?;
                stream.synchronize()?;
                for member in &members {
                    comparisons += compare(&stream, member, len)?;
                }
            }
            if phase == 2 {
                for member in &members {
                    let w = download(&stream, &member.new.weight)?;
                    let h = download(&stream, &member.new.health)?;
                    let s = download(&stream, &member.new.shadow)?;
                    ensure!(
                        w[..len].iter().all(|v| v.is_finite())
                            && h[..len / 256].iter().all(|v| v.to_bits() == 0)
                            && s[..len].iter().all(|v| v.is_infinite()),
                        "scoped policy changed finite-F32/BF16-infinity health boundary"
                    );
                }
            }
            if phase == 4 {
                for (index, member) in members.iter().enumerate() {
                    let health = download(&stream, &member.new.health)?;
                    ensure!(
                        health[..len / 256].iter().enumerate().all(
                            |(group, v)| if group == index {
                                v.is_nan()
                            } else {
                                v.to_bits() == 0
                            }
                        ),
                        "scoped policy changed independent per-member health group ownership"
                    );
                }
            }
            drop(capture);
            for (member, feed) in members.iter().zip(&feeds) {
                readonly_checks += readonly(&stream, &member.input, feed)?;
            }
        }
        let registers_reduced = functions[2].num_regs()? < functions[1].num_regs()?;
        let capacity_increased = functions[2]
            .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?
            > functions[1].occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
        reports.push(serde_json::json!({"elements_per_member":len,"members":4,"ordinary_canonical_and_eager_PTX_byte_identical":true,
            "resources":resources,"registers_reduced_vs_eager":registers_reduced,"capacity_increased_vs_eager":capacity_increased,
            "array_comparisons":comparisons,"readonly_checks":readonly_checks,"feeds":6,"graph_replays_per_feed":2,
            "poison_before_every_replay":true,"all65536_BF16_patterns":true,"isolated_health_group_per_member":true,
            "finite_F32_to_BF16_infinity_health_unchanged":true,"nonNaN_bits_and_NaN_classes_exact":true,"performance_claim":false}));
    }
    std::fs::write(
        root.join("qualification.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "qualified_math":true,"timing_performed":false,"production_default_lowering_changed":false,
            "scope":"One CFG(test) scoped-load resource/math experiment only. Resource capacity is not measured occupancy or performance.","cases":reports,
        }))?,
    )?;
    Ok(())
}
