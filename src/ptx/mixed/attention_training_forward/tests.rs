//! Local dense-leaf qualification; no Mixed flag or execution hook.
use super::*;
use crate::tile::{Expr, ScalarMemory, Stmt};

#[test]
fn training_attention_union_arena_spans_and_temporal_aliases_are_bounded() -> Result<()> {
    for query_tile in [16, 32] {
        let g = Geometry::new(2, query_tile)?;
        assert_eq!(
            g.shared_bytes(),
            if query_tile == 32 {
                44 * 1024
            } else {
                26 * 1024
            }
        );
        for include_qk in [false, true] {
            let ir = ir(g, include_qk)?;
            ir.validate_layouts()?;
            let mut arenas = 0;
            let mut pviews = 0;
            for stmt in &ir.body.stmts {
                match stmt {
                    Stmt::AllocSharedArena { bytes, .. } => {
                        arenas += 1;
                        assert_eq!(*bytes, g.shared_bytes());
                    }
                    Stmt::ViewSharedArena {
                        byte_offset,
                        dtype,
                        rows,
                        cols,
                        layout,
                        ..
                    } => {
                        assert!(byte_offset.is_multiple_of(32));
                        assert!(
                            *byte_offset + layout.allocation_bytes(*rows, *cols, *dtype)
                                <= g.shared_bytes()
                        );
                        if *rows == query_tile
                            && *cols == KEYS
                            && *dtype == DType::BF16
                            && layout.row_stride == SEQUENCE
                        {
                            pviews += 1;
                            let actual_end = byte_offset + ((query_tile - 1) * SEQUENCE + KEYS) * 2;
                            assert!(actual_end <= g.score_bytes());
                            let padded_end =
                                byte_offset + layout.allocation_bytes(*rows, *cols, *dtype);
                            assert!(
                                padded_end <= g.key_offset(),
                                "padding may overlap deadQ but never liveV"
                            );
                        }
                    }
                    _ => {}
                }
            }
            assert_eq!(arenas, 1);
            assert_eq!(pviews, 8);
            let qkloop = ir
                .body
                .stmts
                .iter()
                .find_map(|s| match s {
                    Stmt::ForLoop { loop_var, body, .. }
                        if loop_var.starts_with("qk_key_tiles") =>
                    {
                        Some(body)
                    }
                    _ => None,
                })
                .unwrap();
            let mma = qkloop
                .stmts
                .iter()
                .position(|s| matches!(s, Stmt::MatMul { .. }))
                .unwrap();
            assert!(matches!(qkloop.stmts[mma - 1], Stmt::Barrier));
            assert!(matches!(qkloop.stmts[mma + 1], Stmt::Barrier));
            assert!(matches!(qkloop.stmts[mma + 2], Stmt::ConvertLayout { .. }));
            assert!(matches!(qkloop.stmts[mma + 3], Stmt::Barrier));
            assert!(matches!(qkloop.stmts.last(), Some(Stmt::Barrier)));
            let normalize = ir
                .body
                .stmts
                .iter()
                .find_map(|s| match s {
                    Stmt::ForLoop { loop_var, body, .. }
                        if loop_var.starts_with("normalize_rows") =>
                    {
                        Some(body)
                    }
                    _ => None,
                })
                .unwrap();
            let first_store = normalize
                .stmts
                .iter()
                .position(|s| matches!(s, Stmt::StoreScalar { .. }))
                .unwrap();
            let loads = normalize.stmts[..first_store]
                .iter()
                .filter(|s| {
                    matches!(
                        s,
                        Stmt::LoadScalar {
                            source: ScalarMemory::Shared(_),
                            ..
                        }
                    )
                })
                .count();
            assert_eq!(loads, 16, "read complete row before inplaceP16 writes");
            assert_eq!(
                normalize
                    .stmts
                    .iter()
                    .filter(|s| matches!(
                        s,
                        Stmt::WarpReduce {
                            broadcast: false,
                            ..
                        }
                    ))
                    .count(),
                2
            );
            assert_eq!(
                normalize
                    .stmts
                    .iter()
                    .filter(|s| matches!(s, Stmt::WarpBroadcastLeader { .. }))
                    .count(),
                2
            );
            let pv = ir
                .body
                .stmts
                .iter()
                .enumerate()
                .filter_map(|(i, s)| matches!(s, Stmt::MatMul { .. }).then_some(i))
                .collect::<Vec<_>>();
            assert_eq!(pv.len(), 8);
            for i in pv {
                assert!(matches!(ir.body.stmts[i - 1], Stmt::Barrier));
                assert!(matches!(ir.body.stmts[i + 1], Stmt::Barrier));
            }
        }
    }
    assert!(Geometry::new(0, 32).is_err());
    assert!(Geometry::new(65536, 32).is_err());
    assert!(Geometry::new(64, 31).is_err());
    Ok(())
}

fn eval(e: &Expr, env: &HashMap<String, u64>, x: u64, y: u64, bx: u64, by: u64) -> u64 {
    match e {
        Expr::Const(v) => *v as u64,
        Expr::Var(v) => env[v],
        Expr::ThreadIdx(Dim::X) => x,
        Expr::ThreadIdx(Dim::Y) => y,
        Expr::ThreadIdx(Dim::Z) => 0,
        Expr::BlockIdx(Dim::X) => bx,
        Expr::BlockIdx(Dim::Y) => by,
        Expr::BlockIdx(Dim::Z) => 0,
        Expr::BlockDim(Dim::X) => 32,
        Expr::BlockDim(Dim::Y) => 8,
        Expr::BlockDim(Dim::Z) => 1,
        Expr::Add(a, b) => eval(a, env, x, y, bx, by).wrapping_add(eval(b, env, x, y, bx, by)),
        Expr::Sub(a, b) => eval(a, env, x, y, bx, by).wrapping_sub(eval(b, env, x, y, bx, by)),
        Expr::Mul(a, b) => eval(a, env, x, y, bx, by).wrapping_mul(eval(b, env, x, y, bx, by)),
        Expr::FloorDiv(a, n) => eval(a, env, x, y, bx, by) / *n as u64,
        Expr::Mod(a, n) => eval(a, env, x, y, bx, by) % *n as u64,
        Expr::ShiftRight(a, n) => eval(a, env, x, y, bx, by) >> n,
        Expr::BitAnd(a, n) => eval(a, env, x, y, bx, by) & n,
    }
}

#[test]
fn training_attention_union_global_coordinates_cover_outputs_and_keep_k16_order() -> Result<()> {
    fn visit(
        block: &Block,
        env: &mut HashMap<String, u64>,
        thread: (u64, u64, u64, u64),
        hits: &mut HashMap<String, Vec<u16>>,
    ) {
        let outer = env.clone();
        let (x, y, bx, by) = thread;
        for stmt in &block.stmts {
            match stmt {
                Stmt::LetIndex { name, value } => {
                    let v = eval(value, env, x, y, bx, by);
                    env.insert(name.clone(), v);
                }
                Stmt::ForLoop {
                    loop_var,
                    start,
                    end,
                    body,
                    ..
                } => {
                    let start = eval(start, env, x, y, bx, by);
                    let end = eval(end, env, x, y, bx, by);
                    for i in start..end {
                        env.insert(loop_var.clone(), i);
                        visit(body, env, thread, hits);
                    }
                }
                Stmt::LoadScalar {
                    source: ScalarMemory::Global(name),
                    index,
                    ..
                }
                | Stmt::StoreScalar {
                    target: ScalarMemory::Global(name),
                    index,
                    ..
                } => {
                    let index = usize::try_from(eval(index, env, x, y, bx, by)).unwrap();
                    let list = hits.get_mut(name).unwrap();
                    assert!(index < list.len(), "{name} index{index}");
                    list[index] += 1;
                }
                Stmt::If { .. } => panic!("fullgeometry must not skip causal keys or warp lanes"),
                _ => {}
            }
        }
        *env = outer;
    }
    for query_tile in [16, 32] {
        let g = Geometry::new(2, query_tile)?;
        let ir = ir(g, true)?;
        let mut hits = HashMap::new();
        for name in ["query", "key", "value", "output"] {
            hits.insert(name.into(), vec![0_u16; 2 * SEQUENCE * WIDTH]);
        }
        for name in ["qk", "probability", "narrowed"] {
            hits.insert(name.into(), vec![0_u16; 2 * SEQUENCE * SEQUENCE]);
        }
        hits.insert("mask".into(), vec![0_u16; SEQUENCE * SEQUENCE]);
        for by in 0..2 {
            for bx in 0..SEQUENCE / query_tile {
                for y in 0..8 {
                    for x in 0..32 {
                        visit(
                            &ir.body,
                            &mut HashMap::new(),
                            (x, y, bx as u64, by),
                            &mut hits,
                        );
                    }
                }
            }
        }
        for name in ["query", "qk", "probability", "narrowed", "output"] {
            assert!(hits[name].iter().all(|&n| n == 1), "{name}");
        }
        for name in ["key", "value"] {
            assert!(
                hits[name]
                    .iter()
                    .all(|&n| usize::from(n) == SEQUENCE / query_tile),
                "{name}"
            );
        }
        assert!(hits["mask"].iter().all(|&n| n == 2));
        assert_eq!(
            (0..8)
                .flat_map(|chunk| (0..64).step_by(16).map(move |k| chunk * 64 + k))
                .collect::<Vec<_>>(),
            (0..512).step_by(16).collect::<Vec<_>>()
        );
    }
    Ok(())
}

fn compile(
    stream: &Arc<CudaStream>,
    ir: TileIR,
    tag: &str,
) -> Result<cudarc::driver::CudaFunction> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles
        .region_kernels
        .push(crate::tile::graph::TileRegionKernel { region_id: 0, ir });
    tiles.optimize_indices()?;
    let target = PtxTarget::from_context(stream.context())?;
    let ptx = PtxGraph::from(tiles).with_target(target);
    let source = ptx.to_ptx();
    if let Some(path) = std::env::var_os("TNSR_ATTENTION_TRAINING_FORWARD_AUDIT") {
        std::fs::create_dir_all(&path)?;
        std::fs::write(
            std::path::Path::new(&path).join(format!("{tag}.ptx")),
            &source,
        )?;
    }
    let module = stream.context().load_module(Ptx::from_src(source))?;
    Ok(module.load_function(ptx.region_kernel_name(0).unwrap())?)
}
fn kernel(
    function: cudarc::driver::CudaFunction,
    config: LaunchConfig,
    pointers: Vec<u64>,
) -> Kernel {
    Kernel {
        owner: None,
        label: "disconnected exact training forward leaf".into(),
        function,
        config,
        pointers,
        zero_before: None,
    }
}
fn same_bits<T: std::fmt::Debug>(
    name: &str,
    a: &[T],
    b: &[T],
    bits: impl Fn(&T) -> u32,
    nan: impl Fn(&T) -> bool,
) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            (nan(a) && nan(b)) || bits(a) == bits(b),
            "{name} index{i}: {a:?}/{b:?}"
        );
    }
}

#[test]
fn training_attention_union_dense_leaf_exact_checkpoints_resources_and_replay() -> Result<()> {
    qualify_leaf(InputLayout::DenseHeadMajor, 2)
}

#[test]
fn training_attention_union_sequence_major_leaf_exact_checkpoints_resources_and_replay()
-> Result<()> {
    qualify_leaf(InputLayout::SequenceMajor { heads: 8 }, 64)
}

fn qualify_leaf(input_layout: InputLayout, bh: usize) -> Result<()> {
    use crate::tensor::TensorExpr;
    use half::bf16;
    if !super::super::tests::cuda_available() {
        return Ok(());
    }
    let context = CudaContext::new(0)?;
    // All allocations/kernels use this one ordered private stream, matching
    // ResidentAdam's graph-capture contract and avoiding cross-stream events.
    unsafe {
        context.disable_event_tracking();
    }
    let stream = context.new_stream()?;
    let (batch, heads, source_shape, label) = match input_layout {
        InputLayout::DenseHeadMajor => (1, bh, vec![1, bh, 512, 64], "dense"),
        InputLayout::SequenceMajor { heads } => (
            bh / heads,
            heads,
            vec![bh / heads, 512, heads * 64],
            "sequence-major",
        ),
    };
    let projected = |node, shape: Vec<usize>| {
        let expression = TensorExpr::node_ref(node, shape);
        match input_layout {
            InputLayout::DenseHeadMajor => expression,
            InputLayout::SequenceMajor { .. } => expression
                .reshape(vec![batch, 512, heads, 64])
                .permute(vec![0, 2, 1, 3]),
        }
    };
    let input_len = bh * SEQUENCE * WIDTH;
    let score_len = bh * SEQUENCE * SEQUENCE;
    let mut q = stream.alloc_zeros::<bf16>(input_len)?;
    let mut k = stream.alloc_zeros::<bf16>(input_len)?;
    let mut v = stream.alloc_zeros::<bf16>(input_len)?;
    let mask_host = (0..SEQUENCE * SEQUENCE)
        .map(|i| {
            if i % SEQUENCE > i / SEQUENCE {
                f32::NEG_INFINITY
            } else {
                0.
            }
        })
        .collect::<Vec<_>>();
    let mask = stream.memcpy_stod(&mask_host)?;
    let qk_ref = stream.alloc_zeros::<bf16>(score_len)?;
    let p32_ref = stream.alloc_zeros::<f32>(score_len)?;
    let p16_ref = stream.alloc_zeros::<bf16>(score_len)?;
    let out_ref = stream.alloc_zeros::<bf16>(input_len)?;
    let binding16 = |values: &CudaSlice<bf16>| Binding {
        pointer: values.device_ptr(&stream).0,
        len: values.len(),
        dtype: DType::BF16,
    };
    let baseline_options = crate::ptx::linear::program::ProgramKernelOptions {
        shared_memory_reuse: true,
        narrow_bf16_tile: true,
        batched_attention_bf16_tile_policy:
            crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid,
        ..Default::default()
    };
    let mut qk_graph = TensorGraph::<bf16>::new();
    let qid = qk_graph.graph.add_node(TensorGraphNode::Input {
        name: "q",
        shape: source_shape.clone(),
    });
    let kid = qk_graph.graph.add_node(TensorGraphNode::Input {
        name: "k",
        shape: source_shape.clone(),
    });
    let qk_expr = projected(qid, source_shape.clone())
        .matmul(projected(kid, source_shape.clone()).transpose());
    qk_graph.add_expr(&qk_expr, false);
    let qk_program = Program::<BF16>::new_bound_with_kernel_options(
        &stream,
        qk_graph,
        HashMap::from([(qid, binding16(&q)), (kid, binding16(&k))]),
        Some(binding16(&qk_ref)),
        MatMulPrecision::NativeHalf,
        baseline_options,
    )?;
    let row = kernel(
        compile(
            &stream,
            super::super::attention_softmax::row_ir(0.125)?,
            "baseline-row",
        )?,
        LaunchConfig {
            grid_dim: (u32::try_from(bh * 512)?, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        },
        vec![
            qk_ref.device_ptr(&stream).0,
            mask.device_ptr(&stream).0,
            p32_ref.device_ptr(&stream).0,
            p16_ref.device_ptr(&stream).0,
        ],
    );
    let mut pv_graph = TensorGraph::<bf16>::new();
    let pid = pv_graph.graph.add_node(TensorGraphNode::Input {
        name: "p",
        shape: vec![batch, heads, 512, 512],
    });
    let vid = pv_graph.graph.add_node(TensorGraphNode::Input {
        name: "v",
        shape: source_shape.clone(),
    });
    pv_graph.add_expr(
        &TensorExpr::node_ref(pid, vec![batch, heads, 512, 512])
            .matmul(projected(vid, source_shape.clone())),
        false,
    );
    let pv_program = Program::<BF16>::new_bound_with_kernel_options(
        &stream,
        pv_graph,
        HashMap::from([(pid, binding16(&p16_ref)), (vid, binding16(&v))]),
        Some(binding16(&out_ref)),
        MatMulPrecision::NativeHalf,
        baseline_options,
    )?;
    if matches!(input_layout, InputLayout::SequenceMajor { .. }) {
        ensure!(
            qk_program.batched_attention_bf16_tile_counts.scores == 1
                && pv_program.batched_attention_bf16_tile_counts.query_width == 1,
            "sequence-major leaf baseline must install actual attention rectangles"
        );
    }
    let mut resources = Vec::new();
    let mut validated_geometries = Vec::new();
    let query_tiles: &[usize] = if matches!(input_layout, InputLayout::DenseHeadMajor) {
        &[32, 16]
    } else {
        &[32]
    };
    let checkpoint_variants: &[bool] = if matches!(input_layout, InputLayout::DenseHeadMajor) {
        &[true, false]
    } else {
        &[true]
    };
    for &query_tile in query_tiles {
        let g = Geometry::new(bh, query_tile)?;
        let mut variants = Vec::new();
        for &dump in checkpoint_variants {
            let qk = stream.alloc_zeros::<bf16>(score_len)?;
            let p32 = stream.alloc_zeros::<f32>(score_len)?;
            let p16 = stream.alloc_zeros::<bf16>(score_len)?;
            let out = stream.alloc_zeros::<bf16>(input_len)?;
            let tag = match input_layout {
                InputLayout::DenseHeadMajor => format!("union-q{query_tile}-dump{dump}"),
                InputLayout::SequenceMajor { heads } => {
                    format!("union-sequence-h{heads}-q{query_tile}-dump{dump}")
                }
            };
            let f = compile(&stream, ir_with_input_layout(g, dump, input_layout)?, &tag)?;
            let admit = admitted(&f, g)?;
            resources.push(serde_json::json!({"query_tile":query_tile,"qk_checkpoint":dump,"registers":f.num_regs()?,"shared":f.shared_size_bytes()?,"local":f.local_size_bytes()?,"block":[32,8,1],"api_capacity":f.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?,"prototype_admitted":admit,"scope":"actual driver resource limit, not achieved occupancy"}));
            if let Some(path) = std::env::var_os("TNSR_ATTENTION_TRAINING_FORWARD_AUDIT") {
                std::fs::create_dir_all(&path)?;
                std::fs::write(
                    std::path::Path::new(&path).join(format!("{label}-leaf-resources.json")),
                    serde_json::to_vec_pretty(&resources)?,
                )?;
            }
            if !admit {
                continue;
            }
            let mut args = vec![
                q.device_ptr(&stream).0,
                k.device_ptr(&stream).0,
                v.device_ptr(&stream).0,
                mask.device_ptr(&stream).0,
            ];
            if dump {
                args.push(qk.device_ptr(&stream).0);
            }
            args.extend([
                p32.device_ptr(&stream).0,
                p16.device_ptr(&stream).0,
                out.device_ptr(&stream).0,
            ]);
            let leaf = kernel(f, g.launch()?, args);
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let launched = leaf.run(&stream);
            let ended = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            launched?;
            let captured = Captured {
                graph: Some(ended?.context("empty prototype capture")?),
                stream: stream.clone(),
            };
            variants.push((dump, leaf, captured, qk, p32, p16, out));
        }
        if variants.len() != checkpoint_variants.len() {
            continue;
        }
        validated_geometries.push(query_tile);
        let masked_value_index = match input_layout {
            InputLayout::DenseHeadMajor => 511 * 64,
            InputLayout::SequenceMajor { heads } => 511 * heads * 64,
        };
        for revision in 0..5 {
            let make = |salt: usize| {
                (0..input_len)
                    .map(|i| {
                        bf16::from_f32(match revision {
                            0 => ((i * 7 + salt) % 17) as f32 / 32. - 0.25,
                            1 => ((i * 131 + salt) % 257) as f32 / 97. - 1.23,
                            2 => {
                                if (i + salt).is_multiple_of(2) {
                                    0.
                                } else {
                                    -0.
                                }
                            }
                            4 => {
                                if salt <= 2 {
                                    bf16::from_bits(0x7f7f).to_f32()
                                } else {
                                    0.125
                                }
                            }
                            _ => {
                                if (salt == 3 && i == masked_value_index) || (salt == 1 && i == 0) {
                                    f32::INFINITY
                                } else {
                                    ((i + salt) % 7) as f32 / 32.
                                }
                            }
                        })
                    })
                    .collect::<Vec<_>>()
            };
            stream.memcpy_htod(&make(1), &mut q)?;
            stream.memcpy_htod(&make(2), &mut k)?;
            stream.memcpy_htod(&make(3), &mut v)?;
            for kernel in &qk_program.kernels {
                kernel.run(&stream)?;
            }
            row.run(&stream)?;
            for kernel in &pv_program.kernels {
                kernel.run(&stream)?;
            }
            let qk_old = stream.memcpy_dtov(&qk_ref)?;
            let p32_old = stream.memcpy_dtov(&p32_ref)?;
            let p16_old = stream.memcpy_dtov(&p16_ref)?;
            let out_old = stream.memcpy_dtov(&out_ref)?;
            for (dump, _kernel, captured, qk, p32, p16, out) in &variants {
                for replay in 0..2 {
                    captured
                        .graph
                        .as_ref()
                        .context("prototype capture missing")?
                        .launch()?;
                    let label = format!("q{query_tile}/dump{dump}/revision{revision}/run{replay}");
                    if *dump {
                        same_bits(
                            &format!("{label} QK16"),
                            &stream.memcpy_dtov(qk)?,
                            &qk_old,
                            |v| u32::from(v.to_bits()),
                            |v| v.is_nan(),
                        );
                    }
                    same_bits(
                        &format!("{label} P32"),
                        &stream.memcpy_dtov(p32)?,
                        &p32_old,
                        |v| v.to_bits(),
                        |v| v.is_nan(),
                    );
                    same_bits(
                        &format!("{label} P16"),
                        &stream.memcpy_dtov(p16)?,
                        &p16_old,
                        |v| u32::from(v.to_bits()),
                        |v| v.is_nan(),
                    );
                    same_bits(
                        &format!("{label} O16"),
                        &stream.memcpy_dtov(out)?,
                        &out_old,
                        |v| u32::from(v.to_bits()),
                        |v| v.is_nan(),
                    );
                }
            }
        }
    }
    if let Some(path) = std::env::var_os("TNSR_ATTENTION_TRAINING_FORWARD_AUDIT") {
        std::fs::create_dir_all(&path)?;
        std::fs::write(
            std::path::Path::new(&path).join(format!("{label}-leaf-qualification.json")),
            serde_json::to_vec_pretty(&serde_json::json!({
            "resources":resources,"validated_query_tiles":validated_geometries,"batch_heads":bh,"input_layout":format!("{input_layout:?}"),"source_shape":source_shape,
            "actual_graph_replays_per_validated_variant":10,"checkpoint_comparison":"QK16/P32/P16/O16 vs ordinary viewed BMM+old128row+BMM; all requested variants exact",
            "nan_policy":"compare NaN categories, all other bits including signedzero exact","performance_claim":false,
            "binding_scope":"explicit leaf input layout, not actual QKV graph runtime rewrite activation"}))?,
        )?;
    }
    ensure!(
        !validated_geometries.is_empty(),
        "neither querytile passed prototype resource admission"
    );
    Ok(())
}

/// Explicitly requested diagnostic, disconnected from Mixed execution.
#[test]
fn training_attention_union_captured_component_comparison() -> Result<()> {
    use crate::tensor::TensorExpr;
    use half::bf16;
    use std::time::{Duration, Instant};
    if std::env::var_os("TNSR_ATTENTION_COMPONENT").is_none() {
        return Ok(());
    }
    ensure!(
        super::super::tests::cuda_available(),
        "component requires CUDA"
    );
    let reverse = std::env::var_os("TNSR_ATTENTION_COMPONENT_REVERSE").is_some();
    let sequence_major = std::env::var_os("TNSR_ATTENTION_COMPONENT_SEQUENCE_MAJOR").is_some();
    let diagnostic_screen =
        sequence_major || std::env::var_os("TNSR_ATTENTION_COMPONENT_DIAGNOSTIC").is_some();
    let context = CudaContext::new(0)?;
    // One private ordered stream owns all buffers and captured graphs.
    unsafe {
        context.disable_event_tracking();
    }
    let stream = context.new_stream()?;
    let directory = std::path::PathBuf::from(
        std::env::var_os("TNSR_ATTENTION_TRAINING_FORWARD_AUDIT")
            .context("component audit directory required")?,
    );
    std::fs::create_dir_all(&directory)?;
    let mut results = Vec::new();
    for bh in [64usize, 128, 256] {
        let input_layout = if sequence_major {
            InputLayout::SequenceMajor { heads: 8 }
        } else {
            InputLayout::DenseHeadMajor
        };
        let source_shape = if sequence_major {
            vec![bh / 8, 512, 8 * 64]
        } else {
            vec![bh / 8, 8, 512, 64]
        };
        let projected = |node, shape: Vec<usize>| {
            let expression = TensorExpr::node_ref(node, shape);
            if sequence_major {
                expression
                    .reshape(vec![bh / 8, 512, 8, 64])
                    .permute(vec![0, 2, 1, 3])
            } else {
                expression
            }
        };
        let input_len = bh * SEQUENCE * WIDTH;
        let score_len = bh * SEQUENCE * SEQUENCE;
        let make = |salt: usize| {
            (0..input_len)
                .map(|i| bf16::from_f32(((i * 131 + salt) % 257) as f32 / 397. - 0.3))
                .collect::<Vec<_>>()
        };
        let q = stream.memcpy_stod(&make(1))?;
        let k = stream.memcpy_stod(&make(2))?;
        let v = stream.memcpy_stod(&make(3))?;
        let mask_host = (0..SEQUENCE * SEQUENCE)
            .map(|i| {
                if i % SEQUENCE > i / SEQUENCE {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
            .collect::<Vec<_>>();
        let mask = stream.memcpy_stod(&mask_host)?;
        let qk_ref = stream.alloc_zeros::<bf16>(score_len)?;
        let p32_ref = stream.alloc_zeros::<f32>(score_len)?;
        let p16_ref = stream.alloc_zeros::<bf16>(score_len)?;
        let out_ref = stream.alloc_zeros::<bf16>(input_len)?;
        let binding16 = |values: &CudaSlice<bf16>| Binding {
            pointer: values.device_ptr(&stream).0,
            len: values.len(),
            dtype: DType::BF16,
        };
        let baseline_options = crate::ptx::linear::program::ProgramKernelOptions {
            shared_memory_reuse: true,
            narrow_bf16_tile: true,
            batched_attention_bf16_tile_policy:
                crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid,
            ..Default::default()
        };
        let mut qk_graph = TensorGraph::<bf16>::new();
        let qid = qk_graph.graph.add_node(TensorGraphNode::Input {
            name: "q",
            shape: source_shape.clone(),
        });
        let kid = qk_graph.graph.add_node(TensorGraphNode::Input {
            name: "k",
            shape: source_shape.clone(),
        });
        let qk_expr = projected(qid, source_shape.clone())
            .matmul(projected(kid, source_shape.clone()).transpose());
        qk_graph.add_expr(&qk_expr, false);
        let qk_program = Program::<BF16>::new_bound_with_kernel_options(
            &stream,
            qk_graph,
            HashMap::from([(qid, binding16(&q)), (kid, binding16(&k))]),
            Some(binding16(&qk_ref)),
            MatMulPrecision::NativeHalf,
            baseline_options,
        )?;
        let row = kernel(
            compile(
                &stream,
                super::super::attention_softmax::row_ir(0.125)?,
                "baseline-row",
            )?,
            LaunchConfig {
                grid_dim: (u32::try_from(bh * 512)?, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            vec![
                qk_ref.device_ptr(&stream).0,
                mask.device_ptr(&stream).0,
                p32_ref.device_ptr(&stream).0,
                p16_ref.device_ptr(&stream).0,
            ],
        );
        let mut pv_graph = TensorGraph::<bf16>::new();
        let pid = pv_graph.graph.add_node(TensorGraphNode::Input {
            name: "p",
            shape: vec![bh / 8, 8, 512, 512],
        });
        let vid = pv_graph.graph.add_node(TensorGraphNode::Input {
            name: "v",
            shape: source_shape.clone(),
        });
        pv_graph.add_expr(
            &TensorExpr::node_ref(pid, vec![bh / 8, 8, 512, 512])
                .matmul(projected(vid, source_shape.clone())),
            false,
        );
        let pv_program = Program::<BF16>::new_bound_with_kernel_options(
            &stream,
            pv_graph,
            HashMap::from([(pid, binding16(&p16_ref)), (vid, binding16(&v))]),
            Some(binding16(&out_ref)),
            MatMulPrecision::NativeHalf,
            baseline_options,
        )?;

        ensure!(
            qk_program.batched_attention_bf16_tile_counts.scores == 1
                && pv_program.batched_attention_bf16_tile_counts.query_width == 1,
            "actual baseline batched policies not installed"
        );
        std::fs::write(
            directory.join(format!("bh{bh}-qk-plan.txt")),
            &qk_program.plan,
        )?;
        std::fs::write(
            directory.join(format!("bh{bh}-pv-plan.txt")),
            &pv_program.plan,
        )?;
        let capture = |kernels: &[Kernel]| -> Result<Captured> {
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let launched = (|| -> Result<()> {
                for _ in 0..8 {
                    for kernel in kernels {
                        kernel.run(&stream)?;
                    }
                }
                Ok(())
            })();
            let ended = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            launched?;
            Ok(Captured {
                graph: Some(ended?.context("empty component capture")?),
                stream: stream.clone(),
            })
        };
        let baseline_kernels = qk_program
            .kernels
            .iter()
            .cloned()
            .chain(std::iter::once(row))
            .chain(pv_program.kernels.iter().cloned())
            .collect::<Vec<_>>();
        let baseline = capture(&baseline_kernels)?;
        baseline.graph.as_ref().unwrap().launch()?;
        let qk_old = if diagnostic_screen {
            Some(stream.memcpy_dtov(&qk_ref)?)
        } else {
            None
        };
        let p32_old = stream.memcpy_dtov(&p32_ref)?;
        let p16_old = stream.memcpy_dtov(&p16_ref)?;
        let out_old = stream.memcpy_dtov(&out_ref)?;
        let resource = |kernel: &Kernel| -> Result<serde_json::Value> {
            Ok(
                serde_json::json!({"label":kernel.label,"grid":[kernel.config.grid_dim.0,kernel.config.grid_dim.1,kernel.config.grid_dim.2],"block":[kernel.config.block_dim.0,kernel.config.block_dim.1,kernel.config.block_dim.2],"registers":kernel.function.num_regs()?,"shared":kernel.function.shared_size_bytes()?,"local":kernel.function.local_size_bytes()?,"api_capacity":kernel.function.occupancy_max_active_blocks_per_multiprocessor(kernel.config.block_dim.0*kernel.config.block_dim.1*kernel.config.block_dim.2,0,None)?}),
            )
        };
        let baseline_resources = baseline_kernels
            .iter()
            .map(resource)
            .collect::<Result<Vec<_>>>()?;
        let mut requested = if diagnostic_screen {
            vec![(32, true), (32, false)]
        } else {
            vec![(32, false), (16, false)]
        };
        if sequence_major {
            requested.truncate(1);
        }
        if reverse {
            requested.reverse();
        }
        for (query_tile, dump) in requested {
            let g = Geometry::new(bh, query_tile)?;
            let qk = if dump {
                Some(stream.alloc_zeros::<bf16>(score_len)?)
            } else {
                None
            };
            let p32 = stream.alloc_zeros::<f32>(score_len)?;
            let p16 = stream.alloc_zeros::<bf16>(score_len)?;
            let out = stream.alloc_zeros::<bf16>(input_len)?;
            let f = compile(
                &stream,
                ir_with_input_layout(g, dump, input_layout)?,
                &format!("component-bh{bh}-q{query_tile}-dump{dump}"),
            )?;
            ensure!(admitted(&f, g)?, "component kernel not admitted");
            let mut pointers = vec![
                q.device_ptr(&stream).0,
                k.device_ptr(&stream).0,
                v.device_ptr(&stream).0,
                mask.device_ptr(&stream).0,
            ];
            if let Some(qk) = &qk {
                pointers.push(qk.device_ptr(&stream).0);
            }
            pointers.extend([
                p32.device_ptr(&stream).0,
                p16.device_ptr(&stream).0,
                out.device_ptr(&stream).0,
            ]);
            let leaf = kernel(f, g.launch()?, pointers);
            let candidate = capture(std::slice::from_ref(&leaf))?;
            let validate = || -> Result<()> {
                if let (Some(qk), Some(expected)) = (&qk, &qk_old) {
                    same_bits(
                        "component QK16",
                        &stream.memcpy_dtov(qk)?,
                        expected,
                        |x| u32::from(x.to_bits()),
                        |x| x.is_nan(),
                    );
                }
                same_bits(
                    "component P32",
                    &stream.memcpy_dtov(&p32)?,
                    &p32_old,
                    |x| x.to_bits(),
                    |x| x.is_nan(),
                );
                same_bits(
                    "component P16",
                    &stream.memcpy_dtov(&p16)?,
                    &p16_old,
                    |x| u32::from(x.to_bits()),
                    |x| x.is_nan(),
                );
                same_bits(
                    "component O16",
                    &stream.memcpy_dtov(&out)?,
                    &out_old,
                    |x| u32::from(x.to_bits()),
                    |x| x.is_nan(),
                );
                Ok(())
            };
            candidate.graph.as_ref().unwrap().launch()?;
            validate()?;
            let graphs = [&baseline, &candidate];
            let started = Instant::now();
            let mut warm_pairs = 0;
            while started.elapsed() < Duration::from_millis(500) {
                for mode in if (warm_pairs % 2 == 1) ^ reverse {
                    [1, 0]
                } else {
                    [0, 1]
                } {
                    graphs[mode].graph.as_ref().unwrap().launch()?;
                }
                stream.synchronize()?;
                warm_pairs += 1;
            }
            let warm_ms = started.elapsed().as_secs_f64() * 1000.;
            let mut pairs = Vec::new();
            for pair in 0..32 {
                let order = if (pair % 2 == 1) ^ reverse {
                    [1, 0]
                } else {
                    [0, 1]
                };
                let mut us = [0.; 2];
                for mode in order {
                    let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    graphs[mode].graph.as_ref().unwrap().launch()?;
                    let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    us[mode] = f64::from(start.elapsed_ms(&end)?) * 1000. / 8.;
                }
                pairs.push(serde_json::json!({"pair":pair,"order":order,"baseline_us":us[0],"candidate_us":us[1]}));
            }
            validate()?;
            same_bits(
                "baseline replay P32",
                &stream.memcpy_dtov(&p32_ref)?,
                &p32_old,
                |x| x.to_bits(),
                |x| x.is_nan(),
            );
            same_bits(
                "baseline replay P16",
                &stream.memcpy_dtov(&p16_ref)?,
                &p16_old,
                |x| u32::from(x.to_bits()),
                |x| x.is_nan(),
            );
            same_bits(
                "baseline replay O16",
                &stream.memcpy_dtov(&out_ref)?,
                &out_old,
                |x| u32::from(x.to_bits()),
                |x| x.is_nan(),
            );
            results.push(serde_json::json!({"batch":bh/8,"heads":8,"batch_heads":bh,"query_tile":query_tile,"qk_checkpoint":dump,"input_layout":format!("{input_layout:?}"),"input_source_shape":source_shape,"candidate_resources":resource(&leaf)?,"baseline_resources":baseline_resources,"baseline_batched_policy":"TallScoresByGrid","baseline_score_tall":qk_program.batched_attention_bf16_tile_counts.tall,"baseline_pv_tall":pv_program.batched_attention_bf16_tile_counts.tall,"graph_repeats":8,"warmup_pairs":warm_pairs,"warmup_ms":warm_ms,"pairs":pairs,"checks":"full exact finite P32/P16/O16 and optional QK16 before and after timing","scope":"resident captured forward component on explicitly labelled physical inputs; checkpoints retained; not full model or parity"}));
            std::fs::write(
                directory.join("component-results.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"reverse":reverse,"sequence_major":sequence_major,"diagnostic_screen":diagnostic_screen,"results":results}),
                )?,
            )?;
        }
    }
    Ok(())
}

mod sequence_major;
