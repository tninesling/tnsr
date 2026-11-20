//! Local qualification for the exact warp-row IR builders.
use super::*;
use crate::tile::Stmt;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Dag {
    Zero,
    NegativeInfinity,
    Leaf(usize),
    Pair(bool, Box<Dag>, Box<Dag>),
    Fma(Box<Dag>, usize),
}
fn pair(max: bool, a: Dag, b: Dag) -> Dag {
    Dag::Pair(max, Box::new(a), Box::new(b))
}
fn local(lane: usize, max: bool) -> Dag {
    (0..4).fold(
        if max {
            Dag::NegativeInfinity
        } else {
            Dag::Zero
        },
        |acc, p| pair(max, acc, Dag::Leaf(lane + 128 * p)),
    )
}
fn old_tree(max: bool) -> Dag {
    let mut v: Vec<_> = (0..128).map(|lane| local(lane, max)).collect();
    for stride in [64, 32, 16, 8, 4, 2, 1] {
        for lane in 0..stride {
            v[lane] = pair(max, v[lane].clone(), v[lane + stride].clone());
        }
    }
    v.remove(0)
}
fn virtual_warp(max: bool) -> Vec<Dag> {
    let mut v: Vec<_> = (0..32)
        .map(|lane| {
            pair(
                max,
                pair(max, local(lane, max), local(lane + 64, max)),
                pair(max, local(lane + 32, max), local(lane + 96, max)),
            )
        })
        .collect();
    for stride in [16, 8, 4, 2, 1] {
        let old = v.clone();
        for lane in 0..32 {
            v[lane] = pair(
                max,
                old[lane].clone(),
                if lane + stride < 32 {
                    old[lane + stride].clone()
                } else if max {
                    Dag::NegativeInfinity
                } else {
                    Dag::Zero
                },
            );
        }
    }
    v
}

#[test]
fn virtual128_exact_ordered_dag_and_explicit_leader_result() {
    for max in [false, true] {
        let canonical = old_tree(max);
        let lanes = virtual_warp(max);
        assert_eq!(canonical, lanes[0]);
        // Canonical down-reduce only promises lane0. An explicit broadcast
        // copies THIS ordered tree; it does not reinterpret every lane's value.
        assert!(lanes.iter().skip(1).any(|n| n != &canonical));
        let broadcast = vec![lanes[0].clone(); 32];
        assert!(broadcast.iter().all(|n| n == &canonical));
    }
}

#[test]
fn virtual128_compact_dot_has_exact_four_term_ffma_dependency_dag() {
    fn dot(lane: usize) -> Dag {
        (0..4).fold(Dag::Zero, |a, p| Dag::Fma(Box::new(a), lane + 128 * p))
    }
    let mut old: Vec<_> = (0..128).map(dot).collect();
    for stride in [64, 32, 16, 8, 4, 2, 1] {
        for lane in 0..stride {
            old[lane] = pair(false, old[lane].clone(), old[lane + stride].clone());
        }
    }
    let mut warp: Vec<_> = (0..32)
        .map(|lane| {
            pair(
                false,
                pair(false, dot(lane), dot(lane + 64)),
                pair(false, dot(lane + 32), dot(lane + 96)),
            )
        })
        .collect();
    for stride in [16, 8, 4, 2, 1] {
        let previous = warp.clone();
        for lane in 0..32 {
            warp[lane] = pair(
                false,
                previous[lane].clone(),
                if lane + stride < 32 {
                    previous[lane + stride].clone()
                } else {
                    Dag::Zero
                },
            );
        }
    }
    assert_eq!(old[0], warp[0]);
}

fn max_num(a: f32, b: f32) -> f32 {
    // Explicit CPU oracle model; exact PTX max/nonfinite classes are separately
    // required on the device, rather than inferred from Rust's equal-zero tie.
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a == 0. && b == 0. {
        f32::from_bits((a.to_bits() & b.to_bits()) & 0x8000_0000)
    } else if a >= b {
        a
    } else {
        b
    }
}
fn op(max: bool, a: f32, b: f32) -> f32 {
    if max { max_num(a, b) } else { a + b }
}
fn old_reduce(values: &[f32], max: bool) -> f32 {
    let mut partial = [if max { f32::NEG_INFINITY } else { 0. }; 128];
    for (t, v) in partial.iter_mut().enumerate() {
        for p in 0..4 {
            *v = op(max, *v, values[t + 128 * p]);
        }
    }
    for stride in [64, 32, 16, 8, 4, 2, 1] {
        for t in 0..stride {
            partial[t] = op(max, partial[t], partial[t + stride]);
        }
    }
    partial[0]
}
fn warp_reduce_virtual_partials(partial: &[[f32; 4]; 32], max: bool) -> f32 {
    let mut v = std::array::from_fn::<_, 32, _>(|l| {
        op(
            max,
            op(max, partial[l][0], partial[l][2]),
            op(max, partial[l][1], partial[l][3]),
        )
    });
    for stride in [16, 8, 4, 2, 1] {
        let old = v;
        for lane in 0..32 {
            v[lane] = op(
                max,
                old[lane],
                if lane + stride < 32 {
                    old[lane + stride]
                } else if max {
                    f32::NEG_INFINITY
                } else {
                    0.
                },
            );
        }
    }
    v[0]
}
fn new_reduce(values: &[f32], max: bool) -> f32 {
    let mut partial = [[if max { f32::NEG_INFINITY } else { 0. }; 4]; 32];
    for lane in 0..32 {
        for group in 0..4 {
            for p in 0..4 {
                partial[lane][group] = op(
                    max,
                    partial[lane][group],
                    values[lane + 32 * group + 128 * p],
                );
            }
        }
    }
    warp_reduce_virtual_partials(&partial, max)
}
fn equivalent(a: f32, b: f32) {
    assert!(
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
        "{a:?} vs {b:?}"
    );
}

#[test]
fn virtual128_cpu_sum_max_ffma_and_softmax_nonfinite_order() {
    for special in [
        None,
        Some(0.),
        Some(-0.),
        Some(f32::INFINITY),
        Some(f32::NEG_INFINITY),
        Some(f32::NAN),
    ] {
        let mut values: Vec<_> = (0..512)
            .map(|k| match k % 4 {
                0 => 1e20,
                1 => -1e20,
                2 => (k % 17) as f32 / 128.,
                _ => -0.,
            })
            .collect();
        if let Some(x) = special {
            values[0] = x;
            values[511] = x;
        }
        for max in [false, true] {
            equivalent(old_reduce(&values, max), new_reduce(&values, max));
        }
    }
    for value in [0., -0., 1., -1., f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let values = vec![value; 512];
        for max in [false, true] {
            equivalent(old_reduce(&values, max), new_reduce(&values, max));
        }
    }
    let zeros: Vec<_> = (0..512)
        .map(|k| if k % 2 == 0 { 0. } else { -0. })
        .collect();
    for max in [false, true] {
        equivalent(old_reduce(&zeros, max), new_reduce(&zeros, max));
    }
    for query in [0, 1, 31, 32, 127, 128, 255, 511] {
        let score: Vec<_> = (0..512)
            .map(|key| {
                half::bf16::from_f32((key % 37) as f32 / 16. - 1.).to_f32() * 0.125
                    + if key > query { f32::NEG_INFINITY } else { 0. }
            })
            .collect();
        let old_max = old_reduce(&score, true);
        let new_max = new_reduce(&score, true);
        equivalent(old_max, new_max);
        let old_exp: Vec<_> = score.iter().map(|x| (x - old_max).exp()).collect();
        let new_exp: Vec<_> = score.iter().map(|x| (x - new_max).exp()).collect();
        let old_sum = old_reduce(&old_exp, false);
        let new_sum = new_reduce(&new_exp, false);
        equivalent(old_sum, new_sum);
        let p: Vec<_> = old_exp.iter().map(|x| x / old_sum).collect();
        for key in 0..512 {
            equivalent(p[key], new_exp[key] / new_sum);
        }
        let g: Vec<_> = (0..512)
            .map(|k| half::bf16::from_f32((k % 29) as f32 / 32. - 0.5).to_f32())
            .collect();
        for fused in [false, true] {
            let mut old = [0.0f32; 128];
            let mut new = [[0.0f32; 4]; 32];
            for (t, accumulator) in old.iter_mut().enumerate() {
                for part in 0..4 {
                    let key = t + 128 * part;
                    *accumulator = if fused {
                        p[key].mul_add(g[key], *accumulator)
                    } else {
                        *accumulator + p[key] * g[key]
                    };
                }
            }
            for (lane, partials) in new.iter_mut().enumerate() {
                for (group, accumulator) in partials.iter_mut().enumerate() {
                    for part in 0..4 {
                        let key = lane + 32 * group + 128 * part;
                        *accumulator = if fused {
                            p[key].mul_add(g[key], *accumulator)
                        } else {
                            *accumulator + p[key] * g[key]
                        };
                    }
                }
            }
            for stride in [64, 32, 16, 8, 4, 2, 1] {
                for t in 0..stride {
                    let peer = old[t + stride];
                    old[t] += peer;
                }
            }
            let dot = warp_reduce_virtual_partials(&new, false);
            equivalent(old[0], dot);
            for key in 0..512 {
                equivalent(
                    (p[key] * (g[key] - old[0])) * 0.125,
                    (p[key] * (g[key] - dot)) * 0.125,
                );
            }
        }
    }
}

#[test]
fn warp_geometry_requires_full_rows_and_strict_initial_shape() {
    assert_eq!(
        geometry(&[8, 8, 512, 512], 64),
        Some(WarpRowsLaunch {
            rows: 32768,
            grid: (4096, 1, 1),
            block: (32, 8, 1)
        })
    );
    for (shape, width) in [
        (vec![1, 8, 512, 512], 64),
        (vec![8, 8, 511, 512], 64),
        (vec![8, 8, 512, 511], 64),
        (vec![8, 8, 512, 512], 65),
        (vec![0, 64, 512, 512], 64),
        (vec![64, 512, 512], 64),
        (vec![usize::MAX, 64, 512, 512], 64),
    ] {
        assert!(geometry(&shape, width).is_none());
    }
}

#[test]
fn warp_ir_dtype_zero_shared_no_barriers_canonical_down_and_distinct_broadcast() -> Result<()> {
    for (ir, reductions, broadcasts) in [(forward_ir(0.125)?, 2, 2), (backward_ir()?, 1, 1)] {
        assert_eq!(ir.shared_mem_bytes, 0);
        ir.validate_layouts()?;
        assert!(!ir.body.stmts.iter().any(|s| matches!(
            s,
            Stmt::AllocTile { .. } | Stmt::Barrier | Stmt::If { .. } | Stmt::ForLoop { .. }
        )));
        assert_eq!(
            ir.body
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
            reductions
        );
        assert!(!ir.body.stmts.iter().any(|s| matches!(
            s,
            Stmt::WarpReduce {
                broadcast: true,
                ..
            }
        )));
        assert_eq!(
            ir.body
                .stmts
                .iter()
                .filter(|s| matches!(s, Stmt::WarpBroadcastLeader { .. }))
                .count(),
            broadcasts
        );
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
        tiles.optimize_indices()?;
        let source = PtxGraph::from(tiles).module_source();
        assert!(!source.contains("bar.sync") && !source.contains(".shared "));
        assert!(!source.contains("shfl.sync.bfly"));
        assert_eq!(source.matches("shfl.sync.down.b32").count(), 5 * reductions);
        assert_eq!(source.matches("shfl.sync.idx.b32").count(), broadcasts);
        assert!(source.contains("0x1f, 0xffffffff;"));
        assert!(source.contains("cvt.rn.bf16.f32"));
    }
    assert!(forward_ir(0.25).is_err());
    Ok(())
}

// Future direct-GPU protocol lives in gpu_tests.rs. It compares BOTH generated
// kernels against the existing128-thread row_ir implementations on identical
// physical typed buffers. Exact all-nonNaN bits, zero signs and NaN classes are
// required before any timing or resident/full-model integration is considered.
mod gpu_tests;

#[test]
fn warp_row_physical_abi_and_inplace_score_read_before_write() -> Result<()> {
    let old = super::super::attention_softmax::row_ir(0.125)?;
    let new = forward_ir(0.125)?;
    let params = |ir: &TileIR| {
        ir.params
            .iter()
            .map(|p| (p.name.clone(), p.dtype, p.is_input))
            .collect::<Vec<_>>()
    };
    assert_eq!(params(&old), params(&new));
    let old = super::super::attention_softmax_backward::row_ir()?;
    let new = backward_ir()?;
    assert_eq!(params(&old), params(&new));

    let mut b = ScalarBuilder::default();
    let scores = b.var();
    b.stmts.push(Stmt::AllocTile {
        var: scores,
        space: crate::tile::MemorySpace::Shared,
        layout: crate::tile::TileLayout::Shared(crate::tile::SharedLayout::row_major(COLUMNS)),
        dtype: DType::BF16,
        rows: 32,
        cols: COLUMNS,
    });
    forward_row(
        &mut b,
        ForwardRowBindings {
            scores: ScalarMemory::Shared(scores),
            score_row: Expr::ThreadIdx(Dim::Y),
            mask: ScalarMemory::Global("mask".into()),
            mask_row: Expr::ThreadIdx(Dim::Y),
            probability: ScalarMemory::Global("probability".into()),
            probability_row: Expr::ThreadIdx(Dim::Y),
            narrowed: ScalarMemory::Shared(scores),
            narrowed_row: Expr::ThreadIdx(Dim::Y),
            narrowed_checkpoint: Some((
                ScalarMemory::Global("narrowed".into()),
                Expr::ThreadIdx(Dim::Y),
            )),
        },
    );
    let reads = b
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            matches!(s,Stmt::LoadScalar {source:ScalarMemory::Shared(v),..} if *v==scores)
                .then_some(i)
        })
        .collect::<Vec<_>>();
    let stores = b
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            matches!(s,Stmt::StoreScalar {target:ScalarMemory::Shared(v),..} if *v==scores)
                .then_some(i)
        })
        .collect::<Vec<_>>();
    assert_eq!(reads.len(), 16);
    assert_eq!(stores.len(), 16);
    assert!(reads[15] < stores[0]);
    let ir = TileIR {
        kernel_name: "warp_row_inplace_builder".into(),
        params: vec![
            KernelParam {
                name: "mask".into(),
                dtype: DType::F32,
                is_input: true,
            },
            KernelParam {
                name: "probability".into(),
                dtype: DType::F32,
                is_input: false,
            },
            KernelParam {
                name: "narrowed".into(),
                dtype: DType::BF16,
                is_input: false,
            },
        ],
        body: Block { stmts: b.stmts },
        shared_mem_bytes: 32 * COLUMNS * 2,
    };
    ir.validate_layouts()?;
    Ok(())
}
