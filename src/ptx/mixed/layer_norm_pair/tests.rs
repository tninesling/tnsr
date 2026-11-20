use super::*;
pub(super) fn decoder(batch: usize) -> super::super::test_decoder::Model {
    decoder_with_vocabulary(batch, 31)
}
pub(super) fn decoder_with_vocabulary(
    batch: usize,
    vocab: usize,
) -> super::super::test_decoder::Model {
    decoder_with_dimensions(batch, 512, 8, vocab)
}
pub(super) fn decoder_with_dimensions(
    batch: usize,
    width: usize,
    heads: usize,
    vocab: usize,
) -> super::super::test_decoder::Model {
    decoder_with_shape(batch, width, heads, 512, vocab)
}
pub(super) fn decoder_with_shape(
    batch: usize,
    width: usize,
    heads: usize,
    sequence: usize,
    vocab: usize,
) -> super::super::test_decoder::Model {
    use super::super::test_decoder::{Config, synthetic};
    synthetic(Config {
        width,
        layers: 4,
        heads,
        hidden: 64,
        sequence,
        batch,
        vocab,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.0,
    })
}
#[test]
fn actual_decoder_has_nine_closed_f32_inner_pairs_and_preserved_interfaces() {
    for batch in [8, 32] {
        let model = decoder(batch);
        let (g, _) = model.training_graph().unwrap();
        let pairs = recognize(&g).unwrap();
        assert_eq!(pairs.len(), 9);
        let mut claimed = HashSet::new();
        for pair in pairs {
            assert_eq!(pair.shape, vec![batch, 512, 512]);
            assert!(matches!(
                g[pair.inputs[0]],
                TensorGraphNode::Cast {
                    dtype: ValueDType::F32,
                    ..
                }
            ));
            assert!(!pair.members.contains(&pair.inputs[0]));
            assert!(pair.members.is_disjoint(&claimed));
            claimed.extend(&pair.members);
            assert!(
                pair.members
                    .iter()
                    .all(|n| !matches!(g[*n], TensorGraphNode::ReduceAxis { axis: 0 | 1, .. }))
            );
            assert!(
                ssa::checkpoint_candidates(&pair)
                    .unwrap()
                    .requires_baseline_contraction_audit
            );
            assert!(
                g.graph
                    .neighbors_directed(pair.normalized, Direction::Outgoing)
                    .any(|n| !pair.members.contains(&n))
            );
        }
    }
}
#[test]
fn pair_rejects_internal_escapes_n_escape_and_unsupported_sequence() {
    let model = decoder(8);
    let (g, _) = model.training_graph().unwrap();
    let pair = recognize(&g).unwrap().remove(0);
    for node in [
        pair.mean,
        pair.variance_plus_epsilon,
        pair.inverse_std,
        pair.normalized,
    ] {
        let mut escaped = g.clone();
        let out = escaped.graph.add_node(TensorGraphNode::Reshape {
            shape: g[node].shape().clone(),
        });
        escaped.graph.add_edge(node, out, 0);
        assert_eq!(recognize(&escaped).unwrap().len(), 8);
    }
    let (small, _) = decoder_with_shape(1, 512, 8, 128, 31)
        .training_graph()
        .unwrap();
    assert!(recognize(&small).unwrap().is_empty());
}
#[test]
fn row_tree_preserves_lane_stride_and_primitive_variance_order() {
    let x: Vec<_> = (0..512)
        .map(|i| match i % 4 {
            0 => 1e10,
            1 => 1.,
            2 => -1e10,
            _ => -1.,
        })
        .collect();
    let gamma = vec![1.; 512];
    let beta = vec![0.; 512];
    let g: Vec<_> = (0..512).map(|i| (i % 13) as f32 / 32.).collect();
    let saved = ssa::forward(&x, &gamma, &beta, 1e-5);
    let dx = ssa::backward(&x, &gamma, &g, &saved);
    assert_eq!(saved.output.len(), 512);
    assert!(dx.iter().all(|v| v.is_finite()));
    let mut lanes = [0_f32; 128];
    for i in 0..128 {
        for k in 0..4 {
            lanes[i] += x[i + 128 * k];
        }
    }
    for s in [64, 32, 16, 8, 4, 2, 1] {
        let previous = lanes;
        for i in 0..s {
            lanes[i] = previous[i] + previous[i + s];
        }
    }
    assert_eq!(ssa::sum(&x).to_bits(), lanes[0].to_bits());
    assert_eq!(saved.normalized.len(), 512);
}
#[test]
fn row_ir_requires_tail_audit_and_keeps_all_cta_lanes_in_two_exact_trees() {
    let (g, _) = decoder(8).training_graph().unwrap();
    let pair = recognize(&g).unwrap().remove(0);
    assert!(planned_contiguous(&pair, &g));
    assert!(kernel::backward(&pair, kernel::TailBoundary::Unresolved, DType::F32).is_err());
    fn inspect(stmts: &[crate::tile::Stmt]) -> (usize, usize, usize) {
        use crate::tile::Stmt;
        let mut counts = (0, 0, 0);
        for s in stmts {
            match s {
                Stmt::Barrier => counts.0 += 1,
                Stmt::BoundsCheck { .. } => counts.1 += 1,
                Stmt::AllocTile { .. } => counts.2 += 1,
                Stmt::If { body, .. } => {
                    let c = inspect(&body.stmts);
                    assert_eq!(c.0, 0, "barrier must remain uniform");
                    counts.1 += c.1;
                    counts.2 += c.2;
                }
                _ => {}
            }
        }
        counts
    }
    for ir in [
        kernel::forward(&pair).unwrap(),
        kernel::backward(&pair, kernel::TailBoundary::SeparateMeanTerm, DType::F32).unwrap(),
        kernel::backward(
            &pair,
            kernel::TailBoundary::OriginalPointwiseOpportunity,
            DType::F32,
        )
        .unwrap(),
    ] {
        assert_eq!(inspect(&ir.body.stmts), (16, 0, 3));
        assert_eq!(ir.shared_mem_bytes, 1536);
        assert!(ir.params.iter().all(|p| p.dtype == DType::F32));
    }
}
#[test]
fn runtime_boundary_proof_rejects_reorder_oob_and_predicates() {
    let (g, _) = decoder(8).training_graph().unwrap();
    let pair = recognize(&g).unwrap().remove(0);
    let mut bindings = HashMap::new();
    for &n in &pair.storage_roots {
        let shape = g[n].shape().clone();
        bindings.insert(
            n,
            Value {
                binding: Binding {
                    pointer: n.index() as u64 + 1,
                    len: shape.iter().product(),
                    dtype: DType::F32,
                },
                source_shape: shape.clone(),
                view: VirtualTensor::identity(n, shape, DType::F32),
                chain: Vec::new(),
            },
        );
    }
    assert!(contiguous_bindings(&pair, &bindings, false).is_some());
    assert!(contiguous_bindings(&pair, &bindings, true).is_some());
    let incoming = pair.storage_roots[3];
    let mut promoted = bindings.clone();
    let value = promoted.get_mut(&incoming).unwrap();
    value.binding.dtype = DType::BF16;
    value.view.dtype = DType::BF16;
    assert_eq!(
        contiguous_bindings(&pair, &promoted, true).unwrap()[2]
            .binding
            .dtype,
        DType::BF16
    );
    let n = pair.storage_roots[0];
    let mut bad = bindings.clone();
    bad.get_mut(&n).unwrap().view.predicate = Some(crate::tile::IndexPredicate::And(Vec::new()));
    assert!(contiguous_bindings(&pair, &bad, false).is_none());
    let mut bad = bindings.clone();
    bad.get_mut(&n).unwrap().view.out_of_bounds = Some(crate::tile::ScalarValue::F32(0));
    assert!(contiguous_bindings(&pair, &bad, false).is_none());
}

#[test]
fn explicit_rounding_is_local_and_ptx_preserves_ordinary_affine_opportunities() {
    let (g, _) = decoder(8).training_graph().unwrap();
    let pair = recognize(&g).unwrap().remove(0);
    for ir in [
        kernel::forward(&pair).unwrap(),
        kernel::backward(
            &pair,
            kernel::TailBoundary::OriginalPointwiseOpportunity,
            DType::BF16,
        )
        .unwrap(),
    ] {
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
        tiles.optimize_indices().unwrap();
        let ptx = PtxGraph::from(tiles).to_ptx();
        assert!(ptx.contains("mul.rn.f32"));
        assert!(ptx.contains("add.rn.f32"));
        assert!(ptx.contains("sub.rn.f32"));
        assert!(
            ptx.contains("mul.f32"),
            "legitimate original contraction opportunity remains"
        );
        assert!(ptx.contains("add.f32"));
    }
}

#[test]
fn width_and_row_coverage_guards_are_checked_and_bounded() {
    for shape in [
        [1, 512, 1024],
        [2, 512, 512],
        [8, 512, 512],
        [2, 128, 512],
        [2, 128, 1024],
    ] {
        assert!(supported_geometry(&shape));
    }
    for shape in [
        [0, 512, 512],
        [1, 128, 512],
        [3, 128, 512],
        [2, 128, 2048],
        [1, 512, 256],
        [1, 512, 2048],
        [usize::MAX, 512, 1024],
    ] {
        assert!(!supported_geometry(&shape));
    }
    assert!(!supported_geometry(&[512, 1024]));
}
#[test]
fn actual_decoder_small_batch_and_wide_templates_keep_nine_closed_pairs() {
    for (batch, width, heads) in [(1, 1024, 16), (2, 512, 8), (8, 512, 8)] {
        let (graph, _) = decoder_with_dimensions(batch, width, heads, 31)
            .training_graph()
            .unwrap();
        let pairs = recognize(&graph).unwrap();
        assert_eq!(pairs.len(), 9, "B{batch} W{width}");
        for pair in pairs {
            assert_eq!(pair.shape, vec![batch, 512, width]);
            assert!(
                pair.program
                    .iter()
                    .any(|op| op.kind == Kind::Constant((1.0_f32 / width as f32).to_bits()))
            );
            let stats = lifetime::stat_layouts(&pair).unwrap();
            assert!(
                stats
                    .iter()
                    .all(|value| value.source_shape == [batch, 512, 1])
            );
            let ir = kernel::forward(&pair).unwrap();
            assert_eq!(ir.shared_mem_bytes, 1536);
        }
    }
}
#[test]
fn wide_row_tree_visits_eight_features_in_ascending_order() {
    for width in [512, 1024] {
        let x = (0..width)
            .map(|i| match i % 4 {
                0 => 1e10,
                1 => 1.,
                2 => -1e10,
                _ => -1.,
            })
            .collect::<Vec<f32>>();
        let mut lanes = [0_f32; 128];
        for lane in 0..128 {
            for k in 0..width / 128 {
                lanes[lane] += x[lane + 128 * k];
            }
        }
        for stride in [64, 32, 16, 8, 4, 2, 1] {
            let previous = lanes;
            for lane in 0..stride {
                lanes[lane] = previous[lane] + previous[lane + stride];
            }
        }
        assert_eq!(ssa::sum(&x).to_bits(), lanes[0].to_bits());
        let gamma = vec![1.; width];
        let beta = vec![0.; width];
        let incoming = vec![0.125; width];
        let saved = ssa::forward(&x, &gamma, &beta, 1e-5);
        assert_eq!(ssa::backward(&x, &gamma, &incoming, &saved).len(), width);
    }
}
#[test]
fn w512_ir_remains_identical_across_new_small_and_existing_batches() {
    let (large, _) = decoder(8).training_graph().unwrap();
    let old = recognize(&large).unwrap().remove(0);
    let (small, _) = decoder(2).training_graph().unwrap();
    let new = recognize(&small).unwrap().remove(0);
    assert_eq!(
        render_ptx(kernel::forward(&old).unwrap()),
        render_ptx(kernel::forward(&new).unwrap())
    );
    for tail in [
        kernel::TailBoundary::SeparateMeanTerm,
        kernel::TailBoundary::OriginalPointwiseOpportunity,
    ] {
        for dtype in [DType::F32, DType::BF16] {
            assert_eq!(
                render_ptx(kernel::backward(&old, tail, dtype).unwrap()),
                render_ptx(kernel::backward(&new, tail, dtype).unwrap())
            );
        }
    }
}

fn render_ptx(ir: crate::tile::TileIR) -> String {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
    tiles.optimize_indices().unwrap();
    PtxGraph::from(tiles).to_ptx()
}

#[test]
fn actual_s128_b2_templates_preserve_all_nine_closed_pairs() {
    for (width, heads) in [(512, 8), (1024, 16)] {
        let (graph, _) = decoder_with_shape(2, width, heads, 128, 31)
            .training_graph()
            .unwrap();
        let pairs = recognize(&graph).unwrap();
        assert_eq!(pairs.len(), 9);
        let (old_graph, _) = decoder_with_dimensions(2, width, heads, 31)
            .training_graph()
            .unwrap();
        let old = recognize(&old_graph).unwrap().remove(0);
        for pair in pairs {
            assert_eq!(pair.shape, vec![2, 128, width]);
            assert!(
                lifetime::stat_layouts(&pair)
                    .unwrap()
                    .iter()
                    .all(|stat| stat.source_shape == [2, 128, 1])
            );
            assert_eq!(
                render_ptx(kernel::forward(&pair).unwrap()),
                render_ptx(kernel::forward(&old).unwrap())
            );
            for tail in [
                kernel::TailBoundary::SeparateMeanTerm,
                kernel::TailBoundary::OriginalPointwiseOpportunity,
            ] {
                for dtype in [DType::F32, DType::BF16] {
                    assert_eq!(
                        render_ptx(kernel::backward(&pair, tail, dtype).unwrap()),
                        render_ptx(kernel::backward(&old, tail, dtype).unwrap())
                    );
                }
            }
        }
    }
}
