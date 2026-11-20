//! Original graph identities remain the keys. Row storage is a new physical
//! representation of uniform a/r, never a new mathematical graph operation.
use super::*;
#[derive(Clone)]
pub(super) struct StatLayout {
    pub node: NodeIndex,
    pub source_shape: Vec<usize>,
    pub view: VirtualTensor,
    pub chain: Vec<TensorGraphNode<f32>>,
}
pub(super) fn stat_layouts(pair: &LayerNormPair) -> Result<Vec<StatLayout>> {
    ensure!(
        supported_geometry(&pair.shape),
        "LayerNorm stat geometry invalid"
    );
    let row = vec![pair.shape[0], pair.shape[1], 1];
    [pair.mean, pair.variance_plus_epsilon, pair.inverse_std]
        .into_iter()
        .map(|node| {
            let mut view = VirtualTensor::identity(node, row.clone(), DType::F32);
            let chain = if node == pair.mean {
                Vec::new()
            } else {
                view = view.broadcast_axis(2, pair.shape[2])?;
                vec![TensorGraphNode::BroadcastAxis {
                    axis: 2,
                    shape: pair.shape.clone(),
                }]
            };
            Ok(StatLayout {
                node,
                source_shape: row.clone(),
                view,
                chain,
            })
        })
        .collect()
}
/// Only newly introduced backward reads extend the old graph's direct last
/// consumers. n's existing gamma column use remains ordinary graph liveness;
/// beta is forward-only. Pinning all births to zero would defeat prefix reuse.
pub(super) fn extend_last_consumers(
    pair: &LayerNormPair,
    positions: &HashMap<NodeIndex, usize>,
    last: &mut HashMap<NodeIndex, usize>,
) -> Result<()> {
    let end = *positions
        .get(&pair.backward)
        .context("LayerNorm backward absent from topology")?;
    let reads = [
        pair.inputs[0],
        pair.inputs[1],
        pair.inputs[3],
        pair.storage_roots[0],
        pair.storage_roots[1],
        pair.storage_roots[3],
        pair.mean,
        pair.variance_plus_epsilon,
        pair.inverse_std,
    ];
    for n in reads {
        last.entry(n)
            .and_modify(|old| *old = (*old).max(end))
            .or_insert(end);
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_stats_keep_original_ids_and_new_reads_extend_only_lifetime_end() {
        let model = super::super::tests::decoder(8);
        let (g, _) = model.training_graph().unwrap();
        let pair = recognize(&g).unwrap().remove(0);
        let stats = stat_layouts(&pair).unwrap();
        assert_eq!(
            stats.iter().map(|s| s.node).collect::<Vec<_>>(),
            vec![pair.mean, pair.variance_plus_epsilon, pair.inverse_std]
        );
        for s in stats {
            assert_eq!(s.source_shape, vec![8, 512, 1]);
            assert_eq!(s.view.source, s.node);
            assert_eq!(s.view.dtype, DType::F32);
            assert!(s.view.predicate.is_none());
            assert!(s.view.out_of_bounds.is_none());
            if s.node == pair.mean {
                assert_eq!(s.view.shape, vec![8, 512, 1]);
                assert!(s.chain.is_empty());
            } else {
                assert_eq!(s.view.shape, vec![8, 512, 512]);
                assert_eq!(s.chain.len(), 1);
            }
        }
        let topo = g.toposort();
        let positions: HashMap<_, _> = topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let mut last = HashMap::new();
        for (i, &n) in topo.iter().enumerate() {
            for input in g.inputs(n) {
                last.insert(input, i);
            }
        }
        let n_last = last.get(&pair.normalized).copied();
        let beta_last = last.get(&pair.inputs[2]).copied();
        extend_last_consumers(&pair, &positions, &mut last).unwrap();
        let end = positions[&pair.backward];
        assert!(last[&pair.mean] >= end);
        assert!(last[&pair.storage_roots[0]] >= end);
        assert_eq!(last.get(&pair.normalized).copied(), n_last);
        assert_eq!(last.get(&pair.inputs[2]).copied(), beta_last);
    }
}
