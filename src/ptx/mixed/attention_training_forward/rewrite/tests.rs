use super::*;
fn b(pointer: u64, len: usize, dtype: DType) -> Binding {
    Binding {
        pointer,
        len,
        dtype,
    }
}
#[test]
fn attention_write_ranges_reject_alias_overflow_and_partial_overlap() {
    let output = [
        b(1024, 64, DType::BF16),
        b(2048, 64, DType::F32),
        b(4096, 64, DType::BF16),
        b(8192, 64, DType::BF16),
    ];
    assert!(disjoint(&output, &[b(16384, 64, DType::BF16)]));
    assert!(!disjoint(&output, &[b(1100, 64, DType::BF16)]));
    let mut alias = output;
    alias[2] = b(1120, 64, DType::BF16);
    assert!(!disjoint(&alias, &[]));
    assert!(!disjoint(&[b(u64::MAX - 1, 2, DType::BF16)], &[]));
    assert!(!disjoint(&[b(0, 1, DType::F32)], &[]));
    assert!(!disjoint(&[b(1025, 1, DType::F32)], &[]));
}

#[test]
fn attention_runtime_resource_gate_rejects_diagnostic_one_block_variant() {
    assert!(resources_fit(108, 0, 45056, 256, 2, 49152));
    assert!(resources_fit(128, 0, 45056, 256, 2, 49152));
    for (regs, local, shared, threads, capacity, limit) in [
        (142, 0, 45056, 256, 1, 49152),
        (129, 0, 45056, 256, 2, 49152),
        (108, 1, 45056, 256, 2, 49152),
        (108, 0, 45060, 256, 2, 49152),
        (108, 0, 45056, 255, 2, 49152),
        (108, 0, 45056, 256, 1, 49152),
        (108, 0, 45056, 256, 2, 44000),
    ] {
        assert!(!resources_fit(
            regs, local, shared, threads, capacity, limit
        ));
    }
}
fn preparation_with_batch(batch: usize) -> Preparation {
    let n = NodeIndex::new;
    let r = Region {
        query: n(0),
        key: n(1),
        value: n(2),
        mask: n(3),
        qk: n(4),
        probability: n(5),
        narrowed: n(6),
        output: n(7),
        scale: 0.125,
        shape: vec![batch, 8, 512, 512],
        members: [n(4), n(5), n(6), n(7)].into_iter().collect(),
        early_checkpoint_aliases: HashSet::new(),
    };
    Preparation {
        enabled: true,
        needed: if policy_eligible(&r) {
            nodes(&r).into_iter().collect()
        } else {
            HashSet::new()
        },
        regions: vec![r],
        values: HashMap::new(),
        groups: HashMap::new(),
    }
}
#[test]
fn attention_prospective_reads_and_future_writes_survive_pruning_remap_and_ln_reindex() {
    let mut p = preparation_with_batch(16);
    let mut values = HashMap::new();
    for i in 0..8 {
        let n = NodeIndex::new(i);
        values.insert(
            n,
            Value {
                binding: b(4096 + i as u64 * 128, 4, DType::F32),
                view: VirtualTensor::identity(n, vec![4], DType::F32),
                source_shape: vec![4],
                chain: Vec::new(),
            },
        );
    }
    p.capture_values(&values);
    drop(values);
    assert_eq!(p.values.len(), 8);
    // No PV yet: reserve the current prefix's end, without inventing index end.
    let before = p.prospective_from_owners([Some(NodeIndex::new(4)), None].into_iter());
    assert_eq!(before.len(), 8);
    assert!(before.iter().all(|(i, _)| *i == 1));
    let original = p.prospective_from_owners(
        [
            None,
            Some(NodeIndex::new(4)),
            None,
            Some(NodeIndex::new(5)),
            Some(NodeIndex::new(7)),
        ]
        .into_iter(),
    );
    assert!(original.iter().all(|(i, _)| *i == 4));
    p.remap(&HashMap::from([(4096, 32768)]));
    // A preceding LN rewrite removed original positions; semantic owner resolves
    // the new PV position. Every checkpoint WRITE is present, not just Q/K/V.
    let changed =
        p.prospective_from_owners([Some(NodeIndex::new(5)), Some(NodeIndex::new(7))].into_iter());
    assert_eq!(changed.len(), 8);
    assert!(changed.iter().all(|(i, _)| *i == 1));
    assert!(changed.contains(&(1, 32768)));
    let positions = (0..8).map(|i| (NodeIndex::new(i), i)).collect();
    let mut last = HashMap::new();
    p.extend_last_consumers(&positions, &mut last).unwrap();
    assert!((0..8).all(|i| last[&NodeIndex::new(i)] == 7));
}

#[test]
fn attention_bh64_policy_fallback_reserves_nothing_before_publication() {
    let mut p = preparation_with_batch(8);
    assert!(p.needed.is_empty());
    let positions = (0..8).map(|i| (NodeIndex::new(i), i)).collect();
    let mut last = HashMap::new();
    p.extend_last_consumers(&positions, &mut last).unwrap();
    assert!(last.is_empty());
    let n = NodeIndex::new(0);
    p.capture_values(&HashMap::from([(
        n,
        Value {
            binding: b(4096, 4, DType::F32),
            view: VirtualTensor::identity(n, vec![4], DType::F32),
            source_shape: vec![4],
            chain: Vec::new(),
        },
    )]));
    assert!(p.values.is_empty());
    assert!(
        p.prospective_from_owners([None, Some(NodeIndex::new(7))].into_iter())
            .is_empty()
    );
}
