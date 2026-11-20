//! Pure physical-slot planning, shared by final and prefix compaction.
use super::*;

pub(super) struct Allocation {
    pub binding: Binding,
    pub pinned: bool,
    pub future_live: bool,
}
pub(super) struct Slot {
    pub original: usize,
    binding: Binding,
    last: usize,
    pinned: bool,
}

pub(super) fn plan_slots(
    allocations: &[Allocation],
    uses: impl IntoIterator<Item = (usize, u64)>,
    end: usize,
) -> Result<(Vec<Slot>, HashMap<u64, u64>)> {
    let mut indices = HashMap::new();
    for (index, a) in allocations.iter().enumerate() {
        ensure!(
            indices.insert(a.binding.pointer, index).is_none(),
            "duplicate mixed allocation pointer"
        );
    }
    let mut ranges = vec![(end, 0); allocations.len()];
    for (step, pointer) in uses {
        ensure!(step < end, "mixed allocation use outside prefix");
        if let Some(&index) = indices.get(&pointer) {
            ranges[index].0 = ranges[index].0.min(step);
            ranges[index].1 = ranges[index].1.max(step);
        }
    }
    for (a, range) in allocations.iter().zip(&mut ranges) {
        if a.pinned {
            *range = (0, end);
        } else if a.future_live {
            // The value is needed beyond this prefix. Its actual birth may
            // still follow another allocation's death, permitting safe reuse.
            range.1 = end;
        }
    }
    let mut order: Vec<_> = (0..allocations.len()).collect();
    order.sort_by_key(|&i| ranges[i].0);
    let mut slots: Vec<Slot> = Vec::new();
    let mut remap = HashMap::new();
    for index in order {
        let a = &allocations[index];
        let (first, last) = ranges[index];
        // Same-kernel inputs and outputs overlap. The physical lifetime is
        // convex across prior aliases; this may forgo reuse but cannot shorten it.
        let available = if a.pinned {
            None
        } else {
            slots
                .iter()
                .enumerate()
                .filter(|(_, s)| {
                    !s.pinned
                        && s.last < first
                        && s.binding.dtype == a.binding.dtype
                        && s.binding.len >= a.binding.len
                })
                .min_by_key(|(_, s)| s.binding.len)
                .map(|(i, _)| i)
        };
        let slot = if let Some(i) = available {
            slots[i].last = last;
            i
        } else {
            slots.push(Slot {
                original: index,
                binding: a.binding,
                last,
                pinned: a.pinned,
            });
            slots.len() - 1
        };
        remap.insert(a.binding.pointer, slots[slot].binding.pointer);
    }
    Ok((slots, remap))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn allocation(
        pointer: u64,
        len: usize,
        dtype: DType,
        pinned: bool,
        future_live: bool,
    ) -> Allocation {
        Allocation {
            binding: Binding {
                pointer,
                len,
                dtype,
            },
            pinned,
            future_live,
        }
    }
    #[test]
    fn prefix_future_live_keeps_actual_birth_and_blocks_later_overwrite() {
        let a = [
            allocation(1, 32, DType::F32, false, false),
            allocation(2, 16, DType::F32, false, true),
            allocation(3, 16, DType::F32, false, false),
        ];
        let (slots, map) = plan_slots(&a, [(0, 1), (1, 1), (2, 2), (3, 3)], 4).unwrap();
        assert_eq!(slots.len(), 2);
        assert_eq!(map[&2], 1); // Future-live value reuses storage before its birth.
        assert_ne!(map[&3], 1); // Its future alias must survive subsequent writes.
    }
    #[test]
    fn physical_convex_spans_same_step_pinning_dtype_and_capacity_are_respected() {
        let a = [
            allocation(1, 32, DType::F32, false, false),
            allocation(2, 16, DType::F32, false, false),
            allocation(3, 32, DType::BF16, false, false),
            allocation(4, 64, DType::F32, false, false),
            allocation(5, 16, DType::F32, true, false),
        ];
        let (_, map) = plan_slots(
            &a,
            [(0, 1), (4, 1), (2, 2), (5, 2), (6, 3), (7, 4), (8, 5)],
            9,
        )
        .unwrap();
        assert_ne!(map[&1], map[&2]); // A physical slot's disconnected uses remain convex.
        assert_ne!(map[&2], map[&3]);
        assert_eq!(map[&4], 4); // No smaller slot can hold it.
        assert_eq!(map[&5], 5); // True source/sink is pinned from execution start.
        let (_, same) = plan_slots(&a[..2], [(0, 1), (1, 1), (1, 2)], 2).unwrap();
        assert_ne!(same[&1], same[&2]);
    }
    #[test]
    fn zero_before_is_a_use_and_repeated_prefix_remap_is_safe() {
        let a = [
            allocation(1, 32, DType::F32, false, false),
            allocation(2, 16, DType::F32, false, true),
        ];
        // The first use of 2 is scatter zeroing, prior to its later scatter.
        let (_, map) = plan_slots(&a, [(0, 1), (1, 2), (2, 2)], 3).unwrap();
        assert_eq!(map[&2], 1);
        let next = [
            allocation(1, 32, DType::F32, false, true),
            allocation(3, 16, DType::F32, false, false),
        ];
        let (_, map) = plan_slots(&next, [(0, 1), (1, 1), (2, 1), (3, 3)], 4).unwrap();
        assert_eq!(map[&1], 1);
        assert_eq!(map[&3], 3);
    }
    #[test]
    fn generated_intervals_never_share_overlapping_or_mismatched_slots() {
        for seed in 0..32_usize {
            let end = 25;
            let mut uses = Vec::new();
            let mut spans = Vec::new();
            let mut allocations = Vec::new();
            for index in 0..40_usize {
                let first = (index * 7 + seed * 3) % 19;
                let last = first + (index + seed) % 6;
                let pinned = index % 11 == 0;
                let live = index % 5 == 0;
                let pointer = u64::try_from(index + 1).unwrap();
                allocations.push(allocation(
                    pointer,
                    8 << (index % 4),
                    if index % 3 == 0 {
                        DType::BF16
                    } else {
                        DType::F32
                    },
                    pinned,
                    live,
                ));
                uses.extend([(first, pointer), (last, pointer)]);
                spans.push(if pinned {
                    (0, end)
                } else {
                    (first, if live { end } else { last })
                });
            }
            let (_, map) = plan_slots(&allocations, uses, end).unwrap();
            for (i, a) in allocations.iter().enumerate() {
                let owner = &allocations[usize::try_from(map[&a.binding.pointer] - 1).unwrap()];
                assert_eq!(owner.binding.dtype, a.binding.dtype);
                assert!(owner.binding.len >= a.binding.len);
                if a.pinned {
                    assert_eq!(map[&a.binding.pointer], a.binding.pointer);
                }
                for (j, b) in allocations[..i].iter().enumerate() {
                    if map[&a.binding.pointer] == map[&b.binding.pointer] {
                        assert!(spans[i].1 < spans[j].0 || spans[j].1 < spans[i].0);
                        assert!(!a.pinned && !b.pinned);
                    }
                }
            }
        }
    }
}
