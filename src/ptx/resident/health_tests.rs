//! Execute the actual compact-health suffix in lockstep, including shared-store
//! visibility and full-active-warp requirements, without initializing CUDA.
use super::*;
use std::collections::HashMap;

struct Machine {
    threads: usize,
    block: usize,
    registers: HashMap<TileVar, Vec<f32>>,
    shared: [Option<f32>; 32],
    pending: Vec<(usize, f32)>,
}
impl Machine {
    fn index(&self, expression: &Expr, thread: usize) -> usize {
        match expression {
            Expr::Const(v) => usize::try_from(*v).unwrap(),
            Expr::ThreadIdx(Dim::X) => thread,
            Expr::BlockIdx(Dim::X) => self.block,
            Expr::BlockDim(Dim::X) => self.threads,
            Expr::Add(a, b) => self.index(a, thread) + self.index(b, thread),
            Expr::Mul(a, b) => self.index(a, thread) * self.index(b, thread),
            Expr::ShiftRight(a, n) => self.index(a, thread) >> n,
            Expr::BitAnd(a, n) => self.index(a, thread) & usize::try_from(*n).unwrap(),
            _ => panic!("unexpected health index"),
        }
    }
    fn execute(
        &mut self,
        statements: &[Stmt],
        active: &[bool],
        output: &mut [f32],
        writes: &mut [usize],
    ) {
        for statement in statements {
            match statement {
                Stmt::WarpReduce {
                    dest,
                    src,
                    op: crate::tensor::ReduceOp::Sum,
                    broadcast: false,
                } => {
                    let source = self.registers[src].clone();
                    let mut values = vec![0.0; self.threads];
                    for base in (0..self.threads).step_by(32) {
                        let participating = active[base..base + 32].iter().filter(|v| **v).count();
                        assert!(
                            participating == 0 || participating == 32,
                            "partial active warp at collective"
                        );
                        if participating == 0 {
                            continue;
                        }
                        let mut lanes = source[base..base + 32].to_vec();
                        for distance in [16, 8, 4, 2, 1] {
                            let previous = lanes.clone();
                            for lane in 0..32 {
                                lanes[lane] = previous[lane]
                                    + previous
                                        .get(lane + distance)
                                        .copied()
                                        .unwrap_or(previous[lane]);
                            }
                        }
                        values[base..base + 32].copy_from_slice(&lanes);
                    }
                    self.registers.insert(*dest, values);
                }
                Stmt::If {
                    condition: ScalarPredicate::IndexLt { lhs, rhs },
                    body,
                } => {
                    let selected = (0..self.threads)
                        .map(|thread| {
                            active[thread] && self.index(lhs, thread) < self.index(rhs, thread)
                        })
                        .collect::<Vec<_>>();
                    self.execute(&body.stmts, &selected, output, writes);
                }
                Stmt::StoreScalar {
                    target: ScalarMemory::Shared(_),
                    index,
                    value,
                } => {
                    for (thread, enabled) in active.iter().enumerate() {
                        if *enabled {
                            let index = self.index(index, thread);
                            assert!(index < self.shared.len());
                            assert!(!self.pending.iter().any(|(slot, _)| *slot == index));
                            self.pending.push((index, self.registers[value][thread]));
                        }
                    }
                }
                Stmt::Barrier => {
                    assert!(active.iter().all(|v| *v), "nonuniform block barrier");
                    for (index, value) in self.pending.drain(..) {
                        self.shared[index] = Some(value);
                    }
                }
                Stmt::SetScalar {
                    dest,
                    value: ScalarExpr::Constant(value),
                } => {
                    let values = self
                        .registers
                        .entry(*dest)
                        .or_insert_with(|| vec![0.0; self.threads]);
                    for (thread, enabled) in active.iter().enumerate() {
                        if *enabled {
                            values[thread] = *value;
                        }
                    }
                }
                Stmt::LoadScalar {
                    dest,
                    source: ScalarMemory::Shared(_),
                    index,
                } => {
                    assert!(
                        self.pending.is_empty(),
                        "shared load lacks publication barrier"
                    );
                    for (thread, enabled) in active.iter().enumerate() {
                        if *enabled {
                            let index = self.index(index, thread);
                            let value = self.shared[index].expect("read of unwritten warp partial");
                            self.registers.get_mut(dest).unwrap()[thread] = value;
                        }
                    }
                }
                Stmt::StoreGlobalPredicated {
                    dest_param,
                    src,
                    element_index,
                    row,
                    col,
                    layout,
                } => {
                    assert_eq!(dest_param, "health");
                    for (thread, enabled) in active.iter().enumerate() {
                        if *enabled
                            && self.index(row, thread) < layout.rows
                            && self.index(col, thread) < layout.cols
                        {
                            let index = self.index(element_index, thread);
                            assert!(index < output.len());
                            output[index] = self.registers[src][thread];
                            writes[index] += 1;
                        }
                    }
                }
                _ => panic!("unexpected compact-health statement"),
            }
        }
    }
}

#[test]
fn health_partials_execute_full_warps_and_preserve_tail_and_member_bounds() {
    for threads in [256, 1024] {
        for len in [
            1usize, 3, 17, 31, 32, 33, 127, 255, 256, 257, 511, 1023, 1024, 1025, 2049, 4097,
        ] {
            let count = len.div_ceil(HEALTH_GROUP_SIZE);
            let mut builder = TileIRBuilder::new();
            let scratch = HealthScratch::new(&mut builder);
            let health = reg(&mut builder);
            let offset = 3;
            let total = count + offset + 5;
            let statements = scratch.statements(health, offset, count, total).unwrap();
            for bad in [
                None,
                Some(0),
                Some(len - 1),
                Some((len - 1).min(255)),
                Some((len - 1).min(1024)),
            ] {
                let mut output = vec![42.0; total];
                let mut writes = vec![0; total];
                // Extra CTAs model a short member in a max-length batched grid.
                for block in 0..len.div_ceil(threads) + 2 {
                    let values = (0..threads)
                        .map(|lane| {
                            if bad == Some(block * threads + lane) {
                                f32::NAN
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let mut machine = Machine {
                        threads,
                        block,
                        registers: HashMap::from([(health, values)]),
                        shared: [None; 32],
                        pending: Vec::new(),
                    };
                    machine.execute(&statements, &vec![true; threads], &mut output, &mut writes);
                }
                for index in 0..total {
                    if (offset..offset + count).contains(&index) {
                        assert_eq!(writes[index], 1);
                        assert_eq!(
                            output[index].is_nan(),
                            bad.is_some_and(|bad| index - offset == bad / HEALTH_GROUP_SIZE)
                        );
                        if !output[index].is_nan() {
                            assert_eq!(output[index].to_bits(), 0);
                        }
                    } else {
                        assert_eq!(writes[index], 0);
                        assert_eq!(output[index], 42.0);
                    }
                }
            }
        }
    }
}

fn inspect(block: &Block, thread_guarded: bool) -> (usize, usize) {
    let mut barriers = 0;
    let mut reductions = 0;
    for statement in &block.stmts {
        match statement {
            Stmt::BoundsCheck { .. } => panic!("early return before a compact-health collective"),
            Stmt::Barrier => {
                assert!(!thread_guarded);
                barriers += 1;
            }
            Stmt::WarpReduce { .. } => reductions += 1,
            Stmt::If { condition, body } => {
                let lane_guard = matches!(
                    condition,
                    ScalarPredicate::IndexLt {
                        lhs: Expr::Add(_, _),
                        ..
                    }
                );
                // The only Add-root guard is the per-thread global linear index;
                // its body must contain neither barriers nor warp collectives.
                let counts = inspect(body, thread_guarded || lane_guard);
                if lane_guard {
                    assert_eq!(counts, (0, 0));
                }
                barriers += counts.0;
                reductions += counts.1;
            }
            _ => {}
        }
    }
    (barriers, reductions)
}

#[test]
fn health_kernel_layouts_and_collectives_validate_offline() {
    for len in [1, 33, 255, 256, 257, 1023, 1024, 1025, 4097] {
        let mut ir = adam(len).unwrap();
        ir.validate_layouts().unwrap();
        assert_eq!(ir.shared_mem_bytes, 128);
        assert_eq!(inspect(&ir.body, false), (1, 2));
        ir.optimize_indices().unwrap();
        ir.validate_layouts().unwrap();
    }
    let mut batch = adam_batch(&[(0, 1, 0), (1, 257, 1), (258, 33, 3)], 1024, 1024).unwrap();
    batch.validate_layouts().unwrap();
    assert_eq!(batch.shared_mem_bytes, 128);
    assert_eq!(inspect(&batch.body, false), (3, 6));
    batch.optimize_indices().unwrap();
    batch.validate_layouts().unwrap();
    assert!(adam_batch(&[(0, 257, 0), (257, 3, 1)], 1024, 1024).is_err());
    assert!(adam_batch(&[(0, 257, 0), (256, 3, 2)], 1024, 1024).is_err());
    assert!(adam_batch(&[(0, 257, 0), (257, 3, 2)], 1024, 2).is_err());
}

#[test]
fn standalone_adam_actual_ir_health_groups_match_both_launch_policies() {
    for len in [1_usize, 17, 255, 256, 257, 1023, 1024, 1025, 4097, 8193] {
        let ir = adam(len).unwrap();
        let (start, health) = ir
            .body
            .stmts
            .iter()
            .enumerate()
            .find_map(|(index, stmt)| {
                if let Stmt::WarpReduce { src, .. } = stmt {
                    Some((index, *src))
                } else {
                    None
                }
            })
            .unwrap();
        let statements = &ir.body.stmts[start..];
        let groups = len.div_ceil(HEALTH_GROUP_SIZE);
        for bad in [
            None,
            Some(0),
            Some(len - 1),
            Some((len - 1).min(256)),
            Some((len - 1).min(1024)),
        ] {
            let mut policies = Vec::new();
            for enabled in [false, true] {
                let config = super::super::launch::standalone_adam(len, enabled).unwrap();
                let threads = usize::try_from(config.block_dim.0).unwrap();
                let mut output = vec![42.0; groups];
                let mut writes = vec![0; groups];
                for block in 0..usize::try_from(config.grid_dim.0).unwrap() {
                    let values = (0..threads)
                        .map(|thread| {
                            if bad == Some(block * threads + thread) {
                                f32::NAN
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let mut machine = Machine {
                        threads,
                        block,
                        registers: HashMap::from([(health, values)]),
                        shared: [None; 32],
                        pending: Vec::new(),
                    };
                    machine.execute(statements, &vec![true; threads], &mut output, &mut writes);
                }
                assert!(writes.iter().all(|count| *count == 1));
                let flags: Vec<_> = output.iter().map(|v| v.is_nan()).collect();
                for (group, flag) in flags.iter().enumerate() {
                    assert_eq!(
                        *flag,
                        bad.is_some_and(|index| index / HEALTH_GROUP_SIZE == group)
                    );
                    if !flag {
                        assert_eq!(output[group].to_bits(), 0);
                    }
                }
                policies.push(flags);
            }
            assert_eq!(policies[0], policies[1]);
        }
    }
}
