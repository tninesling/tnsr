use super::*;
use std::collections::HashMap;

struct Scan {
    keys: usize,
    columns: usize,
}

impl NormalizedScan for Scan {
    fn keys(&self, _: &mut ScalarBuilder, _: &Expr) -> Result<Expr> {
        constant(self.keys)
    }

    fn score(&self, builder: &mut ScalarBuilder, _: Expr, key: Expr) -> Result<TileVar> {
        Ok(builder.load(ScalarMemory::Global("scores".into()), key))
    }

    fn accumulate(
        &self,
        builder: &mut ScalarBuilder,
        output: Expr,
        key: Expr,
        probability: TileVar,
        numerator: TileVar,
    ) -> Result<()> {
        let value = builder.load(
            ScalarMemory::Global("values".into()),
            key * self.columns + output,
        );
        builder.set(
            numerator,
            arithmetic(
                BinaryOp::Add,
                numerator,
                arithmetic(BinaryOp::Mul, probability, value),
            ),
        );
        Ok(())
    }
}

/// Execute the emitted scalar scan IR in lockstep, not a second implementation
/// of online softmax. Shared writes become visible only at a uniform barrier,
/// so missing publication/retirement barriers fail these tests as well.
struct Machine {
    registers: Vec<HashMap<TileVar, f32>>,
    globals: HashMap<String, Vec<f32>>,
    shared: HashMap<TileVar, Vec<f32>>,
    pending: HashMap<(TileVar, usize), f32>,
    indices: HashMap<String, usize>,
    barriers: usize,
}

impl Machine {
    fn index(&self, expr: &Expr, lane: usize) -> usize {
        match expr {
            Expr::Const(n) => usize::try_from(*n).unwrap(),
            Expr::Var(name) => self.indices[name],
            Expr::ThreadIdx(Dim::X) => lane,
            Expr::BlockDim(Dim::X) => self.registers.len(),
            Expr::BlockIdx(_) | Expr::ThreadIdx(_) => 0,
            Expr::BlockDim(_) => 1,
            Expr::Add(a, b) => self.index(a, lane) + self.index(b, lane),
            Expr::Mul(a, b) => self.index(a, lane) * self.index(b, lane),
            Expr::Sub(a, b) => self.index(a, lane) - self.index(b, lane),
            Expr::FloorDiv(a, n) => self.index(a, lane) / n,
            Expr::Mod(a, n) => self.index(a, lane) % n,
            Expr::ShiftRight(a, n) => self.index(a, lane) >> n,
            Expr::BitAnd(a, n) => self.index(a, lane) & usize::try_from(*n).unwrap(),
        }
    }

    fn predicate(&self, predicate: &ScalarPredicate, lane: usize) -> bool {
        match predicate {
            ScalarPredicate::IndexLt { lhs, rhs } => self.index(lhs, lane) < self.index(rhs, lane),
            ScalarPredicate::Equal { lhs, rhs } => self.value(lhs, lane) == self.value(rhs, lane),
        }
    }

    fn value(&self, expr: &ScalarExpr, lane: usize) -> f32 {
        match expr {
            ScalarExpr::Constant(n) => *n,
            ScalarExpr::Var(var) => self.registers[lane][var],
            ScalarExpr::Unary { op, value } => match op {
                UnaryOp::Exp => self.value(value, lane).exp(),
                UnaryOp::Log => self.value(value, lane).ln(),
                _ => panic!("unexpected scan unary op: {op:?}"),
            },
            ScalarExpr::Binary { op, lhs, rhs } => {
                let lhs = self.value(lhs, lane);
                let rhs = self.value(rhs, lane);
                match op {
                    ScalarBinaryOp::Max => lhs.max(rhs),
                    ScalarBinaryOp::AddRn => lhs + rhs,
                    ScalarBinaryOp::SubRn => lhs - rhs,
                    ScalarBinaryOp::MulRn => lhs * rhs,
                    ScalarBinaryOp::Arithmetic(BinaryOp::Add) => lhs + rhs,
                    ScalarBinaryOp::Arithmetic(BinaryOp::Sub) => lhs - rhs,
                    ScalarBinaryOp::Arithmetic(BinaryOp::Mul) => lhs * rhs,
                    ScalarBinaryOp::Arithmetic(BinaryOp::Div) => lhs / rhs,
                }
            }
            ScalarExpr::Select {
                condition,
                then_value,
                else_value,
            } => self.value(
                if self.predicate(condition, lane) {
                    then_value
                } else {
                    else_value
                },
                lane,
            ),
        }
    }

    fn block(&mut self, block: &Block, mut active: Vec<bool>) {
        for stmt in &block.stmts {
            match stmt {
                Stmt::AllocTile {
                    var, rows, cols, ..
                } => {
                    self.shared.insert(*var, vec![f32::NAN; rows * cols]);
                }
                Stmt::SetScalar { dest, value } => {
                    for (lane, &enabled) in active.iter().enumerate() {
                        if enabled {
                            let result = self.value(value, lane);
                            self.registers[lane].insert(*dest, result);
                        }
                    }
                }
                Stmt::LoadScalar {
                    dest,
                    source,
                    index,
                } => {
                    for (lane, &enabled) in active.iter().enumerate() {
                        if enabled {
                            let index = self.index(index, lane);
                            let value = match source {
                                ScalarMemory::Global(name) => self.globals[name][index],
                                ScalarMemory::Shared(var) => self.shared[var][index],
                            };
                            self.registers[lane].insert(*dest, value);
                        }
                    }
                }
                Stmt::StoreScalar {
                    target,
                    index,
                    value,
                } => {
                    for (lane, &enabled) in active.iter().enumerate() {
                        if enabled {
                            let index = self.index(index, lane);
                            let value = self.registers[lane][value];
                            match target {
                                ScalarMemory::Global(name) => {
                                    self.globals.get_mut(name).unwrap()[index] = value;
                                }
                                ScalarMemory::Shared(var) => {
                                    assert!(self.pending.insert((*var, index), value).is_none());
                                }
                            }
                        }
                    }
                }
                Stmt::Barrier => {
                    assert!(active.iter().all(|enabled| *enabled), "divergent barrier");
                    for ((var, index), value) in self.pending.drain() {
                        self.shared.get_mut(&var).unwrap()[index] = value;
                    }
                    self.barriers += 1;
                }
                Stmt::WarpReduce {
                    dest,
                    src,
                    op,
                    broadcast,
                } => {
                    assert!(!broadcast);
                    assert!(matches!(op, ReduceOp::Max));
                    for base in (0..active.len()).step_by(32) {
                        if !active[base] {
                            continue;
                        }
                        assert!(active[base..base + 32].iter().all(|enabled| *enabled));
                        let mut values: Vec<_> = (base..base + 32)
                            .map(|lane| self.registers[lane][src])
                            .collect();
                        for distance in [16, 8, 4, 2, 1] {
                            let old = values.clone();
                            for lane in 0..32 {
                                values[lane] = old[lane].max(
                                    old.get(lane + distance)
                                        .copied()
                                        .unwrap_or(f32::NEG_INFINITY),
                                );
                            }
                        }
                        for (offset, value) in values.into_iter().enumerate() {
                            self.registers[base + offset].insert(*dest, value);
                        }
                    }
                }
                Stmt::If { condition, body } => {
                    let enabled = active
                        .iter()
                        .enumerate()
                        .map(|(lane, enabled)| *enabled && self.predicate(condition, lane))
                        .collect();
                    self.block(body, enabled);
                }
                Stmt::ForLoop {
                    loop_var,
                    start,
                    end,
                    carries,
                    body,
                } => {
                    for (lane, &enabled) in active.iter().enumerate() {
                        if enabled {
                            for carry in carries {
                                let value = self.value(&carry.initial, lane);
                                self.registers[lane].insert(carry.var, value);
                            }
                        }
                    }
                    let first = active.iter().position(|enabled| *enabled).unwrap_or(0);
                    let start = self.index(start, first);
                    let end = self.index(end, first);
                    for i in start..end {
                        self.indices.insert(loop_var.clone(), i);
                        self.block(body, active.clone());
                    }
                    self.indices.remove(loop_var);
                }
                Stmt::BoundsCheck { extent } => {
                    for (lane, enabled) in active.iter_mut().enumerate() {
                        *enabled &= lane < *extent;
                    }
                }
                _ => panic!(
                    "unexpected scan statement: {:?}",
                    std::mem::discriminant(stmt)
                ),
            }
        }
    }
}

fn execute(
    scores: &[f32],
    columns: usize,
    mode: ScanOutput,
    sharing: bool,
    case: usize,
) -> (Vec<f32>, usize) {
    let scan = Scan {
        keys: scores.len(),
        columns,
    };
    let (block, _) = if sharing {
        lower_scan_with_weight_sharing(&scan, columns, columns, mode, true)
    } else {
        lower_scan(&scan, columns, columns, mode)
    }
    .unwrap();
    let threads = columns.max(32).next_power_of_two();
    let mut machine = Machine {
        registers: vec![HashMap::new(); threads],
        globals: HashMap::from([
            ("scores".into(), scores.to_vec()),
            (
                "values".into(),
                (0..scores.len() * columns)
                    .map(|i| match (case, i) {
                        (7, 0) => f32::INFINITY,
                        (8, 0) => f32::NAN,
                        _ => ((i % 19) as f32 - 9.0) * 0.13,
                    })
                    .collect(),
            ),
            ("output_0".into(), vec![f32::NAN; columns]),
            ("log_normalizer".into(), vec![f32::NAN; 1]),
        ]),
        shared: HashMap::new(),
        pending: HashMap::new(),
        indices: HashMap::new(),
        barriers: 0,
    };
    machine.block(&block, vec![true; threads]);
    assert!(machine.pending.is_empty());
    let mut result = machine.globals.remove("output_0").unwrap();
    if mode == ScanOutput::Partial {
        result.extend(machine.globals.remove("log_normalizer").unwrap());
    }
    (result, machine.barriers)
}

#[test]
fn shared_online_weights_preserve_emitted_scan_results_and_barriers() {
    for columns in [1, 16, 31, 32, 33, 64, 128, 129] {
        for keys in [0, 1, 31, 32, 33, 65] {
            for case in 0..9 {
                let mut scores: Vec<_> = (0..keys).map(|i| ((i % 17) as f32 - 8.) * 0.7).collect();
                match case {
                    1 => scores.fill(f32::NEG_INFINITY),
                    2 => scores
                        .iter_mut()
                        .take(32)
                        .for_each(|x| *x = f32::NEG_INFINITY),
                    3 if keys > 0 => scores[keys - 1] = 1000.,
                    4 if keys > 0 => scores[keys - 1] = f32::NAN,
                    5 if keys > 0 => scores[keys - 1] = f32::INFINITY,
                    6 => scores
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, x)| *x = if i % 2 == 0 { -0. } else { 0. }),
                    7 | 8 if keys > 0 => scores[0] = f32::NEG_INFINITY,
                    _ => {}
                }
                for mode in [
                    ScanOutput::Normalized,
                    ScanOutput::Normalizer,
                    ScanOutput::Partial,
                ] {
                    let (baseline, old_barriers) = execute(&scores, columns, mode, false, case);
                    let (candidate, new_barriers) = execute(&scores, columns, mode, true, case);
                    for (a, b) in baseline.into_iter().zip(candidate) {
                        assert!(
                            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                            "columns={columns}, keys={keys}, case={case}: {a:?} != {b:?}"
                        );
                    }
                    let extra = if columns <= 128 { keys.div_ceil(32) } else { 0 };
                    assert_eq!(new_barriers, old_barriers + extra);
                }
            }
        }
    }
}
