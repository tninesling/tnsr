//! Generated normalized scan over cached and current K/V, with a runtime bound.
use crate::tensor::BinaryOp;
use crate::tile::{
    DType, Expr, KernelParam, LoopCarry, ScalarBinaryOp, ScalarBuilder, ScalarExpr, ScalarMemory,
    ScalarPredicate, Stmt, TileIR, TileVar,
    normalized_schedule::{NormalizedScan, ScanOutput, lower_scan},
};
use anyhow::{Result, ensure};

pub(super) struct AttentionScan {
    pub batch: usize,
    pub heads: usize,
    pub width: usize,
    pub chunk: usize,
    pub capacity: usize,
    pub partitions: usize,
    pub dtype: DType,
}

fn index(n: usize) -> Expr {
    Expr::Const(n as i64)
}
fn math(op: BinaryOp, a: impl Into<ScalarExpr>, b: impl Into<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::binary(ScalarBinaryOp::Arithmetic(op), a, b)
}
fn prefix() -> Expr {
    Expr::Var("prefix".into())
}

impl AttentionScan {
    pub fn ir(&self) -> Result<TileIR> {
        ensure!(
            self.heads > 0 && self.width.is_multiple_of(self.heads),
            "invalid decoder head shape"
        );
        ensure!(
            self.chunk > 0 && self.chunk <= self.capacity,
            "invalid decoder scan chunk"
        );
        let columns = self.width / self.heads;
        let extent = self.batch * self.chunk * self.width * self.partitions;
        ensure!(
            extent <= i64::MAX as usize,
            "decoder scan extent exceeds i64"
        );
        let (body, shared_mem_bytes) = lower_scan(
            self,
            extent,
            columns,
            if self.partitions > 1 {
                ScanOutput::Partial
            } else {
                ScanOutput::Normalized
            },
        )?;
        let mut params: Vec<_> = ["query", "key", "value", "cached_key", "cached_value"]
            .into_iter()
            .map(|name| KernelParam {
                name: name.into(),
                dtype: self.dtype,
                is_input: true,
            })
            .collect();
        params.push(KernelParam {
            name: "length".into(),
            dtype: DType::U32,
            is_input: true,
        });
        params.push(KernelParam {
            name: "output_0".into(),
            dtype: if self.partitions > 1 {
                DType::F32
            } else {
                self.dtype
            },
            is_input: false,
        });
        if self.partitions > 1 {
            params.push(KernelParam {
                name: "log_normalizer".into(),
                dtype: DType::F32,
                is_input: false,
            });
        }
        Ok(TileIR {
            kernel_name: "decoder_attention".into(),
            params,
            body,
            shared_mem_bytes,
        })
    }

    // Fibers are [batch, head, query]; projected buffers are [batch, chunk, width].
    fn coordinates(&self, fiber: Expr) -> (Expr, Expr, Expr) {
        let fiber = Expr::FloorDiv(Box::new(fiber), self.partitions);
        let batch = Expr::FloorDiv(Box::new(fiber.clone()), self.heads * self.chunk);
        let head = Expr::Mod(
            Box::new(Expr::FloorDiv(Box::new(fiber.clone()), self.chunk)),
            self.heads,
        );
        let query = Expr::Mod(Box::new(fiber), self.chunk);
        (batch, head, query)
    }

    fn key(&self, fiber: Expr, key: Expr) -> Expr {
        if self.partitions == 1 {
            key
        } else {
            key + Expr::Mul(
                Box::new(Expr::Mod(Box::new(fiber), self.partitions)),
                Box::new(Expr::Var("span".into())),
            )
        }
    }

    pub(super) fn load_kv(
        &self,
        b: &mut ScalarBuilder,
        kind: &str,
        batch: Expr,
        head: Expr,
        key: Expr,
        column: Expr,
    ) -> Result<TileVar> {
        let result = b.value(ScalarExpr::Constant(0.));
        let feature = head * (self.width / self.heads) + column;
        b.when(
            ScalarPredicate::IndexLt {
                lhs: key.clone(),
                rhs: prefix(),
            },
            |b| {
                let offset = batch.clone() * (self.capacity * self.width)
                    + key.clone() * self.width
                    + feature.clone();
                let value = b.load(ScalarMemory::Global(format!("cached_{kind}")), offset);
                b.set(result, value);
                Ok(())
            },
        )?;
        let new_key = Expr::Sub(Box::new(key), Box::new(prefix()));
        // Unsigned subtraction wraps for cached positions, excluding this load.
        b.when(
            ScalarPredicate::IndexLt {
                lhs: new_key.clone(),
                rhs: index(self.chunk),
            },
            |b| {
                let offset = batch * (self.chunk * self.width) + new_key * self.width + feature;
                let value = b.load(ScalarMemory::Global(kind.into()), offset);
                b.set(result, value);
                Ok(())
            },
        )?;
        Ok(result)
    }
}

impl NormalizedScan for AttentionScan {
    fn keys(&self, b: &mut ScalarBuilder, fiber: &Expr) -> Result<Expr> {
        b.stmts.push(Stmt::LoadIndex {
            name: "prefix".into(),
            source: "length".into(),
            index: index(0),
        });
        let (_, _, query) = self.coordinates(fiber.clone());
        if self.partitions == 1 {
            Ok(prefix() + query + index(1))
        } else {
            b.stmts.push(Stmt::LoadIndex {
                name: "span".into(),
                source: "length".into(),
                index: index(1),
            });
            b.stmts.push(Stmt::LoadIndex {
                name: "scan_length".into(),
                source: "length".into(),
                index: index(2)
                    + query * self.partitions
                    + Expr::Mod(Box::new(fiber.clone()), self.partitions),
            });
            Ok(Expr::Var("scan_length".into()))
        }
    }
    fn score(&self, b: &mut ScalarBuilder, fiber: Expr, key: Expr) -> Result<TileVar> {
        let key = self.key(fiber.clone(), key);
        let (batch, head, query) = self.coordinates(fiber);
        let columns = self.width / self.heads;
        let sum = b.var();
        let name = b.loop_name("dot");
        b.fold(
            name,
            index(columns),
            vec![LoopCarry {
                var: sum,
                initial: ScalarExpr::Constant(0.),
            }],
            |b, column| {
                let offset = batch.clone() * (self.chunk * self.width)
                    + query.clone() * self.width
                    + head.clone() * columns
                    + column.clone();
                let q = b.load(ScalarMemory::Global("query".into()), offset);
                let k = self.load_kv(b, "key", batch.clone(), head.clone(), key, column)?;
                b.set(sum, math(BinaryOp::Add, sum, math(BinaryOp::Mul, q, k)));
                Ok(())
            },
        )?;
        Ok(b.value(math(
            BinaryOp::Mul,
            sum,
            ScalarExpr::Constant((columns as f32).sqrt().recip()),
        )))
    }
    fn accumulate(
        &self,
        b: &mut ScalarBuilder,
        output: Expr,
        key: Expr,
        probability: TileVar,
        numerator: TileVar,
    ) -> Result<()> {
        let columns = self.width / self.heads;
        let fiber = Expr::FloorDiv(Box::new(output.clone()), columns);
        let column = Expr::Mod(Box::new(output), columns);
        let key = self.key(fiber.clone(), key);
        let (batch, head, _) = self.coordinates(fiber);
        let v = self.load_kv(b, "value", batch, head, key, column)?;
        b.set(
            numerator,
            math(
                BinaryOp::Add,
                numerator,
                math(BinaryOp::Mul, probability, v),
            ),
        );
        Ok(())
    }
}

/// Merge either normalized/LSE partials or raw numerator/max/denominator partials.
pub(super) struct MergeScan {
    pub rows: usize,
    pub columns: usize,
    pub partitions: usize,
    pub dtype: DType,
    pub raw_partials: bool,
}
impl MergeScan {
    pub fn ir(&self) -> Result<TileIR> {
        if self.partitions <= 8 || self.raw_partials {
            return self.pointwise_ir();
        }
        let (body, shared_mem_bytes) = lower_scan(
            self,
            self.rows * self.columns,
            self.columns,
            ScanOutput::Normalized,
        )?;
        let params = [
            ("partials", true),
            ("normalizers", true),
            ("output_0", false),
        ]
        .into_iter()
        .map(|(name, is_input)| KernelParam {
            name: name.into(),
            dtype: if is_input { DType::F32 } else { self.dtype },
            is_input,
        })
        .collect();
        Ok(TileIR {
            kernel_name: "decoder_merge".into(),
            params,
            body,
            shared_mem_bytes,
        })
    }
    // A handful of partitions is cheaper to merge per output element than
    // through a cooperative CTA: no shared score tile or block barriers.
    fn pointwise_ir(&self) -> Result<TileIR> {
        let mut b = ScalarBuilder::default();
        b.stmts.push(Stmt::BoundsCheck {
            extent: self.rows * self.columns,
        });
        let output = Expr::Mul(
            Box::new(Expr::BlockIdx(crate::tile::Dim::X)),
            Box::new(Expr::BlockDim(crate::tile::Dim::X)),
        ) + Expr::ThreadIdx(crate::tile::Dim::X);
        let row = Expr::FloorDiv(Box::new(output.clone()), self.columns);
        let column = Expr::Mod(Box::new(output.clone()), self.columns);
        let maximum = b.value(ScalarExpr::Constant(f32::NEG_INFINITY));
        let mut scores = Vec::new();
        for part in 0..self.partitions {
            let score = b.load(
                ScalarMemory::Global("normalizers".into()),
                (row.clone() * self.partitions + index(part))
                    * if self.raw_partials {
                        super::prefill::STATISTICS_WIDTH
                    } else {
                        1usize
                    },
            );
            b.set(
                maximum,
                ScalarExpr::binary(ScalarBinaryOp::Max, maximum, score),
            );
            scores.push(score);
        }
        let numerator = b.value(ScalarExpr::Constant(0.));
        let denominator = b.value(ScalarExpr::Constant(0.));
        for (part, score) in scores.into_iter().enumerate() {
            let weight = b.value(ScalarExpr::unary(
                crate::tensor::UnaryOp::Exp,
                math(BinaryOp::Sub, score, maximum),
            ));
            let value = b.load(
                ScalarMemory::Global("partials".into()),
                (row.clone() * self.partitions + index(part)) * self.columns + column.clone(),
            );
            b.set(
                numerator,
                math(BinaryOp::Add, numerator, math(BinaryOp::Mul, weight, value)),
            );
            let contribution = if self.raw_partials {
                let den = b.load(
                    ScalarMemory::Global("normalizers".into()),
                    (row.clone() * self.partitions + index(part))
                        * super::prefill::STATISTICS_WIDTH
                        + index(1),
                );
                math(BinaryOp::Mul, weight, den)
            } else {
                weight.into()
            };
            b.set(denominator, math(BinaryOp::Add, denominator, contribution));
        }
        let value = b.value(math(BinaryOp::Div, numerator, denominator));
        b.store(ScalarMemory::Global("output_0".into()), output, value);
        Ok(TileIR {
            kernel_name: "decoder_merge".into(),
            params: [
                ("partials", DType::F32, true),
                ("normalizers", DType::F32, true),
                ("output_0", self.dtype, false),
            ]
            .into_iter()
            .map(|(name, dtype, is_input)| KernelParam {
                name: name.into(),
                dtype,
                is_input,
            })
            .collect(),
            body: crate::tile::Block { stmts: b.stmts },
            shared_mem_bytes: 0,
        })
    }
}
impl NormalizedScan for MergeScan {
    fn keys(&self, _: &mut ScalarBuilder, _: &Expr) -> Result<Expr> {
        Ok(index(self.partitions))
    }
    fn score(&self, b: &mut ScalarBuilder, fiber: Expr, key: Expr) -> Result<TileVar> {
        Ok(b.load(
            ScalarMemory::Global("normalizers".into()),
            fiber * self.partitions + key,
        ))
    }
    fn accumulate(
        &self,
        b: &mut ScalarBuilder,
        output: Expr,
        key: Expr,
        probability: TileVar,
        numerator: TileVar,
    ) -> Result<()> {
        let fiber = Expr::FloorDiv(Box::new(output.clone()), self.columns);
        let column = Expr::Mod(Box::new(output), self.columns);
        let value = b.load(
            ScalarMemory::Global("partials".into()),
            fiber * (self.partitions * self.columns) + key * self.columns + column,
        );
        b.set(
            numerator,
            math(
                BinaryOp::Add,
                numerator,
                math(BinaryOp::Mul, probability, value),
            ),
        );
        Ok(())
    }
}
