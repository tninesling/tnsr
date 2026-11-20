//! Qualified eight-warp NT128 proof/injection shared by Mixed and its probe.
//! The independent production plain selector stays strict.
use super::*;

fn same_access(a: &VirtualTensor, b: &VirtualTensor) -> Result<bool> {
    Ok(a.source == b.source
        && a.dtype == b.dtype
        && a.shape == b.shape
        && a.predicate.is_none()
        && a.out_of_bounds.is_none()
        && a.access.clone().normalize_in_domain(&a.shape)?
            == b.access.clone().normalize_in_domain(&b.shape)?)
}
pub(in crate::ptx::mixed) fn bias_slot(r: &MatMulRegion) -> Result<usize> {
    ensure!(
        r.batch_shape.is_empty()
            && r.output_shape == [r.m, r.n]
            && r.lhs_shape == [r.m, r.k]
            && r.rhs_shape == [r.k, r.n]
            && r.m >= 2048
            && r.n >= 256
            && r.k >= 64
            && r.m.is_multiple_of(128)
            && r.n.is_multiple_of(128)
            && r.k.is_multiple_of(32)
            && r.inputs.len() == 3
            && r.outputs.len() == 1
            && r.epilogue_operations.len() == 1,
        "requires full ordinary128 tiles and one linear bias epilogue"
    );
    let operation = &r.epilogue_operations[0];
    let RegionOpKind::Binary {
        op: BinaryOp::Add,
        lhs,
        rhs,
    } = operation.kind
    else {
        anyhow::bail!("requires exactly Add");
    };
    ensure!(
        lhs == r.matmul_value && rhs != lhs && r.outputs[0].value == operation.output,
        "requires original C+bias operand/output order"
    );
    let a = r
        .inputs
        .iter()
        .find(|i| i.value == r.lhs_value)
        .context("A missing")?;
    let b = r
        .inputs
        .iter()
        .find(|i| i.value == r.rhs_value)
        .context("B missing")?;
    let bias = r
        .inputs
        .iter()
        .position(|i| i.value == rhs)
        .context("bias missing")?;
    ensure!(
        bias == 2
            && a.value != b.value
            && r.inputs[0].value == r.lhs_value
            && r.inputs[1].value == r.rhs_value,
        "requires actual A/B/bias ABI order"
    );
    ensure!(
        a.source_shape == [r.m, r.k]
            && b.source_shape == [r.n, r.k]
            && r.inputs[bias].source_shape == [r.n],
        "requires whole dense physical A/W/bias buffers"
    );
    let expected_a = VirtualTensor::identity(a.tensor.source, vec![r.m, r.k], DType::BF16);
    let expected_b =
        VirtualTensor::identity(b.tensor.source, vec![r.n, r.k], DType::BF16).permute(&[1, 0])?;
    let expected_bias =
        VirtualTensor::identity(r.inputs[bias].tensor.source, vec![r.n], DType::BF16)
            .reshape(vec![1, r.n])?
            .broadcast_axis(0, r.m)?;
    ensure!(
        same_access(&a.tensor, &expected_a)?
            && same_access(&b.tensor, &expected_b)?
            && same_access(&r.inputs[bias].tensor, &expected_bias)?,
        "requires exact whole-volume Arow/Bcol/bias[col] maps"
    );
    crate::tile::ordinary_bf16_block::aligned_operand(a, &r.lhs_shape, false)
        .context("A copy alignment")?;
    crate::tile::ordinary_bf16_block::aligned_operand(b, &r.rhs_shape, true)
        .context("B copy alignment")?;
    Ok(bias)
}

/// All recursive accepted defs AND uses, including mutable scalars, loop carries
/// and predicates. Every other statement is rejected, never silently skipped.
struct Variables {
    maximum: Option<usize>,
}
impl Variables {
    fn var(&mut self, v: TileVar) {
        self.maximum = Some(self.maximum.map_or(v.0, |m| m.max(v.0)));
    }
    fn scalar(&mut self, e: &ScalarExpr) {
        match e {
            ScalarExpr::Constant(_) => {}
            ScalarExpr::Var(v) => self.var(*v),
            ScalarExpr::Unary { value, .. } => self.scalar(value),
            ScalarExpr::Binary { lhs, rhs, .. } => {
                self.scalar(lhs);
                self.scalar(rhs);
            }
            ScalarExpr::Select {
                condition,
                then_value,
                else_value,
            } => {
                self.predicate(condition);
                self.scalar(then_value);
                self.scalar(else_value);
            }
        }
    }
    fn predicate(&mut self, p: &ScalarPredicate) {
        match p {
            ScalarPredicate::IndexLt { .. } => {}
            ScalarPredicate::Equal { lhs, rhs } => {
                self.scalar(lhs);
                self.scalar(rhs);
            }
        }
    }
    fn memory(&mut self, m: &ScalarMemory) {
        if let ScalarMemory::Shared(v) = m {
            self.var(*v);
        }
    }
    fn block(&mut self, b: &Block) -> Result<()> {
        for s in &b.stmts {
            match s {
                Stmt::AllocTile { var, .. } => self.var(*var),
                Stmt::AllocSharedArena { arena, .. } => self.var(*arena),
                Stmt::ViewSharedArena { dest, arena, .. } => {
                    self.var(*dest);
                    self.var(*arena);
                }
                Stmt::AliasShared { dest, src, .. }
                | Stmt::ConvertLayout { dest, src, .. }
                | Stmt::ExtractRegister { dest, src, .. }
                | Stmt::LoadSharedToReg { dest, src }
                | Stmt::StoreAccumulatorWindow { dest, src, .. }
                | Stmt::WarpBroadcastLeader { dest, src }
                | Stmt::WarpReduce { dest, src, .. } => {
                    self.var(*dest);
                    self.var(*src);
                }
                Stmt::SelectSharedStage {
                    dest,
                    first,
                    second,
                    ..
                } => {
                    self.var(*dest);
                    self.var(*first);
                    self.var(*second);
                }
                Stmt::AsyncCopy { dest, .. }
                | Stmt::Load { dest, .. }
                | Stmt::LoadGlobalToSharedPredicated { dest, .. }
                | Stmt::LoadGlobalPredicated { dest, .. } => self.var(*dest),
                Stmt::Store { src, .. } | Stmt::StoreGlobalPredicated { src, .. } => self.var(*src),
                Stmt::Zero { tile } => self.var(*tile),
                Stmt::MatMul { dest, a, b, .. }
                | Stmt::Add { dest, a, b }
                | Stmt::Sub { dest, a, b }
                | Stmt::Mul { dest, a, b }
                | Stmt::Div { dest, a, b }
                | Stmt::Gt { dest, a, b } => {
                    self.var(*dest);
                    self.var(*a);
                    self.var(*b);
                }
                Stmt::Neg { dest, src }
                | Stmt::Exp { dest, src }
                | Stmt::Log { dest, src }
                | Stmt::Sqrt { dest, src }
                | Stmt::Relu { dest, src } => {
                    self.var(*dest);
                    self.var(*src);
                }
                Stmt::Mask {
                    dest,
                    values,
                    condition,
                } => {
                    self.var(*dest);
                    self.var(*values);
                    self.var(*condition);
                }
                Stmt::SetScalar { dest, value } => {
                    self.var(*dest);
                    self.scalar(value);
                }
                Stmt::LoadScalar { dest, source, .. } => {
                    self.var(*dest);
                    self.memory(source);
                }
                Stmt::StoreScalar { target, value, .. } => {
                    self.memory(target);
                    self.var(*value);
                }
                Stmt::ForLoop { carries, body, .. } => {
                    for carry in carries {
                        self.var(carry.var);
                        self.scalar(&carry.initial);
                    }
                    self.block(body)?;
                }
                Stmt::If { condition, body } => {
                    self.predicate(condition);
                    self.block(body)?;
                }
                Stmt::BoundsCheck { .. }
                | Stmt::LetIndex { .. }
                | Stmt::LoadIndex { .. }
                | Stmt::AsyncCommit
                | Stmt::AsyncWait
                | Stmt::Barrier => {}
                _ => anyhow::bail!("unrecognized IR pattern in fresh-variable scan: {s:?}"),
            }
        }
        Ok(())
    }
}
pub(in crate::ptx::mixed) fn fresh_variable(body: &Block) -> Result<TileVar> {
    let mut vars = Variables { maximum: None };
    vars.block(body)?;
    Ok(TileVar(
        vars.maximum
            .map_or(Some(0), |n| n.checked_add(1))
            .context("fresh variable ID overflow")?,
    ))
}
fn inject(block: &mut Block, bias: TileVar, param: &str) -> Result<usize> {
    let mut out = Vec::new();
    let mut stores = 0;
    for mut stmt in std::mem::take(&mut block.stmts) {
        match &mut stmt {
            Stmt::ForLoop { body, .. } => stores += inject(body, bias, param)?,
            Stmt::If { .. } => anyhow::bail!("conditional output path not admitted"),
            Stmt::StoreGlobalPredicated {
                dest_param,
                src,
                col,
                ..
            } => {
                ensure!(dest_param == "output_0", "unexpected output name");
                ensure!(
                    matches!(out.last(),Some(Stmt::ConvertLayout{dest,coordinates:Some(_),..})if *dest==*src),
                    "store must immediately consume its original F32 shared extraction"
                );
                out.push(Stmt::LoadScalar {
                    dest: bias,
                    source: ScalarMemory::Global(param.into()),
                    index: col.clone(),
                });
                out.push(Stmt::SetScalar {
                    dest: *src,
                    value: ScalarExpr::binary(ScalarBinaryOp::AddRn, *src, bias),
                });
                stores += 1;
            }
            _ => {}
        }
        out.push(stmt);
    }
    block.stmts = out;
    Ok(stores)
}
/// No ordinary selector is changed: independently guarded postcompile leaf.
pub(in crate::ptx::mixed) fn candidate(
    r: &MatMulRegion,
    target: PtxTarget,
) -> Result<(TileIR, MatMulSchedule, TileIR)> {
    let slot = bias_slot(r)?;
    ensure!(
        crate::tile::large_tile_policy::square128_streamed::region_candidate(
            r,
            MatMulSharedLayout::Padded,
            MatMulPipeline::Auto,
            true,
            target.matmul_resources
        )
        .is_none(),
        "production plain selector must still reject bias"
    );
    let mut scheduled = r.clone();
    scheduled.schedule = crate::tile::large_tile_policy::square128_streamed::geometry(
        r.schedule,
        &r.output_shape,
        target.matmul_resources,
    )
    .context("square geometry guard")?;
    scheduled.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        target.supports_async_copy(),
        target.matmul_resources,
    )?;
    let schedule = scheduled.schedule;
    ensure!(
        crate::tile::large_tile_policy::square128_streamed::configured_fits(
            schedule,
            target.matmul_resources
        ) && !schedule.operand_layouts.0.column_major
            && schedule.operand_layouts.1.column_major
            && schedule.shared_mem_bytes() == 40960,
        "configured NT resource/pipeline proof failed"
    );
    // This clone exists only to expose the existing plain window IR. Keep all
    // three params/order/maps; restore exactly the proved bias at every store.
    scheduled.epilogue_operations.clear();
    scheduled.outputs[0].value = scheduled.matmul_value;
    let mut graph = TileGraph::from_graph(&TensorGraph::<f32>::new());
    graph.add_matmul_regions(&[scheduled])?;
    let plain = graph.matmul_region_kernels[0].ir.clone();
    plain.validate_layouts()?;
    let mut ir = plain.clone();
    ensure!(
        ir.params.len() == 4
            && ir
                .params
                .iter()
                .map(|p| p.name.as_str())
                .eq(["input_0", "input_1", "input_2", "output_0"])
            && ir.params.iter().all(|p| p.dtype == DType::BF16),
        "original physical ABI changed"
    );
    let bias = fresh_variable(&ir.body)?;
    ensure!(
        inject(&mut ir.body, bias, &format!("input_{slot}"))? == 4,
        "exact four-window output path required"
    );
    ir.kernel_name = "test_only_fused_nt_linear_bias_square128_v1".into();
    ir.validate_layouts()?;
    Ok((ir, schedule, plain))
}
