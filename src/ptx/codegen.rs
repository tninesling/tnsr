use std::fmt;

use itertools::Itertools;

use super::instructions::*;
use super::types::*;

#[derive(Clone, Debug)]
pub struct Label<'a> {
    pub name: &'a str,
    pub body: bumpalo::collections::Vec<'a, Inst<'a>>,
}

#[derive(Clone, Debug)]
pub struct Function<'a> {
    pub name: &'a str,
    pub params: bumpalo::collections::Vec<'a, (&'a str, Type)>,
    pub body: bumpalo::collections::Vec<'a, Inst<'a>>,
    pub labels: bumpalo::collections::Vec<'a, Label<'a>>,
    pub predicate_registers: bumpalo::collections::Vec<'a, &'a str>,
    pub f32_registers: bumpalo::collections::Vec<'a, &'a str>,
    pub i32_registers: bumpalo::collections::Vec<'a, &'a str>,
    pub i64_registers: bumpalo::collections::Vec<'a, &'a str>,
    pub shared_memory: bumpalo::collections::Vec<'a, (&'a str, usize)>,
    pub arena: &'a bumpalo::Bump,
}

impl<'a> Function<'a> {
    pub fn new(name: &'a str, arena: &'a bumpalo::Bump) -> Self {
        Self {
            name,
            params: bumpalo::collections::Vec::new_in(arena),
            body: bumpalo::collections::Vec::new_in(arena),
            labels: bumpalo::collections::Vec::new_in(arena),
            predicate_registers: bumpalo::collections::Vec::new_in(arena),
            f32_registers: bumpalo::collections::Vec::new_in(arena),
            i32_registers: bumpalo::collections::Vec::new_in(arena),
            i64_registers: bumpalo::collections::Vec::new_in(arena),
            shared_memory: bumpalo::collections::Vec::new_in(arena),
            arena,
        }
    }

    pub fn add_inst(&mut self, inst: Inst<'a>) {
        self.body.push(inst);
    }

    pub fn add_param_u64(&mut self, name: &'a str) -> Operand<'a, U64> {
        self.params.push((name, Type::U64));
        Operand::addr(name)
    }

    pub fn add_label(&mut self, label: Label<'a>) {
        self.labels.push(label);
    }

    pub fn add_predicate_register(&mut self) -> Operand<'a, Pred> {
        let idx = self.predicate_registers.len();
        let name = bumpalo::format!(in self.arena, "%p{}", idx);
        let name_str = name.into_bump_str();
        self.predicate_registers.push(name_str);
        Operand::pred(name_str)
    }

    pub fn add_f32_register(&mut self) -> Operand<'a, F32> {
        let idx = self.f32_registers.len();
        let name = bumpalo::format!(in self.arena, "%f{}", idx);
        let name_str = name.into_bump_str();
        self.f32_registers.push(name_str);
        Operand::reg(name_str)
    }

    pub fn add_i32_register(&mut self) -> Operand<'a, I32> {
        let idx = self.i32_registers.len();
        let name = bumpalo::format!(in self.arena, "%r{}", idx);
        let name_str = name.into_bump_str();
        self.i32_registers.push(name_str);
        Operand::reg(name_str)
    }

    pub fn add_u32_register(&mut self) -> Operand<'a, U32> {
        let idx = self.i32_registers.len();
        let name = bumpalo::format!(in self.arena, "%r{}", idx);
        let name_str = name.into_bump_str();
        self.i32_registers.push(name_str);
        Operand::reg(name_str)
    }

    pub fn add_i64_register(&mut self) -> Operand<'a, I64> {
        let idx = self.i64_registers.len();
        let name = bumpalo::format!(in self.arena, "%rd{}", idx);
        let name_str = name.into_bump_str();
        self.i64_registers.push(name_str);
        Operand::reg(name_str)
    }

    pub fn add_u64_register(&mut self) -> Operand<'a, U64> {
        let idx = self.i64_registers.len();
        let name = bumpalo::format!(in self.arena, "%rd{}", idx);
        let name_str = name.into_bump_str();
        self.i64_registers.push(name_str);
        Operand::reg(name_str)
    }

    pub fn add_b64_register(&mut self) -> Operand<'a, B64> {
        let idx = self.i64_registers.len();
        let name = bumpalo::format!(in self.arena, "%rd{}", idx);
        let name_str = name.into_bump_str();
        self.i64_registers.push(name_str);
        Operand::reg(name_str)
    }

    pub fn add_b32_register(&mut self) -> Operand<'a, B32> {
        let idx = self.i32_registers.len();
        let name = bumpalo::format!(in self.arena, "%r{}", idx);
        let name_str = name.into_bump_str();
        self.i32_registers.push(name_str);
        Operand::reg(name_str)
    }

    pub fn add_shared_memory(&mut self, name: &'a str, size: usize) {
        self.shared_memory.push((name, size));
    }

    /// Add a parameter, load it, and convert to global address in one call
    ///
    /// This combines the common pattern of:
    /// - Adding a u64 parameter
    /// - Loading it into a register
    /// - Converting to a global address
    pub fn add_global_ptr_param(&mut self, name: &'a str) -> Operand<'a, U64> {
        let param = self.add_param_u64(name);
        let ptr = self.add_u64_register();
        self.add_inst(Inst::LdParamU64 {
            dst: ptr.clone(),
            addr: param,
        });
        let global = self.add_u64_register();
        self.add_inst(Inst::ConvertToGlobal {
            dst: global.clone(),
            src: ptr,
        });
        global
    }

    /// Add a u64 parameter and load it into a register
    ///
    /// Use this for scalar values like lengths, not pointers.
    pub fn add_u64_param_value(&mut self, name: &'a str) -> Operand<'a, U64> {
        let param = self.add_param_u64(name);
        let reg = self.add_u64_register();
        self.add_inst(Inst::LdParamU64 {
            dst: reg.clone(),
            addr: param,
        });
        reg
    }

    /// Capture instructions generated by a closure
    ///
    /// This allows executing a closure that adds instructions to the function body,
    /// then extracting those instructions as a separate Vec.
    /// Useful for building loop bodies or conditionally including instruction sequences.
    pub fn capture_instructions<F, R>(
        &mut self,
        f: F,
    ) -> (R, bumpalo::collections::Vec<'a, Inst<'a>>)
    where
        F: FnOnce(&mut Self) -> R,
    {
        let body_len_before = self.body.len();
        let result = f(self);
        let mut instructions = bumpalo::collections::Vec::new_in(self.arena);
        instructions.extend(self.body.drain(body_len_before..));
        (result, instructions)
    }
}

impl<'a> fmt::Display for Function<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.params.is_empty() {
            writeln!(f, ".visible .entry {}() {{", self.name)?;
        } else {
            writeln!(
                f,
                ".visible .entry {}(\n{}\n)\n{{",
                self.name,
                self.params
                    .iter()
                    .map(|(name, ty)| format!("    .param .{ty} {name}"))
                    .join(",\n")
            )?;
        }
        for (name, size) in &self.shared_memory {
            writeln!(f, "    .shared .align 16 .b8 {name}[{size}];")?;
        }
        if !self.predicate_registers.is_empty() {
            writeln!(f, "    .reg .pred %p<{}>;", self.predicate_registers.len())?;
        }
        if !self.f32_registers.is_empty() {
            writeln!(f, "    .reg .f32 %f<{}>;", self.f32_registers.len())?;
        }
        if !self.i32_registers.is_empty() {
            writeln!(f, "    .reg .b32 %r<{}>;", self.i32_registers.len())?;
        }
        if !self.i64_registers.is_empty() {
            writeln!(f, "    .reg .b64 %rd<{}>;", self.i64_registers.len())?;
        }
        writeln!(f, "\n")?;

        // Track whether we're inside a loop (after the first label)
        let mut in_loop = false;

        for inst in &self.body {
            match inst {
                Inst::Label(name) => {
                    // Add newline before labels for better readability
                    writeln!(f, "\n    {name}:")?;
                    in_loop = true;
                }
                _ => {
                    if in_loop {
                        writeln!(f, "      {inst}")?;
                    } else {
                        writeln!(f, "    {inst}")?;
                    }
                }
            }
        }
        for label in &self.labels {
            writeln!(f, "\n{}:", label.name)?;
            for inst in &label.body {
                writeln!(f, "    {inst}")?;
            }
        }
        writeln!(f, "\n}}")
    }
}

#[derive(Clone, Debug)]
pub struct Module<'a> {
    pub functions: Vec<Function<'a>>,
    target: (u32, u32),
}

impl Default for Module<'_> {
    fn default() -> Self {
        Self {
            functions: Vec::new(),
            target: (8, 0),
        }
    }
}

impl<'a> Module<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_target(target: (u32, u32)) -> Self {
        Self {
            functions: Vec::new(),
            target,
        }
    }

    pub fn add_function(&mut self, func: Function<'a>) {
        self.functions.push(func);
    }
}

impl<'a> fmt::Display for Module<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, ".version 8.0")?;
        writeln!(f, ".target sm_{}{}", self.target.0, self.target.1)?;
        writeln!(f, ".address_size 64")?;
        writeln!(f)?;

        for func in &self.functions {
            writeln!(f, "{func}\n")?;
        }

        Ok(())
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::F32 => write!(f, "f32"),
            Type::F16 => write!(f, "f16"),
            Type::BF16 => write!(f, "bf16"),
            Type::TF32 => write!(f, "tf32"),
            Type::FP8_E4M3 => write!(f, "e4m3x2"),
            Type::FP8_E5M2 => write!(f, "e5m2x2"),
            Type::I32 => write!(f, "s32"),
            Type::I64 => write!(f, "s64"),
            Type::U32 => write!(f, "u32"),
            Type::U64 => write!(f, "u64"),
            Type::Bitmask => write!(f, "b64"),
            Type::Pred => write!(f, "pred"),
        }
    }
}

impl fmt::Display for VecWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VecWidth::Scalar => Ok(()),
            VecWidth::V2 => write!(f, ".v2"),
            VecWidth::V4 => write!(f, ".v4"),
        }
    }
}

impl<'a, T: PtxType> fmt::Display for Operand<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Reg(name, _) => write!(f, "{name}"),
            Operand::Pred(name) => write!(f, "{name}"),
            Operand::ImmI32(value) => write!(f, "{value}"),
            Operand::ImmU64(value) => write!(f, "{value}"),
            Operand::ImmF32(value) => write!(f, "0f{:08X}", value.to_bits()),
            Operand::Addr(name) => write!(f, "[{name}]"),
            Operand::Symbol(name) => write!(f, "{name}"),
        }
    }
}

impl<'a, DstT: PtxType, SrcT: PtxType> fmt::Display for ConvertInst<'a, DstT, SrcT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cvt.{}.{} {}, {};",
            self.dst.ty(),
            self.src.ty(),
            self.dst,
            self.src
        )
    }
}

impl<'a, T: PtxType> fmt::Display for FmaInst<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mad.lo.{} {}, {}, {}, {};",
            self.dst.ty(),
            self.dst,
            self.a,
            self.b,
            self.c
        )
    }
}

impl<'a> fmt::Display for AddInst<'a, I32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "add.s32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for AddInst<'a, I64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "add.s64 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for AddInst<'a, U64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "add.u64 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for MulInst<'a, I32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mul.lo.s32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for MulInst<'a, U64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mul.lo.u64 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for MulInst<'a, F32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mul.f32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for MaxInst<'a, F32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "max.f32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for NegInst<'a, F32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "neg.f32 {}, {};", self.dst, self.src)
    }
}

impl<'a> fmt::Display for Ex2Inst<'a, F32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ex2.approx.f32 {}, {};", self.dst, self.src)
    }
}

impl<'a> fmt::Display for Lg2Inst<'a, F32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lg2.approx.f32 {}, {};", self.dst, self.src)
    }
}

impl<'a> fmt::Display for AddInst<'a, F32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "add.f32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for SubInst<'a, U64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sub.u64 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for SubInst<'a, F32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sub.f32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for DivInst<'a, U64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "div.u64 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for DivInst<'a, F32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "div.approx.f32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for ShiftLeftInst<'a, B64, U64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shl.b64 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for ShiftLeftInst<'a, U64, U64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shl.b64 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for ShiftLeftInst<'a, I32, I32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shl.b32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for ShiftRightInst<'a, I32, I32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shr.s32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for AndInst<'a, I32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "and.b32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for SetpInst<'a, U64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op_str = match self.op {
            CompareOp::Ge => "ge",
            CompareOp::Lt => "lt",
            CompareOp::Gt => "gt",
            CompareOp::Ne => "ne",
            CompareOp::Eq => "eq",
        };
        write!(f, "setp.{op_str}.u64 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for SetpInst<'a, I32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op_str = match self.op {
            CompareOp::Ge => "ge",
            CompareOp::Lt => "lt",
            CompareOp::Gt => "gt",
            CompareOp::Ne => "ne",
            CompareOp::Eq => "eq",
        };
        write!(f, "setp.{op_str}.s32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for SetpInst<'a, F32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op_str = match self.op {
            CompareOp::Ge => "ge",
            CompareOp::Lt => "lt",
            CompareOp::Gt => "gt",
            CompareOp::Ne => "ne",
            CompareOp::Eq => "eq",
        };
        write!(f, "setp.{op_str}.f32 {}, {}, {};", self.dst, self.a, self.b)
    }
}

impl<'a> fmt::Display for SelpInst<'a, F32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "selp.f32 {}, {}, {}, {};",
            self.dst, self.a, self.b, self.pred
        )
    }
}

impl<'a> fmt::Display for SelpInst<'a, U64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "selp.u64 {}, {}, {}, {};",
            self.dst, self.a, self.b, self.pred
        )
    }
}

impl<'a> fmt::Display for Inst<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Inst::ConvertU64U32(inst) => write!(f, "{inst}"),
            Inst::ConvertU64I32(inst) => write!(f, "{inst}"),
            Inst::ConvertU64F32(inst) => {
                write!(f, "cvt.rzi.u64.f32 {}, {};", inst.dst, inst.src)
            }
            Inst::ConvertI32U64(inst) => write!(f, "{inst}"),
            Inst::ConvertToGlobal { dst, src } => write!(f, "cvta.to.global.u64 {dst}, {src};"),
            Inst::MovI32 { dst, src } => write!(f, "mov.u32 {dst}, {src};"),
            Inst::MovU32 { dst, src } => write!(f, "mov.u32 {dst}, {src};"),
            Inst::MovU64 { dst, src } => write!(f, "mov.u64 {dst}, {src};"),
            Inst::MovF32 { dst, src } => write!(f, "mov.f32 {dst}, {src};"),
            Inst::MovB32 { dst, src } => write!(f, "mov.b32 {dst}, {src};"),
            Inst::MovF32B32 { dst, src } => write!(f, "mov.b32 {dst}, {src};"),
            Inst::LdGlobalF32 { dst, addr, vec } => match vec {
                VecWidth::Scalar => write!(f, "ld.global.f32 {}, [{addr}];", dst[0]),
                VecWidth::V2 => write!(
                    f,
                    "ld.global.v2.f32 {{{}}}, [{addr}];",
                    dst.iter().join(", ")
                ),
                VecWidth::V4 => write!(
                    f,
                    "ld.global.v4.f32 {{{}}}, [{addr}];",
                    dst.iter().join(", ")
                ),
            },
            Inst::LdParamU64 { dst, addr } => write!(f, "ld.param.u64 {dst}, {addr};"),
            Inst::StGlobalF32 { addr, src, vec } => match vec {
                VecWidth::Scalar => write!(f, "st.global.f32 [{addr}], {};", src.iter().join(", ")),
                _ => write!(
                    f,
                    "st.global{vec}.f32 [{addr}], {{{}}};",
                    src.iter().join(", ")
                ),
            },
            Inst::AtomicAddGlobalF32(inst) => write!(
                f,
                "atom.global.add.f32 {}, [{}], {};",
                inst.dst, inst.addr, inst.value
            ),

            Inst::AddI32(inst) => write!(f, "{inst}"),
            Inst::AddI64(inst) => write!(f, "{inst}"),
            Inst::AddU64(inst) => write!(f, "{inst}"),
            Inst::AddF32(inst) => write!(f, "{inst}"),
            Inst::SubU64(inst) => write!(f, "{inst}"),
            Inst::SubF32(inst) => write!(f, "{inst}"),
            Inst::MulI32(inst) => write!(f, "{inst}"),
            Inst::MulU64(inst) => write!(f, "{inst}"),
            Inst::MulF32(inst) => write!(f, "{inst}"),
            Inst::DivU64(inst) => write!(f, "{inst}"),
            Inst::DivF32(inst) => write!(f, "{inst}"),
            Inst::MaxF32(inst) => write!(f, "{inst}"),
            Inst::NegF32(inst) => write!(f, "{inst}"),
            Inst::Ex2F32(inst) => write!(f, "{inst}"),
            Inst::Lg2F32(inst) => write!(f, "{inst}"),
            Inst::FmaI32(inst) => write!(f, "{inst}"),
            Inst::ShiftLeftB64(inst) => write!(f, "{inst}"),
            Inst::ShiftLeftU64(inst) => write!(f, "{inst}"),
            Inst::ShiftLeftI32(inst) => write!(f, "{inst}"),
            Inst::ShiftRightI32(inst) => write!(f, "{inst}"),
            Inst::AndI32(inst) => write!(f, "{inst}"),
            Inst::SetpU64(inst) => write!(f, "{inst}"),
            Inst::SetpI32(inst) => write!(f, "{inst}"),
            Inst::SetpF32(inst) => write!(f, "{inst}"),
            Inst::SelpF32(inst) => write!(f, "{inst}"),
            Inst::SelpU64(inst) => write!(f, "{inst}"),
            Inst::ShflSyncDownB32 {
                dst,
                predicate,
                src,
                offset,
            } => write!(
                f,
                "shfl.sync.down.b32 {dst}|{predicate}, {src}, {offset}, 0x1f, 0xffffffff;"
            ),

            Inst::Bra { condition, target } => write!(f, "@{condition} bra {target};"),
            Inst::BraUni { target } => write!(f, "bra.uni {target};"),
            Inst::Label(name) => write!(f, "{name}:"),
            Inst::Ret => write!(f, "ret;"),
            Inst::Trap => write!(f, "trap;"),

            Inst::BarSync { barrier_id } => write!(f, "bar.sync {barrier_id};"),

            Inst::LdSharedF32 { dst, addr, vec } => match vec {
                VecWidth::Scalar => write!(f, "ld.shared.f32 {}, [{addr}];", dst.iter().join(", ")),
                _ => write!(
                    f,
                    "ld.shared{vec}.f32 {{{}}}, [{addr}];",
                    dst.iter().join(", ")
                ),
            },
            Inst::StSharedF32 { addr, src, vec } => match vec {
                VecWidth::Scalar => write!(f, "st.shared.f32 [{addr}], {};", src.iter().join(", ")),
                _ => write!(
                    f,
                    "st.shared{vec}.f32 [{addr}], {{{}}};",
                    src.iter().join(", ")
                ),
            },
            Inst::StSharedB32 { addr, src } => write!(f, "st.shared.b32 [{addr}], {src};"),
            Inst::ConvertTf32F32 { dst, src } => {
                write!(f, "cvt.rna.tf32.f32 {dst}, {src};")
            }
            Inst::LdMatrix { frags, addr } => {
                write!(
                    f,
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b32 {{{}}}, [{addr}];",
                    frags.iter().join(", ")
                )
            }

            Inst::WmmaLoadA {
                frags,
                addr,
                stride,
            } => {
                write!(
                    f,
                    "wmma.load.a.sync.aligned.m16n16k8.row.shared.tf32 {{{}}}, [{}], {};",
                    frags.iter().join(", "),
                    addr,
                    stride
                )
            }
            Inst::WmmaLoadB {
                frags,
                addr,
                stride,
            } => {
                write!(
                    f,
                    "wmma.load.b.sync.aligned.m16n16k8.row.shared.tf32 {{{}}}, [{}], {};",
                    frags.iter().join(", "),
                    addr,
                    stride
                )
            }
            Inst::WmmaMma {
                d_frags,
                a_frags,
                b_frags,
                c_frags,
            } => {
                write!(
                    f,
                    "wmma.mma.sync.aligned.m16n16k8.row.row.f32.tf32.tf32.f32 {{{}}}, {{{}}}, {{{}}}, {{{}}};",
                    d_frags.iter().join(", "),
                    a_frags.iter().join(", "),
                    b_frags.iter().join(", "),
                    c_frags.iter().join(", ")
                )
            }
            Inst::MmaM16N8K16 {
                d_frags,
                a_frags,
                b_frags,
                c_frags,
            } => {
                write!(
                    f,
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{}}}, {{{}}}, {{{}}}, {{{}}};",
                    d_frags.iter().join(", "),
                    a_frags.iter().join(", "),
                    b_frags.iter().join(", "),
                    c_frags.iter().join(", ")
                )
            }
            Inst::MmaM16N8K8Tf32 {
                d_frags,
                a_frags,
                b_frags,
                c_frags,
            } => {
                write!(
                    f,
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {{{}}}, {{{}}}, {{{}}}, {{{}}};",
                    d_frags.iter().join(", "),
                    a_frags.iter().join(", "),
                    b_frags.iter().join(", "),
                    c_frags.iter().join(", ")
                )
            }
            Inst::WmmaStore {
                addr,
                frags,
                stride,
            } => {
                write!(
                    f,
                    "wmma.store.d.sync.aligned.m16n16k8.row.shared.f32 [{}], {{{}}}, {};",
                    addr,
                    frags.iter().join(", "),
                    stride
                )
            }
        }
    }
}
