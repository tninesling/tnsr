use std::fmt;

use itertools::Itertools;

#[derive(Clone, Copy, Debug)]
pub enum Type {
    F32,
    F16,
    I32,
    U32,
    Pred,
}

#[derive(Clone, Copy, Debug)]
pub enum VecWidth {
    Scalar,
    V2,
    V4,
}

#[derive(Clone, Debug)]
pub enum Operand {
    Reg(String, Type),
    Pred(String),
    ImmI32(i32),
    ImmF32(f32),
    Addr(String), // label or symbol
}

#[derive(Clone, Debug)]
pub enum Inst {
    // Data movement
    Mov {
        dst: Operand,
        src: Operand,
    },
    LdGlobal {
        dst: Vec<Operand>,
        addr: Operand,
        ty: Type,
        vec: VecWidth,
    },
    StGlobal {
        addr: Operand,
        src: Vec<Operand>,
        ty: Type,
        vec: VecWidth,
    },

    // Math
    Add {
        dst: Operand,
        a: Operand,
        b: Operand,
        ty: Type,
    },
    Mul {
        dst: Operand,
        a: Operand,
        b: Operand,
        ty: Type,
    },
    Fma {
        dst: Operand,
        a: Operand,
        b: Operand,
        c: Operand,
        ty: Type,
    },
    Max {
        dst: Operand,
        a: Operand,
        b: Operand,
        ty: Type,
    },

    // Control
    SetpLtU32 {
        dst: Operand,
        a: Operand,
        b: Operand,
    },
    Bra {
        target: String,
    },
    Label(String),
    Ret,

    // Barrier
    BarSync {
        barrier_id: u32,
    },
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub body: Vec<Inst>,
}

#[derive(Clone, Debug)]
pub struct Module {
    pub functions: Vec<Function>,
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, ".version 8.0")?;
        writeln!(f, ".target sm_80")?;
        writeln!(f, ".address_size 64")?;
        writeln!(f)?;

        for func in &self.functions {
            writeln!(f, "{func}\n")?;
        }

        Ok(())
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, ".visible .entry {}() {{", self.name)?;
        for inst in &self.body {
            writeln!(f, "  {inst}")?;
        }
        writeln!(f, "}}")
    }
}

impl fmt::Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LdGlobal { dst, addr, ty, vec } => {
                let dst_regs = dst.iter().join(", ");
                write!(f, "ld.global{vec}.{ty} {{ {dst_regs} }}, [{addr}];",)
            }
            Self::Add { dst, a, b, ty } => write!(f, "add.{ty} {dst}, {a}, {b};"),
            Self::StGlobal { addr, src, ty, vec } => {
                let src_regs = src.iter().join(", ");
                write!(f, "st.global{vec}.{ty} [{addr}], {{ {src_regs} }};")
            }
            _ => todo!(),
        }
    }
}

impl fmt::Display for VecWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar => write!(f, ""),
            Self::V2 => write!(f, ".v2"),
            Self::V4 => write!(f, ".v4"),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::I32 => write!(f, "s32"),
            Self::U32 => write!(f, "u32"),
            Self::Pred => write!(f, "pred"),
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(name, _) => write!(f, "{name}"),
            Self::Pred(name) => write!(f, "{name}"),
            Self::ImmI32(v) => write!(f, "{v}"),
            Self::ImmF32(v) => write!(f, "{v}"),
            Self::Addr(sym) => write!(f, "[{sym}]"),
        }
    }
}
