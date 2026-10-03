use std::marker::PhantomData;
use std::sync::LazyLock;

use super::types::*;

pub static GRID_DIM: LazyLock<Dim3<'static>> = LazyLock::new(|| Dim3 {
    x: Operand::reg("%nctaid.x"),
    y: Operand::reg("%nctaid.y"),
    z: Operand::reg("%nctaid.z"),
});

pub static BLOCK_ID: LazyLock<Dim3<'static>> = LazyLock::new(|| Dim3 {
    x: Operand::reg("%ctaid.x"),
    y: Operand::reg("%ctaid.y"),
    z: Operand::reg("%ctaid.z"),
});

pub static BLOCK_DIM: LazyLock<Dim3<'static>> = LazyLock::new(|| Dim3 {
    x: Operand::reg("%ntid.x"),
    y: Operand::reg("%ntid.y"),
    z: Operand::reg("%ntid.z"),
});

pub static THREAD_ID: LazyLock<Dim3<'static>> = LazyLock::new(|| Dim3 {
    x: Operand::reg("%tid.x"),
    y: Operand::reg("%tid.y"),
    z: Operand::reg("%tid.z"),
});

pub struct Dim3<'a> {
    pub x: Operand<'a, U32>,
    pub y: Operand<'a, U32>,
    pub z: Operand<'a, U32>,
}

#[derive(Clone, Copy, Debug)]
pub enum VecWidth {
    Scalar,
    V2,
    V4,
}

#[derive(Clone, Debug)]
pub enum Operand<'a, T: PtxType> {
    Reg(&'a str, PhantomData<T>),
    Pred(&'a str),
    ImmI32(i32),
    ImmU64(u64),
    ImmF32(f32),
    Addr(&'a str),   // Parameter address (formatted with brackets [name])
    Symbol(&'a str), // Symbol reference (formatted without brackets)
}

impl<'a, T: PtxType> Operand<'a, T> {
    pub fn reg(name: &'a str) -> Self {
        Operand::Reg(name, PhantomData)
    }

    pub fn imm_i32(value: i32) -> Self {
        Operand::ImmI32(value)
    }

    pub fn imm_u64(value: u64) -> Self {
        Operand::ImmU64(value)
    }

    pub fn imm_f32(value: f32) -> Self {
        Operand::ImmF32(value)
    }

    pub fn addr(name: &'a str) -> Self {
        Operand::Addr(name)
    }

    pub fn symbol(name: &'a str) -> Self {
        Operand::Symbol(name)
    }

    pub fn pred(name: &'a str) -> Self {
        Operand::Pred(name)
    }

    pub fn ty(&self) -> Type {
        T::as_type()
    }
}

#[derive(Clone, Debug)]
pub struct ConvertInst<'a, DstT: PtxType, SrcT: PtxType> {
    pub dst: Operand<'a, DstT>,
    pub src: Operand<'a, SrcT>,
    _phantom: PhantomData<(DstT, SrcT)>,
}

#[derive(Clone, Debug)]
pub struct AtomicAddInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub addr: Operand<'a, U64>,
    pub value: Operand<'a, T>,
}

impl<'a, T: PtxType> AtomicAddInst<'a, T> {
    pub fn new(dst: Operand<'a, T>, addr: Operand<'a, U64>, value: Operand<'a, T>) -> Self {
        Self { dst, addr, value }
    }
}

impl<'a, DstT: PtxType, SrcT: PtxType> ConvertInst<'a, DstT, SrcT> {
    pub fn new(dst: Operand<'a, DstT>, src: Operand<'a, SrcT>) -> Self {
        Self {
            dst,
            src,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BinaryInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub a: Operand<'a, T>,
    pub b: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> BinaryInst<'a, T> {
    pub fn new(dst: Operand<'a, T>, a: Operand<'a, T>, b: Operand<'a, T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UnaryInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub src: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> UnaryInst<'a, T> {
    pub fn new(dst: Operand<'a, T>, src: Operand<'a, T>) -> Self {
        Self {
            dst,
            src,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AddInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub a: Operand<'a, T>,
    pub b: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> AddInst<'a, T> {
    pub fn new(dst: Operand<'a, T>, a: Operand<'a, T>, b: Operand<'a, T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MulInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub a: Operand<'a, T>,
    pub b: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> MulInst<'a, T> {
    pub fn new(dst: Operand<'a, T>, a: Operand<'a, T>, b: Operand<'a, T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MaxInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub a: Operand<'a, T>,
    pub b: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> MaxInst<'a, T> {
    pub fn new(dst: Operand<'a, T>, a: Operand<'a, T>, b: Operand<'a, T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NegInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub src: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> NegInst<'a, T> {
    pub fn new(dst: Operand<'a, T>, src: Operand<'a, T>) -> Self {
        Self {
            dst,
            src,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Ex2Inst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub src: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> Ex2Inst<'a, T> {
    pub fn new(dst: Operand<'a, T>, src: Operand<'a, T>) -> Self {
        Self {
            dst,
            src,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Lg2Inst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub src: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> Lg2Inst<'a, T> {
    pub fn new(dst: Operand<'a, T>, src: Operand<'a, T>) -> Self {
        Self {
            dst,
            src,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub a: Operand<'a, T>,
    pub b: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> SubInst<'a, T> {
    pub fn new(dst: Operand<'a, T>, a: Operand<'a, T>, b: Operand<'a, T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DivInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub a: Operand<'a, T>,
    pub b: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> DivInst<'a, T> {
    pub fn new(dst: Operand<'a, T>, a: Operand<'a, T>, b: Operand<'a, T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CompareOp {
    Ge,
    Lt,
    Gt,
    Ne,
    Eq,
}

#[derive(Clone, Debug)]
pub struct ShiftLeftInst<'a, DstT: PtxType, SrcT: PtxType> {
    pub dst: Operand<'a, DstT>,
    pub a: Operand<'a, SrcT>,
    pub b: Operand<'a, I32>,
    _phantom: PhantomData<(DstT, SrcT)>,
}

impl<'a, DstT: PtxType, SrcT: PtxType> ShiftLeftInst<'a, DstT, SrcT> {
    pub fn new(dst: Operand<'a, DstT>, a: Operand<'a, SrcT>, b: Operand<'a, I32>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShiftRightInst<'a, DstT: PtxType, SrcT: PtxType> {
    pub dst: Operand<'a, DstT>,
    pub a: Operand<'a, SrcT>,
    pub b: Operand<'a, I32>,
    _phantom: PhantomData<(DstT, SrcT)>,
}

impl<'a, DstT: PtxType, SrcT: PtxType> ShiftRightInst<'a, DstT, SrcT> {
    pub fn new(dst: Operand<'a, DstT>, a: Operand<'a, SrcT>, b: Operand<'a, I32>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AndInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub a: Operand<'a, T>,
    pub b: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> AndInst<'a, T> {
    pub fn new(dst: Operand<'a, T>, a: Operand<'a, T>, b: Operand<'a, T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SetpInst<'a, T: PtxType> {
    pub dst: Operand<'a, Pred>,
    pub a: Operand<'a, T>,
    pub b: Operand<'a, T>,
    pub op: CompareOp,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> SetpInst<'a, T> {
    pub fn new(
        dst: Operand<'a, Pred>,
        a: Operand<'a, T>,
        b: Operand<'a, T>,
        op: CompareOp,
    ) -> Self {
        Self {
            dst,
            a,
            b,
            op,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FmaInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub a: Operand<'a, T>,
    pub b: Operand<'a, T>,
    pub c: Operand<'a, T>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> FmaInst<'a, T> {
    pub fn new(
        dst: Operand<'a, T>,
        a: Operand<'a, T>,
        b: Operand<'a, T>,
        c: Operand<'a, T>,
    ) -> Self {
        Self {
            dst,
            a,
            b,
            c,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SelpInst<'a, T: PtxType> {
    pub dst: Operand<'a, T>,
    pub a: Operand<'a, T>,
    pub b: Operand<'a, T>,
    pub pred: Operand<'a, Pred>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PtxType> SelpInst<'a, T> {
    pub fn new(
        dst: Operand<'a, T>,
        a: Operand<'a, T>,
        b: Operand<'a, T>,
        pred: Operand<'a, Pred>,
    ) -> Self {
        Self {
            dst,
            a,
            b,
            pred,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Inst<'a> {
    ConvertU64U32(ConvertInst<'a, U64, U32>),
    ConvertU64I32(ConvertInst<'a, U64, I32>),
    ConvertU64F32(ConvertInst<'a, U64, F32>),
    ConvertI32U64(ConvertInst<'a, I32, U64>),
    ConvertF32F16(ConvertInst<'a, F32, F16>),
    ConvertF16F32(ConvertInst<'a, F16, F32>),
    ConvertF32BF16(ConvertInst<'a, F32, BF16>),
    ConvertBF16F32(ConvertInst<'a, BF16, F32>),
    ConvertToGlobal {
        dst: Operand<'a, U64>,
        src: Operand<'a, U64>,
    },
    MovI32 {
        dst: Operand<'a, I32>,
        src: Operand<'a, U32>,
    },
    MovU32 {
        dst: Operand<'a, U32>,
        src: Operand<'a, U32>,
    },
    MovU64 {
        dst: Operand<'a, U64>,
        src: Operand<'a, U64>,
    },
    MovF32 {
        dst: Operand<'a, F32>,
        src: Operand<'a, F32>,
    },
    MovB32 {
        dst: Operand<'a, B32>,
        src: Operand<'a, F32>,
    },
    MovF32B32 {
        dst: Operand<'a, F32>,
        src: Operand<'a, B32>,
    },
    LdGlobalF32 {
        dst: Vec<Operand<'a, F32>>,
        addr: Operand<'a, U64>,
        vec: VecWidth,
    },
    LdGlobalF16 {
        dst: Operand<'a, F16>,
        addr: Operand<'a, U64>,
    },
    LdGlobalBF16 {
        dst: Operand<'a, BF16>,
        addr: Operand<'a, U64>,
    },
    LdParamU64 {
        dst: Operand<'a, U64>,
        addr: Operand<'a, U64>,
    },
    StGlobalF32 {
        addr: Operand<'a, U64>,
        src: Vec<Operand<'a, F32>>,
        vec: VecWidth,
    },
    StGlobalF16 {
        addr: Operand<'a, U64>,
        src: Operand<'a, F16>,
    },
    StGlobalBF16 {
        addr: Operand<'a, U64>,
        src: Operand<'a, BF16>,
    },
    AtomicAddGlobalF32(AtomicAddInst<'a, F32>),

    AddI32(AddInst<'a, I32>),
    AddI64(AddInst<'a, I64>),
    AddU64(AddInst<'a, U64>),
    AddF32(AddInst<'a, F32>),
    SubU64(SubInst<'a, U64>),
    SubF32(SubInst<'a, F32>),
    MulI32(MulInst<'a, I32>),
    MulU64(MulInst<'a, U64>),
    MulF32(MulInst<'a, F32>),
    DivU64(DivInst<'a, U64>),
    DivF32(DivInst<'a, F32>),
    MaxF32(MaxInst<'a, F32>),
    NegF32(NegInst<'a, F32>),
    Ex2F32(Ex2Inst<'a, F32>),
    Lg2F32(Lg2Inst<'a, F32>),
    FmaI32(FmaInst<'a, I32>),
    ShiftLeftB64(ShiftLeftInst<'a, B64, U64>),
    ShiftLeftU64(ShiftLeftInst<'a, U64, U64>),
    ShiftLeftI32(ShiftLeftInst<'a, I32, I32>),
    ShiftRightI32(ShiftRightInst<'a, I32, I32>),
    AndI32(AndInst<'a, I32>),
    SetpU64(SetpInst<'a, U64>),
    SetpI32(SetpInst<'a, I32>),
    SetpF32(SetpInst<'a, F32>),
    SelpF32(SelpInst<'a, F32>),
    SelpU64(SelpInst<'a, U64>),
    ShflSyncDownB32 {
        dst: Operand<'a, B32>,
        predicate: Operand<'a, Pred>,
        src: Operand<'a, B32>,
        offset: u32,
    },

    Bra {
        condition: Operand<'a, Pred>,
        target: &'a str,
    },
    BraUni {
        target: &'a str,
    },
    Label(&'a str),
    Ret,
    Trap,

    BarSync {
        barrier_id: u32,
    },

    LdSharedF32 {
        dst: Vec<Operand<'a, F32>>,
        addr: Operand<'a, U64>,
        vec: VecWidth,
    },
    LdSharedF16 {
        dst: Operand<'a, F16>,
        addr: Operand<'a, U64>,
    },
    LdSharedBF16 {
        dst: Operand<'a, BF16>,
        addr: Operand<'a, U64>,
    },
    StSharedF32 {
        addr: Operand<'a, U64>,
        src: Vec<Operand<'a, F32>>,
        vec: VecWidth,
    },
    StSharedF16 {
        addr: Operand<'a, U64>,
        src: Operand<'a, F16>,
    },
    StSharedBF16 {
        addr: Operand<'a, U64>,
        src: Operand<'a, BF16>,
    },
    StSharedB32 {
        addr: Operand<'a, U64>,
        src: Operand<'a, B32>,
    },
    ConvertTf32F32 {
        dst: Operand<'a, B32>,
        src: Operand<'a, F32>,
    },
    LdMatrix {
        frags: Vec<Operand<'a, B32>>,
        addr: Operand<'a, U64>,
    },

    WmmaLoadA {
        dtype: crate::tile::DType,
        frags: Vec<Operand<'a, B32>>,
        addr: Operand<'a, U64>,
        stride: Operand<'a, I32>,
    },
    WmmaLoadB {
        dtype: crate::tile::DType,
        frags: Vec<Operand<'a, B32>>,
        addr: Operand<'a, U64>,
        stride: Operand<'a, I32>,
    },
    WmmaMma {
        dtype: crate::tile::DType,
        d_frags: Vec<Operand<'a, F32>>,
        a_frags: Vec<Operand<'a, B32>>,
        b_frags: Vec<Operand<'a, B32>>,
        c_frags: Vec<Operand<'a, F32>>,
    },
    MmaM16N8K16 {
        d_frags: Vec<Operand<'a, F32>>,
        a_frags: Vec<Operand<'a, I32>>,
        b_frags: Vec<Operand<'a, I32>>,
        c_frags: Vec<Operand<'a, F32>>,
    },
    MmaM16N8K8Tf32 {
        d_frags: Vec<Operand<'a, F32>>,
        a_frags: Vec<Operand<'a, B32>>,
        b_frags: Vec<Operand<'a, B32>>,
        c_frags: Vec<Operand<'a, F32>>,
    },
    WmmaStore {
        dtype: crate::tile::DType,
        addr: Operand<'a, U64>,
        frags: Vec<Operand<'a, F32>>,
        stride: Operand<'a, I32>,
    },
}

impl<'a> Inst<'a> {
    pub fn convert_u64_u32(dst: Operand<'a, U64>, src: Operand<'a, U32>) -> Self {
        Inst::ConvertU64U32(ConvertInst::new(dst, src))
    }

    pub fn convert_u64_i32(dst: Operand<'a, U64>, src: Operand<'a, I32>) -> Self {
        Inst::ConvertU64I32(ConvertInst::new(dst, src))
    }

    pub fn convert_u64_f32(dst: Operand<'a, U64>, src: Operand<'a, F32>) -> Self {
        Inst::ConvertU64F32(ConvertInst::new(dst, src))
    }

    pub fn convert_i32_u64(dst: Operand<'a, I32>, src: Operand<'a, U64>) -> Self {
        Inst::ConvertI32U64(ConvertInst::new(dst, src))
    }

    pub fn convert_f32_f16(dst: Operand<'a, F32>, src: Operand<'a, F16>) -> Self {
        Inst::ConvertF32F16(ConvertInst::new(dst, src))
    }

    pub fn convert_f16_f32(dst: Operand<'a, F16>, src: Operand<'a, F32>) -> Self {
        Inst::ConvertF16F32(ConvertInst::new(dst, src))
    }

    pub fn convert_f32_bf16(dst: Operand<'a, F32>, src: Operand<'a, BF16>) -> Self {
        Inst::ConvertF32BF16(ConvertInst::new(dst, src))
    }

    pub fn convert_bf16_f32(dst: Operand<'a, BF16>, src: Operand<'a, F32>) -> Self {
        Inst::ConvertBF16F32(ConvertInst::new(dst, src))
    }

    pub fn convert_to_global(dst: Operand<'a, U64>, src: Operand<'a, U64>) -> Self {
        Inst::ConvertToGlobal { dst, src }
    }

    pub fn mov_i32(dst: Operand<'a, I32>, src: Operand<'a, U32>) -> Self {
        Inst::MovI32 { dst, src }
    }

    pub fn mov_u32(dst: Operand<'a, U32>, src: Operand<'a, U32>) -> Self {
        Inst::MovU32 { dst, src }
    }

    pub fn mov_u64(dst: Operand<'a, U64>, src: Operand<'a, U64>) -> Self {
        Inst::MovU64 { dst, src }
    }

    pub fn mov_f32(dst: Operand<'a, F32>, src: Operand<'a, F32>) -> Self {
        Inst::MovF32 { dst, src }
    }

    pub fn mov_b32(dst: Operand<'a, B32>, src: Operand<'a, F32>) -> Self {
        Inst::MovB32 { dst, src }
    }

    pub fn mov_f32_b32(dst: Operand<'a, F32>, src: Operand<'a, B32>) -> Self {
        Inst::MovF32B32 { dst, src }
    }

    pub fn load_global_scalar_f32(dst: Operand<'a, F32>, addr: Operand<'a, U64>) -> Self {
        Inst::LdGlobalF32 {
            dst: vec![dst],
            addr,
            vec: VecWidth::Scalar,
        }
    }

    pub fn load_param_u64(dst: Operand<'a, U64>, addr: Operand<'a, U64>) -> Self {
        Inst::LdParamU64 { dst, addr }
    }

    pub fn store_global_scalar_f32(addr: Operand<'a, U64>, src: Operand<'a, F32>) -> Self {
        Inst::StGlobalF32 {
            addr,
            src: vec![src],
            vec: VecWidth::Scalar,
        }
    }

    pub fn atomic_add_global_f32(
        dst: Operand<'a, F32>,
        addr: Operand<'a, U64>,
        value: Operand<'a, F32>,
    ) -> Self {
        Inst::AtomicAddGlobalF32(AtomicAddInst::new(dst, addr, value))
    }

    pub fn add_i32(dst: Operand<'a, I32>, a: Operand<'a, I32>, b: Operand<'a, I32>) -> Self {
        Inst::AddI32(AddInst::new(dst, a, b))
    }

    pub fn add_i64(dst: Operand<'a, I64>, a: Operand<'a, I64>, b: Operand<'a, I64>) -> Self {
        Inst::AddI64(AddInst::new(dst, a, b))
    }

    pub fn add_u64(dst: Operand<'a, U64>, a: Operand<'a, U64>, b: Operand<'a, U64>) -> Self {
        Inst::AddU64(AddInst::new(dst, a, b))
    }

    pub fn add_f32(dst: Operand<'a, F32>, a: Operand<'a, F32>, b: Operand<'a, F32>) -> Self {
        Inst::AddF32(AddInst::new(dst, a, b))
    }

    pub fn sub_u64(dst: Operand<'a, U64>, a: Operand<'a, U64>, b: Operand<'a, U64>) -> Self {
        Inst::SubU64(SubInst::new(dst, a, b))
    }

    pub fn sub_f32(dst: Operand<'a, F32>, a: Operand<'a, F32>, b: Operand<'a, F32>) -> Self {
        Inst::SubF32(SubInst::new(dst, a, b))
    }

    pub fn mul_i32(dst: Operand<'a, I32>, a: Operand<'a, I32>, b: Operand<'a, I32>) -> Self {
        Inst::MulI32(MulInst::new(dst, a, b))
    }

    pub fn mul_u64(dst: Operand<'a, U64>, a: Operand<'a, U64>, b: Operand<'a, U64>) -> Self {
        Inst::MulU64(MulInst::new(dst, a, b))
    }

    pub fn mul_f32(dst: Operand<'a, F32>, a: Operand<'a, F32>, b: Operand<'a, F32>) -> Self {
        Inst::MulF32(MulInst::new(dst, a, b))
    }

    pub fn div_u64(dst: Operand<'a, U64>, a: Operand<'a, U64>, b: Operand<'a, U64>) -> Self {
        Inst::DivU64(DivInst::new(dst, a, b))
    }

    pub fn div_f32(dst: Operand<'a, F32>, a: Operand<'a, F32>, b: Operand<'a, F32>) -> Self {
        Inst::DivF32(DivInst::new(dst, a, b))
    }

    pub fn max_f32(dst: Operand<'a, F32>, a: Operand<'a, F32>, b: Operand<'a, F32>) -> Self {
        Inst::MaxF32(MaxInst::new(dst, a, b))
    }

    pub fn fma_i32(
        dst: Operand<'a, I32>,
        a: Operand<'a, I32>,
        b: Operand<'a, I32>,
        c: Operand<'a, I32>,
    ) -> Inst<'a> {
        Inst::FmaI32(FmaInst::new(dst, a, b, c))
    }

    pub fn neg_f32(dst: Operand<'a, F32>, src: Operand<'a, F32>) -> Self {
        Inst::NegF32(NegInst::new(dst, src))
    }

    pub fn ex2_f32(dst: Operand<'a, F32>, src: Operand<'a, F32>) -> Self {
        Inst::Ex2F32(Ex2Inst::new(dst, src))
    }

    pub fn lg2_f32(dst: Operand<'a, F32>, src: Operand<'a, F32>) -> Self {
        Inst::Lg2F32(Lg2Inst::new(dst, src))
    }

    pub fn shift_left_b64(dst: Operand<'a, B64>, a: Operand<'a, U64>, b: Operand<'a, I32>) -> Self {
        Inst::ShiftLeftB64(ShiftLeftInst::new(dst, a, b))
    }

    pub fn shift_left_u64(dst: Operand<'a, U64>, a: Operand<'a, U64>, b: Operand<'a, I32>) -> Self {
        Inst::ShiftLeftU64(ShiftLeftInst::new(dst, a, b))
    }

    pub fn shift_left_i32(dst: Operand<'a, I32>, a: Operand<'a, I32>, b: Operand<'a, I32>) -> Self {
        Inst::ShiftLeftI32(ShiftLeftInst::new(dst, a, b))
    }

    pub fn shift_right_i32(
        dst: Operand<'a, I32>,
        a: Operand<'a, I32>,
        b: Operand<'a, I32>,
    ) -> Self {
        Inst::ShiftRightI32(ShiftRightInst::new(dst, a, b))
    }

    pub fn and_i32(dst: Operand<'a, I32>, a: Operand<'a, I32>, b: Operand<'a, I32>) -> Self {
        Inst::AndI32(AndInst::new(dst, a, b))
    }

    pub fn setp_ge_u64(dst: Operand<'a, Pred>, a: Operand<'a, U64>, b: Operand<'a, U64>) -> Self {
        Inst::SetpU64(SetpInst::new(dst, a, b, CompareOp::Ge))
    }

    pub fn setp_lt_u64(dst: Operand<'a, Pred>, a: Operand<'a, U64>, b: Operand<'a, U64>) -> Self {
        Inst::SetpU64(SetpInst::new(dst, a, b, CompareOp::Lt))
    }

    pub fn setp_eq_u64(dst: Operand<'a, Pred>, a: Operand<'a, U64>, b: Operand<'a, U64>) -> Self {
        Inst::SetpU64(SetpInst::new(dst, a, b, CompareOp::Eq))
    }

    pub fn setp_ne_u64(dst: Operand<'a, Pred>, a: Operand<'a, U64>, b: Operand<'a, U64>) -> Self {
        Inst::SetpU64(SetpInst::new(dst, a, b, CompareOp::Ne))
    }

    pub fn setp_lt_i32(dst: Operand<'a, Pred>, a: Operand<'a, I32>, b: Operand<'a, I32>) -> Self {
        Inst::SetpI32(SetpInst::new(dst, a, b, CompareOp::Lt))
    }

    pub fn setp_gt_f32(dst: Operand<'a, Pred>, a: Operand<'a, F32>, b: Operand<'a, F32>) -> Self {
        Inst::SetpF32(SetpInst::new(dst, a, b, CompareOp::Gt))
    }

    pub fn setp_ne_f32(dst: Operand<'a, Pred>, a: Operand<'a, F32>, b: Operand<'a, F32>) -> Self {
        Inst::SetpF32(SetpInst::new(dst, a, b, CompareOp::Ne))
    }

    pub fn setp_eq_f32(dst: Operand<'a, Pred>, a: Operand<'a, F32>, b: Operand<'a, F32>) -> Self {
        Inst::SetpF32(SetpInst::new(dst, a, b, CompareOp::Eq))
    }

    pub fn selp_f32(
        dst: Operand<'a, F32>,
        a: Operand<'a, F32>,
        b: Operand<'a, F32>,
        pred: Operand<'a, Pred>,
    ) -> Self {
        Inst::SelpF32(SelpInst::new(dst, a, b, pred))
    }

    pub fn shfl_sync_down_b32(
        dst: Operand<'a, B32>,
        predicate: Operand<'a, Pred>,
        src: Operand<'a, B32>,
        offset: u32,
    ) -> Self {
        Inst::ShflSyncDownB32 {
            dst,
            predicate,
            src,
            offset,
        }
    }

    pub fn selp_u64(
        dst: Operand<'a, U64>,
        a: Operand<'a, U64>,
        b: Operand<'a, U64>,
        pred: Operand<'a, Pred>,
    ) -> Self {
        Inst::SelpU64(SelpInst::new(dst, a, b, pred))
    }

    pub fn bra(condition: Operand<'a, Pred>, target: &'a str) -> Self {
        Inst::Bra { condition, target }
    }

    pub fn label(name: &'a str) -> Self {
        Inst::Label(name)
    }

    pub fn wmma_load_a(
        frags: Vec<Operand<'a, B32>>,
        addr: Operand<'a, U64>,
        stride: Operand<'a, I32>,
    ) -> Self {
        Inst::WmmaLoadA {
            dtype: crate::tile::DType::TF32,
            frags,
            addr,
            stride,
        }
    }

    pub fn wmma_load_b(
        frags: Vec<Operand<'a, B32>>,
        addr: Operand<'a, U64>,
        stride: Operand<'a, I32>,
    ) -> Self {
        Inst::WmmaLoadB {
            dtype: crate::tile::DType::TF32,
            frags,
            addr,
            stride,
        }
    }

    pub fn wmma_mma(
        d_frags: Vec<Operand<'a, F32>>,
        a_frags: Vec<Operand<'a, B32>>,
        b_frags: Vec<Operand<'a, B32>>,
        c_frags: Vec<Operand<'a, F32>>,
    ) -> Self {
        Inst::WmmaMma {
            dtype: crate::tile::DType::TF32,
            d_frags,
            a_frags,
            b_frags,
            c_frags,
        }
    }

    pub fn convert_tf32_f32(dst: Operand<'a, B32>, src: Operand<'a, F32>) -> Self {
        Inst::ConvertTf32F32 { dst, src }
    }

    pub fn mma_m16n8k16(
        d_frags: Vec<Operand<'a, F32>>,
        a_frags: Vec<Operand<'a, I32>>,
        b_frags: Vec<Operand<'a, I32>>,
        c_frags: Vec<Operand<'a, F32>>,
    ) -> Self {
        Inst::MmaM16N8K16 {
            d_frags,
            a_frags,
            b_frags,
            c_frags,
        }
    }

    pub fn mma_m16n8k8_tf32(
        d_frags: Vec<Operand<'a, F32>>,
        a_frags: Vec<Operand<'a, B32>>,
        b_frags: Vec<Operand<'a, B32>>,
        c_frags: Vec<Operand<'a, F32>>,
    ) -> Self {
        Inst::MmaM16N8K8Tf32 {
            d_frags,
            a_frags,
            b_frags,
            c_frags,
        }
    }

    pub fn ldmatrix(frags: Vec<Operand<'a, B32>>, addr: Operand<'a, U64>) -> Self {
        Inst::LdMatrix { frags, addr }
    }

    pub fn load_shared_scalar_f32(dst: Operand<'a, F32>, addr: Operand<'a, U64>) -> Self {
        Inst::LdSharedF32 {
            dst: vec![dst],
            addr,
            vec: VecWidth::Scalar,
        }
    }

    pub fn store_shared_v4_f32(addr: Operand<'a, U64>, src: Vec<Operand<'a, F32>>) -> Self {
        Inst::StSharedF32 {
            addr,
            src,
            vec: VecWidth::V4,
        }
    }

    pub fn store_global_v4_f32(addr: Operand<'a, U64>, src: Vec<Operand<'a, F32>>) -> Self {
        Inst::StGlobalF32 {
            addr,
            src,
            vec: VecWidth::V4,
        }
    }

    pub fn wmma_store(
        addr: Operand<'a, U64>,
        frags: Vec<Operand<'a, F32>>,
        stride: Operand<'a, I32>,
    ) -> Self {
        Inst::WmmaStore {
            dtype: crate::tile::DType::TF32,
            addr,
            frags,
            stride,
        }
    }
}
