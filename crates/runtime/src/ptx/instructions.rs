use std::marker::PhantomData;
use std::sync::LazyLock;

use super::types::*;

pub static GRID_DIM: LazyLock<Dim3> = LazyLock::new(|| Dim3 {
    x: Operand::reg("%nctaid.x"),
    y: Operand::reg("%nctaid.y"),
    z: Operand::reg("%nctaid.z"),
});

pub static BLOCK_ID: LazyLock<Dim3> = LazyLock::new(|| Dim3 {
    x: Operand::reg("%ctaid.x"),
    y: Operand::reg("%ctaid.y"),
    z: Operand::reg("%ctaid.z"),
});

pub static BLOCK_DIM: LazyLock<Dim3> = LazyLock::new(|| Dim3 {
    x: Operand::reg("%ntid.x"),
    y: Operand::reg("%ntid.y"),
    z: Operand::reg("%ntid.z"),
});

pub static THREAD_ID: LazyLock<Dim3> = LazyLock::new(|| Dim3 {
    x: Operand::reg("%tid.x"),
    y: Operand::reg("%tid.y"),
    z: Operand::reg("%tid.z"),
});

pub struct Dim3 {
    pub x: Operand<U32>,
    pub y: Operand<U32>,
    pub z: Operand<U32>,
}

#[derive(Clone, Copy, Debug)]
pub enum VecWidth {
    Scalar,
    V2,
    V4,
}

#[derive(Clone, Debug)]
pub enum Operand<T: PtxType> {
    Reg(String, PhantomData<T>),
    Pred(String),
    ImmI32(i32),
    ImmU64(u64),
    ImmF32(f32),
    Addr(String),      // Parameter address (formatted with brackets [name])
    Symbol(String),    // Symbol reference (formatted without brackets)
}

impl<T: PtxType> Operand<T> {
    pub fn reg(name: impl Into<String>) -> Self {
        Operand::Reg(name.into(), PhantomData)
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

    pub fn addr(name: impl Into<String>) -> Self {
        Operand::Addr(name.into())
    }

    pub fn symbol(name: impl Into<String>) -> Self {
        Operand::Symbol(name.into())
    }

    pub fn pred(name: impl Into<String>) -> Self {
        Operand::Pred(name.into())
    }

    pub fn ty(&self) -> Type {
        T::as_type()
    }
}

#[derive(Clone, Debug)]
pub struct ConvertInst<DstT: PtxType, SrcT: PtxType> {
    pub dst: Operand<DstT>,
    pub src: Operand<SrcT>,
    _phantom: PhantomData<(DstT, SrcT)>,
}

impl<DstT: PtxType, SrcT: PtxType> ConvertInst<DstT, SrcT> {
    pub fn new(dst: Operand<DstT>, src: Operand<SrcT>) -> Self {
        Self {
            dst,
            src,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BinaryInst<T: PtxType> {
    pub dst: Operand<T>,
    pub a: Operand<T>,
    pub b: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> BinaryInst<T> {
    pub fn new(dst: Operand<T>, a: Operand<T>, b: Operand<T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UnaryInst<T: PtxType> {
    pub dst: Operand<T>,
    pub src: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> UnaryInst<T> {
    pub fn new(dst: Operand<T>, src: Operand<T>) -> Self {
        Self {
            dst,
            src,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AddInst<T: PtxType> {
    pub dst: Operand<T>,
    pub a: Operand<T>,
    pub b: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> AddInst<T> {
    pub fn new(dst: Operand<T>, a: Operand<T>, b: Operand<T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MulInst<T: PtxType> {
    pub dst: Operand<T>,
    pub a: Operand<T>,
    pub b: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> MulInst<T> {
    pub fn new(dst: Operand<T>, a: Operand<T>, b: Operand<T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MaxInst<T: PtxType> {
    pub dst: Operand<T>,
    pub a: Operand<T>,
    pub b: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> MaxInst<T> {
    pub fn new(dst: Operand<T>, a: Operand<T>, b: Operand<T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NegInst<T: PtxType> {
    pub dst: Operand<T>,
    pub src: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> NegInst<T> {
    pub fn new(dst: Operand<T>, src: Operand<T>) -> Self {
        Self {
            dst,
            src,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Ex2Inst<T: PtxType> {
    pub dst: Operand<T>,
    pub src: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> Ex2Inst<T> {
    pub fn new(dst: Operand<T>, src: Operand<T>) -> Self {
        Self {
            dst,
            src,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Lg2Inst<T: PtxType> {
    pub dst: Operand<T>,
    pub src: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> Lg2Inst<T> {
    pub fn new(dst: Operand<T>, src: Operand<T>) -> Self {
        Self {
            dst,
            src,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubInst<T: PtxType> {
    pub dst: Operand<T>,
    pub a: Operand<T>,
    pub b: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> SubInst<T> {
    pub fn new(dst: Operand<T>, a: Operand<T>, b: Operand<T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DivInst<T: PtxType> {
    pub dst: Operand<T>,
    pub a: Operand<T>,
    pub b: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> DivInst<T> {
    pub fn new(dst: Operand<T>, a: Operand<T>, b: Operand<T>) -> Self {
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
pub struct ShiftLeftInst<DstT: PtxType, SrcT: PtxType> {
    pub dst: Operand<DstT>,
    pub a: Operand<SrcT>,
    pub b: Operand<I32>,
    _phantom: PhantomData<(DstT, SrcT)>,
}

impl<DstT: PtxType, SrcT: PtxType> ShiftLeftInst<DstT, SrcT> {
    pub fn new(dst: Operand<DstT>, a: Operand<SrcT>, b: Operand<I32>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShiftRightInst<DstT: PtxType, SrcT: PtxType> {
    pub dst: Operand<DstT>,
    pub a: Operand<SrcT>,
    pub b: Operand<I32>,
    _phantom: PhantomData<(DstT, SrcT)>,
}

impl<DstT: PtxType, SrcT: PtxType> ShiftRightInst<DstT, SrcT> {
    pub fn new(dst: Operand<DstT>, a: Operand<SrcT>, b: Operand<I32>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AndInst<T: PtxType> {
    pub dst: Operand<T>,
    pub a: Operand<T>,
    pub b: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> AndInst<T> {
    pub fn new(dst: Operand<T>, a: Operand<T>, b: Operand<T>) -> Self {
        Self {
            dst,
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SetpInst<T: PtxType> {
    pub dst: Operand<Pred>,
    pub a: Operand<T>,
    pub b: Operand<T>,
    pub op: CompareOp,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> SetpInst<T> {
    pub fn new(dst: Operand<Pred>, a: Operand<T>, b: Operand<T>, op: CompareOp) -> Self {
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
pub struct FmaInst<T: PtxType> {
    pub dst: Operand<T>,
    pub a: Operand<T>,
    pub b: Operand<T>,
    pub c: Operand<T>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> FmaInst<T> {
    pub fn new(dst: Operand<T>, a: Operand<T>, b: Operand<T>, c: Operand<T>) -> Self {
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
pub struct SelpInst<T: PtxType> {
    pub dst: Operand<T>,
    pub a: Operand<T>,
    pub b: Operand<T>,
    pub pred: Operand<Pred>,
    _phantom: PhantomData<T>,
}

impl<T: PtxType> SelpInst<T> {
    pub fn new(dst: Operand<T>, a: Operand<T>, b: Operand<T>, pred: Operand<Pred>) -> Self {
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
pub enum Inst {
    ConvertU64U32(ConvertInst<U64, U32>),
    ConvertU64I32(ConvertInst<U64, I32>),
    ConvertI32U64(ConvertInst<I32, U64>),
    ConvertToGlobal {
        dst: Operand<U64>,
        src: Operand<U64>,
    },
    MovI32 {
        dst: Operand<I32>,
        src: Operand<U32>,
    },
    MovU32 {
        dst: Operand<U32>,
        src: Operand<U32>,
    },
    MovU64 {
        dst: Operand<U64>,
        src: Operand<U64>,
    },
    MovF32 {
        dst: Operand<F32>,
        src: Operand<F32>,
    },
    MovB32 {
        dst: Operand<B32>,
        src: Operand<F32>,
    },
    LdGlobalF32 {
        dst: Vec<Operand<F32>>,
        addr: Operand<U64>,
        vec: VecWidth,
    },
    LdParamU64 {
        dst: Operand<U64>,
        addr: Operand<U64>,
    },
    StGlobalF32 {
        addr: Operand<U64>,
        src: Vec<Operand<F32>>,
        vec: VecWidth,
    },

    AddI32(AddInst<I32>),
    AddI64(AddInst<I64>),
    AddU64(AddInst<U64>),
    AddF32(AddInst<F32>),
    SubU64(SubInst<U64>),
    SubF32(SubInst<F32>),
    MulI32(MulInst<I32>),
    MulU64(MulInst<U64>),
    MulF32(MulInst<F32>),
    DivU64(DivInst<U64>),
    DivF32(DivInst<F32>),
    MaxF32(MaxInst<F32>),
    NegF32(NegInst<F32>),
    Ex2F32(Ex2Inst<F32>),
    Lg2F32(Lg2Inst<F32>),
    FmaI32(FmaInst<I32>),
    ShiftLeftB64(ShiftLeftInst<B64, U64>),
    ShiftLeftU64(ShiftLeftInst<U64, U64>),
    ShiftLeftI32(ShiftLeftInst<I32, I32>),
    ShiftRightI32(ShiftRightInst<I32, I32>),
    AndI32(AndInst<I32>),
    SetpU64(SetpInst<U64>),
    SetpI32(SetpInst<I32>),
    SetpF32(SetpInst<F32>),
    SelpF32(SelpInst<F32>),
    SelpU64(SelpInst<U64>),

    Bra {
        condition: Operand<Pred>,
        target: String,
    },
    BraUni {
        target: String,
    },
    Label(String),
    Ret,
    Trap,

    BarSync {
        barrier_id: u32,
    },

    LdSharedF32 {
        dst: Vec<Operand<F32>>,
        addr: Operand<U64>,
        vec: VecWidth,
    },
    StSharedF32 {
        addr: Operand<U64>,
        src: Vec<Operand<F32>>,
        vec: VecWidth,
    },
    LdMatrix {
        frags: Vec<Operand<B32>>,
        addr: Operand<U64>,
    },

    WmmaLoadA {
        frags: Vec<Operand<F32>>,
        addr: Operand<U64>,
        stride: Operand<I32>,
    },
    WmmaLoadB {
        frags: Vec<Operand<F32>>,
        addr: Operand<U64>,
        stride: Operand<I32>,
    },
    WmmaLoadC {
        frags: Vec<Operand<F32>>,
        addr: Operand<U64>,
        stride: Operand<I32>,
    },
    WmmaMma {
        d_frags: Vec<Operand<F32>>,
        a_frags: Vec<Operand<F32>>,
        b_frags: Vec<Operand<F32>>,
        c_frags: Vec<Operand<F32>>,
    },
    MmaM16N8K16 {
        d_frags: Vec<Operand<F32>>,
        a_frags: Vec<Operand<I32>>,
        b_frags: Vec<Operand<I32>>,
        c_frags: Vec<Operand<F32>>,
    },
    MmaM16N8K8Tf32 {
        d_frags: Vec<Operand<F32>>,
        a_frags: Vec<Operand<B32>>,
        b_frags: Vec<Operand<B32>>,
        c_frags: Vec<Operand<F32>>,
    },
    WmmaStore {
        addr: Operand<U64>,
        frags: Vec<Operand<F32>>,
        stride: Operand<I32>,
    },
}

impl Inst {
    pub fn convert_u64_u32(dst: Operand<U64>, src: Operand<U32>) -> Self {
        Inst::ConvertU64U32(ConvertInst::new(dst, src))
    }

    pub fn convert_u64_i32(dst: Operand<U64>, src: Operand<I32>) -> Self {
        Inst::ConvertU64I32(ConvertInst::new(dst, src))
    }

    pub fn convert_i32_u64(dst: Operand<I32>, src: Operand<U64>) -> Self {
        Inst::ConvertI32U64(ConvertInst::new(dst, src))
    }

    pub fn convert_to_global(dst: Operand<U64>, src: Operand<U64>) -> Self {
        Inst::ConvertToGlobal { dst, src }
    }

    pub fn mov_i32(dst: Operand<I32>, src: Operand<U32>) -> Self {
        Inst::MovI32 { dst, src }
    }

    pub fn mov_u32(dst: Operand<U32>, src: Operand<U32>) -> Self {
        Inst::MovU32 { dst, src }
    }

    pub fn mov_u64(dst: Operand<U64>, src: Operand<U64>) -> Self {
        Inst::MovU64 { dst, src }
    }

    pub fn mov_f32(dst: Operand<F32>, src: Operand<F32>) -> Self {
        Inst::MovF32 { dst, src }
    }

    pub fn mov_b32(dst: Operand<B32>, src: Operand<F32>) -> Self {
        Inst::MovB32 { dst, src }
    }

    pub fn load_global_scalar_f32(dst: Operand<F32>, addr: Operand<U64>) -> Self {
        Inst::LdGlobalF32 {
            dst: vec![dst],
            addr,
            vec: VecWidth::Scalar,
        }
    }

    pub fn load_param_u64(dst: Operand<U64>, addr: Operand<U64>) -> Self {
        Inst::LdParamU64 { dst, addr }
    }

    pub fn store_global_scalar_f32(addr: Operand<U64>, src: Operand<F32>) -> Self {
        Inst::StGlobalF32 {
            addr,
            src: vec![src],
            vec: VecWidth::Scalar,
        }
    }

    pub fn add_i32(dst: Operand<I32>, a: Operand<I32>, b: Operand<I32>) -> Self {
        Inst::AddI32(AddInst::new(dst, a, b))
    }

    pub fn add_i64(dst: Operand<I64>, a: Operand<I64>, b: Operand<I64>) -> Self {
        Inst::AddI64(AddInst::new(dst, a, b))
    }

    pub fn add_u64(dst: Operand<U64>, a: Operand<U64>, b: Operand<U64>) -> Self {
        Inst::AddU64(AddInst::new(dst, a, b))
    }

    pub fn add_f32(dst: Operand<F32>, a: Operand<F32>, b: Operand<F32>) -> Self {
        Inst::AddF32(AddInst::new(dst, a, b))
    }

    pub fn sub_u64(dst: Operand<U64>, a: Operand<U64>, b: Operand<U64>) -> Self {
        Inst::SubU64(SubInst::new(dst, a, b))
    }

    pub fn sub_f32(dst: Operand<F32>, a: Operand<F32>, b: Operand<F32>) -> Self {
        Inst::SubF32(SubInst::new(dst, a, b))
    }

    pub fn mul_i32(dst: Operand<I32>, a: Operand<I32>, b: Operand<I32>) -> Self {
        Inst::MulI32(MulInst::new(dst, a, b))
    }

    pub fn mul_u64(dst: Operand<U64>, a: Operand<U64>, b: Operand<U64>) -> Self {
        Inst::MulU64(MulInst::new(dst, a, b))
    }

    pub fn mul_f32(dst: Operand<F32>, a: Operand<F32>, b: Operand<F32>) -> Self {
        Inst::MulF32(MulInst::new(dst, a, b))
    }

    pub fn div_u64(dst: Operand<U64>, a: Operand<U64>, b: Operand<U64>) -> Self {
        Inst::DivU64(DivInst::new(dst, a, b))
    }

    pub fn div_f32(dst: Operand<F32>, a: Operand<F32>, b: Operand<F32>) -> Self {
        Inst::DivF32(DivInst::new(dst, a, b))
    }

    pub fn max_f32(dst: Operand<F32>, a: Operand<F32>, b: Operand<F32>) -> Self {
        Inst::MaxF32(MaxInst::new(dst, a, b))
    }

    pub fn fma_i32(dst: Operand<I32>, a: Operand<I32>, b: Operand<I32>, c: Operand<I32>) -> Inst {
        Inst::FmaI32(FmaInst::new(dst, a, b, c))
    }

    pub fn neg_f32(dst: Operand<F32>, src: Operand<F32>) -> Self {
        Inst::NegF32(NegInst::new(dst, src))
    }

    pub fn ex2_f32(dst: Operand<F32>, src: Operand<F32>) -> Self {
        Inst::Ex2F32(Ex2Inst::new(dst, src))
    }

    pub fn lg2_f32(dst: Operand<F32>, src: Operand<F32>) -> Self {
        Inst::Lg2F32(Lg2Inst::new(dst, src))
    }

    pub fn shift_left_b64(dst: Operand<B64>, a: Operand<U64>, b: Operand<I32>) -> Self {
        Inst::ShiftLeftB64(ShiftLeftInst::new(dst, a, b))
    }

    pub fn shift_left_u64(dst: Operand<U64>, a: Operand<U64>, b: Operand<I32>) -> Self {
        Inst::ShiftLeftU64(ShiftLeftInst::new(dst, a, b))
    }

    pub fn shift_left_i32(dst: Operand<I32>, a: Operand<I32>, b: Operand<I32>) -> Self {
        Inst::ShiftLeftI32(ShiftLeftInst::new(dst, a, b))
    }

    pub fn shift_right_i32(dst: Operand<I32>, a: Operand<I32>, b: Operand<I32>) -> Self {
        Inst::ShiftRightI32(ShiftRightInst::new(dst, a, b))
    }

    pub fn and_i32(dst: Operand<I32>, a: Operand<I32>, b: Operand<I32>) -> Self {
        Inst::AndI32(AndInst::new(dst, a, b))
    }

    pub fn setp_ge_u64(dst: Operand<Pred>, a: Operand<U64>, b: Operand<U64>) -> Self {
        Inst::SetpU64(SetpInst::new(dst, a, b, CompareOp::Ge))
    }

    pub fn setp_lt_u64(dst: Operand<Pred>, a: Operand<U64>, b: Operand<U64>) -> Self {
        Inst::SetpU64(SetpInst::new(dst, a, b, CompareOp::Lt))
    }

    pub fn setp_eq_u64(dst: Operand<Pred>, a: Operand<U64>, b: Operand<U64>) -> Self {
        Inst::SetpU64(SetpInst::new(dst, a, b, CompareOp::Eq))
    }

    pub fn setp_lt_i32(dst: Operand<Pred>, a: Operand<I32>, b: Operand<I32>) -> Self {
        Inst::SetpI32(SetpInst::new(dst, a, b, CompareOp::Lt))
    }

    pub fn setp_gt_f32(dst: Operand<Pred>, a: Operand<F32>, b: Operand<F32>) -> Self {
        Inst::SetpF32(SetpInst::new(dst, a, b, CompareOp::Gt))
    }

    pub fn setp_ne_f32(dst: Operand<Pred>, a: Operand<F32>, b: Operand<F32>) -> Self {
        Inst::SetpF32(SetpInst::new(dst, a, b, CompareOp::Ne))
    }

    pub fn setp_eq_f32(dst: Operand<Pred>, a: Operand<F32>, b: Operand<F32>) -> Self {
        Inst::SetpF32(SetpInst::new(dst, a, b, CompareOp::Eq))
    }

    pub fn selp_f32(
        dst: Operand<F32>,
        a: Operand<F32>,
        b: Operand<F32>,
        pred: Operand<Pred>,
    ) -> Self {
        Inst::SelpF32(SelpInst::new(dst, a, b, pred))
    }

    pub fn selp_u64(
        dst: Operand<U64>,
        a: Operand<U64>,
        b: Operand<U64>,
        pred: Operand<Pred>,
    ) -> Self {
        Inst::SelpU64(SelpInst::new(dst, a, b, pred))
    }

    pub fn bra(condition: Operand<Pred>, target: &str) -> Self {
        Inst::Bra {
            condition,
            target: target.to_string(),
        }
    }

    pub fn label(name: &str) -> Self {
        Inst::Label(name.to_string())
    }

    pub fn wmma_load_a(frags: Vec<Operand<F32>>, addr: Operand<U64>, stride: Operand<I32>) -> Self {
        Inst::WmmaLoadA {
            frags,
            addr,
            stride,
        }
    }

    pub fn wmma_load_b(frags: Vec<Operand<F32>>, addr: Operand<U64>, stride: Operand<I32>) -> Self {
        Inst::WmmaLoadB {
            frags,
            addr,
            stride,
        }
    }

    pub fn wmma_load_c(frags: Vec<Operand<F32>>, addr: Operand<U64>, stride: Operand<I32>) -> Self {
        Inst::WmmaLoadC {
            frags,
            addr,
            stride,
        }
    }

    pub fn wmma_mma(
        d_frags: Vec<Operand<F32>>,
        a_frags: Vec<Operand<F32>>,
        b_frags: Vec<Operand<F32>>,
        c_frags: Vec<Operand<F32>>,
    ) -> Self {
        Inst::WmmaMma {
            d_frags,
            a_frags,
            b_frags,
            c_frags,
        }
    }

    pub fn mma_m16n8k16(
        d_frags: Vec<Operand<F32>>,
        a_frags: Vec<Operand<I32>>,
        b_frags: Vec<Operand<I32>>,
        c_frags: Vec<Operand<F32>>,
    ) -> Self {
        Inst::MmaM16N8K16 {
            d_frags,
            a_frags,
            b_frags,
            c_frags,
        }
    }

    pub fn mma_m16n8k8_tf32(
        d_frags: Vec<Operand<F32>>,
        a_frags: Vec<Operand<B32>>,
        b_frags: Vec<Operand<B32>>,
        c_frags: Vec<Operand<F32>>,
    ) -> Self {
        Inst::MmaM16N8K8Tf32 {
            d_frags,
            a_frags,
            b_frags,
            c_frags,
        }
    }

    pub fn ldmatrix(frags: Vec<Operand<B32>>, addr: Operand<U64>) -> Self {
        Inst::LdMatrix { frags, addr }
    }

    pub fn load_shared_scalar_f32(dst: Operand<F32>, addr: Operand<U64>) -> Self {
        Inst::LdSharedF32 {
            dst: vec![dst],
            addr,
            vec: VecWidth::Scalar,
        }
    }

    pub fn store_shared_v4_f32(addr: Operand<U64>, src: Vec<Operand<F32>>) -> Self {
        Inst::StSharedF32 {
            addr,
            src,
            vec: VecWidth::V4,
        }
    }

    pub fn store_global_v4_f32(addr: Operand<U64>, src: Vec<Operand<F32>>) -> Self {
        Inst::StGlobalF32 {
            addr,
            src,
            vec: VecWidth::V4,
        }
    }

    pub fn wmma_store(addr: Operand<U64>, frags: Vec<Operand<F32>>, stride: Operand<I32>) -> Self {
        Inst::WmmaStore {
            addr,
            frags,
            stride,
        }
    }
}
