#[derive(Clone, Copy, Debug)]
pub struct F32;
#[derive(Clone, Copy, Debug)]
pub struct F16;
#[derive(Clone, Copy, Debug)]
pub struct BF16;
#[derive(Clone, Copy, Debug)]
pub struct TF32;
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct FP8_E4M3;
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct FP8_E5M2;
#[derive(Clone, Copy, Debug)]
pub struct I32;
#[derive(Clone, Copy, Debug)]
pub struct I64;
#[derive(Clone, Copy, Debug)]
pub struct U32;
#[derive(Clone, Copy, Debug)]
pub struct U64;
#[derive(Clone, Copy, Debug)]
pub struct Pred;
#[derive(Clone, Copy, Debug)]
pub struct B32;
#[derive(Clone, Copy, Debug)]
pub struct B64;

#[derive(Clone, Copy, Debug)]
pub enum Type {
    F32,
    F16,
    BF16,
    TF32,
    #[allow(non_camel_case_types)]
    FP8_E4M3,
    #[allow(non_camel_case_types)]
    FP8_E5M2,
    I32,
    I64,
    U32,
    U64,
    Bitmask,
    Pred,
}

pub trait PtxType: 'static {
    fn as_type() -> Type;
}

impl PtxType for F32 {
    fn as_type() -> Type {
        Type::F32
    }
}

impl PtxType for F16 {
    fn as_type() -> Type {
        Type::F16
    }
}

impl PtxType for BF16 {
    fn as_type() -> Type {
        Type::BF16
    }
}

impl PtxType for TF32 {
    fn as_type() -> Type {
        Type::TF32
    }
}

impl PtxType for FP8_E4M3 {
    fn as_type() -> Type {
        Type::FP8_E4M3
    }
}

impl PtxType for FP8_E5M2 {
    fn as_type() -> Type {
        Type::FP8_E5M2
    }
}

impl PtxType for I32 {
    fn as_type() -> Type {
        Type::I32
    }
}

impl PtxType for I64 {
    fn as_type() -> Type {
        Type::I64
    }
}

impl PtxType for U32 {
    fn as_type() -> Type {
        Type::U32
    }
}

impl PtxType for U64 {
    fn as_type() -> Type {
        Type::U64
    }
}

impl PtxType for Pred {
    fn as_type() -> Type {
        Type::Pred
    }
}

impl PtxType for B32 {
    fn as_type() -> Type {
        Type::Bitmask
    }
}

impl PtxType for B64 {
    fn as_type() -> Type {
        Type::Bitmask
    }
}

pub trait ArithmeticType: PtxType {}
pub trait FloatType: ArithmeticType {}

impl ArithmeticType for F32 {}
impl ArithmeticType for F16 {}
impl ArithmeticType for BF16 {}
impl ArithmeticType for I32 {}
impl ArithmeticType for I64 {}
impl ArithmeticType for U32 {}
impl ArithmeticType for U64 {}

impl FloatType for F32 {}
impl FloatType for F16 {}
impl FloatType for BF16 {}

/// PTX scalar marker and its corresponding host storage type.
pub trait CudaDType: PtxType + Clone + Copy + 'static {
    type HostType: crate::tile::TileDType
        + num_traits::Float
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default
        + Send
        + Sync
        + 'static;
}

impl CudaDType for F32 {
    type HostType = f32;
}

impl CudaDType for F16 {
    type HostType = half::f16;
}

impl CudaDType for BF16 {
    type HostType = half::bf16;
}
