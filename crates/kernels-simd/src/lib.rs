mod avx2_gemm;
mod naive_gemm;

pub use avx2_gemm::Avx2Gemm;
pub use naive_gemm::NaiveGemm;
