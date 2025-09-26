use kernels_core::Gemm;
use kernels_core::Mat;
use kernels_core::MatMut;

pub struct NaiveGemm;

impl Gemm for NaiveGemm {
    fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &Mat,
        lda: usize,
        b: &Mat,
        ldb: usize,
        beta: f32,
        c: &mut MatMut,
        ldc: usize,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a.data[i * lda + p] * b.data[p * ldb + j];
                }
                c.data[i * ldc + j] = alpha * sum + beta * c.data[i * ldc + j];
            }
        }
    }
}
