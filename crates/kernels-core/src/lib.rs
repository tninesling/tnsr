/// Row-major matrices with dynamic dims.
pub struct Mat<'a> {
    pub rows: usize,
    pub cols: usize,
    /// stride = cols for contiguous row-major
    pub data: &'a [f32],
}

pub struct MatMut<'a> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a mut [f32],
}

/// GEMM trait: C = alpha * A @ B + beta * C
pub trait Gemm {
    /// A: m x k
    /// B: k x n
    /// C: m x n
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
    );
}

impl<T> Gemm for T
where
    T: 'static + Clone + Gemm + Send + Sync,
{
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
        <T as Gemm>::gemm(self, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
}
