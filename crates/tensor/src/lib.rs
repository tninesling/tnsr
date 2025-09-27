use kernels_core::Gemm;
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Arc<Vec<f32>>,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, init: f32) -> Self {
        Self {
            rows,
            cols,
            data: Arc::new(vec![init; rows * cols]),
        }
    }

    /// Naive elementwise access
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        Arc::make_mut(&mut self.data)[row * self.cols + col] = val;
    }

    /// Matrix multiply using runtime-dispatched GEMM
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.cols, other.rows);
        let mut c = Tensor::new(self.rows, other.cols, 0.0);

        // Prepare view wrappers
        let a_view = kernels_core::Mat {
            rows: self.rows,
            cols: self.cols,
            data: &self.data,
        };
        let b_view = kernels_core::Mat {
            rows: other.rows,
            cols: other.cols,
            data: &other.data,
        };
        let mut c_view = kernels_core::MatMut {
            rows: c.rows,
            cols: c.cols,
            data: Arc::make_mut(&mut c.data).as_mut_slice(),
        };

        // Dispatch GEMM via runtime
        let gemm = runtime::get_preferred_gemm();
        gemm.gemm(
            self.rows,
            other.cols,
            self.cols,
            1.0,
            &a_view,
            self.cols,
            &b_view,
            other.cols,
            0.0,
            &mut c_view,
            c.cols,
        );

        c
    }

    pub fn matmul_with_gemm(&self, other: &Tensor, gemm: &dyn Gemm) -> Tensor {
        assert_eq!(self.cols, other.rows);
        let mut c = Tensor::new(self.rows, other.cols, 0.0);

        // Prepare view wrappers
        let a_view = kernels_core::Mat {
            rows: self.rows,
            cols: self.cols,
            data: &self.data,
        };
        let b_view = kernels_core::Mat {
            rows: other.rows,
            cols: other.cols,
            data: &other.data,
        };
        let mut c_view = kernels_core::MatMut {
            rows: c.rows,
            cols: c.cols,
            data: Arc::make_mut(&mut c.data).as_mut_slice(),
        };

        gemm.gemm(
            self.rows,
            other.cols,
            self.cols,
            1.0,
            &a_view,
            self.cols,
            &b_view,
            other.cols,
            0.0,
            &mut c_view,
            c.cols,
        );

        c
    }
}
