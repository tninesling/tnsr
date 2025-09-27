use std::cell;

use cust::launch;
use cust::memory::CopyDestination as _;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::stream::Stream;
use cust::util::SliceExt as _;
use kernels_core::Gemm;
use kernels_core::Mat;
use kernels_core::MatMut;

pub struct TiledGemm<'m, 's> {
    pub module: &'m Module,
    pub stream: &'s Stream,
}

impl Gemm for TiledGemm<'_, '_> {
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
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        let kernel_cell = cell::LazyCell::new(|| {
            self.module
                .get_function("tiled_gemm")
                .expect("Kernel \"tiled_gemm\" not found")
        });
        let kernel = &*kernel_cell;
        let stream = self.stream;

        let (_, block_size) = kernel
            .suggested_launch_configuration(0, 0.into())
            .expect("Failed to get launch configuration");
        let block_size = block_size as usize;
        let (block_size_x, block_size_y) = if block_size > m * n {
            (block_size.div_ceil(m) as u32, m as u32)
        } else {
            (1, block_size as u32)
        };
        let (grid_size_x, grid_size_y) = (
            (m as u32).div_ceil(block_size_x),
            (n as u32).div_ceil(block_size_y),
        );

        let a_gpu = a.data.as_dbuf().unwrap();
        let b_gpu = b.data.as_dbuf().unwrap();
        let mut c_gpu = unsafe { DeviceBuffer::uninitialized(m * n).expect("DeviceBuffer init") };

        unsafe {
            launch!(
                kernel<<<
                    (grid_size_x, grid_size_y),
                    (block_size_x, block_size_y),
                    0,
                    stream
                >>>(
                    a_gpu.as_device_ptr(),
                    a.len(),
                    b_gpu.as_device_ptr(),
                    b.len(),
                    c_gpu.as_device_ptr(),
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                )
            )
            .expect("Kernel launch failed");
        };

        c_gpu.copy_to(&mut c.data).unwrap();
    }
}
