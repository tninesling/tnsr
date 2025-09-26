#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;
use kernels_core::Gemm;
use kernels_core::Mat;
use kernels_core::MatMut;

use crate::naive_gemm::NaiveGemm;

pub struct Avx2Gemm;

impl Avx2Gemm {
    #[inline(always)]
    unsafe fn micro_kernel_4x8(
        k: usize,
        a: *const f32,
        lda: isize,
        b: *const f32,
        ldb: isize,
        c: *mut f32,
        ldc: isize,
        alpha: f32,
    ) {
        unsafe {
            // c_block is 4 x 8
            let mut c0 = _mm256_setzero_ps(); // holds 8 floats
            let mut c1 = _mm256_setzero_ps();
            let mut c2 = _mm256_setzero_ps();
            let mut c3 = _mm256_setzero_ps();

            for p in 0..k {
                // load B row (8 floats)
                let bptr = b.offset((p as isize) * ldb);
                let bvec = _mm256_loadu_ps(bptr);

                // load A scalar rows (4 scalars broadcast)
                let a0 = _mm256_set1_ps(*a.offset(0 * lda + p as isize));
                let a1 = _mm256_set1_ps(*a.offset(1 * lda + p as isize));
                let a2 = _mm256_set1_ps(*a.offset(2 * lda + p as isize));
                let a3 = _mm256_set1_ps(*a.offset(3 * lda + p as isize));

                c0 = _mm256_fmadd_ps(a0, bvec, c0);
                c1 = _mm256_fmadd_ps(a1, bvec, c1);
                c2 = _mm256_fmadd_ps(a2, bvec, c2);
                c3 = _mm256_fmadd_ps(a3, bvec, c3);
            }

            // scale by alpha and store back into c
            let alpha_v = _mm256_set1_ps(alpha);
            c0 = _mm256_mul_ps(c0, alpha_v);
            c1 = _mm256_mul_ps(c1, alpha_v);
            c2 = _mm256_mul_ps(c2, alpha_v);
            c3 = _mm256_mul_ps(c3, alpha_v);

            // store rows
            _mm256_storeu_ps(c.offset(0 * ldc), c0);
            _mm256_storeu_ps(c.offset(1 * ldc), c1);
            _mm256_storeu_ps(c.offset(2 * ldc), c2);
            _mm256_storeu_ps(c.offset(3 * ldc), c3);
        }
    }
}

impl Gemm for Avx2Gemm {
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
        // Simple blocking: iterate over i in steps of 4, j in steps of 8
        assert!(is_x86_feature_detected!("avx2"), "avx2 not available");
        unsafe {
            for i in (0..m).step_by(4) {
                for j in (0..n).step_by(8) {
                    // compute block sizes (handle tails)
                    let mr = (4).min(m - i);
                    let nr = (8).min(n - j);
                    // For simplicity, handle only full-block case here,
                    // fall back to naive for tails.
                    if mr == 4 && nr == 8 {
                        let a_ptr = a.data.as_ptr().add(i * lda) as *const f32;
                        let b_ptr = b.data.as_ptr().add(j) as *const f32;
                        let c_ptr = c.data.as_mut_ptr().add(i * ldc + j) as *mut f32;
                        Avx2Gemm::micro_kernel_4x8(
                            k,
                            a_ptr,
                            lda as isize,
                            b_ptr,
                            ldb as isize,
                            c_ptr,
                            ldc as isize,
                            alpha,
                        );
                    } else {
                        // fallback naive for edges:
                        NaiveGemm.gemm(
                            mr,
                            nr,
                            k,
                            alpha,
                            &Mat {
                                rows: mr,
                                cols: k,
                                data: &a.data[i * lda..],
                            },
                            lda,
                            &Mat {
                                rows: k,
                                cols: nr,
                                data: &b.data[j..],
                            },
                            ldb,
                            beta,
                            &mut MatMut {
                                rows: mr,
                                cols: nr,
                                data: &mut c.data[i * ldc + j..],
                            },
                            ldc,
                        );
                    }
                }
            }
        }
    }
}
