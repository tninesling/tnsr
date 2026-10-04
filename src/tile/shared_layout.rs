//! Shared-memory address maps. Logical coordinates stay independent of padding
//! and bank swizzling; every producer and consumer uses this descriptor.
use anyhow::{Result, anyhow, ensure};

use super::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedLayout {
    pub row_stride: usize,
    /// XOR these row bits into the column. Zero preserves row-major storage.
    pub xor_mask: usize,
}

impl SharedLayout {
    pub const fn row_major(cols: usize) -> Self {
        Self {
            row_stride: cols,
            xor_mask: 0,
        }
    }

    pub fn padded(cols: usize, padding: usize) -> Result<Self> {
        let row_stride = cols
            .checked_add(padding)
            .ok_or_else(|| anyhow!("shared row stride overflows usize"))?;
        Ok(Self {
            row_stride,
            xor_mask: 0,
        })
    }

    /// Preserve contiguous groups of `group_width` columns. WMMA uses groups
    /// of 16 columns; scalar accesses may swizzle individual elements.
    pub fn swizzled(cols: usize, group_width: usize) -> Result<Self> {
        ensure!(
            cols.is_power_of_two() && group_width.is_power_of_two() && group_width <= cols,
            "shared swizzle requires power-of-two columns and groups within the tile"
        );
        Ok(Self {
            row_stride: cols,
            xor_mask: cols - group_width,
        })
    }

    pub fn validate(self, rows: usize, cols: usize, dtype: DType) -> Result<()> {
        ensure!(
            self.row_stride >= cols,
            "shared row stride is smaller than the tile"
        );
        ensure!(
            self.xor_mask == 0 || (cols.is_power_of_two() && self.xor_mask < cols),
            "shared XOR swizzle exceeds the column domain"
        );
        ensure!(
            rows.checked_mul(self.row_stride)
                .and_then(|size| size.checked_mul(dtype.size_bytes()))
                .and_then(|size| size.checked_add(31))
                .is_some(),
            "shared allocation size overflows usize"
        );
        Ok(())
    }

    pub fn offset(self, row: usize, col: usize) -> usize {
        row * self.row_stride + (col ^ (row & self.xor_mask))
    }

    /// Every allocation starts at a 32-byte boundary, including padded tiles.
    pub fn allocation_bytes(self, rows: usize, dtype: DType) -> usize {
        (rows * self.row_stride * dtype.size_bytes()).next_multiple_of(32)
    }

    /// Register-fragment size determines WMMA pointer and row alignment.
    pub const fn wmma_alignment(dtype: DType) -> usize {
        match dtype {
            DType::F16 | DType::F32 => 32,
            DType::TF32 | DType::BF16 => 16,
        }
    }

    pub fn supports_wmma(self, dtype: DType) -> bool {
        let alignment = Self::wmma_alignment(dtype);
        self.xor_mask & 15 == 0
            && self
                .row_stride
                .checked_mul(dtype.size_bytes())
                .is_some_and(|stride| stride.is_multiple_of(alignment))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    #[test]
    fn maps_are_bounded_and_bijective_for_element_widths_and_edge_tiles() {
        for dtype in [DType::F32, DType::TF32, DType::F16, DType::BF16] {
            for rows in [1, 3, 16, 17, 32, 33] {
                for cols in [16, 32, 64] {
                    for layout in [
                        SharedLayout::row_major(cols),
                        SharedLayout::padded(cols, 1).unwrap(),
                        SharedLayout::swizzled(cols, 1).unwrap(),
                        SharedLayout::swizzled(cols, 16).unwrap(),
                    ] {
                        layout.validate(rows, cols, dtype).unwrap();
                        let mut addresses = BTreeSet::new();
                        for row in 0..rows {
                            for col in 0..cols {
                                let offset = layout.offset(row, col);
                                assert!(
                                    (offset + 1) * dtype.size_bytes()
                                        <= layout.allocation_bytes(rows, dtype)
                                );
                                assert!(addresses.insert(offset));
                            }
                        }
                        assert_eq!(addresses.len(), rows * cols);
                        assert!(layout.allocation_bytes(rows, dtype).is_multiple_of(32));
                    }
                }
            }
        }
    }

    #[test]
    fn wmma_submatrices_keep_regular_rows_and_aligned_fragment_bases() {
        for (dtype, padding, alignment, k) in [
            (DType::TF32, 4, 16, 8),
            (DType::F16, 16, 32, 16),
            (DType::BF16, 8, 16, 16),
        ] {
            for layout in [
                SharedLayout::padded(32, padding).unwrap(),
                SharedLayout::swizzled(32, 16).unwrap(),
            ] {
                assert!(layout.supports_wmma(dtype));
                for (height, width, row_step, col_step) in [(16, k, 16, k), (k, 16, k, 16)] {
                    for base_row in (0..32).step_by(row_step) {
                        for base_col in (0..32).step_by(col_step) {
                            let base = layout.offset(base_row, base_col);
                            assert!((base * dtype.size_bytes()).is_multiple_of(alignment));
                            for row in 0..height {
                                for col in 0..width {
                                    assert_eq!(
                                        layout.offset(base_row + row, base_col + col),
                                        base + row * layout.row_stride + col
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            !SharedLayout::swizzled(32, 1)
                .unwrap()
                .supports_wmma(DType::TF32)
        );
        assert!(
            !SharedLayout::padded(32, 1)
                .unwrap()
                .supports_wmma(DType::F16)
        );
    }

    // Analytic bank model: a bank conflict is multiple distinct 32-bit words
    // in one bank. This is evidence about staging, not a hardware counter claim.
    fn bank_multiplicity(layout: SharedLayout, dtype: DType, rows: usize) -> usize {
        let mut banks: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for lane in 0..32 {
            let word = layout.offset(lane % rows, lane / rows) * dtype.size_bytes() / 4;
            banks.entry(word % 32).or_default().insert(word);
        }
        banks.values().map(BTreeSet::len).max().unwrap()
    }

    #[test]
    fn padding_and_xor_reduce_transposed_staging_bank_multiplicity() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            assert_eq!(
                bank_multiplicity(SharedLayout::swizzled(16, 1).unwrap(), dtype, 16),
                1
            );
            assert!(bank_multiplicity(SharedLayout::row_major(16), dtype, 16) > 1);
        }
        assert_eq!(
            bank_multiplicity(SharedLayout::row_major(32), DType::TF32, 32),
            32
        );
        assert_eq!(
            bank_multiplicity(SharedLayout::padded(32, 4).unwrap(), DType::TF32, 32),
            4
        );
        assert_eq!(
            bank_multiplicity(SharedLayout::swizzled(32, 16).unwrap(), DType::TF32, 32),
            16
        );
    }

    #[test]
    fn rejects_invalid_domains_strides_and_overflow() {
        assert!(SharedLayout::swizzled(31, 16).is_err());
        assert!(SharedLayout::swizzled(16, 32).is_err());
        assert!(SharedLayout::padded(usize::MAX, 1).is_err());
        assert!(
            SharedLayout::row_major(15)
                .validate(16, 16, DType::F32)
                .is_err()
        );
        assert!(
            SharedLayout {
                row_stride: 32,
                xor_mask: 32
            }
            .validate(16, 32, DType::F32)
            .is_err()
        );
        assert!(
            SharedLayout::row_major(usize::MAX)
                .validate(2, 16, DType::F32)
                .is_err()
        );
    }
}
