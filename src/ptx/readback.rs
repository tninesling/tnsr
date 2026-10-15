//! Compile affine virtual readback maps once, then copy with strided iteration.
use crate::tile::{IndexExpr, VirtualTensor};
use anyhow::{Context, Result, ensure};

#[derive(Clone)]
struct Affine {
    offset: i64,
    strides: Vec<i64>,
}
impl Affine {
    fn constant(value: i64, rank: usize) -> Self {
        Self {
            offset: value,
            strides: vec![0; rank],
        }
    }
    fn bounds(&self, shape: &[usize]) -> Option<(i64, i64)> {
        let (mut lo, mut hi) = (self.offset, self.offset);
        for (&stride, &extent) in self.strides.iter().zip(shape) {
            let span = stride.checked_mul(i64::try_from(extent.checked_sub(1)?).ok()?)?;
            lo = lo.checked_add(span.min(0))?;
            hi = hi.checked_add(span.max(0))?;
        }
        Some((lo, hi))
    }
    fn scale(mut self, factor: i64) -> Option<Self> {
        self.offset = self.offset.checked_mul(factor)?;
        for stride in &mut self.strides {
            *stride = stride.checked_mul(factor)?;
        }
        Some(self)
    }
    fn combine(mut self, other: Self, subtract: bool) -> Option<Self> {
        let op = |a: i64, b: i64| {
            if subtract {
                a.checked_sub(b)
            } else {
                a.checked_add(b)
            }
        };
        self.offset = op(self.offset, other.offset)?;
        for (a, b) in self.strides.iter_mut().zip(other.strides) {
            *a = op(*a, b)?;
        }
        Some(self)
    }
    fn is_constant(&self) -> bool {
        self.strides.iter().all(|&x| x == 0)
    }
    fn compile(expr: &IndexExpr, shape: &[usize]) -> Option<Self> {
        use IndexExpr::*;
        let result = match expr {
            Const(value) => Self::constant(*value, shape.len()),
            IterDim(axis) => {
                let extent = *shape.get(*axis)?;
                let mut value = Self::constant(0, shape.len());
                value.strides[*axis] = if extent > 1 { 1 } else { 0 };
                value
            }
            Symbol(_) => return None,
            Add(a, b) | Sub(a, b) => Self::compile(a, shape)?
                .combine(Self::compile(b, shape)?, matches!(expr, Sub(..)))?,
            Mul(a, b) => {
                let a = Self::compile(a, shape)?;
                let b = Self::compile(b, shape)?;
                if a.is_constant() {
                    b.scale(a.offset)?
                } else if b.is_constant() {
                    a.scale(b.offset)?
                } else {
                    return None;
                }
            }
            FloorDiv(value, divisor) if *divisor > 0 => {
                let mut value = Self::compile(value, shape)?;
                if value.strides.iter().all(|x| x % divisor == 0) {
                    value.offset = value.offset.div_euclid(*divisor);
                    for stride in &mut value.strides {
                        *stride /= divisor;
                    }
                    value
                } else {
                    let (lo, hi) = value.bounds(shape)?;
                    let quotient = lo.div_euclid(*divisor);
                    if quotient != hi.div_euclid(*divisor) {
                        return None;
                    }
                    Self::constant(quotient, shape.len())
                }
            }
            Mod(value, modulus) if *modulus > 0 => {
                let mut value = Self::compile(value, shape)?;
                if value.strides.iter().all(|x| x % modulus == 0) {
                    Self::constant(value.offset.rem_euclid(*modulus), shape.len())
                } else {
                    let (lo, hi) = value.bounds(shape)?;
                    let quotient = lo.div_euclid(*modulus);
                    if quotient != hi.div_euclid(*modulus) {
                        return None;
                    }
                    value.offset = value.offset.checked_sub(quotient.checked_mul(*modulus)?)?;
                    value
                }
            }
            _ => return None,
        };
        // Every subexpression must remain representable under the original
        // checked i64 evaluator; reassociation must not conceal overflow.
        result.bounds(shape)?;
        Some(result)
    }
}

struct StridedCopy {
    offset: i64,
    strides: Vec<i64>,
    extents: Vec<usize>,
    block: usize,
}
impl StridedCopy {
    fn compile(view: &VirtualTensor, source: &[usize]) -> Result<Option<Self>> {
        ensure!(
            view.access.results.len() == source.len(),
            "virtual output source rank mismatch"
        );
        let mut linear = Affine::constant(0, view.shape.len());
        let mut source_stride = 1i64;
        for (expr, &extent) in view.access.results.iter().zip(source).rev() {
            let Some(coordinate) = Affine::compile(expr, &view.shape) else {
                return Ok(None);
            };
            let Some((lo, hi)) = coordinate.bounds(&view.shape) else {
                return Ok(None);
            };
            ensure!(
                lo >= 0 && usize::try_from(hi).is_ok_and(|hi| hi < extent),
                "virtual output coordinate is out of bounds"
            );
            let Some(term) = coordinate.scale(source_stride) else {
                return Ok(None);
            };
            let Some(sum) = linear.combine(term, false) else {
                return Ok(None);
            };
            linear = sum;
            let Some(next) = i64::try_from(extent)
                .ok()
                .and_then(|x| source_stride.checked_mul(x))
            else {
                return Ok(None);
            };
            source_stride = next;
        }
        let mut extents = Vec::new();
        let mut strides = Vec::new();
        for (&extent, &stride) in view.shape.iter().zip(&linear.strides) {
            if extent > 1 {
                extents.push(extent);
                strides.push(stride);
            }
        }
        let mut block = 1usize;
        while strides
            .last()
            .is_some_and(|&stride| usize::try_from(stride) == Ok(block))
        {
            let extent = extents.pop().context("strided extent missing")?;
            strides.pop();
            block = block
                .checked_mul(extent)
                .context("readback block size overflow")?;
        }
        Ok(Some(Self {
            offset: linear.offset,
            strides,
            extents,
            block,
        }))
    }
    fn copy<T: Copy>(&self, storage: Vec<T>, len: usize) -> Result<Vec<T>> {
        if self.offset == 0 && self.block == len && storage.len() == len {
            return Ok(storage);
        }
        let mut output = Vec::with_capacity(len);
        let mut coordinates = vec![0; self.extents.len()];
        let resets: Vec<i64> = self
            .strides
            .iter()
            .zip(&self.extents)
            .map(|(&stride, &extent)| {
                stride
                    .checked_mul(i64::try_from(extent - 1).context("readback extent exceeds i64")?)
                    .context("readback stride reset overflow")
            })
            .collect::<Result<_>>()?;
        let mut offset = self.offset;
        let blocks = len / self.block;
        for index in 0..blocks {
            let begin = usize::try_from(offset).context("negative readback offset")?;
            let end = begin
                .checked_add(self.block)
                .context("readback block overflow")?;
            output.extend_from_slice(
                storage
                    .get(begin..end)
                    .context("readback storage out of bounds")?,
            );
            if index + 1 == blocks {
                break;
            }
            for axis in (0..coordinates.len()).rev() {
                coordinates[axis] += 1;
                if coordinates[axis] < self.extents[axis] {
                    offset = offset
                        .checked_add(self.strides[axis])
                        .context("readback stride overflow")?;
                    break;
                }
                coordinates[axis] = 0;
                offset = offset
                    .checked_sub(resets[axis])
                    .context("readback reset overflow")?;
            }
        }
        Ok(output)
    }
}
fn count(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .context("readback shape overflow")
}

pub(super) fn materialize_host_view<T: Copy>(
    storage: Vec<T>,
    view: &VirtualTensor,
    source_shape: &[usize],
) -> Result<Vec<T>> {
    ensure!(
        view.predicate.is_none(),
        "Predicated virtual outputs are not yet supported at the host boundary"
    );
    ensure!(
        count(source_shape)? == storage.len(),
        "virtual output storage length disagrees with shape"
    );
    let len = count(&view.shape)?;
    if len == 0 {
        return Ok(Vec::new());
    }
    if let Some(plan) = StridedCopy::compile(view, source_shape)? {
        return plan.copy(storage, len);
    }
    reference(storage, view, source_shape)
}

fn reference<T: Copy>(
    storage: Vec<T>,
    view: &VirtualTensor,
    source_shape: &[usize],
) -> Result<Vec<T>> {
    let len = count(&view.shape)?;
    let mut output = Vec::with_capacity(len);
    for linear in 0..len {
        let mut remainder = linear;
        let mut coordinate = vec![0; view.shape.len()];
        for axis in (0..view.shape.len()).rev() {
            coordinate[axis] = remainder % view.shape[axis];
            remainder /= view.shape[axis];
        }
        let source = view.source_index(&coordinate)?;
        ensure!(
            source.len() == source_shape.len(),
            "virtual output source rank mismatch"
        );
        let mut offset = 0usize;
        for (&coordinate, &extent) in source.iter().zip(source_shape) {
            ensure!(
                coordinate < extent,
                "virtual output source coordinate exceeds extent"
            );
            offset = offset
                .checked_mul(extent)
                .and_then(|x| x.checked_add(coordinate))
                .context("readback source index overflow")?;
        }
        output.push(
            *storage
                .get(offset)
                .context("virtual output source index out of bounds")?,
        );
    }
    Ok(output)
}

#[cfg(test)]
#[path = "readback/tests.rs"]
mod tests;
