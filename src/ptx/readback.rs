//! Compile flattened virtual readback addresses, then copy with strided iteration.
use crate::tile::{IndexExpr, VirtualTensor};
use anyhow::{Context, Result, ensure};
use std::collections::HashMap;

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
    /// Split an affine numerator into divisor multiples and a bounded residual.
    /// If the residual has one quotient over the full rectangular domain, both
    /// floor division and modulo are affine. Truncating coefficient division
    /// keeps small negative coefficients (e.g. a reversed low digit) small.
    fn div_mod(&self, divisor: i64, shape: &[usize]) -> Option<(Self, Self)> {
        if divisor <= 0 {
            return None;
        }
        let d = i128::from(divisor);
        let quotient_strides: Vec<i64> = self.strides.iter().map(|&v| v / divisor).collect();
        let remainder_strides: Vec<i64> = self.strides.iter().map(|&v| v % divisor).collect();
        let remainder_offset = i128::from(self.offset % divisor);
        let (mut lo, mut hi) = (remainder_offset, remainder_offset);
        for (&stride, &extent) in remainder_strides.iter().zip(shape) {
            let span =
                i128::from(stride).checked_mul(i128::try_from(extent.checked_sub(1)?).ok()?)?;
            lo = lo.checked_add(span.min(0))?;
            hi = hi.checked_add(span.max(0))?;
        }
        let carry = lo.div_euclid(d);
        if carry != hi.div_euclid(d) {
            return None;
        }
        let quotient = Self {
            offset: i64::try_from(i128::from(self.offset / divisor).checked_add(carry)?).ok()?,
            strides: quotient_strides,
        };
        let remainder = Self {
            offset: i64::try_from(remainder_offset.checked_sub(carry.checked_mul(d)?)?).ok()?,
            strides: remainder_strides,
        };
        quotient.bounds(shape)?;
        remainder.bounds(shape)?;
        Some((quotient, remainder))
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
                let value = Self::compile(value, shape)?;
                if let Some((quotient, _)) = value.div_mod(*divisor, shape) {
                    quotient
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
                let value = Self::compile(value, shape)?;
                if let Some((_, remainder)) = value.div_mod(*modulus, shape) {
                    remainder
                } else {
                    let (lo, hi) = value.bounds(shape)?;
                    let quotient = lo.div_euclid(*modulus);
                    if quotient != hi.div_euclid(*modulus) {
                        return None;
                    }
                    let offset = i128::from(value.offset)
                        .checked_sub(i128::from(quotient).checked_mul(i128::from(*modulus))?)?;
                    Self {
                        offset: i64::try_from(offset).ok()?,
                        strides: value.strides,
                    }
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

/// Bounds of every original checked-i64 subexpression must be valid before
/// quotient/remainder cancellation can replace the coordinate evaluator.
struct CheckedDomain<'a> {
    shape: &'a [usize],
    bounds: HashMap<IndexExpr, Option<(i64, i64)>>,
}
impl CheckedDomain<'_> {
    fn bounds(&mut self, expression: &IndexExpr) -> Option<(i64, i64)> {
        if let Some(bounds) = self.bounds.get(expression) {
            return *bounds;
        }
        // Affine compilation can prove tighter correlated ranges, e.g.
        // `(100*i + 1) % 100` is constant even though its operand spans rows.
        if let Some(affine) = Affine::compile(expression, self.shape) {
            let bounds = affine.bounds(self.shape);
            self.bounds.insert(expression.clone(), bounds);
            return bounds;
        }
        use IndexExpr::*;
        let result = match expression {
            Const(value) => Some((*value, *value)),
            IterDim(axis) => self
                .shape
                .get(*axis)
                .and_then(|&extent| Some((0, i64::try_from(extent.checked_sub(1)?).ok()?))),
            Symbol(_) => None,
            Add(a, b) | Sub(a, b) | Mul(a, b) => {
                let (alo, ahi) = self.bounds(a)?;
                let (blo, bhi) = self.bounds(b)?;
                let (alo, ahi, blo, bhi) = (
                    i128::from(alo),
                    i128::from(ahi),
                    i128::from(blo),
                    i128::from(bhi),
                );
                let (lo, hi) = match expression {
                    Add(..) => (alo.checked_add(blo)?, ahi.checked_add(bhi)?),
                    Sub(..) => (alo.checked_sub(bhi)?, ahi.checked_sub(blo)?),
                    _ => {
                        let products = [
                            alo.checked_mul(blo)?,
                            alo.checked_mul(bhi)?,
                            ahi.checked_mul(blo)?,
                            ahi.checked_mul(bhi)?,
                        ];
                        (*products.iter().min()?, *products.iter().max()?)
                    }
                };
                Some((i64::try_from(lo).ok()?, i64::try_from(hi).ok()?))
            }
            FloorDiv(value, divisor) if *divisor > 0 => {
                let (lo, hi) = self.bounds(value)?;
                Some((lo.div_euclid(*divisor), hi.div_euclid(*divisor)))
            }
            Mod(value, divisor) if *divisor > 0 => {
                let (lo, hi) = self.bounds(value)?;
                if lo.div_euclid(*divisor) == hi.div_euclid(*divisor) {
                    Some((lo.rem_euclid(*divisor), hi.rem_euclid(*divisor)))
                } else {
                    Some((0, divisor - 1))
                }
            }
            _ => None,
        };
        self.bounds.insert(expression.clone(), result);
        result
    }
}

/// Exact symbolic linear combination of affine terms and quotient atoms.
/// Coefficients are i128 only during compilation; accepted copies still use
/// checked i64 offsets. This avoids introducing runtime overflow from expanding
/// `x % d = x - d * floor(x / d)` for negative values near i64::MIN.
struct LinearAddress {
    offset: i128,
    strides: Vec<i128>,
    atoms: HashMap<IndexExpr, i128>,
}
impl LinearAddress {
    fn constant(value: i128, rank: usize) -> Self {
        Self {
            offset: value,
            strides: vec![0; rank],
            atoms: HashMap::new(),
        }
    }
    fn scale(mut self, factor: i128) -> Option<Self> {
        self.offset = self.offset.checked_mul(factor)?;
        for stride in &mut self.strides {
            *stride = stride.checked_mul(factor)?;
        }
        for coefficient in self.atoms.values_mut() {
            *coefficient = coefficient.checked_mul(factor)?;
        }
        Some(self)
    }
    fn add(mut self, other: Self) -> Option<Self> {
        self.offset = self.offset.checked_add(other.offset)?;
        for (a, b) in self.strides.iter_mut().zip(other.strides) {
            *a = a.checked_add(b)?;
        }
        for (atom, coefficient) in other.atoms {
            let value = self.atoms.entry(atom).or_default();
            *value = value.checked_add(coefficient)?;
        }
        self.atoms.retain(|_, coefficient| *coefficient != 0);
        Some(self)
    }
    fn quotient(value: &IndexExpr, divisor: i64, domain: &mut CheckedDomain<'_>) -> Option<Self> {
        if divisor <= 0 {
            return None;
        }
        if let Some(affine) = Affine::compile(
            &IndexExpr::FloorDiv(Box::new(value.clone()), divisor),
            domain.shape,
        ) {
            return Some(Self {
                offset: i128::from(affine.offset),
                strides: affine.strides.into_iter().map(i128::from).collect(),
                atoms: HashMap::new(),
            });
        }
        let (lo, hi) = domain.bounds(value)?;
        let quotient = lo.div_euclid(divisor);
        if quotient == hi.div_euclid(divisor) {
            return Some(Self::constant(i128::from(quotient), domain.shape.len()));
        }
        if let IndexExpr::FloorDiv(inner, prior) = value
            && *prior > 0
            && let Some(product) = prior.checked_mul(divisor)
        {
            return Self::quotient(inner, product, domain);
        }
        // A composed reshape can reconstruct a flattened index before decoding
        // it again. Canonicalize that affine argument so quotient atoms cancel
        // even when the original coordinates came from multiple reshapes.
        let value = match Self::compile(value, domain).and_then(Self::affine) {
            Some(affine) => {
                if let Some((quotient, _)) = affine.div_mod(divisor, domain.shape) {
                    return Some(Self {
                        offset: i128::from(quotient.offset),
                        strides: quotient.strides.into_iter().map(i128::from).collect(),
                        atoms: HashMap::new(),
                    });
                }
                let (lo, hi) = affine.bounds(domain.shape)?;
                let quotient = lo.div_euclid(divisor);
                if quotient == hi.div_euclid(divisor) {
                    return Some(Self::constant(i128::from(quotient), domain.shape.len()));
                }
                let mut terms = (affine.offset != 0)
                    .then_some(IndexExpr::Const(affine.offset))
                    .into_iter()
                    .collect::<Vec<_>>();
                for (axis, stride) in affine.strides.into_iter().enumerate() {
                    if stride == 0 {
                        continue;
                    }
                    let coordinate = IndexExpr::IterDim(axis);
                    terms.push(if stride == 1 {
                        coordinate
                    } else {
                        IndexExpr::Mul(Box::new(coordinate), Box::new(IndexExpr::Const(stride)))
                    });
                }
                let mut terms = terms.into_iter();
                let first = terms.next().unwrap_or(IndexExpr::Const(0));
                terms.fold(first, |a, b| IndexExpr::Add(Box::new(a), Box::new(b)))
            }
            None => value.clone(),
        };
        let mut result = Self::constant(0, domain.shape.len());
        result
            .atoms
            .insert(IndexExpr::FloorDiv(Box::new(value), divisor), 1);
        Some(result)
    }
    fn compile(expression: &IndexExpr, domain: &mut CheckedDomain<'_>) -> Option<Self> {
        // This gate applies to the original expression, before any cancellation.
        domain.bounds(expression)?;
        if let Some(affine) = Affine::compile(expression, domain.shape) {
            return Some(Self {
                offset: i128::from(affine.offset),
                strides: affine.strides.into_iter().map(i128::from).collect(),
                atoms: HashMap::new(),
            });
        }
        use IndexExpr::*;
        match expression {
            Add(a, b) => Self::compile(a, domain)?.add(Self::compile(b, domain)?),
            Sub(a, b) => Self::compile(a, domain)?.add(Self::compile(b, domain)?.scale(-1)?),
            Mul(a, b) => {
                let (lo, hi) = domain.bounds(a)?;
                if lo == hi {
                    return Self::compile(b, domain)?.scale(i128::from(lo));
                }
                let (lo, hi) = domain.bounds(b)?;
                if lo == hi {
                    return Self::compile(a, domain)?.scale(i128::from(lo));
                }
                None
            }
            FloorDiv(value, divisor) => Self::quotient(value, *divisor, domain),
            Mod(value, divisor) if *divisor > 0 => Self::compile(value, domain)?
                .add(Self::quotient(value, *divisor, domain)?.scale(-i128::from(*divisor))?),
            _ => None,
        }
    }
    fn affine(self) -> Option<Affine> {
        if !self.atoms.is_empty() {
            return None;
        }
        Some(Affine {
            offset: i64::try_from(self.offset).ok()?,
            strides: self
                .strides
                .into_iter()
                .map(|v| i64::try_from(v).ok())
                .collect::<Option<_>>()?,
        })
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
        let mut domain = CheckedDomain {
            shape: &view.shape,
            bounds: HashMap::new(),
        };
        let mut linear = LinearAddress::constant(0, view.shape.len());
        let mut source_stride = 1i64;
        for (expr, &extent) in view.access.results.iter().zip(source).rev() {
            let Some((lo, hi)) = domain.bounds(expr) else {
                return Ok(None);
            };
            if lo < 0 || !usize::try_from(hi).is_ok_and(|hi| hi < extent) {
                // Nonlinear intervals can be conservative. Defer uncertain
                // bounds to the exact evaluator rather than rejecting a valid
                // map whose reachable coordinates occupy a smaller subset.
                return Ok(None);
            }
            let Some(coordinate) = LinearAddress::compile(expr, &mut domain) else {
                return Ok(None);
            };
            let Some(term) = coordinate.scale(i128::from(source_stride)) else {
                return Ok(None);
            };
            let Some(sum) = linear.add(term) else {
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
        let Some(linear) = linear.affine() else {
            return Ok(None);
        };
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
    fn copy_positive_matrix<T: Copy>(&self, storage: &[T], len: usize) -> Result<Option<Vec<T>>> {
        if self.block != 1
            || self.extents.len() != 2
            || self.strides.len() != 2
            || self.offset < 0
            || self.strides.iter().any(|&stride| stride < 0)
        {
            return Ok(None);
        }
        let (rows, cols) = (self.extents[0], self.extents[1]);
        ensure!(rows > 0 && cols > 0, "readback matrix has an empty extent");
        ensure!(
            rows.checked_mul(cols) == Some(len),
            "readback matrix length disagrees with extents"
        );
        let offset =
            usize::try_from(self.offset).context("readback matrix offset exceeds usize")?;
        let row_stride =
            usize::try_from(self.strides[0]).context("readback row stride exceeds usize")?;
        let col_stride =
            usize::try_from(self.strides[1]).context("readback column stride exceeds usize")?;
        let last = (rows - 1)
            .checked_mul(row_stride)
            .and_then(|row| {
                (cols - 1)
                    .checked_mul(col_stride)
                    .and_then(|col| row.checked_add(col))
            })
            .and_then(|span| offset.checked_add(span))
            .context("readback matrix address overflow")?;
        ensure!(
            last < storage.len(),
            "readback matrix storage out of bounds"
        );
        let mut output = Vec::new();
        output
            .try_reserve_exact(len)
            .context("allocating readback matrix output")?;
        // Every coordinate is nonnegative and bounded by its final row/column.
        // The checked maximum above proves each multiplication/addition below
        // fits usize and every safe storage index is in bounds. Extending an
        // exact-size row avoids per-element slice copies and odometer updates.
        for row in 0..rows {
            let base = offset + row * row_stride;
            output.extend((0..cols).map(|col| storage[base + col * col_stride]));
        }
        Ok(Some(output))
    }

    fn copy<T: Copy>(&self, storage: Vec<T>, len: usize) -> Result<Vec<T>> {
        if self.offset == 0 && self.block == len && storage.len() == len {
            return Ok(storage);
        }
        if let Some(output) = self.copy_positive_matrix(&storage, len)? {
            return Ok(output);
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
    // Common physical outputs need neither address proof nor copying. Require
    // the full map rank: is_identity alone also accepts a vacuous/prefix map.
    if view.shape == source_shape
        && view.access.results.len() == view.shape.len()
        && view.access.is_identity()
    {
        return Ok(storage);
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
