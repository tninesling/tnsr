//! Prove aligned physical copy groups from the entire affine access map.
//! Sampling adjacent indices is insufficient for reshapes with wrap boundaries.
use anyhow::{Context, Result, ensure};

use super::{IndexExpr, IndexMap, RegionInput};

pub(super) fn copy_orientation(input: &RegionInput, group: usize) -> Result<Option<bool>> {
    ensure!(group > 0, "matrix copy group must be positive");
    let group_i64 = i64::try_from(group).context("matrix copy group exceeds i64")?;
    ensure!(
        input.tensor.access.results.len() == input.source_shape.len(),
        "matrix access map rank mismatch"
    );
    let shape = &input.tensor.shape;
    if shape.len() < 2 || input.tensor.predicate.is_some() || input.tensor.out_of_bounds.is_some() {
        return Ok(None);
    }
    let mut linear = IndexExpr::Const(0);
    for (coordinate, &extent) in input.tensor.access.results.iter().zip(&input.source_shape) {
        linear = IndexExpr::Add(
            Box::new(IndexExpr::Mul(
                Box::new(linear),
                Box::new(IndexExpr::Const(
                    i64::try_from(extent).context("physical matrix extent exceeds i64")?,
                )),
            )),
            Box::new(coordinate.clone()),
        );
    }
    let map = IndexMap {
        results: vec![linear],
    }
    .normalize_in_domain(shape)?;
    let Some(linear) = reconstruct(&map.results[0]) else {
        return Ok(None);
    };
    let Some((constant, strides)) = affine(&linear, shape) else {
        return Ok(None);
    };
    let aligned = |axis: usize| {
        strides[axis] == 1
            && shape[axis].is_multiple_of(group)
            && constant.rem_euclid(group_i64) == 0
            && strides
                .iter()
                .enumerate()
                .all(|(i, stride)| i == axis || (*stride >= 0 && stride.rem_euclid(group_i64) == 0))
    };
    Ok(if aligned(shape.len() - 1) {
        Some(false)
    } else if aligned(shape.len() - 2) {
        Some(true)
    } else {
        None
    })
}

// Flatten physical row-major coordinates before checking strides. Real model
// buffers often retain batch axes: (x/d)*d + x%d must reconstruct x, including
// nested quotients. This is symbolic integer algebra used only for a proof;
// checked i128 coefficients reject any unrepresentable result conservatively.
fn reconstruct(expr: &IndexExpr) -> Option<IndexExpr> {
    fn collect(expr: &IndexExpr, scale: i128, terms: &mut Vec<(IndexExpr, i128)>) -> Option<()> {
        match expr {
            IndexExpr::Add(a, b) => {
                collect(a, scale, terms)?;
                collect(b, scale, terms)?;
            }
            IndexExpr::Sub(a, b) => {
                collect(a, scale, terms)?;
                collect(b, scale.checked_neg()?, terms)?;
            }
            IndexExpr::Mul(a, b) => match (&**a, &**b) {
                (IndexExpr::Const(c), value) | (value, IndexExpr::Const(c)) => {
                    collect(value, scale.checked_mul(i128::from(*c))?, terms)?
                }
                _ => return None,
            },
            IndexExpr::Mod(value, d) if *d > 0 => {
                collect(value, scale, terms)?;
                collect(
                    &IndexExpr::FloorDiv(value.clone(), *d),
                    scale.checked_mul(-i128::from(*d))?,
                    terms,
                )?;
            }
            IndexExpr::FloorDiv(value, d) if *d > 0 => {
                let value = reconstruct(value)?;
                let quotient = if let IndexExpr::FloorDiv(inner, a) = value {
                    IndexExpr::FloorDiv(inner, a.checked_mul(*d)?)
                } else if *d == 1 {
                    value
                } else {
                    IndexExpr::FloorDiv(Box::new(value), *d)
                };
                if let Some((_, coefficient)) = terms.iter_mut().find(|(term, _)| *term == quotient)
                {
                    *coefficient = coefficient.checked_add(scale)?;
                } else {
                    terms.push((quotient, scale));
                }
            }
            IndexExpr::Const(c) => {
                if let Some((_, coefficient)) = terms
                    .iter_mut()
                    .find(|(term, _)| *term == IndexExpr::Const(1))
                {
                    *coefficient = coefficient.checked_add(scale.checked_mul(i128::from(*c))?)?;
                } else {
                    terms.push((IndexExpr::Const(1), scale.checked_mul(i128::from(*c))?));
                }
            }
            IndexExpr::IterDim(_) => {
                if let Some((_, coefficient)) = terms.iter_mut().find(|(term, _)| term == expr) {
                    *coefficient = coefficient.checked_add(scale)?;
                } else {
                    terms.push((expr.clone(), scale));
                }
            }
            _ => return None,
        }
        Some(())
    }
    let mut terms = Vec::new();
    collect(expr, 1, &mut terms)?;
    let mut result = None;
    for (term, scale) in terms.into_iter().filter(|(_, scale)| *scale != 0) {
        let scale = i64::try_from(scale).ok()?;
        let term = if term == IndexExpr::Const(1) {
            IndexExpr::Const(scale)
        } else if scale == 1 {
            term
        } else {
            IndexExpr::Mul(Box::new(term), Box::new(IndexExpr::Const(scale)))
        };
        result = Some(if let Some(sum) = result {
            IndexExpr::Add(Box::new(sum), Box::new(term))
        } else {
            term
        });
    }
    Some(result.unwrap_or(IndexExpr::Const(0)))
}

fn affine(expr: &IndexExpr, shape: &[usize]) -> Option<(i64, Vec<i64>)> {
    let rank = shape.len();
    match expr {
        IndexExpr::Const(value) => Some((*value, vec![0; rank])),
        IndexExpr::IterDim(axis) => {
            let mut strides = vec![0; rank];
            *strides.get_mut(*axis)? = 1;
            Some((0, strides))
        }
        IndexExpr::Add(a, b) | IndexExpr::Sub(a, b) => {
            let (ca, sa) = affine(a, shape)?;
            let (cb, sb) = affine(b, shape)?;
            let combine = |a: i64, b: i64| {
                if matches!(expr, IndexExpr::Add(..)) {
                    a.checked_add(b)
                } else {
                    a.checked_sub(b)
                }
            };
            Some((
                combine(ca, cb)?,
                sa.into_iter()
                    .zip(sb)
                    .map(|(a, b)| combine(a, b))
                    .collect::<Option<_>>()?,
            ))
        }
        IndexExpr::Mul(a, b) => {
            let (constant, value) = match (&**a, &**b) {
                (IndexExpr::Const(c), value) | (value, IndexExpr::Const(c)) => (*c, value),
                _ => return None,
            };
            let (offset, strides) = affine(value, shape)?;
            Some((
                offset.checked_mul(constant)?,
                strides
                    .into_iter()
                    .map(|s| s.checked_mul(constant))
                    .collect::<Option<_>>()?,
            ))
        }
        IndexExpr::FloorDiv(value, divisor) | IndexExpr::Mod(value, divisor) if *divisor > 0 => {
            let (offset, strides) = affine(value, shape)?;
            let remainder = offset.rem_euclid(*divisor);
            let residuals: Vec<i64> = strides.iter().map(|s| s.rem_euclid(*divisor)).collect();
            let max =
                residuals
                    .iter()
                    .zip(shape)
                    .try_fold(remainder, |sum, (&stride, &extent)| {
                        let bound = i64::try_from(extent.checked_sub(1)?).ok()?;
                        sum.checked_add(stride.checked_mul(bound)?)
                    })?;
            if max >= *divisor {
                return None;
            }
            if matches!(expr, IndexExpr::FloorDiv(..)) {
                Some((
                    offset.div_euclid(*divisor),
                    strides
                        .into_iter()
                        .map(|s| s.div_euclid(*divisor))
                        .collect(),
                ))
            } else {
                Some((remainder, residuals))
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{DType, RegionValue, VirtualTensor};
    use petgraph::graph::NodeIndex;

    fn input(shape: Vec<usize>) -> RegionInput {
        let node = NodeIndex::new(0);
        RegionInput {
            node,
            value: RegionValue(0),
            tensor: VirtualTensor::identity(node, shape.clone(), DType::BF16),
            source_shape: shape,
        }
    }

    #[test]
    fn proves_transposed_and_interleaved_split_groups() -> Result<()> {
        let mut a = input(vec![64, 64, 256]);
        a.tensor = a.tensor.reshape(vec![4096, 256])?;
        a.tensor = a
            .tensor
            .permute(&[1, 0])?
            .reshape(vec![256, 4, 1024])?
            .permute(&[1, 0, 2])?;
        assert_eq!(copy_orientation(&a, 8)?, Some(true));
        let mut b = input(vec![64, 64, 768]);
        b.tensor = b.tensor.reshape(vec![4096, 768])?;
        b.tensor = b.tensor.reshape(vec![4, 1024, 768])?;
        assert_eq!(copy_orientation(&b, 8)?, Some(false));
        let mut w = input(vec![768, 256]);
        w.tensor = w.tensor.permute(&[1, 0])?;
        assert_eq!(copy_orientation(&w, 8)?, Some(true));
        Ok(())
    }

    #[test]
    fn rejects_partial_groups_misaligned_strides_and_reshape_wraps() -> Result<()> {
        assert_eq!(copy_orientation(&input(vec![256, 683]), 8)?, None);
        let mut a = input(vec![4096, 683]);
        a.tensor = a
            .tensor
            .permute(&[1, 0])?
            .reshape(vec![683, 4, 1024])?
            .permute(&[1, 0, 2])?;
        assert_eq!(copy_orientation(&a, 8)?, None);
        // The first adjacent pair is physically contiguous, but later logical
        // rows wrap a source row. A local stride sample would miss this.
        let mut wrapped = input(vec![24, 40]);
        wrapped.tensor = wrapped.tensor.permute(&[1, 0])?.reshape(vec![30, 32])?;
        assert_eq!(copy_orientation(&wrapped, 8)?, None);
        Ok(())
    }
    #[test]
    fn proven_groups_match_every_physical_index_after_reshape_and_permutation() -> Result<()> {
        let mut accepted = 0;
        for source_shape in [vec![8, 16], vec![16, 24], vec![3, 8, 16]] {
            let base = input(source_shape.clone());
            let total: usize = source_shape.iter().product();
            let axes: Vec<usize> = (0..source_shape.len()).rev().collect();
            for start in [base.tensor.clone(), base.tensor.permute(&axes)?] {
                let candidates = [
                    start.clone(),
                    start.reshape(vec![total / 8, 8])?,
                    start.reshape(vec![total / 16, 16])?,
                ];
                for candidate in candidates {
                    let axes: Vec<usize> = (0..candidate.shape.len()).rev().collect();
                    for view in [candidate.clone(), candidate.permute(&axes)?] {
                        let mut input = base.clone();
                        input.tensor = view;
                        let Some(transposed) = copy_orientation(&input, 8)? else {
                            continue;
                        };
                        accepted += 1;
                        let axis = input.tensor.shape.len() - if transposed { 2 } else { 1 };
                        let offset = |coordinates: &[usize]| -> Result<usize> {
                            Ok(input
                                .tensor
                                .source_index(coordinates)?
                                .iter()
                                .zip(&source_shape)
                                .fold(0, |offset, (&index, &extent)| offset * extent + index))
                        };
                        for linear in 0..total {
                            let mut remainder = linear;
                            let mut coordinate = vec![0; input.tensor.shape.len()];
                            for (c, &extent) in coordinate.iter_mut().zip(&input.tensor.shape).rev()
                            {
                                *c = remainder % extent;
                                remainder /= extent;
                            }
                            if !coordinate[axis].is_multiple_of(8) {
                                continue;
                            }
                            let base = offset(&coordinate)?;
                            assert!(base.is_multiple_of(8));
                            for lane in 0..8 {
                                coordinate[axis] += lane;
                                assert_eq!(offset(&coordinate)?, base + lane);
                                coordinate[axis] -= lane;
                            }
                        }
                    }
                }
            }
        }
        assert!(accepted > 3);
        Ok(())
    }
}
