//! Exact physical input descriptors for the disconnected training leaf.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::ptx::mixed) enum InputLayout {
    DenseHeadMajor,
    SequenceMajor { heads: usize },
}

impl InputLayout {
    pub(super) fn validate(self, geometry: Geometry) -> Result<()> {
        if let Self::SequenceMajor { heads } = self {
            ensure!(
                heads > 0
                    && geometry.batch_heads.is_multiple_of(heads)
                    && geometry.query_tile == 32,
                "sequence-major requires q32 and heads dividing the complete batch-head extent"
            );
        }
        Ok(())
    }

    /// Dense uses the original expression tree, preserving its generated PTX.
    pub(super) fn index(self, head: Expr, sequence: Expr, feature: Expr) -> Expr {
        match self {
            Self::DenseHeadMajor => head * (SEQUENCE * WIDTH) + sequence * WIDTH + feature,
            Self::SequenceMajor { heads } => {
                ((div(head.clone(), heads) * SEQUENCE + sequence) * heads + rem(head, heads))
                    * WIDTH
                    + feature
            }
        }
    }
}

fn equivalent(actual: &VirtualTensor, expected: &VirtualTensor) -> Option<bool> {
    Some(
        actual.source == expected.source
            && actual.shape == expected.shape
            && actual.dtype == expected.dtype
            && actual.predicate.is_none()
            && actual.out_of_bounds.is_none()
            && actual
                .access
                .clone()
                .normalize_in_domain(&actual.shape)
                .ok()?
                == expected
                    .access
                    .clone()
                    .normalize_in_domain(&expected.shape)
                    .ok()?,
    )
}

/// Proof over the entire normalized map, not sampled coordinates. This does not
/// by itself prove output/input disjointness or allocation lifetime at a rewrite.
pub(super) fn prove(
    value: &Value,
    batch: usize,
    heads: usize,
    transposed_key: bool,
) -> Option<InputLayout> {
    let bh = batch.checked_mul(heads)?;
    let width = heads.checked_mul(WIDTH)?;
    let len = bh.checked_mul(SEQUENCE)?.checked_mul(WIDTH)?;
    if batch == 0
        || heads == 0
        || value.binding.dtype != DType::BF16
        || value.view.dtype != DType::BF16
        || value.binding.len != len
        || value.binding.pointer == 0
        || !value.binding.pointer.is_multiple_of(2)
        || value
            .source_shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))?
            != len
        || value.view.predicate.is_some()
        || value.view.out_of_bounds.is_some()
    {
        return None;
    }
    let bytes = u64::try_from(len.checked_mul(2)?).ok()?;
    value.binding.pointer.checked_add(bytes)?;
    let logical = vec![batch, heads, SEQUENCE, WIDTH];
    let actual = if transposed_key {
        if value.view.shape != [batch, heads, WIDTH, SEQUENCE] {
            return None;
        }
        value.view.permute(&[0, 1, 3, 2]).ok()?
    } else {
        value.view.clone()
    };
    if actual.shape != logical {
        return None;
    }
    let identity =
        VirtualTensor::identity(value.view.source, value.source_shape.clone(), DType::BF16);
    let (layout, expected) = if value.source_shape == [batch, SEQUENCE, width] {
        (
            InputLayout::SequenceMajor { heads },
            identity
                .reshape(vec![batch, SEQUENCE, heads, WIDTH])
                .ok()?
                .permute(&[0, 2, 1, 3])
                .ok()?,
        )
    } else if value.source_shape == [batch, SEQUENCE, heads, WIDTH] {
        (
            InputLayout::SequenceMajor { heads },
            identity.permute(&[0, 2, 1, 3]).ok()?,
        )
    } else if value.source_shape == [batch, heads, SEQUENCE, WIDTH] {
        (InputLayout::DenseHeadMajor, identity)
    } else if value.source_shape == [bh, SEQUENCE, WIDTH] {
        (InputLayout::DenseHeadMajor, identity.reshape(logical).ok()?)
    } else {
        return None;
    };
    equivalent(&actual, &expected)?.then_some(layout)
}

pub(super) fn prove_triplet(
    query: &Value,
    key: &Value,
    value: &Value,
    batch: usize,
    heads: usize,
) -> Option<InputLayout> {
    let q = prove(query, batch, heads, false)?;
    (prove(key, batch, heads, true)? == q && prove(value, batch, heads, false)? == q).then_some(q)
}

#[cfg(test)]
mod tests;
