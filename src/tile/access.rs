use anyhow::{Context, Result};

use crate::graph::NodeIndex;
use crate::tensor::{ReduceOp, Shape};

use super::DType;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Extent {
    Static(usize),
    Symbol(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IterKind {
    Parallel,
    Reduction(ReduceOp),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IterDim {
    pub extent: Extent,
    pub kind: IterKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IterDomain {
    pub dimensions: Vec<IterDim>,
}

impl IterDomain {
    pub fn parallel(shape: &[usize]) -> Self {
        Self {
            dimensions: shape
                .iter()
                .map(|&extent| IterDim {
                    extent: Extent::Static(extent),
                    kind: IterKind::Parallel,
                })
                .collect(),
        }
    }
}

/// Structural integer expression used by virtual tensor access maps.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexExpr {
    IterDim(usize),
    Symbol(usize),
    Const(i64),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    FloorDiv(Box<Self>, i64),
    Mod(Box<Self>, i64),
}

impl IndexExpr {
    pub fn operation_count(&self) -> usize {
        match self {
            Self::IterDim(_) | Self::Symbol(_) | Self::Const(_) => 0,
            Self::Add(lhs, rhs) | Self::Sub(lhs, rhs) | Self::Mul(lhs, rhs) => {
                1 + lhs.operation_count() + rhs.operation_count()
            }
            Self::FloorDiv(value, _) | Self::Mod(value, _) => 1 + value.operation_count(),
        }
    }

    pub fn normalize(self) -> Result<Self> {
        super::index_egraph::normalize_map(self)
    }

    pub fn evaluate(&self, iteration: &[i64], symbols: &[i64]) -> Result<i64> {
        match self {
            Self::IterDim(dimension) => iteration
                .get(*dimension)
                .copied()
                .with_context(|| format!("iteration dimension {dimension} is unavailable")),
            Self::Symbol(symbol) => symbols
                .get(*symbol)
                .copied()
                .with_context(|| format!("index symbol {symbol} is unavailable")),
            Self::Const(value) => Ok(*value),
            Self::Add(lhs, rhs) => lhs
                .evaluate(iteration, symbols)?
                .checked_add(rhs.evaluate(iteration, symbols)?)
                .context("index addition overflowed"),
            Self::Sub(lhs, rhs) => lhs
                .evaluate(iteration, symbols)?
                .checked_sub(rhs.evaluate(iteration, symbols)?)
                .context("index subtraction overflowed"),
            Self::Mul(lhs, rhs) => lhs
                .evaluate(iteration, symbols)?
                .checked_mul(rhs.evaluate(iteration, symbols)?)
                .context("index multiplication overflowed"),
            Self::FloorDiv(value, divisor) => {
                anyhow::ensure!(*divisor > 0, "index floor-divisor must be positive");
                Ok(value.evaluate(iteration, symbols)?.div_euclid(*divisor))
            }
            Self::Mod(value, modulus) => {
                anyhow::ensure!(*modulus > 0, "index modulus must be positive");
                Ok(value.evaluate(iteration, symbols)?.rem_euclid(*modulus))
            }
        }
    }

    fn substitute(&self, dimensions: &[Self]) -> Result<Self> {
        let expression = match self {
            Self::IterDim(dimension) => dimensions
                .get(*dimension)
                .cloned()
                .with_context(|| format!("cannot compose iteration dimension {dimension}"))?,
            Self::Symbol(symbol) => Self::Symbol(*symbol),
            Self::Const(value) => Self::Const(*value),
            Self::Add(lhs, rhs) => Self::Add(
                Box::new(lhs.substitute(dimensions)?),
                Box::new(rhs.substitute(dimensions)?),
            ),
            Self::Sub(lhs, rhs) => Self::Sub(
                Box::new(lhs.substitute(dimensions)?),
                Box::new(rhs.substitute(dimensions)?),
            ),
            Self::Mul(lhs, rhs) => Self::Mul(
                Box::new(lhs.substitute(dimensions)?),
                Box::new(rhs.substitute(dimensions)?),
            ),
            Self::FloorDiv(value, divisor) => {
                Self::FloorDiv(Box::new(value.substitute(dimensions)?), *divisor)
            }
            Self::Mod(value, modulus) => {
                Self::Mod(Box::new(value.substitute(dimensions)?), *modulus)
            }
        };
        expression.normalize()
    }
}

/// Maps one logical output index to coordinates in another tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexMap {
    pub results: Vec<IndexExpr>,
}

impl IndexMap {
    pub fn identity(rank: usize) -> Self {
        Self {
            results: (0..rank).map(IndexExpr::IterDim).collect(),
        }
    }

    pub fn operation_count(&self) -> usize {
        self.results.iter().map(IndexExpr::operation_count).sum()
    }

    pub fn is_identity(&self) -> bool {
        self.results
            .iter()
            .enumerate()
            .all(|(dimension, result)| result == &IndexExpr::IterDim(dimension))
    }

    /// Compose `next -> current` with this `current -> source` map.
    pub fn compose(&self, next: &Self) -> Result<Self> {
        Ok(Self {
            results: self
                .results
                .iter()
                .map(|expression| expression.substitute(&next.results))
                .collect::<Result<_>>()?,
        })
    }

    /// Canonicalize accesses under the static bounds of their iteration domain.
    pub fn normalize_in_domain(&self, shape: &[usize]) -> Result<Self> {
        Ok(Self {
            results: self
                .results
                .iter()
                .cloned()
                .map(|e| super::index_egraph::normalize_map_in_domain(e, shape))
                .collect::<Result<_>>()?,
        })
    }

    pub fn evaluate(&self, iteration: &[i64], symbols: &[i64]) -> Result<Vec<i64>> {
        self.results
            .iter()
            .map(|expression| expression.evaluate(iteration, symbols))
            .collect()
    }

    pub fn reshape(input_shape: &[usize], output_shape: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            element_count(input_shape)? == element_count(output_shape)?,
            "reshape access map must preserve element count"
        );
        // Adding/removing singleton axes preserves every nontrivial coordinate.
        // Keep these maps explicit instead of flattening and decoding them again.
        let input_axes: Vec<_> = input_shape.iter().copied().filter(|&d| d != 1).collect();
        let output_axes: Vec<_> = output_shape
            .iter()
            .enumerate()
            .filter(|(_, d)| **d != 1)
            .collect();
        if input_axes
            .iter()
            .copied()
            .eq(output_axes.iter().map(|(_, d)| **d))
        {
            let mut axes = output_axes.iter();
            let results = input_shape
                .iter()
                .map(|&extent| {
                    if extent == 1 {
                        Ok(IndexExpr::Const(0))
                    } else {
                        let (dimension, _) =
                            axes.next().context("reshape coordinate is missing")?;
                        Ok(IndexExpr::IterDim(*dimension))
                    }
                })
                .collect::<Result<_>>()?;
            return Ok(Self { results });
        }
        let output_strides = row_major_strides(output_shape)?;
        let input_strides = row_major_strides(input_shape)?;
        let linear = output_strides.iter().enumerate().fold(
            IndexExpr::Const(0),
            |linear, (dimension, stride)| {
                IndexExpr::Add(
                    Box::new(linear),
                    Box::new(IndexExpr::Mul(
                        Box::new(IndexExpr::IterDim(dimension)),
                        Box::new(IndexExpr::Const(*stride)),
                    )),
                )
            },
        );
        let results = input_shape
            .iter()
            .zip(input_strides)
            .map(|(&extent, stride)| {
                let coordinate = IndexExpr::FloorDiv(Box::new(linear.clone()), stride.max(1));
                IndexExpr::Mod(
                    Box::new(coordinate),
                    usize_to_i64(extent.max(1), "shape extent")?,
                )
                .normalize()
            })
            .collect::<Result<_>>()?;
        Ok(Self { results })
    }

    pub fn permute(axes: &[usize]) -> Result<Self> {
        let mut inverse = vec![usize::MAX; axes.len()];
        for (output_dimension, &input_dimension) in axes.iter().enumerate() {
            anyhow::ensure!(
                input_dimension < axes.len(),
                "permutation axis {input_dimension} is out of bounds"
            );
            anyhow::ensure!(
                inverse[input_dimension] == usize::MAX,
                "permutation axis {input_dimension} is duplicated"
            );
            inverse[input_dimension] = output_dimension;
        }
        Ok(Self {
            results: inverse.into_iter().map(IndexExpr::IterDim).collect(),
        })
    }

    pub fn broadcast(rank: usize, axis: usize) -> Result<Self> {
        anyhow::ensure!(axis < rank, "broadcast axis {axis} is out of bounds");
        Ok(Self {
            results: (0..rank)
                .map(|dimension| {
                    if dimension == axis {
                        IndexExpr::Const(0)
                    } else {
                        IndexExpr::IterDim(dimension)
                    }
                })
                .collect(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Boolean bounds or masking predicate over an iteration domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexPredicate {
    Compare {
        op: CompareOp,
        lhs: IndexExpr,
        rhs: IndexExpr,
    },
    And(Vec<Self>),
}

impl IndexPredicate {
    pub fn evaluate(&self, iteration: &[i64], symbols: &[i64]) -> Result<bool> {
        match self {
            Self::Compare { op, lhs, rhs } => {
                let lhs = lhs.evaluate(iteration, symbols)?;
                let rhs = rhs.evaluate(iteration, symbols)?;
                Ok(match op {
                    CompareOp::Eq => lhs == rhs,
                    CompareOp::Ne => lhs != rhs,
                    CompareOp::Lt => lhs < rhs,
                    CompareOp::Le => lhs <= rhs,
                    CompareOp::Gt => lhs > rhs,
                    CompareOp::Ge => lhs >= rhs,
                })
            }
            Self::And(predicates) => {
                for predicate in predicates {
                    if !predicate.evaluate(iteration, symbols)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    fn substitute(&self, dimensions: &[IndexExpr]) -> Result<Self> {
        match self {
            Self::Compare { op, lhs, rhs } => Ok(Self::Compare {
                op: *op,
                lhs: lhs.substitute(dimensions)?,
                rhs: rhs.substitute(dimensions)?,
            }),
            Self::And(predicates) => Ok(Self::And(
                predicates
                    .iter()
                    .map(|predicate| predicate.substitute(dimensions))
                    .collect::<Result<_>>()?,
            )),
        }
    }

    fn and(lhs: Option<Self>, rhs: Option<Self>) -> Option<Self> {
        let mut predicates = Vec::new();
        for predicate in [lhs, rhs].into_iter().flatten() {
            match predicate {
                Self::And(children) => predicates.extend(children),
                predicate => predicates.push(predicate),
            }
        }
        match predicates.len() {
            0 => None,
            1 => predicates.pop(),
            _ => Some(Self::And(predicates)),
        }
    }
}

/// Hashable scalar literal used for masked virtual reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarValue {
    F16(u16),
    BF16(u16),
    F32(u32),
}

/// Immutable, normalized read view over a graph value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VirtualTensor {
    pub source: NodeIndex,
    pub shape: Shape,
    pub dtype: DType,
    pub access: IndexMap,
    pub predicate: Option<IndexPredicate>,
    pub out_of_bounds: Option<ScalarValue>,
}

impl VirtualTensor {
    pub fn identity(source: NodeIndex, shape: Shape, dtype: DType) -> Self {
        Self {
            source,
            access: IndexMap::identity(shape.len()),
            shape,
            dtype,
            predicate: None,
            out_of_bounds: None,
        }
    }

    pub fn view(
        &self,
        shape: Shape,
        access_to_current: IndexMap,
        predicate: Option<IndexPredicate>,
    ) -> Result<Self> {
        anyhow::ensure!(
            access_to_current.results.len() == self.shape.len(),
            "view map has {} results for source rank {}",
            access_to_current.results.len(),
            self.shape.len()
        );
        let source_predicate = self
            .predicate
            .as_ref()
            .map(|predicate| predicate.substitute(&access_to_current.results))
            .transpose()?;
        Ok(Self {
            source: self.source,
            shape,
            dtype: self.dtype,
            access: self.access.compose(&access_to_current)?,
            predicate: IndexPredicate::and(source_predicate, predicate),
            out_of_bounds: self.out_of_bounds,
        })
    }

    pub fn reshape(&self, shape: Shape) -> Result<Self> {
        self.view(shape.clone(), IndexMap::reshape(&self.shape, &shape)?, None)
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        anyhow::ensure!(axes.len() == self.shape.len(), "permutation rank mismatch");
        let access = IndexMap::permute(axes)?;
        let shape = axes.iter().map(|&axis| self.shape[axis]).collect();
        self.view(shape, access, None)
    }

    pub fn broadcast_axis(&self, axis: usize, target_size: usize) -> Result<Self> {
        anyhow::ensure!(axis < self.shape.len(), "broadcast axis is out of bounds");
        anyhow::ensure!(
            self.shape[axis] == 1,
            "only dimensions of extent one can be broadcast"
        );
        let mut shape = self.shape.clone();
        shape[axis] = target_size;
        self.view(shape, IndexMap::broadcast(self.shape.len(), axis)?, None)
    }

    pub fn source_index(&self, iteration: &[usize]) -> Result<Vec<usize>> {
        anyhow::ensure!(
            iteration.len() == self.shape.len(),
            "iteration rank does not match virtual tensor rank"
        );
        for (dimension, (&coordinate, &extent)) in iteration.iter().zip(&self.shape).enumerate() {
            anyhow::ensure!(
                coordinate < extent,
                "iteration coordinate {coordinate} is out of bounds for dimension {dimension} with extent {extent}"
            );
        }
        let iteration = iteration
            .iter()
            .map(|&value| usize_to_i64(value, "iteration coordinate"))
            .collect::<Result<Vec<_>>>()?;
        self.access
            .evaluate(&iteration, &[])?
            .into_iter()
            .map(|coordinate| {
                usize::try_from(coordinate).context("virtual tensor produced a negative index")
            })
            .collect()
    }
}

fn row_major_strides(shape: &[usize]) -> Result<Vec<i64>> {
    let mut stride = 1usize;
    let mut strides = vec![0i64; shape.len()];
    for (dimension, &extent) in shape.iter().enumerate().rev() {
        strides[dimension] = usize_to_i64(stride, "row-major stride")?;
        stride = stride
            .checked_mul(extent)
            .context("row-major stride overflowed usize")?;
    }
    Ok(strides)
}

fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &extent| {
        count
            .checked_mul(extent)
            .context("shape size overflowed usize")
    })
}

fn usize_to_i64(value: usize, description: &str) -> Result<i64> {
    i64::try_from(value).with_context(|| format!("{description} does not fit in i64"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    fn coordinates(shape: &[usize]) -> Vec<Vec<usize>> {
        let count: usize = shape.iter().product();
        let strides = row_major_strides(shape).unwrap();
        (0..count)
            .map(|linear| {
                shape
                    .iter()
                    .zip(&strides)
                    .map(|(&extent, &stride)| (linear / stride as usize) % extent)
                    .collect()
            })
            .collect()
    }

    fn linear_index(index: &[usize], shape: &[usize]) -> usize {
        index
            .iter()
            .zip(row_major_strides(shape).unwrap())
            .map(|(&coordinate, stride)| coordinate * stride as usize)
            .sum()
    }

    #[test]
    fn reshape_maps_every_coordinate_to_same_linear_element() {
        for (input_shape, output_shape) in [
            (vec![2, 3, 4], vec![6, 4]),
            (vec![2, 3, 4], vec![4, 3, 2]),
            (vec![24], vec![2, 2, 2, 3]),
            (vec![1, 2, 1, 3, 1], vec![2, 3]),
            (vec![2, 3], vec![1, 2, 1, 3, 1]),
            (vec![1, 1], vec![1, 1, 1]),
            (vec![2, 1, 3], vec![1, 2, 3, 1]),
        ] {
            let map = IndexMap::reshape(&input_shape, &output_shape).unwrap();
            for output_index in coordinates(&output_shape) {
                let iteration: Vec<i64> = output_index.iter().map(|&value| value as i64).collect();
                let input_index: Vec<usize> = map
                    .evaluate(&iteration, &[])
                    .unwrap()
                    .into_iter()
                    .map(|value| value as usize)
                    .collect();
                assert_eq!(
                    linear_index(&input_index, &input_shape),
                    linear_index(&output_index, &output_shape)
                );
            }
        }
    }

    #[test]
    fn composed_views_resolve_to_one_base_tensor() {
        let source = NodeIndex::new(7);
        let view = VirtualTensor::identity(source, vec![2, 3, 4], DType::F32)
            .reshape(vec![1, 4, 6])
            .unwrap()
            .permute(&[0, 2, 1])
            .unwrap()
            .broadcast_axis(0, 5)
            .unwrap();
        assert_eq!(view.source, source);
        assert_eq!(view.shape, vec![5, 6, 4]);
        for index in coordinates(&view.shape) {
            let source_index = view.source_index(&index).unwrap();
            assert_eq!(
                linear_index(&source_index, &[2, 3, 4]),
                index[2] * 6 + index[1]
            );
        }
    }

    #[test]
    fn predicates_compose_with_view_coordinates() {
        let source = VirtualTensor {
            predicate: Some(IndexPredicate::Compare {
                op: CompareOp::Lt,
                lhs: IndexExpr::IterDim(1),
                rhs: IndexExpr::Const(2),
            }),
            ..VirtualTensor::identity(NodeIndex::new(1), vec![2, 3], DType::F32)
        };
        let transposed = source.permute(&[1, 0]).unwrap();
        let predicate = transposed.predicate.unwrap();
        assert!(predicate.evaluate(&[1, 0], &[]).unwrap());
        assert!(!predicate.evaluate(&[2, 0], &[]).unwrap());
    }

    #[test]
    fn malformed_maps_return_contextual_errors() {
        assert!(IndexMap::reshape(&[2, 3], &[5]).is_err());
        assert!(IndexMap::permute(&[0, 0]).is_err());
        assert!(IndexMap::broadcast(2, 2).is_err());
        assert!(
            IndexExpr::FloorDiv(Box::new(IndexExpr::Const(1)), 0)
                .normalize()
                .is_err()
        );
    }

    #[test]
    fn zero_sized_reshape_has_a_valid_but_unreachable_map() {
        let map = IndexMap::reshape(&[2, 0, 3], &[0, 6]).unwrap();
        assert_eq!(map.results.len(), 3);
        let tensor = VirtualTensor::identity(NodeIndex::new(0), vec![2, 0, 3], DType::F32)
            .reshape(vec![0, 6])
            .unwrap();
        assert!(tensor.source_index(&[0, 0]).is_err());
    }

    #[test]
    fn randomized_permute_reshape_composition_matches_materialized_indexing() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5eed_f051);
        for _ in 0..64 {
            let rank = rng.random_range(2..=5);
            let source_shape: Vec<usize> = (0..rank).map(|_| rng.random_range(1..=4)).collect();
            let mut axes: Vec<usize> = (0..rank).collect();
            axes.shuffle(&mut rng);
            let permuted_shape: Vec<usize> = axes.iter().map(|&axis| source_shape[axis]).collect();
            let count: usize = source_shape.iter().product();
            let divisors: Vec<usize> = (1..=count)
                .filter(|candidate| count.is_multiple_of(*candidate))
                .collect();
            let divisor = divisors[rng.random_range(0..divisors.len())];
            let output_shape = vec![divisor, count / divisor];
            let view = VirtualTensor::identity(NodeIndex::new(0), source_shape.clone(), DType::F32)
                .permute(&axes)
                .unwrap()
                .reshape(output_shape.clone())
                .unwrap();

            for output_index in coordinates(&output_shape) {
                let linear = linear_index(&output_index, &output_shape);
                let mut remainder = linear;
                let mut permuted_index = vec![0; rank];
                for dimension in (0..rank).rev() {
                    permuted_index[dimension] = remainder % permuted_shape[dimension];
                    remainder /= permuted_shape[dimension];
                }
                let mut expected_source = vec![0; rank];
                for (output_dimension, &input_dimension) in axes.iter().enumerate() {
                    expected_source[input_dimension] = permuted_index[output_dimension];
                }
                assert_eq!(view.source_index(&output_index).unwrap(), expected_source);
            }
        }
    }
}
