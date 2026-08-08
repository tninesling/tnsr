//! Batched iteration over in-memory tensors.
//!
//! [`DataLoader`] slices one or more named, row-major tensors along their
//! leading (sample) dimension and yields [`Batch`]es whose contents can be
//! fed directly into [`Executor::execute`](crate::Executor::execute). It
//! supports shuffling, constant-filled placeholder tensors, and padding or
//! dropping the final partial batch.
//!
//! # Example
//!
//! ```rust
//! use tnsr::data::DataLoader;
//!
//! let images = vec![0.0f32; 100 * 784];
//! let labels = vec![0.0f32; 100 * 10];
//!
//! let mut loader = DataLoader::new(100)
//!     .with("images", &images, 784)
//!     .with("labels", &labels, 10)
//!     .batch_size(32)
//!     .shuffle(42)
//!     .pad_last();
//!
//! for batch in loader.iter().unwrap() {
//!     let inputs = batch.into_inputs();
//!     assert_eq!(inputs["images"].len(), 32 * 784);
//! }
//! ```

use std::collections::{HashMap, HashSet};

use anyhow::{Result, ensure};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// How to handle a final batch that is smaller than `batch_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LastBatch {
    /// Yield the final batch at its actual, smaller size.
    #[default]
    Partial,
    /// Pad the final batch to `batch_size` rows with `D::default()`.
    Pad,
    /// Do not yield the final partial batch at all.
    Drop,
}

/// A named tensor participating in batching.
enum TensorSource<'a, D> {
    /// Row-major tensor borrowed from the caller, with `sample_stride`
    /// elements per sample.
    Slice {
        name: String,
        data: &'a [D],
        sample_stride: usize,
    },
    /// Synthetic tensor filled with a constant value, for placeholder inputs
    /// that a graph requires but a pass does not use (e.g. labels during
    /// evaluation).
    Fill {
        name: String,
        sample_stride: usize,
        value: D,
    },
}

impl<D> TensorSource<'_, D> {
    fn name(&self) -> &str {
        match self {
            TensorSource::Slice { name, .. } | TensorSource::Fill { name, .. } => name,
        }
    }

    fn sample_stride(&self) -> usize {
        match self {
            TensorSource::Slice { sample_stride, .. }
            | TensorSource::Fill { sample_stride, .. } => *sample_stride,
        }
    }
}

/// Fluent loader that yields batches of named tensors.
///
/// Construct with [`DataLoader::new`], register tensors with
/// [`DataLoader::with`] and [`DataLoader::with_fill`], configure with
/// [`DataLoader::batch_size`], [`DataLoader::shuffle`],
/// [`DataLoader::pad_last`], and [`DataLoader::drop_last`], then iterate with
/// [`DataLoader::iter`].
///
/// Each call to [`DataLoader::iter`] starts a fresh pass over the dataset.
/// When shuffling is enabled, every pass draws a new permutation, so one
/// loader can be reused across training epochs.
pub struct DataLoader<'a, D> {
    len: usize,
    tensors: Vec<TensorSource<'a, D>>,
    batch_size: usize,
    shuffle: Option<StdRng>,
    last_batch: LastBatch,
}

impl<'a, D> DataLoader<'a, D> {
    /// Create a loader over a dataset of `len` samples.
    ///
    /// Every tensor registered with [`DataLoader::with`] must hold exactly
    /// `len` samples.
    pub fn new(len: usize) -> Self {
        Self {
            len,
            tensors: Vec::new(),
            batch_size: 1,
            shuffle: None,
            last_batch: LastBatch::Partial,
        }
    }

    /// Register a row-major tensor with `sample_stride` elements per sample.
    ///
    /// `data.len()` must equal `len * sample_stride`; this is checked when
    /// iteration starts.
    pub fn with(mut self, name: impl Into<String>, data: &'a [D], sample_stride: usize) -> Self {
        self.tensors.push(TensorSource::Slice {
            name: name.into(),
            data,
            sample_stride,
        });
        self
    }

    /// Register a synthetic tensor filled with `value`.
    ///
    /// Useful for inputs a graph requires but a pass does not use, such as
    /// labels during evaluation.
    pub fn with_fill(mut self, name: impl Into<String>, sample_stride: usize, value: D) -> Self {
        self.tensors.push(TensorSource::Fill {
            name: name.into(),
            sample_stride,
            value,
        });
        self
    }

    /// Set the number of samples per batch (default 1).
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Shuffle the sample order, seeded for reproducibility.
    ///
    /// A fresh permutation is drawn on every call to [`DataLoader::iter`].
    pub fn shuffle(mut self, seed: u64) -> Self {
        self.shuffle = Some(StdRng::seed_from_u64(seed));
        self
    }

    /// Pad the final partial batch to `batch_size` rows with `D::default()`.
    ///
    /// [`Batch::len`] still reports the number of real samples. Useful when
    /// the consumer (e.g. a computation graph) requires a fixed batch
    /// dimension.
    pub fn pad_last(mut self) -> Self {
        self.last_batch = LastBatch::Pad;
        self
    }

    /// Drop the final partial batch instead of yielding it.
    pub fn drop_last(mut self) -> Self {
        self.last_batch = LastBatch::Drop;
        self
    }

    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the dataset contains no samples.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Validate the configuration and start a fresh pass over the dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size` is zero, no tensors are registered,
    /// a tensor name is duplicated, a `sample_stride` is zero, or a slice
    /// tensor does not hold exactly `len * sample_stride` elements.
    pub fn iter(&mut self) -> Result<BatchIter<'_, D>> {
        ensure!(self.batch_size > 0, "batch_size must be at least 1");
        ensure!(
            !self.tensors.is_empty(),
            "DataLoader has no tensors; register one with .with() or .with_fill()"
        );
        let mut names = HashSet::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            ensure!(
                tensor.sample_stride() > 0,
                "Tensor '{}' has sample_stride 0",
                tensor.name()
            );
            ensure!(
                names.insert(tensor.name()),
                "Duplicate tensor name '{}'",
                tensor.name()
            );
            if let TensorSource::Slice {
                name,
                data,
                sample_stride,
            } = tensor
            {
                let expected = self.len * sample_stride;
                ensure!(
                    data.len() == expected,
                    "Tensor '{}' has {} elements, expected {} ({} samples * {} per sample)",
                    name,
                    data.len(),
                    expected,
                    self.len,
                    sample_stride
                );
            }
        }

        let mut len = self.len;
        if self.last_batch == LastBatch::Drop {
            len -= len % self.batch_size;
        }
        let mut order: Vec<usize> = (0..len).collect();
        if let Some(rng) = &mut self.shuffle {
            order.shuffle(rng);
        }

        Ok(BatchIter {
            tensors: &self.tensors,
            order,
            batch_size: self.batch_size,
            pad: self.last_batch == LastBatch::Pad,
            pos: 0,
        })
    }
}

/// Iterator over the batches of one pass through a [`DataLoader`].
///
/// Created by [`DataLoader::iter`].
pub struct BatchIter<'a, D> {
    tensors: &'a [TensorSource<'a, D>],
    order: Vec<usize>,
    batch_size: usize,
    pad: bool,
    pos: usize,
}

impl<D> Iterator for BatchIter<'_, D>
where
    D: Clone + Default,
{
    type Item = Batch<D>;

    fn next(&mut self) -> Option<Batch<D>> {
        let remaining = self.order.len() - self.pos;
        if remaining == 0 {
            return None;
        }
        let n = self.batch_size.min(remaining);
        let indices = self.order[self.pos..self.pos + n].to_vec();
        self.pos += n;
        let rows = if self.pad { self.batch_size } else { n };

        let mut inputs = HashMap::with_capacity(self.tensors.len());
        for tensor in self.tensors {
            let (name, buffer) = match tensor {
                TensorSource::Slice {
                    name,
                    data,
                    sample_stride,
                } => {
                    let mut buffer = Vec::with_capacity(rows * sample_stride);
                    for &idx in &indices {
                        let start = idx * sample_stride;
                        buffer.extend_from_slice(&data[start..start + sample_stride]);
                    }
                    buffer.resize(rows * sample_stride, D::default());
                    (name.clone(), buffer)
                }
                TensorSource::Fill {
                    name,
                    sample_stride,
                    value,
                } => (name.clone(), vec![value.clone(); rows * sample_stride]),
            };
            inputs.insert(name, buffer);
        }

        Some(Batch { inputs, indices })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.order.len() - self.pos).div_ceil(self.batch_size);
        (remaining, Some(remaining))
    }
}

impl<D> ExactSizeIterator for BatchIter<'_, D>
where
    D: Clone + Default,
{
    fn len(&self) -> usize {
        (self.order.len() - self.pos).div_ceil(self.batch_size)
    }
}

/// One batch of samples gathered from a [`DataLoader`].
pub struct Batch<D> {
    inputs: HashMap<String, Vec<D>>,
    indices: Vec<usize>,
}

impl<D> Batch<D> {
    /// The batched tensors by name, each with `rows * sample_stride`
    /// elements.
    pub fn inputs(&self) -> &HashMap<String, Vec<D>> {
        &self.inputs
    }

    /// Consume the batch into the input map expected by
    /// [`Executor::execute`](crate::Executor::execute).
    pub fn into_inputs(self) -> HashMap<String, Vec<D>> {
        self.inputs
    }

    /// Borrow a batched tensor by name.
    pub fn get(&self, name: &str) -> Option<&[D]> {
        self.inputs.get(name).map(Vec::as_slice)
    }

    /// Number of real samples in this batch.
    ///
    /// Smaller than the padded row count when [`DataLoader::pad_last`] pads
    /// the final batch.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether the batch contains no samples.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Dataset indices of the samples in this batch, in batch order.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Samples of two elements each: sample i is [2i, 2i + 1].
    fn sample_data(n: usize) -> Vec<i32> {
        (0..n as i32).flat_map(|i| [2 * i, 2 * i + 1]).collect()
    }

    fn all_indices<D: Clone + Default>(loader: &mut DataLoader<'_, D>) -> Vec<usize> {
        loader
            .iter()
            .unwrap()
            .flat_map(|batch| batch.indices().to_vec())
            .collect()
    }

    #[test]
    fn sequential_batches_include_partial_last() {
        let data = sample_data(10);
        let mut loader = DataLoader::new(10).with("x", &data, 2).batch_size(4);
        let batches: Vec<Batch<i32>> = loader.iter().unwrap().collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 4);
        assert_eq!(
            batches[0].get("x").unwrap(),
            [0, 1, 2, 3, 4, 5, 6, 7].as_slice()
        );
        assert_eq!(batches[2].len(), 2);
        assert_eq!(batches[2].indices(), [8, 9].as_slice());
        assert_eq!(batches[2].get("x").unwrap(), [16, 17, 18, 19].as_slice());
    }

    #[test]
    fn pad_last_pads_to_full_batch() {
        let data = sample_data(10);
        let mut loader = DataLoader::new(10)
            .with("x", &data, 2)
            .batch_size(4)
            .pad_last();
        let batches: Vec<Batch<i32>> = loader.iter().unwrap().collect();
        assert_eq!(batches.len(), 3);
        let last = &batches[2];
        assert_eq!(last.len(), 2);
        assert_eq!(
            last.get("x").unwrap(),
            [16, 17, 18, 19, 0, 0, 0, 0].as_slice()
        );
    }

    #[test]
    fn drop_last_skips_partial_batch() {
        let data = sample_data(10);
        let mut loader = DataLoader::new(10)
            .with("x", &data, 2)
            .batch_size(4)
            .drop_last();
        let batches: Vec<Batch<i32>> = loader.iter().unwrap().collect();
        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(|b| b.len() == 4));
    }

    #[test]
    fn exact_multiple_yields_no_extra_batch() {
        let data = sample_data(8);
        let mut loader = DataLoader::new(8).with("x", &data, 2).batch_size(4);
        assert_eq!(loader.iter().unwrap().count(), 2);
        let mut padded = DataLoader::new(8)
            .with("x", &data, 2)
            .batch_size(4)
            .pad_last();
        assert_eq!(padded.iter().unwrap().count(), 2);
    }

    #[test]
    fn shuffle_is_deterministic_per_seed() {
        let data = sample_data(50);
        let mut a = DataLoader::new(50)
            .with("x", &data, 2)
            .batch_size(7)
            .shuffle(7);
        let mut b = DataLoader::new(50)
            .with("x", &data, 2)
            .batch_size(7)
            .shuffle(7);
        assert_eq!(all_indices(&mut a), all_indices(&mut b));
    }

    #[test]
    fn shuffle_covers_each_sample_once_per_pass() {
        let data = sample_data(100);
        let mut loader = DataLoader::new(100)
            .with("x", &data, 2)
            .batch_size(10)
            .shuffle(1);
        let first = all_indices(&mut loader);
        let second = all_indices(&mut loader);
        for mut pass in [first.clone(), second.clone()] {
            pass.sort_unstable();
            assert_eq!(pass, (0..100).collect::<Vec<_>>());
        }
        // Successive passes draw fresh permutations.
        assert_ne!(first, second);
    }

    #[test]
    fn shuffled_batches_gather_matching_rows() {
        let data = sample_data(10);
        let mut loader = DataLoader::new(10)
            .with("x", &data, 2)
            .batch_size(4)
            .shuffle(3);
        for batch in loader.iter().unwrap() {
            for (row, &idx) in batch.indices().iter().enumerate() {
                let expected = [2 * idx as i32, 2 * idx as i32 + 1];
                assert_eq!(&batch.get("x").unwrap()[row * 2..row * 2 + 2], expected);
            }
        }
    }

    #[test]
    fn fill_tensor_repeats_value() {
        let data = sample_data(5);
        let mut loader = DataLoader::new(5)
            .with("x", &data, 2)
            .with_fill("y", 3, 9)
            .batch_size(4)
            .pad_last();
        let batch = loader.iter().unwrap().next().unwrap();
        assert_eq!(batch.get("y").unwrap(), [9; 12].as_slice());
    }

    #[test]
    fn into_inputs_matches_executor_shape() {
        let data = sample_data(4);
        let mut loader = DataLoader::new(4).with("x", &data, 2).batch_size(4);
        let batch = loader.iter().unwrap().next().unwrap();
        let inputs = batch.into_inputs();
        assert_eq!(inputs["x"].len(), 8);
    }

    #[test]
    fn empty_dataset_yields_no_batches() {
        let data: Vec<i32> = Vec::new();
        let mut loader = DataLoader::new(0).with("x", &data, 2).batch_size(4);
        assert_eq!(loader.iter().unwrap().count(), 0);
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        let data = sample_data(10);

        let mut zero_batch = DataLoader::new(10).with("x", &data, 2).batch_size(0);
        assert!(zero_batch.iter().is_err());

        let mut no_tensors: DataLoader<'_, i32> = DataLoader::new(10);
        assert!(no_tensors.iter().is_err());

        let mut wrong_len = DataLoader::new(9).with("x", &data, 2);
        assert!(wrong_len.iter().is_err());

        let mut zero_stride = DataLoader::new(10).with("x", &data, 0);
        assert!(zero_stride.iter().is_err());

        let mut duplicate = DataLoader::new(10).with("x", &data, 2).with("x", &data, 2);
        assert!(duplicate.iter().is_err());
    }

    #[test]
    fn iterator_reports_exact_size() {
        let data = sample_data(10);
        let mut loader = DataLoader::new(10).with("x", &data, 2).batch_size(4);
        let mut iter = loader.iter().unwrap();
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.len(), 2);
    }
}
