//! Layout-closed paired ABI for the explicit default-off numerical screen.
//! Leaf/component proof and complete native training qualification are separate.
#[cfg(test)]
use crate::graph::NodeIndex;
use crate::tile::{DType, VirtualTensor};
use anyhow::{Context, Result, ensure};

const S: usize = 512;
const D: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    pub batch: usize,
    pub heads: usize,
}
impl Shape {
    pub(super) fn new(batch: usize, heads: usize) -> Result<Self> {
        let bh = batch
            .checked_mul(heads)
            .context("paired leaf BH overflow")?;
        ensure!(
            batch > 0 && heads > 0 && (64..=65535).contains(&bh),
            "paired leaf requires S512/D64 and 64<=BH<=65535"
        );
        Ok(Self { batch, heads })
    }
    pub(super) fn bh(self) -> usize {
        self.batch * self.heads
    }
    pub(super) fn elements(self) -> usize {
        self.bh() * S * D
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReadLayout {
    HeadMajor,
    SequenceMajor,
}

/// A single choice for the COMPLETE pair, never independent per-operand modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Transport {
    Scalar,
    Async16,
}
impl Transport {
    pub(super) fn kernel_name(self, scalar_name: String) -> String {
        match self {
            Self::Scalar => scalar_name,
            Self::Async16 => format!("{scalar_name}_Async16"),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Descriptor {
    pub pointer: u64,
    pub storage_len: usize,
    pub storage_dtype: DType,
    pub source_shape: Vec<usize>,
    pub view: VirtualTensor,
}
fn same_map(a: &VirtualTensor, b: &VirtualTensor) -> Option<bool> {
    Some(
        a.source == b.source
            && a.dtype == b.dtype
            && a.shape == b.shape
            && a.access.clone().normalize_in_domain(&a.shape).ok()?
                == b.access.clone().normalize_in_domain(&b.shape).ok()?,
    )
}

/// Includes the explicit [B*S,H*D] full-volume sequence-major source expected
/// from a rank-two out-projection input-gradient producer. Actual B8/B16/B32
/// descriptors have been observed; runtime admission still re-proves each map.
pub(super) fn prove(shape: Shape, input: &Descriptor, transposed_key: bool) -> Option<ReadLayout> {
    Shape::new(shape.batch, shape.heads).ok()?;
    if input.storage_dtype != DType::BF16
        || input.view.dtype != DType::BF16
        || input.storage_len != shape.elements()
        || input.pointer == 0
        || !input.pointer.is_multiple_of(2)
        || input.view.predicate.is_some()
        || input.view.out_of_bounds.is_some()
        || input
            .source_shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))?
            != shape.elements()
    {
        return None;
    }
    input
        .pointer
        .checked_add(u64::try_from(input.storage_len.checked_mul(2)?).ok()?)?;
    let actual = if transposed_key {
        if input.view.shape != [shape.batch, shape.heads, D, S] {
            return None;
        }
        input.view.permute(&[0, 1, 3, 2]).ok()?
    } else {
        input.view.clone()
    };
    if actual.shape != [shape.batch, shape.heads, S, D] {
        return None;
    }
    let base = VirtualTensor::identity(input.view.source, input.source_shape.clone(), DType::BF16);
    let seq3 = vec![shape.batch, S, shape.heads * D];
    let seq2 = vec![shape.batch * S, shape.heads * D];
    let seq4 = vec![shape.batch, S, shape.heads, D];
    let dense4 = vec![shape.batch, shape.heads, S, D];
    let dense3 = vec![shape.bh(), S, D];
    let dense2 = vec![shape.bh() * S, D];
    let (layout, expected) = if [seq2, seq3, seq4].contains(&input.source_shape) {
        (
            ReadLayout::SequenceMajor,
            base.reshape(vec![shape.batch, S, shape.heads, D])
                .ok()?
                .permute(&[0, 2, 1, 3])
                .ok()?,
        )
    } else if [dense2, dense3, dense4.clone()].contains(&input.source_shape) {
        (ReadLayout::HeadMajor, base.reshape(dense4).ok()?)
    } else {
        return None;
    };
    if same_map(&actual, &expected)? {
        return Some(layout);
    }
    // The actual rank-two out-projection VJP retains both full-volume reshape
    // steps. Compare against that exact canonical construction as well: no
    // address sampling, arbitrary-map admission, offset or broadcast is added.
    if layout == ReadLayout::SequenceMajor {
        let two_step = base
            .reshape(vec![shape.batch, S, shape.heads * D])
            .ok()?
            .reshape(vec![shape.batch, S, shape.heads, D])
            .ok()?
            .permute(&[0, 2, 1, 3])
            .ok()?;
        return same_map(&actual, &two_step)?.then_some(layout);
    }
    None
}

pub(super) struct UncompiledPair {
    pub shape: Shape,
    pub read_layouts: [ReadLayout; 4],
    pub output_shapes: [Vec<usize>; 4],
    pub statistics_shape: Vec<usize>,
    pub global_payload_bytes: usize,
    pub transport: Transport,
}
impl UncompiledPair {
    pub(super) fn new(shape: Shape, inputs: [Descriptor; 4]) -> Result<Self> {
        Self::with_transport(shape, inputs, Transport::Scalar)
    }

    pub(super) fn with_transport(
        shape: Shape,
        inputs: [Descriptor; 4],
        requested: Transport,
    ) -> Result<Self> {
        let mut modes = [ReadLayout::HeadMajor; 4];
        for (i, input) in inputs.iter().enumerate() {
            modes[i] = prove(shape, input, i == 1).with_context(|| {
                format!("paired leaf operand{i} has an unsupported full-volume BF16 layout")
            })?;
        }
        ensure!(
            modes[0] == modes[1] && modes[1] == modes[2],
            "initial paired leaf requires one proven Q/K/V storage family"
        );
        let ordinary = vec![shape.batch, shape.heads, S, D];
        let transposed_key = vec![shape.batch, shape.heads, D, S];
        let statistics_shape = vec![shape.bh(), S, 3]; // m,l from forward; delta from dQ.
        // Strict canonical full-volume proof above excludes offsets and partial
        // storage. Every source stride is a multiple of64 BF16 elements. A
        // single unaligned pointer selects the WHOLE original scalar pair.
        let transport = if requested == Transport::Async16
            && inputs.iter().all(|input| input.pointer.is_multiple_of(16))
        {
            Transport::Async16
        } else {
            Transport::Scalar
        };
        let global_payload_bytes = shape
            .elements()
            .checked_mul(8)
            .and_then(|n| shape.bh().checked_mul(S)?.checked_mul(12)?.checked_add(n))
            .context("paired leaf output/workspace byte extent overflow")?;
        Ok(Self {
            shape,
            read_layouts: modes,
            output_shapes: [ordinary.clone(), ordinary.clone(), transposed_key, ordinary],
            statistics_shape,
            global_payload_bytes,
            transport,
        })
    }
}

/// There is deliberately no forward-only runnable state in this scaffold.
/// All three compiled functions must be present before resource admission and
/// whole-pair capture/execution. Runtime integration remains a separate step.
pub(super) struct PairedFunctions<T> {
    pub forward: T,
    pub d_query_and_delta: T,
    pub d_key_value: T,
}
impl<T> PairedFunctions<T> {
    pub(super) fn from_all(forward: Option<T>, dq: Option<T>, dkv: Option<T>) -> Result<Self> {
        Ok(Self {
            forward: forward.context("paired forward missing")?,
            d_query_and_delta: dq.context("paired dQ/delta missing")?,
            d_key_value: dkv.context("paired dKT/dV missing")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn descriptor(shape: Shape, sequence: bool, key: bool, rank2: bool) -> Descriptor {
        let source_shape = if sequence {
            if rank2 {
                vec![shape.batch * S, shape.heads * D]
            } else {
                vec![shape.batch, S, shape.heads * D]
            }
        } else {
            vec![shape.bh(), S, D]
        };
        let mut view =
            VirtualTensor::identity(NodeIndex::new(1), source_shape.clone(), DType::BF16);
        if sequence {
            view = view
                .reshape(vec![shape.batch, S, shape.heads, D])
                .unwrap()
                .permute(&[0, 2, 1, 3])
                .unwrap();
        } else {
            view = view.reshape(vec![shape.batch, shape.heads, S, D]).unwrap();
        }
        if key {
            view = view.permute(&[0, 1, 3, 2]).unwrap();
        }
        Descriptor {
            pointer: 0x1000,
            storage_len: shape.elements(),
            storage_dtype: DType::BF16,
            source_shape,
            view,
        }
    }
    #[test]
    fn online_training_rank_two_incoming_accepts_only_exact_two_step_canonical_map() -> Result<()> {
        for batch in [8, 16, 32] {
            let shape = Shape::new(batch, 8)?;
            let mut input = descriptor(shape, true, false, true);
            input.view =
                VirtualTensor::identity(input.view.source, input.source_shape.clone(), DType::BF16)
                    .reshape(vec![batch, S, shape.heads * D])?
                    .reshape(vec![batch, S, shape.heads, D])?
                    .permute(&[0, 2, 1, 3])?;
            assert_eq!(prove(shape, &input, false), Some(ReadLayout::SequenceMajor));
            let mut wrong = input.clone();
            wrong.view = wrong.view.permute(&[0, 2, 1, 3])?;
            assert_eq!(prove(shape, &wrong, false), None);
            let mut wrong = input.clone();
            wrong.storage_len += 1;
            assert_eq!(prove(shape, &wrong, false), None);
        }
        Ok(())
    }
    #[test]
    fn online_training_paired_leaf_requires_closed_canonical_inputs_and_three_functions()
    -> Result<()> {
        let shape = Shape::new(8, 8)?;
        let inputs = [
            descriptor(shape, true, false, false),
            descriptor(shape, true, true, false),
            descriptor(shape, true, false, false),
            descriptor(shape, true, false, true),
        ];
        let plan = UncompiledPair::new(shape, inputs.clone())?;
        assert_eq!(plan.read_layouts, [ReadLayout::SequenceMajor; 4]);
        assert_eq!(plan.shape, shape);
        assert_eq!(plan.output_shapes[2], vec![8, 8, 64, 512]);
        assert_eq!(plan.output_shapes[0], vec![8, 8, 512, 64]);
        assert_eq!(plan.statistics_shape, vec![64, 512, 3]);
        assert_eq!(
            plan.global_payload_bytes,
            shape.elements() * 8 + 64 * 512 * 12
        );
        let mut bad = inputs.clone();
        bad[3].storage_len += 1;
        assert!(UncompiledPair::new(shape, bad).is_err());
        let mut bad = inputs;
        bad[0].source_shape = vec![8, 512, 3 * 512];
        assert!(UncompiledPair::new(shape, bad).is_err());
        assert!(PairedFunctions::from_all(Some(1), None, Some(3)).is_err());
        assert!(PairedFunctions::from_all(None, Some(2), Some(3)).is_err());
        assert!(PairedFunctions::from_all(Some(1), Some(2), None).is_err());
        let pair = PairedFunctions::from_all(Some(1), Some(2), Some(3))?;
        assert_eq!(
            (pair.forward, pair.d_query_and_delta, pair.d_key_value),
            (1, 2, 3)
        );
        assert!(Shape::new(1, 8).is_err());
        Ok(())
    }
    #[test]
    fn online_training_paired_leaf_allows_only_explicit_linear_layout_families() -> Result<()> {
        let shape = Shape::new(16, 8)?;
        let inputs = [
            descriptor(shape, false, false, false),
            descriptor(shape, false, true, false),
            descriptor(shape, false, false, false),
            descriptor(shape, true, false, true),
        ];
        let plan = UncompiledPair::new(shape, inputs)?;
        assert_eq!(
            plan.read_layouts,
            [
                ReadLayout::HeadMajor,
                ReadLayout::HeadMajor,
                ReadLayout::HeadMajor,
                ReadLayout::SequenceMajor
            ]
        );
        let larger = Shape::new(32, 8)?;
        let more = [
            descriptor(larger, false, false, false),
            descriptor(larger, false, true, false),
            descriptor(larger, false, false, false),
            descriptor(larger, true, false, true),
        ];
        assert_eq!(
            UncompiledPair::new(larger, more)?.global_payload_bytes,
            plan.global_payload_bytes * 2
        );
        Ok(())
    }
}
