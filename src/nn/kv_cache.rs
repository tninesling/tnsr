//! Incremental GPT inference with fixed-capacity, per-layer projected K/V storage.
use super::Gpt;
use crate::graph::{NodeIndex, TensorGraph};
use crate::tensor::Parameter;
use crate::{Backend, Executor, SimpleExecutor};
use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, HashMap};

mod graph;
use graph::build_graph;
#[cfg(feature = "cuda")]
mod attention;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod prefill;
#[cfg(feature = "cuda")]
mod stages;

/// Storage at the attention boundary; projections and the rest of GPT remain f32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GptAttentionPrecision {
    #[default]
    F32,
    Bf16,
}

/// CUDA prefill schedule. Scalar is useful as a precision-matched reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GptPrefillMode {
    #[default]
    Auto,
    Scalar,
}

/// Immutable inference precision and scheduling policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct GptDecoderOptions {
    pub attention_precision: GptAttentionPrecision,
    pub prefill: GptPrefillMode,
}

struct DecodeGraph {
    graph: TensorGraph<f32>,
    logits: NodeIndex,
    projected: Vec<NodeIndex>,
    #[cfg(feature = "cuda")]
    attention: Vec<DecodeAttention>,
}

#[cfg(feature = "cuda")]
struct DecodeAttention {
    hidden: NodeIndex,
    query: NodeIndex,
    key: NodeIndex,
    value: NodeIndex,
    output: NodeIndex,
}

enum DecoderBackend {
    Cpu(SimpleExecutor),
    #[cfg(feature = "cuda")]
    Cuda(Box<cuda::CudaDecoder>),
}

/// An inference session bound to one GPT model, batch size and context capacity.
///
/// `decode` accepts row-major `[batch, new_tokens]` token IDs and returns logits
/// for those positions only. All rows advance together. CUDA retains K/V and
/// model weights on device; only token/position/length feeds and logits cross the
/// host boundary. Reset before changing model weights or starting another prompt.
/// The session rejects overflow rather than evicting learned-position history.
pub struct GptDecoder {
    model: Gpt,
    backend: DecoderBackend,
    cache: Vec<Parameter<f32>>,
    parameters: Vec<Parameter<f32>>,
    revisions: Vec<u64>,
    graphs: BTreeMap<(usize, usize), DecodeGraph>,
    batch: usize,
    capacity: usize,
    length: usize,
    failed: bool,
    options: GptDecoderOptions,
}

impl Gpt {
    /// Create a reusable incremental inference session using the full context.
    pub fn decoder(&self, backend: Backend, batch_size: usize) -> Result<GptDecoder> {
        self.decoder_with_capacity(backend, batch_size, self.config.max_sequence_length)
    }

    /// Create a session with a smaller fixed K/V capacity to bound memory use.
    pub fn decoder_with_capacity(
        &self,
        backend: Backend,
        batch_size: usize,
        capacity: usize,
    ) -> Result<GptDecoder> {
        self.decoder_with_options(backend, batch_size, capacity, GptDecoderOptions::default())
    }

    /// Create a session with explicit attention storage and prefill scheduling.
    /// BF16 rounds Q/K/V and attention outputs; softmax statistics accumulate in f32.
    /// CPU uses rounded f32 values as a reference; CUDA stores K/V physically in BF16.
    /// BF16 CUDA attention requires compute capability 8.0 or newer.
    pub fn decoder_with_options(
        &self,
        backend: Backend,
        batch_size: usize,
        capacity: usize,
        options: GptDecoderOptions,
    ) -> Result<GptDecoder> {
        GptDecoder::new(self.clone(), backend, batch_size, capacity, options)
    }

    /// Greedy cached generation with the same sliding-window behavior as `generate`.
    ///
    /// Once full, learned positions must restart at zero: rebuild the retained
    /// context instead of evicting K/V whose hidden states used older positions.
    pub fn generate_cached(
        &self,
        backend: Backend,
        prompt: &[usize],
        max_new_tokens: usize,
    ) -> Result<Vec<usize>> {
        ensure!(
            !prompt.is_empty(),
            "GPT generation prompt must be non-empty"
        );
        ensure!(
            prompt.iter().all(|&t| t < self.config.vocab_size),
            "GPT generation prompt contains an out-of-range token"
        );
        let mut tokens = prompt.to_vec();
        if max_new_tokens == 0 {
            return Ok(tokens);
        }
        let mut decoder = self.decoder(backend, 1)?;
        for _ in 0..max_new_tokens {
            let logits = if decoder.is_empty() || decoder.len() == decoder.capacity() {
                decoder.reset()?;
                let start = tokens.len().saturating_sub(decoder.capacity());
                decoder.decode(&tokens[start..])?
            } else {
                decoder.decode(&tokens[tokens.len() - 1..])?
            };
            let next = logits[logits.len() - self.config.vocab_size..]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .context("GPT vocabulary is empty")?;
            tokens.push(next);
        }
        Ok(tokens)
    }
}

impl GptDecoder {
    fn new(
        model: Gpt,
        backend: Backend,
        batch: usize,
        capacity: usize,
        options: GptDecoderOptions,
    ) -> Result<Self> {
        ensure!(batch > 0, "decoder batch size must be non-zero");
        ensure!(
            capacity > 0 && capacity <= model.config.max_sequence_length,
            "decoder capacity must fit the model context"
        );
        let elements = batch
            .checked_mul(capacity)
            .and_then(|n| n.checked_mul(model.config.embed_dim))
            .context("decoder cache size overflow")?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .context("decoder cache byte size overflow")?;
        ensure!(
            bytes <= isize::MAX as usize,
            "decoder cache exceeds allocation limits"
        );
        bytes
            .checked_mul(model.blocks.len())
            .and_then(|n| n.checked_mul(2))
            .context("decoder total cache size overflow")?;
        let cache: Vec<_> = (0..model.blocks.len() * 2)
            .map(|_| {
                Parameter::new(
                    vec![0.; elements],
                    vec![batch, capacity, model.config.embed_dim],
                )
            })
            .collect();
        let parameters = model_parameters(&model);
        let revisions = parameters
            .iter()
            .map(Parameter::revision)
            .collect::<Result<_>>()?;
        let backend =
            match backend {
                Backend::Cpu => DecoderBackend::Cpu(SimpleExecutor::new()),
                #[cfg(feature = "cuda")]
                Backend::Cuda | Backend::Ptx => DecoderBackend::Cuda(Box::new(
                    cuda::CudaDecoder::new(&cache, &parameters, options)?,
                )),
            };
        Ok(Self {
            model,
            backend,
            cache,
            parameters,
            revisions,
            graphs: BTreeMap::new(),
            batch,
            capacity,
            length: 0,
            failed: false,
            options,
        })
    }

    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn batch_size(&self) -> usize {
        self.batch
    }

    /// Clear the context while retaining recently used graphs/programs for reuse.
    pub fn reset(&mut self) -> Result<()> {
        self.failed = true;
        self.length = 0;
        match &mut self.backend {
            DecoderBackend::Cpu(_) => {
                for cache in &self.cache {
                    cache.set_values(vec![
                        0.;
                        self.batch
                            * self.capacity
                            * self.model.config.embed_dim
                    ])?;
                }
            }
            #[cfg(feature = "cuda")]
            DecoderBackend::Cuda(backend) => backend.reset(&self.cache)?,
        }
        self.failed = false;
        Ok(())
    }

    /// Append a nonempty chunk. Invalid inputs leave the context unchanged.
    /// Execution failures require `reset` before another decode attempt.
    pub fn decode(&mut self, tokens: &[usize]) -> Result<Vec<f32>> {
        ensure!(
            !self.failed,
            "decoder execution failed; reset the cache before reuse"
        );
        ensure!(
            !tokens.is_empty() && tokens.len().is_multiple_of(self.batch),
            "tokens must be a nonempty rectangular [batch, sequence] chunk"
        );
        let chunk = tokens.len() / self.batch;
        ensure!(
            chunk <= self.capacity - self.length,
            "decoder context capacity exceeded"
        );
        ensure!(
            tokens.iter().all(|&t| t < self.model.config.vocab_size),
            "decoder token is outside the vocabulary"
        );
        let revisions = self
            .parameters
            .iter()
            .map(Parameter::revision)
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            self.is_empty() || revisions == self.revisions,
            "model weights changed; reset the decoder cache"
        );
        // CPU graphs slice the exact live prefix. CUDA stages replace attention
        // with a dynamic scan and retain one program per chunk shape.
        let past = match self.backend {
            DecoderBackend::Cpu(_) => self.length,
            #[cfg(feature = "cuda")]
            DecoderBackend::Cuda(_) => self.capacity,
        };
        let shape = (chunk, past);
        if !self.graphs.contains_key(&shape) {
            // Bound compiled shape retention for callers using arbitrary chunk sizes.
            if self.graphs.len() == 8 {
                self.graphs.clear();
            }
            self.graphs.insert(
                shape,
                build_graph(
                    &self.model,
                    &self.cache,
                    self.batch,
                    past,
                    chunk,
                    self.options.attention_precision,
                ),
            );
        }
        let graph = &self.graphs[&shape];
        let mut feeds = HashMap::from([
            ("tokens".into(), tokens.iter().map(|&t| t as f32).collect()),
            (
                "positions".into(),
                (self.length..self.length + chunk)
                    .map(|p| p as f32)
                    .collect(),
            ),
        ]);
        if matches!(self.backend, DecoderBackend::Cpu(_)) {
            let stride = past + chunk;
            let mut mask = vec![f32::NEG_INFINITY; chunk * stride];
            for q in 0..chunk {
                mask[q * stride..q * stride + past + q + 1].fill(0.);
            }
            feeds.insert("mask".into(), mask);
        }
        self.failed = true;
        let logits = match &mut self.backend {
            DecoderBackend::Cpu(executor) => {
                executor.execute(&graph.graph, feeds)?;
                let logits = executor
                    .get_value(graph.logits)
                    .context("decoder logits missing")?;
                let projected = graph
                    .projected
                    .iter()
                    .map(|&n| {
                        executor
                            .get_value(n)
                            .context("decoder projected value missing")
                    })
                    .collect::<Result<Vec<_>>>()?;
                let width = self.model.config.embed_dim;
                for (cache, mut values) in self.cache.iter().zip(projected) {
                    if self.options.attention_precision == GptAttentionPrecision::Bf16 {
                        for value in &mut values {
                            *value = half::bf16::from_f32(*value).to_f32();
                        }
                    }
                    let mut old = cache.values()?;
                    for b in 0..self.batch {
                        let start = (b * self.capacity + self.length) * width;
                        old[start..start + chunk * width]
                            .copy_from_slice(&values[b * chunk * width..(b + 1) * chunk * width]);
                    }
                    cache.set_values(old)?;
                }
                logits
            }
            #[cfg(feature = "cuda")]
            DecoderBackend::Cuda(backend) => backend.decode(
                graph,
                feeds,
                &self.parameters,
                &self.cache,
                self.length,
                chunk,
                self.batch,
                self.capacity,
                self.model.config.embed_dim,
            )?,
        };
        ensure!(
            self.parameters
                .iter()
                .map(Parameter::revision)
                .collect::<Result<Vec<_>>>()?
                == revisions,
            "model weights changed during decoding; reset the cache"
        );
        self.revisions = revisions;
        self.length += chunk;
        self.failed = false;
        Ok(logits)
    }
}

fn model_parameters(model: &Gpt) -> Vec<Parameter<f32>> {
    let mut result = vec![
        model.token_embedding.clone(),
        model.position_embedding.clone(),
        model.final_norm_weight.clone(),
        model.final_norm_bias.clone(),
    ];
    for b in &model.blocks {
        let a = &b.attention;
        result.extend([
            a.query_weight.clone(),
            a.query_bias.clone(),
            a.key_weight.clone(),
            a.key_bias.clone(),
            a.value_weight.clone(),
            a.value_bias.clone(),
            a.output_weight.clone(),
            a.output_bias.clone(),
            b.attention_norm_weight.clone(),
            b.attention_norm_bias.clone(),
            b.feed_forward_norm_weight.clone(),
            b.feed_forward_norm_bias.clone(),
            b.feed_forward_input_weight.clone(),
            b.feed_forward_input_bias.clone(),
            b.feed_forward_output_weight.clone(),
            b.feed_forward_output_bias.clone(),
        ]);
    }
    result
}
