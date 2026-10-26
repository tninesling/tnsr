#[cfg(feature = "cuda")]
use super::DecodeAttention;
use super::{DecodeGraph, GptAttentionPrecision};
use crate::graph::{TensorGraph, TensorGraphNode};
use crate::nn::{Gpt, embedding, layer_norm, linear, softmax};
use crate::tensor::ValueDType;
use crate::tensor::{Parameter, TensorExpr};

fn stash(graph: &mut TensorGraph<f32>, expr: TensorExpr<f32>) -> TensorExpr<f32> {
    let shape = expr.shape().clone();
    let node = graph.add_expr(&expr, false);
    TensorExpr::node_ref(node, shape)
}

pub(super) fn build_graph(
    model: &Gpt,
    cache: &[Parameter<f32>],
    batch: usize,
    past: usize,
    chunk: usize,
    precision: GptAttentionPrecision,
) -> DecodeGraph {
    let mut graph = TensorGraph::new();
    let width = model.config.embed_dim;
    let heads = model.config.num_heads;
    let split = |x: TensorExpr<f32>, sequence| {
        x.reshape(vec![batch, sequence, heads, width / heads])
            .permute(vec![0, 2, 1, 3])
    };
    let mut hidden = stash(
        &mut graph,
        embedding(
            model.token_embedding.clone(),
            TensorExpr::input("tokens", vec![batch, chunk]),
        ) + embedding(
            model.position_embedding.clone(),
            TensorExpr::input("positions", vec![chunk]),
        )
        .reshape(vec![1, chunk, width])
        .broadcast_axis(0, batch),
    );
    let mut projected = Vec::new();
    #[cfg(feature = "cuda")]
    let mut attention = Vec::new();
    for (i, block) in model.blocks.iter().enumerate() {
        let norm = stash(
            &mut graph,
            layer_norm(
                hidden.clone(),
                2,
                block.attention_norm_weight.clone(),
                block.attention_norm_bias.clone(),
                block.epsilon,
            ),
        );
        let a = &block.attention;
        let q = stash(
            &mut graph,
            linear(
                norm.clone(),
                a.query_weight.clone(),
                Some(a.query_bias.clone()),
            ),
        );
        let k = stash(
            &mut graph,
            linear(norm.clone(), a.key_weight.clone(), Some(a.key_bias.clone())),
        );
        let v = stash(
            &mut graph,
            linear(norm, a.value_weight.clone(), Some(a.value_bias.clone())),
        );
        // Explicit identity sinks retain projected K/V without recomputing them.
        for x in [&k, &v] {
            let source = graph.add_expr(x, false);
            let sink = graph.graph.add_node(TensorGraphNode::Reshape {
                shape: vec![batch, chunk, width],
            });
            graph.graph.add_edge(source, sink, 0);
            projected.push(sink);
        }
        let joined = |cache: &Parameter<f32>, new: TensorExpr<f32>| {
            if past == 0 {
                split(new, chunk)
            } else {
                TensorExpr::concat(
                    &[
                        split(TensorExpr::from(cache.clone()).narrow(1, 0, past), past),
                        split(new, chunk),
                    ],
                    2,
                )
            }
        };
        let round = |x: TensorExpr<f32>| {
            if precision == GptAttentionPrecision::Bf16 {
                x.cast(ValueDType::BF16).cast(ValueDType::F32)
            } else {
                x
            }
        };
        let keys = joined(&cache[2 * i], round(k.clone()));
        let values = joined(&cache[2 * i + 1], round(v.clone()));
        let scores = split(round(q.clone()), chunk).matmul(keys.swap_axes(2, 3))
            * (width as f32 / heads as f32).sqrt().recip()
            + TensorExpr::input("mask", vec![chunk, past + chunk]);
        let attended = stash(&mut graph, round(softmax(scores, 3).matmul(values)));
        #[cfg(feature = "cuda")]
        attention.push(DecodeAttention {
            hidden: graph.add_expr(&hidden, false),
            query: graph.add_expr(&q, false),
            key: graph.add_expr(&k, false),
            value: graph.add_expr(&v, false),
            output: graph.add_expr(&attended, false),
        });
        let attended = attended.swap_axes(1, 2).reshape(vec![batch, chunk, width]);
        let residual = stash(
            &mut graph,
            hidden
                + linear(
                    attended,
                    a.output_weight.clone(),
                    Some(a.output_bias.clone()),
                ),
        );
        let norm = layer_norm(
            residual.clone(),
            2,
            block.feed_forward_norm_weight.clone(),
            block.feed_forward_norm_bias.clone(),
            block.epsilon,
        );
        let ff = linear(
            norm,
            block.feed_forward_input_weight.clone(),
            Some(block.feed_forward_input_bias.clone()),
        )
        .relu();
        hidden = stash(
            &mut graph,
            residual
                + linear(
                    ff,
                    block.feed_forward_output_weight.clone(),
                    Some(block.feed_forward_output_bias.clone()),
                ),
        );
    }
    let hidden = layer_norm(
        hidden,
        2,
        model.final_norm_weight.clone(),
        model.final_norm_bias.clone(),
        model.config.layer_norm_epsilon,
    );
    let logits = graph.add_expr(
        &hidden.matmul(TensorExpr::from(model.token_embedding.clone()).transpose()),
        false,
    );
    DecodeGraph {
        graph,
        logits,
        projected,
        #[cfg(feature = "cuda")]
        attention,
    }
}
