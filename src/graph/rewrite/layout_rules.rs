//! Conservative identities that remove complete layout round trips.
//!
//! In contrast to moving a transpose after pointwise work, these rules remove
//! inverse maps or prove that only size-one axes moved. They do not reorder
//! arithmetic, change a broadcast's multiplicity, or cross a cast boundary.
use super::TensorOp;
use super::tensor_analysis::TensorAnalysis;
use egg::{EGraph, Id};

type Graph = EGraph<TensorOp, TensorAnalysis>;

fn axes(graph: &Graph, node: &TensorOp) -> Option<(Id, Vec<usize>)> {
    match node {
        TensorOp::Transpose(input) => {
            let rank = graph[*input].data.as_ref()?.len();
            if rank < 2 {
                return None;
            }
            let mut axes: Vec<usize> = (0..rank).collect();
            axes.swap(rank - 2, rank - 1);
            Some((*input, axes))
        }
        TensorOp::Permute(ids) => {
            let values = ids[1..]
                .iter()
                .map(|&id| {
                    graph[id].nodes.iter().find_map(|node| match node {
                        TensorOp::Num(value) => usize::try_from(*value).ok(),
                        _ => None,
                    })
                })
                .collect::<Option<Vec<usize>>>()?;
            Some((ids[0], values))
        }
        _ => None,
    }
}

fn unary_input(node: &TensorOp) -> Option<Id> {
    match node {
        TensorOp::Neg(input)
        | TensorOp::Exp(input)
        | TensorOp::Log(input)
        | TensorOp::Relu(input) => Some(*input),
        _ => None,
    }
}

fn replace_unary_input(node: &TensorOp, input: Id) -> Option<TensorOp> {
    match node {
        TensorOp::Neg(_) => Some(TensorOp::Neg(input)),
        TensorOp::Exp(_) => Some(TensorOp::Exp(input)),
        TensorOp::Log(_) => Some(TensorOp::Log(input)),
        TensorOp::Relu(_) => Some(TensorOp::Relu(input)),
        _ => None,
    }
}

fn metadata(graph: &mut Graph, input: Id, values: &[usize]) -> Option<Box<[Id]>> {
    let literals: Vec<i32> = values
        .iter()
        .map(|&value| i32::try_from(value).ok())
        .collect::<Option<_>>()?;
    let mut ids = Vec::with_capacity(values.len() + 1);
    ids.push(input);
    ids.extend(
        literals
            .into_iter()
            .map(|value| graph.add(TensorOp::Num(value))),
    );
    Some(ids.into_boxed_slice())
}

fn identity_composition(outer: &[usize], inner: &[usize]) -> bool {
    outer.len() == inner.len()
        && outer
            .iter()
            .enumerate()
            .all(|(axis, &value)| inner.get(value) == Some(&axis))
}

pub(super) fn modify(graph: &mut Graph, id: Id) {
    for node in graph[id].nodes.clone() {
        if let Some((input, outer_axes)) = axes(graph, &node) {
            // Moving only size-one axes leaves flattened row-major values
            // unchanged, even when the resulting tensor shape is different.
            if let Some(shape) = graph[input].data.as_ref() {
                let active_input = (0..shape.len()).filter(|&axis| shape[axis] != 1);
                let active_output = outer_axes.iter().copied().filter(|&axis| shape[axis] != 1);
                if active_input.eq(active_output) {
                    let output_shape = graph[id].data.clone();
                    if output_shape == graph[input].data {
                        graph.union(id, input);
                    } else if let Some(shape) = output_shape
                        && let Some(ids) = metadata(graph, input, &shape)
                    {
                        let next = graph.add(TensorOp::Reshape(ids));
                        graph.union(id, next);
                    }
                }
            }

            for inner in graph[input].nodes.clone() {
                if let Some((source, inner_axes)) = axes(graph, &inner) {
                    // Pure permutation composition already exists. Here unify
                    // the two API representations of the same axis map.
                    let mixed = matches!(node, TensorOp::Transpose(_))
                        != matches!(inner, TensorOp::Transpose(_));
                    if mixed && outer_axes.len() == inner_axes.len() {
                        if identity_composition(&outer_axes, &inner_axes) {
                            graph.union(id, source);
                        } else if graph.find(source) != graph.find(id) {
                            let composed: Vec<usize> = outer_axes
                                .iter()
                                .filter_map(|&axis| inner_axes.get(axis).copied())
                                .collect();
                            if composed.len() == outer_axes.len()
                                && let Some(ids) = metadata(graph, source, &composed)
                            {
                                let next = graph.add(TensorOp::Permute(ids));
                                graph.union(id, next);
                            }
                        }
                    }
                }

                // P^-1(unary(P(x))) is unary(x). No alternative with a moved
                // standalone layout is introduced, so extraction cannot choose
                // the unfavorable transpose-sinking implementation.
                if let Some(unary_source) = unary_input(&inner) {
                    for layout in graph[unary_source].nodes.clone() {
                        if let Some((source, inner_axes)) = axes(graph, &layout)
                            && identity_composition(&outer_axes, &inner_axes)
                            && let Some(unary) = replace_unary_input(&inner, source)
                        {
                            let next = graph.add(unary);
                            graph.union(id, next);
                        }
                    }
                }
            }
        }

        // Reshape round trips through unary pointwise functions are the same
        // identity on flattened values, without any axis-map arithmetic.
        if let TensorOp::Reshape(ids) = node {
            for unary in graph[ids[0]].nodes.clone() {
                if let Some(unary_source) = unary_input(&unary) {
                    for layout in graph[unary_source].nodes.clone() {
                        if let TensorOp::Reshape(inner) = layout
                            && graph[inner[0]].data == graph[id].data
                            && let Some(next) = replace_unary_input(&unary, inner[0])
                        {
                            let next = graph.add(next);
                            graph.union(id, next);
                        }
                    }
                }
            }
        }
    }
}
