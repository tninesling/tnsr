use tensor::Constant;
use tensor::DType;
use tensor::Shape;
use tensor::Tensor;

pub fn linear<D: DType + Default + 'static>(
    x: impl Tensor<D> + 'static,
    w: impl Tensor<D> + 'static,
    b: Option<impl Tensor<D> + 'static>,
) -> Box<dyn Tensor<D>> {
    use tensor::TensorOps;
    let bnode: Box<dyn Tensor<D>> = match b {
        Some(bias) => Box::new(bias),
        None => Box::new(Constant::new(vec![D::default()], vec![])),
    };
    Box::new(x.matmul(w) + bnode)
}

pub fn constant_f32(data: Vec<f32>, shape: Shape) -> Constant<f32> {
    Constant::new(data, shape)
}

#[cfg(test)]
mod tests {
    use runtime::Executor;
    use runtime::SimpleExecutor;
    use tensor::graph::TensorGraph;

    use super::*;

    const EPSILON: f32 = 1e-5;

    fn assert_approx_eq(a: &[f32], b: &[f32]) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if (x - y).abs() > EPSILON {
                panic!("mismatch at {}: {} vs {}", i, x, y);
            }
        }
    }

    #[test]
    fn linear_no_bias() {
        let x = constant_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let w = constant_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let node = linear::<f32>(x, w, Option::<Constant<f32>>::None);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let exec = SimpleExecutor {};
        let out = exec.execute(&graph, Default::default());

        let expected = {
            let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let (m, k, n) = (2usize, 3usize, 2usize);
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    out[i * n + j] = sum;
                }
            }
            out
        };

        assert_approx_eq(&out, &expected);
    }

    #[test]
    fn linear_with_bias_vector() {
        let x = constant_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let w = constant_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let b = constant_f32(vec![10.0, -1.0], vec![2]);

        let node = linear::<f32>(x, w, Some(b));

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let exec = SimpleExecutor {};
        let out = exec.execute(&graph, Default::default());

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bmat = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![10.0f32, -1.0];
        let (m, k, n) = (2usize, 3usize, 2usize);
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * bmat[p * n + j];
                }
                expected[i * n + j] = sum + bias[j];
            }
        }

        assert_approx_eq(&out, &expected);
    }
}
