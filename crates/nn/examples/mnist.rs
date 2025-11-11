use std::time::Instant;

use candle_datasets::vision::mnist;
use clap::Parser;
use rand::prelude::*;
use runtime::Executor;
use runtime::optimizer::SGD;
use tensor::Input;
use tensor::Parameter;
use tensor::graph::TensorGraph;

#[cfg(not(feature = "cuda"))]
use runtime::SimpleExecutor;
#[cfg(feature = "cuda")]
use runtime::cuda::CudaExecutor;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.05)]
    lr: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn one_hot(labels: &[u8], num_classes: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; labels.len() * num_classes];
    for (i, &y) in labels.iter().enumerate() {
        out[i * num_classes + (y as usize)] = 1.0;
    }
    out
}

fn xavier_init(rng: &mut StdRng, fan_in: usize, fan_out: usize) -> Vec<f32> {
    let limit = (6.0f32).sqrt() / ((fan_in + fan_out) as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| rng.random_range(-limit..limit))
        .collect()
}

fn build_mlp_graph(
    batch: usize,
    w1: &Parameter<f32>,
    b1: &Parameter<f32>,
    w2: &Parameter<f32>,
    b2: &Parameter<f32>,
) -> (
    TensorGraph<f32>,
    tensor::graph::NodeIndex,
    tensor::graph::NodeIndex,
) {
    // Inputs
    let images = Input::<f32>::new("images", vec![batch, 784]);
    let labels = Input::<f32>::new("labels", vec![batch, 10]);

    // Forward: h = relu(images @ w1 + b1), logits = h @ w2 + b2
    let h = nn::relu(nn::linear(images.clone(), w1.clone(), Some(b1.clone())));
    let logits = nn::linear(h, w2.clone(), Some(b2.clone()));
    let mut graph = TensorGraph::new();

    // Always build both logits and loss nodes
    let logits_idx = logits.clone().lower_to_graph(&mut graph);
    let loss = nn::cross_entropy_one_hot_logits(logits, labels, 1);
    let loss_idx = loss.lower_to_graph(&mut graph);

    (graph, logits_idx, loss_idx)
}

fn main() {
    let args = Args::parse();

    // Initialize chrome tracing
    let guard = runtime::init_chrome_tracing("mnist_trace.json")
        .expect("Failed to initialize chrome tracing");
    println!("Chrome tracing initialized. Trace will be written to mnist_trace.json");

    let mut rng = StdRng::seed_from_u64(args.seed);

    // Load dataset from HF hub.
    let ds_span = tracing::span!(tracing::Level::INFO, "load_mnist");
    let _enter = ds_span.enter();
    let ds = mnist::load()
        .expect("failed to load MNIST from hub; ensure network access or provide local IDX files");
    drop(_enter);
    println!("MNIST dataset loaded.");

    let train_images_2d: Vec<Vec<f32>> =
        ds.train_images.to_vec2().expect("to_vec2 for train_images");
    let n_train = train_images_2d.len();
    let train_images: Vec<f32> = train_images_2d.into_iter().flatten().collect();
    let train_labels_raw: Vec<u8> = ds.train_labels.to_vec1().expect("to_vec1 for train_labels");
    assert_eq!(n_train, train_labels_raw.len());
    let train_labels_oh = one_hot(&train_labels_raw, 10);
    println!("Training samples: {n_train}");

    let test_images_2d: Vec<Vec<f32>> = ds.test_images.to_vec2().expect("to_vec2 for test_images");
    let n_test = test_images_2d.len();
    let test_images: Vec<f32> = test_images_2d.into_iter().flatten().collect();
    let test_labels_raw: Vec<u8> = ds.test_labels.to_vec1().expect("to_vec1 for test_labels");
    assert_eq!(n_test, test_labels_raw.len());
    println!("Test samples: {n_test}");

    // Parameters
    let w1 = Parameter::new(xavier_init(&mut rng, 784, 128), vec![784, 128]);
    let b1 = Parameter::new(vec![0.0f32; 128], vec![1, 128]);
    let w2 = Parameter::new(xavier_init(&mut rng, 128, 10), vec![128, 10]);
    let b2 = Parameter::new(vec![0.0f32; 10], vec![1, 10]);

    // Graph
    println!("Building computation graph...");
    let (graph, logits_idx, loss_idx) = build_mlp_graph(args.batch_size, &w1, &b1, &w2, &b2);

    let opt = SGD::new(args.lr);

    #[cfg(feature = "cuda")]
    let mut exec = CudaExecutor::new();
    #[cfg(not(feature = "cuda"))]
    let mut exec = SimpleExecutor::new();

    // Training loop
    println!("Starting training for {} epochs...", args.epochs);
    let batches_per_epoch = n_train.div_ceil(args.batch_size);
    let training_start = Instant::now();
    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();
        let epoch_span = tracing::span!(
            tracing::Level::INFO,
            "epoch",
            idx = epoch + 1,
            batches = batches_per_epoch,
            lr = args.lr,
            batch_size = args.batch_size
        );
        let _eg = epoch_span.enter();

        // Training phase
        {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..n_train).collect();
            indices.shuffle(&mut rng);
            let mut epoch_loss = 0.0f32;
            let mut steps = 0usize;

            for b in 0..batches_per_epoch {
                let start = b * args.batch_size;
                let end = ((b + 1) * args.batch_size).min(n_train);
                let bs = end - start;
                if bs == 0 {
                    continue;
                }
                let batch_span = tracing::span!(
                    tracing::Level::TRACE,
                    "batch",
                    idx = b,
                    bs = bs,
                    start = start,
                    end = end
                );
                let _bg = batch_span.enter();
                // Collect batch
                let mut x = vec![0.0f32; args.batch_size * 784];
                let mut y = vec![0.0f32; args.batch_size * 10];
                for (i, idx) in (start..end).enumerate() {
                    let j = indices[idx];
                    let src_x = &train_images[j * 784..(j + 1) * 784];
                    let dst_x = &mut x[i * 784..(i + 1) * 784];
                    dst_x.copy_from_slice(src_x);
                    let src_y = &train_labels_oh[j * 10..(j + 1) * 10];
                    let dst_y = &mut y[i * 10..(i + 1) * 10];
                    dst_y.copy_from_slice(src_y);
                }
                // For last batch if bs < batch_size, pad remaining already zeros; network ignores extra rows statistically; or we could rebuild graphs but we keep it simple.

                let mut inputs = std::collections::HashMap::new();
                inputs.insert("images".to_string(), x);
                inputs.insert("labels".to_string(), y);

                // Forward pass to get loss value
                let loss_value = exec.forward(&graph, inputs.clone());

                // Backward pass to get gradients
                let backward_result = exec.backward(&graph, loss_idx, None);
                let grads = backward_result.grads_by_param;

                if steps.is_multiple_of(50)
                    && let Some(&lv) = loss_value.first()
                {
                    epoch_loss += lv;
                }
                let opt_span = tracing::span!(tracing::Level::TRACE, "optimizer_step");
                let _og = opt_span.enter();
                opt.step(&graph, &grads);
                drop(_og);
                steps += 1;
            }
            println!(
                "epoch {} avg loss ~ {:.4}, execution time {:.2}",
                epoch + 1,
                epoch_loss.max(1e-8) / (steps.max(1) as f32 / 50.0),
                epoch_start.elapsed().as_secs_f32()
            );
        }

        // Evaluate
        let test_batches = n_test.div_ceil(args.batch_size);
        let eval_span = tracing::span!(
            tracing::Level::INFO,
            "evaluate",
            test_batches = test_batches
        );
        let _ev = eval_span.enter();
        let mut correct = 0usize;
        let mut seen = 0usize;
        for b in 0..test_batches {
            let start = b * args.batch_size;
            let end = ((b + 1) * args.batch_size).min(n_test);
            let bs = end - start;
            if bs == 0 {
                continue;
            }
            let infer_span = tracing::span!(
                tracing::Level::TRACE,
                "infer_batch",
                idx = b,
                bs = bs,
                start = start,
                end = end
            );
            let _ib = infer_span.enter();
            let mut x = vec![0.0f32; args.batch_size * 784];
            for (i, j) in (start..end).enumerate() {
                let src = &test_images[j * 784..(j + 1) * 784];
                let dst = &mut x[i * 784..(i + 1) * 784];
                dst.copy_from_slice(src);
            }
            // Dummy labels input (graph expects labels but we don't need loss for inference)
            let dummy_labels = vec![0.0f32; args.batch_size * 10];
            let mut inputs = std::collections::HashMap::new();
            inputs.insert("images".to_string(), x);
            inputs.insert("labels".to_string(), dummy_labels);

            // Forward pass through the unified graph
            exec.forward(&graph, inputs);

            // Extract logits using get_value()
            #[cfg(feature = "cuda")]
            let logits = exec.get_value(logits_idx).unwrap();
            #[cfg(not(feature = "cuda"))]
            let logits = exec.get_value(logits_idx).unwrap().clone();

            for i in 0..bs {
                let row = &logits[i * 10..(i + 1) * 10];
                let pred = row
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_nan())
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                if pred as u8 == test_labels_raw[start + i] {
                    correct += 1;
                }
            }
            seen += bs;
        }
        let acc = correct as f32 / seen as f32;
        println!("test accuracy: {:.2}% ({}/{})", acc * 100.0, correct, seen);
        println!(
            "evaluation time {:.2}s",
            training_start.elapsed().as_secs_f32()
        );

        // End of epoch - _eg guard will be dropped here
    }

    println!("Training completed. Flushing chrome trace...");

    // Explicitly flush the tracing guard to ensure all spans are written
    drop(guard);
    println!("Chrome trace written to mnist_trace.json");
    println!("Open chrome://tracing in Chrome browser and load the trace file to visualize timing");
}
