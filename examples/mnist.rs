use std::time::Instant;

use clap::Parser;
use half::{bf16, f16};
use mnist::MnistBuilder;
use num_traits::Float;
use rand::prelude::*;
use tnsr::nn;
use tnsr::nn::Model;
use tnsr::optimizer::SGD;
use tnsr::tensor::{Input, Parameter};
use tnsr::{Executor, Runtime};

/// MNIST dataset downloader and loader utilities
mod mnist_loader {
    use std::fs;
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};

    use flate2::read::GzDecoder;

    /// Base URL for MNIST dataset files (Google Cloud Storage mirror)
    const MNIST_BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist";

    /// MNIST dataset file names
    const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
    const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
    const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
    const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";

    /// Ensures all MNIST dataset files are downloaded and decompressed in the data directory.
    pub fn ensure_mnist_data(data_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        // Create data directory if it doesn't exist
        fs::create_dir_all(data_dir)?;

        let files = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS];

        for file_name in &files {
            let file_path = data_dir.join(file_name);

            if file_path.exists() {
                println!("  ✓ {} already exists, skipping", file_name);
                continue;
            }

            println!("  Downloading {}...", file_name);
            let gz_file_name = format!("{}.gz", file_name);
            let url = format!("{}/{}", MNIST_BASE_URL, gz_file_name);

            // Download the .gz file
            let response = reqwest::blocking::get(&url)?;
            if !response.status().is_success() {
                return Err(format!(
                    "Failed to download {}: HTTP {}",
                    gz_file_name,
                    response.status()
                )
                .into());
            }

            let gz_bytes = response.bytes()?;

            println!("  Decompressing {}...", gz_file_name);

            // Decompress directly to file
            let mut decoder = GzDecoder::new(&gz_bytes[..]);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;

            // Write decompressed data to file
            let mut output_file = fs::File::create(&file_path)?;
            output_file.write_all(&decompressed)?;

            println!("  ✓ {} ready", file_name);
        }

        Ok(())
    }

    /// Returns the default data directory path (project_root/data)
    pub fn default_data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
    }
}

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
    /// Datatype: f32, f16, or bf16
    #[arg(long, default_value = "f32")]
    dtype: String,
}

fn one_hot<D: Float>(labels: &[u8], num_classes: usize) -> Vec<D> {
    let mut out = vec![D::zero(); labels.len() * num_classes];
    for (i, &y) in labels.iter().enumerate() {
        out[i * num_classes + (y as usize)] = D::one();
    }
    out
}

fn xavier_init<D: Float>(rng: &mut StdRng, fan_in: usize, fan_out: usize) -> Vec<D> {
    let limit = (6.0f32).sqrt() / ((fan_in + fan_out) as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| D::from(rng.random_range(-limit..limit)).unwrap())
        .collect()
}

fn build_mlp_model<D>(
    batch: usize,
    w1: &Parameter<D>,
    b1: &Parameter<D>,
    w2: &Parameter<D>,
    b2: &Parameter<D>,
) -> Model<D>
where
    D: tnsr::tensor::DType + Default + 'static,
{
    // Inputs
    let images = Input::new("images", vec![batch, 784]);
    let labels = Input::new("labels", vec![batch, 10]);

    // Forward: h = relu(images @ w1 + b1), logits = h @ w2 + b2
    let h = nn::relu(nn::linear(images.clone(), w1.clone(), Some(b1.clone())));
    let logits = nn::linear(h, w2.clone(), Some(b2.clone()));
    let loss = nn::cross_entropy_one_hot_logits(logits.clone(), labels, 1);

    // Build model with named outputs
    let mut model = Model::new();
    model.add_output("logits", logits);
    model.add_output("loss", loss.clone());
    model.set_loss(loss);

    model
}

fn main() {
    let args = Args::parse();

    // Initialize chrome tracing
    let guard =
        tnsr::init_chrome_tracing("mnist_trace.json").expect("Failed to initialize chrome tracing");
    println!("Chrome tracing initialized. Trace will be written to mnist_trace.json");

    let mut rng = StdRng::seed_from_u64(args.seed);

    // Ensure MNIST data is downloaded and decompressed
    println!("Checking MNIST dataset...");
    let data_dir = mnist_loader::default_data_dir();
    if let Err(e) = mnist_loader::ensure_mnist_data(&data_dir) {
        eprintln!("Failed to download MNIST dataset: {}", e);
        std::process::exit(1);
    }
    println!("MNIST dataset ready in {}", data_dir.display());

    // Load dataset from the mnist crate
    let ds_span = tracing::span!(tracing::Level::INFO, "load_mnist");
    let _enter = ds_span.enter();
    let mnist_data = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .finalize();
    drop(_enter);
    println!("MNIST dataset loaded.");

    // Dispatch based on dtype
    match args.dtype.as_str() {
        "f32" => run_mnist::<f32>(&args, &mnist_data, &mut rng),
        "f16" => run_mnist::<f16>(&args, &mnist_data, &mut rng),
        "bf16" => run_mnist::<bf16>(&args, &mnist_data, &mut rng),
        _ => {
            eprintln!("Unknown dtype '{}'. Use f32, f16, or bf16", args.dtype);
            std::process::exit(1);
        }
    }

    println!("\nTraining completed. Flushing chrome trace...");
    drop(guard);
    println!("Chrome trace written to mnist_trace.json");
    println!("Open chrome://tracing in Chrome browser and load the trace file to visualize timing");
}

fn run_mnist<D>(args: &Args, mnist_data: &mnist::Mnist, rng: &mut StdRng)
where
    D: tnsr::tensor::DType + Float + Default + Send + Sync + 'static,
{
    println!("Running with dtype: {}", std::any::type_name::<D>());

    // Convert training images from u8 to D and normalize to [0, 1]
    let train_images: Vec<D> = mnist_data
        .trn_img
        .iter()
        .map(|&x| D::from(x as f32 / 255.0).unwrap())
        .collect();
    let train_labels_raw: Vec<u8> = mnist_data.trn_lbl.clone();
    let n_train = train_labels_raw.len();
    assert_eq!(n_train * 784, train_images.len());
    let train_labels_oh = one_hot::<D>(&train_labels_raw, 10);
    println!("Training samples: {n_train}");

    // Convert test images from u8 to D and normalize to [0, 1]
    let test_images: Vec<D> = mnist_data
        .tst_img
        .iter()
        .map(|&x| D::from(x as f32 / 255.0).unwrap())
        .collect();
    let test_labels_raw: Vec<u8> = mnist_data.tst_lbl.clone();
    let n_test = test_labels_raw.len();
    assert_eq!(n_test * 784, test_images.len());
    println!("Test samples: {n_test}");

    println!("Building computation graph...");
    let w1 = Parameter::new(xavier_init(rng, 784, 128), vec![784, 128]);
    let b1 = Parameter::new(vec![D::zero(); 128], vec![1, 128]);
    let w2 = Parameter::new(xavier_init(rng, 128, 10), vec![128, 10]);
    let b2 = Parameter::new(vec![D::zero(); 10], vec![1, 10]);
    let model = build_mlp_model(args.batch_size, &w1, &b1, &w2, &b2);
    let logits_idx = model.get_output("logits").expect("logits output");
    let loss_idx = model.loss().expect("loss node");

    // Augment graph with gradient computation nodes (consumes the graph)
    let grad_graph = model.into_graph().with_gradients(loss_idx);
    let opt = SGD::new(args.lr);
    let mut runtime = Runtime::Cpu(tnsr::SimpleExecutor::new());
    println!("Using backend: {:?}", runtime.backend());

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
            indices.shuffle(rng);
            let mut epoch_loss = D::zero();
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
                let mut x = vec![D::zero(); args.batch_size * 784];
                let mut y = vec![D::zero(); args.batch_size * 10];
                for (i, idx) in (start..end).enumerate() {
                    let j = indices[idx];
                    let src_x = &train_images[j * 784..(j + 1) * 784];
                    let dst_x = &mut x[i * 784..(i + 1) * 784];
                    dst_x.copy_from_slice(src_x);
                    let src_y = &train_labels_oh[j * 10..(j + 1) * 10];
                    let dst_y = &mut y[i * 10..(i + 1) * 10];
                    dst_y.copy_from_slice(src_y);
                }

                let mut inputs = std::collections::HashMap::new();
                inputs.insert("images".to_string(), x);
                inputs.insert("labels".to_string(), y);

                // Forward pass computes both loss and gradients
                let loss_value = runtime.execute(&grad_graph, inputs.clone()).unwrap();
                let grads = runtime.get_gradients(&grad_graph);

                if steps.is_multiple_of(50)
                    && let Some(&lv) = loss_value.first()
                {
                    epoch_loss = epoch_loss + lv;
                }
                let opt_span = tracing::span!(tracing::Level::TRACE, "optimizer_step");
                let _og = opt_span.enter();
                opt.step(&grad_graph, &grads);
                drop(_og);
                steps += 1;
            }
            let avg_loss_f32 = (epoch_loss / D::from(steps.max(1) as f32 / 50.0).unwrap())
                .to_f32()
                .unwrap_or(0.0);
            println!(
                "epoch {} avg loss ~ {:.4}, execution time {:.2}",
                epoch + 1,
                avg_loss_f32,
                epoch_start.elapsed().as_secs_f32()
            );
        }

        // Evaluate on current epoch
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
            let mut x = vec![D::zero(); args.batch_size * 784];
            for (i, j) in (start..end).enumerate() {
                let src = &test_images[j * 784..(j + 1) * 784];
                let dst = &mut x[i * 784..(i + 1) * 784];
                dst.copy_from_slice(src);
            }
            // Dummy labels input
            let dummy_labels = vec![D::zero(); args.batch_size * 10];
            let mut inputs = std::collections::HashMap::new();
            inputs.insert("images".to_string(), x);
            inputs.insert("labels".to_string(), dummy_labels);

            runtime.execute(&grad_graph, inputs).unwrap();
            let logits = runtime.get_value(logits_idx).unwrap();

            for i in 0..bs {
                let row = &logits[i * 10..(i + 1) * 10];
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
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
        println!("total time {:.2}s", training_start.elapsed().as_secs_f32());
    }

    // Final inference graph evaluation
    let infer_graph = grad_graph.without_gradients();
    println!("\nFinal evaluation using inference-only graph...");
    let final_eval_span = tracing::span!(tracing::Level::INFO, "final_evaluation");
    let _fev = final_eval_span.enter();
    let mut correct = 0usize;
    let mut seen = 0usize;
    let test_batches = n_test.div_ceil(args.batch_size);
    for b in 0..test_batches {
        let start = b * args.batch_size;
        let end = ((b + 1) * args.batch_size).min(n_test);
        let bs = end - start;
        if bs == 0 {
            continue;
        }
        let mut x = vec![D::zero(); args.batch_size * 784];
        for (i, j) in (start..end).enumerate() {
            let src = &test_images[j * 784..(j + 1) * 784];
            let dst = &mut x[i * 784..(i + 1) * 784];
            dst.copy_from_slice(src);
        }
        let dummy_labels = vec![D::zero(); args.batch_size * 10];
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("images".to_string(), x);
        inputs.insert("labels".to_string(), dummy_labels);

        runtime.execute(&infer_graph, inputs).unwrap();
        let logits = runtime.get_value(logits_idx).unwrap();

        for i in 0..bs {
            let row = &logits[i * 10..(i + 1) * 10];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if pred as u8 == test_labels_raw[start + i] {
                correct += 1;
            }
        }
        seen += bs;
    }
    let final_acc = correct as f32 / seen as f32;
    println!(
        "Final test accuracy (inference graph): {:.2}% ({}/{})",
        final_acc * 100.0,
        correct,
        seen
    );
}
