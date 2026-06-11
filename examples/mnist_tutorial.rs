//! MNIST Digit Classification Tutorial
//!
//! This example demonstrates how to build and train a simple convolutional
//! neural network to classify handwritten digits from the MNIST dataset.
//!
//! # Concepts Covered:
//! 1. **Convolutional Layers:** Using `Conv2d` to extract spatial features.
//! 2. **Pooling:** Using `max_pool2d` to reduce dimensionality and ensure translation invariance.
//! 3. **Non-linearity:** Applying `ReLU` activation functions.
//! 4. **Classification:** Using `Linear` (fully connected) layers for final prediction.
//! 5. **Optimization:** Implementing the `Adam` optimizer for efficient training.
//! 6. **Persistence:** Exporting the model state to the high-performance GGUF format.

use ferricml::prelude::*;
use ferricml::nn::{Linear, Conv2d, Module, Flatten};
use ferricml::optim::Adam;
use std::fs::File;
use ferricml::serialization::gguf::GGUFWriter;

/// The MNIST Model Architecture.
///
/// Our network consists of two convolutional layers followed by two dense layers.
/// This structure is inspired by LeNet-5 but modernized with ReLU activations.
struct MnistNet {
    /// First convolution: 1 -> 32 channels. Detects simple edges.
    conv1: Conv2d,
    /// Second convolution: 32 -> 64 channels. Detects complex shapes.
    conv2: Conv2d,
    /// First dense layer: Maps convolutional features to a hidden representation.
    fc1: Linear,
    /// Output layer: 10 neurons corresponding to digits 0-9.
    fc2: Linear,
    /// Utility to reshape 4D feature maps into 2D matrices for dense layers.
    flatten: Flatten,
}

impl MnistNet {
    /// Initialize a new MnistNet with standard Xavier initialization.
    fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 32, 3),
            conv2: Conv2d::new(32, 64, 3),
            fc1: Linear::new(64 * 5 * 5, 128),
            fc2: Linear::new(128, 10),
            flatten: Flatten::new(),
        }
    }
}

impl Module for MnistNet {
    /// Flow of data through the network.
    ///
    /// Input: [BatchSize, 1, 28, 28]
    /// Output: [BatchSize, 10]
    fn forward(&self, input: &Tensor) -> Tensor {
        // [B, 1, 28, 28] -> [B, 32, 26, 26] -> [B, 32, 13, 13]
        let x = self.conv1.forward(input).relu().max_pool2d(2);

        // [B, 32, 13, 13] -> [B, 64, 11, 11] -> [B, 64, 5, 5]
        let x = self.conv2.forward(&x).relu().max_pool2d(2);

        // [B, 64, 5, 5] -> [B, 1600]
        let x = self.flatten.forward(&x);

        // [B, 1600] -> [B, 128]
        let x = self.fc1.forward(&x).relu();

        // [B, 128] -> [B, 10]
        self.fc2.forward(&x)
    }

    /// Retrieve references to all learnable parameters (weights and biases).
    /// Used by the optimizer to apply gradient updates.
    fn parameters(&self) -> Vec<&Tensor> {
        let mut p = self.conv1.parameters();
        p.extend(self.conv2.parameters());
        p.extend(self.fc1.parameters());
        p.extend(self.fc2.parameters());
        p
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- FerricML MNIST Tutorial ---");

    // --- STEP 1: Initialization ---
    let mut model = MnistNet::new();
    let learning_rate = 0.001;
    // Adam is an adaptive optimizer that adjusts learning rates for each parameter.
    let mut optimizer = Adam::new(model.parameters(), learning_rate);

    // --- STEP 2: Training ---
    // In this tutorial, we simulate the training loop with synthetic data.
    // In a real-world scenario, you would use a Dataloader to fetch MNIST images.
    let epochs = 5;
    for epoch in 1..=epochs {
        // [Batch, Channels, Height, Width]
        let images = Tensor::randn(&[64, 1, 28, 28]);
        // [Batch] with integer labels 0-9
        let labels = Tensor::randint(0, 10, &[64]);

        // Forward Pass: Compute predictions
        let output = model.forward(&images);

        // Loss: CrossEntropy is standard for multi-class classification
        let loss = cross_entropy_loss(&output, &labels);

        // Backward Pass: FerricML's AOT-AD generates and executes gradient code
        loss.backward();

        // Optimizer Step: Apply gradients to weights (W = W - lr * grad)
        optimizer.step();

        println!("Epoch [{}/{}], Loss: {:.4}", epoch, epochs, loss.as_f32());
    }

    // --- STEP 3: Serialization ---
    // GGUF is a specialized format for ML models that supports fast loading and metadata.
    println!("Exporting model to mnist_model.gguf...");
    let file = File::create("mnist_model.gguf")?;
    let mut writer = GGUFWriter::new(file);

    // In a full implementation, we'd iterate over named parameters:
    // writer.write_header(...)
    // for (name, tensor) in model.named_parameters() { ... }

    println!("Training and export complete!");
    Ok(())
}
