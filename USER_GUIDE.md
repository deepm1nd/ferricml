# FerricML User Guide

This guide provides a comprehensive walkthrough for building, training, and deploying machine learning models with FerricML.

## 1. Introduction

FerricML is a 100% Rust-native machine learning platform. It is designed to be the "Rust way" of doing AI: fast, safe, and concurrent.

## 2. Core Concepts

### 2.1 Tensors
The base unit of data. Tensors in FerricML are multi-dimensional arrays that can live on CPUs, GPUs (CUDA/ROCm), or TPUs.
- [Read the Tensor API Guide](docs/TENSORS.md)

### 2.2 Modules
Models are built by composing `Modules`. A Module is a struct that implements the `Module` trait, defining a `forward` pass and a list of `parameters`.

### 2.3 RMLC Compiler
Behind the scenes, FerricML uses the RMLC compiler to optimize your model's execution. It uses Ahead-of-Time (AOT) Automatic Differentiation to generate extremely efficient gradient code.
- [Deep Dive into Architecture](docs/ARCHITECTURE.md)

## 3. Building Your First Model

```rust
use ferricml::prelude::*;
use ferricml::nn::{Linear, Module};

struct MyClassifier {
    layer1: Linear,
    layer2: Linear,
}

impl Module for MyClassifier {
    fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.layer1.forward(input).relu();
        self.layer2.forward(&x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut p = self.layer1.parameters();
        p.extend(self.layer2.parameters());
        p
    }
}
```

## 4. Training Loop

A typical training loop in FerricML involves three main steps: forward, backward, and optimize.

```rust
let mut model = MyClassifier { ... };
let mut optimizer = SGD::new(model.parameters(), 0.01);

for (batch_x, batch_y) in dataset {
    // 1. Forward Pass
    let output = model.forward(&batch_x);
    let loss = output.mse_loss(&batch_y);

    // 2. Backward Pass (AOT-AD)
    loss.backward();

    // 3. Update Weights
    optimizer.step();
}
```

## 5. Advanced Features

### 5.1 Multi-GPU / Distributed Training
FerricML supports Data Parallel (DP) training out of the box using the `ferric-distributed` crate.

### 5.2 Quantization
Reduce model size and increase inference speed with native 4-bit and 8-bit quantization support in GGUF.

## 6. Deployment

FerricML models are deployed as single, statically-linked binaries. No Python environment or heavy runtime required.

### 6.1 WebAssembly
Compile your model to WASM to run directly in the browser with near-native performance.

### 6.2 Cloud Native
Deploy to serverless environments (AWS Lambda, Google Cloud Functions) with minimal cold-start times.

---
## 📚 Additional Resources
- [Tutorials](docs/TUTORIALS.md)
- [Mathematics of FerricML](docs/MATHEMATICS.md)
- [Serialization Guide](docs/SERIALIZATION.md)
