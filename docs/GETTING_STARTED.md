# Getting Started with FerricML

Welcome to FerricML! This guide will walk you through the installation process and your first steps with the framework.

## 🛠 Prerequisites

To use FerricML, you need to have the Rust toolchain installed. If you don't have it yet, you can install it via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

FerricML requires **Rust 1.75** or newer.

## 📦 Installation

Add FerricML to your project by running:

```bash
cargo add ferricml
```

### Backend Support

FerricML is designed to run on various hardware backends. By default, it uses the CPU. You can enable other backends using Cargo features:

| Backend | Feature Flag | Requirements |
|---------|--------------|--------------|
| **CPU (LLVM)** | `(default)` | None |
| **CUDA** | `cuda` | NVIDIA GPU + CUDA Toolkit |
| **ROCm** | `rocm` | AMD GPU + ROCm Stack |
| **TPU** | `tpu` | XLA / StableHLO environment |

Example for CUDA:
```bash
cargo add ferricml --features cuda
```

## 🏗 Your First Program

Here is a complete example of creating a tensor and performing basic math.

```rust
use ferricml::prelude::*;

fn main() {
    // 1. Initialize the backend (optional, defaults to CPU)
    // Device::set_current(Device::Cuda(0));

    // 2. Create tensors
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Tensor::ones(&[2, 2]);

    // 3. Perform operations
    let c = &a + &b;
    let d = a.matmul(&b);

    println!("Addition Result:\n{:?}", c);
    println!("Matmul Result:\n{:?}", d);
}
```

## 🧠 Training a Simple Model

To train models, you'll use the `Module` trait and an optimizer.

```rust
use ferricml::prelude::*;
use ferricml::nn::{Linear, Module};
use ferricml::optim::SGD;

struct MyNet {
    fc: Linear,
}

impl Module for MyNet {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.fc.forward(input).relu()
    }
}

fn main() {
    let mut model = MyNet { fc: Linear::new(10, 1) };
    let optimizer = SGD::new(model.parameters(), 0.01);

    // Dummy training step
    let input = Tensor::randn(&[1, 10]);
    let target = Tensor::zeros(&[1, 1]);

    let output = model.forward(&input);
    let loss = (output - target).pow(2).mean();

    loss.backward();
    optimizer.step();
}
```

## ⏭ Next Steps

- **Deep Dive into Tensors:** Read the [Tensor API Guide](TENSORS.md).
- **Understanding Autograd:** Learn about [AOT Automatic Differentiation](AUTOGRAD.md).
- **Architecture:** Explore the [RMLC Compiler Architecture](ARCHITECTURE.md).
- **Serialization:** Learn how to [Save and Load Models](SERIALIZATION.md).
