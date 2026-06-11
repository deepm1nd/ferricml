<div align="center">
  <img src="https://raw.githubusercontent.com/deepm1nd/ferricml/website/img/hero_banner.png" alt="FerricML Logo" width="200">
  <h1>FerricML</h1>
  <p><strong>A High-Performance, End-to-End Machine Learning Platform Native to Rust.</strong></p>

  <p>
    <a href="https://github.com/deepm1nd/ferricml/actions"><img src="https://github.com/deepm1nd/ferricml/workflows/CI/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/ferricml"><img src="https://img.shields.io/crates/v/ferricml.svg" alt="Crates.io"></a>
    <a href="https://docs.rs/ferricml"><img src="https://docs.rs/ferricml/badge.svg" alt="Documentation"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  </p>
</div>

---

[FerricML](https://deepm1nd.github.io/ferricml) is a universal machine learning framework implemented natively in Rust, designed for performance, memory safety, and seamless deployment across heterogeneous hardware backends (CPU, CUDA, ROCm, TPU).

Unlike existing frameworks that rely on heavy Python-C++ bindings, FerricML offers a **unified language stack**, providing zero-cost abstractions and compile-time guarantees for building, training, and deploying AI models.

## 🚀 Key Features

- **100% Pure Rust:** Full memory safety and thread safety without the overhead of garbage collection or a Global Interpreter Lock (GIL).
- **AOT Automatic Differentiation:** Ahead-of-Time (AOT) AD for superior performance optimization compared to traditional tape-based methods.
- **Heterogeneous Compute:** First-class support for LLVM (CPU), NVVM/PTX (CUDA), ROCDL (ROCm), and XLA/StableHLO (TPU).
- **Multi-Level IR:** A compiler architecture inspired by MLIR, featuring Tensor, Structured, and Target dialects for deep hardware optimization.
- **Production-Ready Serialization:** Native support for GGUF (v3) and SafeTensors.
- **WASM & Edge Support:** Deploy your models to the web or IoT devices with minimal footprint.

## ⚡ Quick Start

Add FerricML to your `Cargo.toml`:

```toml
[dependencies]
ferricml = "0.1"
```

Create and run a simple model:

```rust
use ferricml::prelude::*;

fn main() {
    // Create tensors with symbolic tracking
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let w = Tensor::randn(&[2, 2]).requires_grad();

    // Perform a forward pass
    let y = x.matmul(&w).relu();

    println!("Output: {:?}", y);

    // Compute gradients (AOT Autograd)
    // y.backward();
    // println!("Gradients: {:?}", w.grad());
}
```

## 🛠 Installation

FerricML supports multiple backends. To enable GPU support, use the corresponding feature flags:

```bash
# CPU (Default)
cargo add ferricml

# CUDA Support
cargo add ferricml --features cuda

# ROCm Support
cargo add ferricml --features rocm
```

For detailed setup instructions, see the [Getting Started Guide](docs/GETTING_STARTED.md).

## 📚 Documentation

- [User Guide](USER_GUIDE.md) - Step-by-step walkthrough.
- [Architecture Deep Dive](docs/ARCHITECTURE.md) - Understanding the RMLC compiler.
- [Validation & Parity](docs/VALIDATION.md) - Numerical equivalence results.
- [API Reference](https://docs.rs/ferricml) - Comprehensive crate documentation.
- [Examples & Tutorials](examples/) - End-to-end model implementations.

## 🤝 Contributing

We welcome contributions! Whether you're fixing bugs, adding new layers, or optimizing backend kernels, please see our [Contributing Guidelines](docs/CONTRIBUTING.md).

## 📄 License

FerricML is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
