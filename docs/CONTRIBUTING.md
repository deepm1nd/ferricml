# Contributing to FerricML

First off, thank you for considering contributing to FerricML! It's people like you who make it a great tool.

## 🌈 Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct (standard Apache/Rust community guidelines).

## 🚀 How Can I Contribute?

### Reporting Bugs
- Use the GitHub Issue Tracker.
- Describe the expected behavior and the actual behavior.
- Provide a minimal reproducible example (code snippet).

### Suggesting Enhancements
- Open an issue with the "feature request" label.
- Explain why the enhancement would be useful and how it fits into the FerricML philosophy (Performance, Safety, Rust-native).

### Pull Requests
1.  **Fork the repo** and create your branch from `main`.
2.  **Add tests!** We strive for high test coverage, especially for new layers or kernels.
3.  **Run validation scripts:** Ensure `./validate_all.sh` passes.
4.  **Format your code:** We use `rustfmt`. Run `cargo fmt` before committing.
5.  **Document your changes:** If you're adding a new public API, add doc comments (`///`).

## 🏗 Repository Structure

- `crates/ferric-core`: The core Tensor engine and symbolic graph.
- `crates/ferric-ir`: Multi-level IR definitions and transformations.
- `crates/ferric-autograd`: AOT Automatic Differentiation logic.
- `crates/ferric-nn`: Neural network layers and modules.
- `crates/ferric-backend-*`: Hardware-specific implementations.
- `crates/ferric-serialization`: GGUF and SafeTensors support.

## 🛠 Development Environment

To set up your environment for development:

```bash
git clone https://github.com/deepm1nd/ferricml.git
cd ferricml
./env_install.sh
cargo build
```

## 🧪 Testing

We use standard Cargo tests. Please ensure all tests pass before submitting a PR:

```bash
cargo test --all-features
```

## 📜 Coding Standards

- **Memory Safety:** Avoid `unsafe` unless absolutely necessary for performance in backend kernels.
- **Performance:** Prefer zero-copy operations and lazy evaluations.
- **Typing:** Leverage Rust's strong type system to catch errors at compile-time (e.g., using newtypes for Dimensions).

## 💎 Getting Help

If you have questions, feel free to open a "Question" issue or join our community discussions.

Happy coding!
