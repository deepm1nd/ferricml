# FerricML Tutorials

Master FerricML through these step-by-step tutorials.

## 🟢 Beginner

### 1. [Linear Regression](../examples/linear_regression.rs)
The "Hello World" of ML. Learn how to create tensors, define a simple linear model, and use Stochastic Gradient Descent (SGD) to fit data.
- **Key Concepts:** Tensors, `requires_grad`, MSE Loss, SGD.

### 2. Basic Tensor Operations
A deep dive into the Tensor API. Learn about reshapes, broadcasts, and common mathematical ops.
- **Status:** [See documentation](TENSORS.md).

## 🟡 Intermediate

### 3. [MNIST Digit Classification](../examples/mnist_tutorial.rs)
Build a Convolutional Neural Network (CNN) to recognize handwritten digits.
- **Key Concepts:** `Module` trait, `Conv2d`, `MaxPool2d`, `CrossEntropy`, `Adam` optimizer.

### 4. Custom Layer Implementation
Learn how to implement your own neural network layers by extending the `Module` trait.

## 🔴 Advanced

### 5. GGUF Model Export
Full lifecycle of a model: Train in FerricML and export to the GGUF format for use in GGUF-compatible inference engines like `llama.cpp`.

### 6. Writing Custom Backend Kernels
Learn how to write and integrate custom hardware kernels for the CUDA or ROCm backends.
