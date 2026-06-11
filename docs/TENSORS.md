# Tensors: The Foundation of FerricML

This guide provides a deep dive into the FerricML Tensor API, covering everything from basic creation to advanced memory management and broadcasting.

## 1. What is a Tensor?

In FerricML, a `Tensor` is a multi-dimensional container of elements of a single data type (typically `f32` or `f16`). Every tensor has a **Shape**, a **Stride**, and a **Storage**.

- **Shape:** The dimensions of the tensor (e.g., `[64, 3, 224, 224]`).
- **Storage:** The raw buffer on the Host (CPU) or Device (GPU).
- **Stride:** The number of elements to skip in memory to reach the next element in each dimension.

## 2. Tensor Creation

### 2.1 From Existing Data
```rust
// From a standard Rust slice
let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

// From a Vec (takes ownership)
let y = Tensor::from_vec(vec![0.0; 100], &[10, 10]);
```

### 2.2 Factory Methods
```rust
let ones = Tensor::ones(&[3, 3]);
let zeros = Tensor::zeros(&[5, 5]);
let eye = Tensor::eye(4); // 4x4 Identity matrix
let arange = Tensor::arange(0, 10, 1); // [0, 1, 2, ..., 9]
```

### 2.3 Random Initialization
FerricML provides robust random initialization routines:
```rust
let rand_uniform = Tensor::rand(&[2, 2]); // [0, 1)
let rand_normal = Tensor::randn(&[2, 2]); // Mean 0, Std 1
```

## 3. Indexing and Slicing

FerricML uses a powerful slicing syntax that minimizes data copying.

```rust
let x = Tensor::randn(&[10, 20]);

// Select the first 5 rows and all columns
let top_half = x.slice(0, 0, 5, 1);

// Select a single row (returns a view)
let row_0 = x.get_dim(0, 0);
```

## 4. Broadcasting Rules

FerricML follows NumPy-style broadcasting, allowing operations between tensors of different shapes if they are compatible. Two dimensions are compatible when:
1. They are equal.
2. One of them is 1.

```rust
let a = Tensor::ones(&[3, 1]);
let b = Tensor::ones(&[1, 5]);
let c = &a + &b; // Result shape is [3, 5]
```

## 5. Mathematical Operations

### 5.1 Unary Operations (Element-wise)
```rust
let y = x.exp();
let y = x.log();
let y = x.sin();
let y = x.relu();
let y = x.abs();
```

### 5.2 Binary Operations
```rust
let sum = &a + &b;
let diff = &a - &b;
let prod = &a * &b;
let div = &a / &b;
```

### 5.3 Reductions
```rust
let total_sum = x.sum();
let mean_val = x.mean();
let max_val = x.max();
let argmax = x.argmax(1); // Argmax along dimension 1
```

### 5.4 Linear Algebra
```rust
let z = x.matmul(&w); // Matrix multiplication
let z = x.dot(&y);    // Dot product for vectors
```

## 6. Memory Management and Devices

FerricML manages memory across different hardware backends automatically, but you can control where data lives:

```rust
let cpu_tensor = Tensor::ones(&[10, 10]);

// Move to GPU
let gpu_tensor = cpu_tensor.to_device(Device::Cuda(0));

// Move back to CPU
let cpu_copy = gpu_tensor.to_device(Device::Cpu);
```

### In-place Operations
To save memory, use in-place variants when possible:
```rust
x.add_assign(&y); // Equivalent to x += y
```

## 7. Gradients and Autograd

To track a tensor for automatic differentiation:
```rust
let mut w = Tensor::randn(&[10, 1]).requires_grad();

// After a forward pass...
// loss.backward();

// Access the gradient
let grad = w.grad().unwrap();
```

---
*For more advanced usage, check the [MNIST Tutorial](../examples/mnist_tutorial.rs) or the [Architecture Deep Dive](ARCHITECTURE.md).*
