# Ahead-of-Time Automatic Differentiation (AOT-AD)

Automatic Differentiation is at the heart of FerricML. Our unique **AOT-AD** approach provides the flexibility of dynamic frameworks with the performance of static graph compilers.

## 1. The Core Concept

In most frameworks (like PyTorch), the backward pass is computed by traversing a dynamic "tape" that records operations during the forward pass. While flexible, this introduces significant runtime overhead.

FerricML's RMLC compiler symbolicly differentiates the computational graph **before execution**. This results in a compiled artifact that contains both the forward and backward passes, fully optimized for the target hardware.

## 2. How it Works

The AOT-AD process follows three main stages:

### 2.1 Symbolic Tracing
When you define a model in FerricML, the RMLC compiler traces the data flow using symbolic tensors. This creates a high-level representation of the mathematical functions being computed.

### 2.2 Reverse-Mode Gradient Generation
RMLC applies the rules of calculus (Chain Rule) to this symbolic graph to generate a corresponding gradient graph.

For an operation $y = f(x)$, RMLC generates $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot f'(x)$.

### 2.3 Joint Optimization
This is where AOT-AD shines. Since the compiler sees both passes simultaneously, it can:
- **Reuse Buffers:** Identify intermediate tensors from the forward pass that are no longer needed after their corresponding backward step and reuse that memory immediately.
- **Kernel Fusion:** Fuse the last layers of the forward pass with the first layers of the backward pass into a single execution unit.
- **Dead Gradient Elimination:** Completely skip gradient calculations for tensors that do not influence the final loss.

## 3. Supported Operations

FerricML provides high-performance symbolic derivatives for:
- **Linear Algebra:** Matrix Multiplication, Vector Dots.
- **Convolutions:** Standard, Dilated, and Transposed convolutions.
- **Activations:** ReLU, Sigmoid, Tanh, Softmax.
- **Reductions:** Sum, Mean, Variance, Max/Min.
- **Normalization:** BatchNorm, LayerNorm, GroupNorm.

## 4. Usage in FerricML

Tracking gradients is simple:

```rust
// 1. Mark tensors that need gradients
let w = Tensor::randn(&[10, 10]).requires_grad();

// 2. Perform forward pass
let y = x.matmul(&w);
let loss = y.mse_loss(&target);

// 3. Trigger AOT-AD
loss.backward();

// 4. Access gradients
let w_grad = w.grad().expect("Gradient should be computed");
```

## 5. Mathematical Stability

To ensure robust training, our AOT-AD implementation includes:
- **Numerical epsilon injection** to prevent division by zero in gradients.
- **Log-space computations** for probability-related gradients to avoid underflow.
- **Safe handling of non-differentiable points** (e.g., at $x=0$ in ReLU).

## 6. Advanced: Custom Gradients

For researchers, FerricML allows you to define custom symbolic derivatives for your own operations, which the RMLC compiler will then optimize alongside the built-in ones.

---
*Next: Explore the [Mathematics of FerricML](MATHEMATICS.md).*
