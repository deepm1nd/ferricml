# Tutorial: Linear Regression in FerricML

Linear regression is the fundamental "Hello World" of machine learning. In this tutorial, we will learn how to find the linear relationship between an input $X$ and an output $Y$, modeled as $Y = W \cdot X + B$.

## 1. Prerequisites

Ensure you have FerricML added to your `Cargo.toml`:

```toml
[dependencies]
ferricml = "0.1"
```

## 2. Setting Up Data

We'll generate synthetic data based on the formula $Y = 3X + 2$.

```rust
use ferricml::prelude::*;

fn main() {
    // Input data (4 samples, 1 feature each)
    let x_train = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[4, 1]);

    // True outputs based on y = 3x + 2
    let y_true = Tensor::from_slice(&[5.0, 8.0, 11.0, 14.0]).reshape(&[4, 1]);
}
```

## 3. Initializing Parameters

In machine learning, we start with random guesses for our weight $W$ and bias $B$ and then improve them. We must call `.requires_grad()` to tell FerricML to track gradients for these tensors.

```rust
let mut w = Tensor::randn(&[1, 1]).requires_grad();
let mut b = Tensor::zeros(&[1]).requires_grad();
```

## 4. The Training Loop

We will perform 100 iterations of the following steps:
1. **Prediction:** Calculate $Y_{pred}$ using our current parameters.
2. **Loss Calculation:** Measure how far $Y_{pred}$ is from $Y_{true}$ using Mean Squared Error (MSE).
3. **Backpropagation:** Use FerricML's AOT-AD to calculate gradients.
4. **Optimization:** Update the parameters to reduce the loss.

```rust
use ferricml::optim::SGD;

let mut optimizer = SGD::new(vec![&w, &b], 0.01);

for step in 0..100 {
    // 1. Forward Pass
    let y_pred = x_train.matmul(&w) + &b;

    // 2. Compute Loss: MSE = mean((pred - true)^2)
    let loss = (&y_pred - &y_true).pow(2).mean();

    // 3. Backward Pass
    loss.backward();

    // 4. Update Parameters
    optimizer.step();

    if step % 10 == 0 {
        println!("Step {}: Loss = {:.4}", step, loss.as_f32());
    }
}
```

## 5. Results

After training, our parameters should be close to their true values (W=3, B=2).

```rust
println!("Final Weight: {:.4}", w.as_f32());
println!("Final Bias: {:.4}", b.as_f32());
```

## 6. Why this is better in FerricML

- **Performance:** FerricML's AOT-AD compiles the entire training step into a single optimized kernel, avoiding the overhead of dynamic graph building.
- **Safety:** Rust ensures that we don't have null pointers or memory leaks in our training loop.
- **Predictability:** The memory for $W, B, X,$ and all intermediate gradients is planned at compile-time.

---
*View the full source code in [examples/linear_regression.rs](../../examples/linear_regression.rs).*
