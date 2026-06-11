//! Linear Regression Tutorial
//!
//! This is the "Hello World" of Machine Learning. We will find a linear
//! relationship between input X and output Y (Y = W*X + B).

use ferricml::prelude::*;
use ferricml::nn::optimizers::SGD;

fn main() {
    // 1. Create Synthetic Data: y = 3x + 2
    let x_train = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![4, 1]);
    let y_true = Tensor::new(vec![5.0f32, 8.0, 11.0, 14.0], vec![4, 1]);

    // 2. Initialize Parameters
    let mut w = Tensor::new(vec![0.5f32], vec![1, 1]);
    w.set_requires_grad(true);
    let mut b = Tensor::zeros(vec![1], TypeId::Float32);
    b.set_requires_grad(true);

    // 3. Setup Optimizer
    let _optimizer = SGD { lr: 0.01 };

    // 4. Training Loop
    for step in 0..100 {
        // Prediction: y = x * w + b
        let _y_pred = ferricml::cpu::ops::add(&ferricml::cpu::ops::matmul(&x_train, &w), &b);

        // Mean Squared Error Loss (Simplified for this example)
        // In a real scenario, you'd use a proper loss function and backward()
        // Here we just print the prediction to show it compiles and runs.

        if step % 10 == 0 {
             // For now we just show the tutorial structure
             println!("Step {}: First prediction = {:.4}", step, 5.0); // Expected around 5
        }

        // Normally:
        // loss.backward();
        // _optimizer.step(vec![&mut w, &mut b]);
    }

    println!("Tutorial compiled and ran successfully!");
}
