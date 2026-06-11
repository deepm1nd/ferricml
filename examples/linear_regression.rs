//! Linear Regression Tutorial
//!
//! This is the "Hello World" of Machine Learning. We will find a linear
//! relationship between input X and output Y (Y = W*X + B).

use ferricml::prelude::*;
use ferricml::optim::SGD;

fn main() {
    // 1. Create Synthetic Data: y = 3x + 2
    let x_train = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[4, 1]);
    let y_true = Tensor::from_slice(&[5.0, 8.0, 11.0, 14.0]).reshape(&[4, 1]);

    // 2. Initialize Parameters
    let mut w = Tensor::randn(&[1, 1]).requires_grad();
    let mut b = Tensor::zeros(&[1]).requires_grad();

    // 3. Setup Optimizer
    let mut optimizer = SGD::new(vec![&w, &b], 0.01);

    // 4. Training Loop
    for step in 0..100 {
        // Prediction
        let y_pred = x_train.matmul(&w) + &b;

        // Mean Squared Error Loss
        let loss = (&y_pred - &y_true).pow(2).mean();

        // Autograd
        loss.backward();

        // Optimizer Step
        optimizer.step();

        if step % 10 == 0 {
            println!("Step {}: Loss = {:.4}", step, loss.as_f32());
        }
    }

    println!("Final Weight: {:.4} (Expected 3.0)", w.as_f32());
    println!("Final Bias: {:.4} (Expected 2.0)", b.as_f32());
}
