# Tutorial: MNIST Digit Classification

The MNIST dataset is a collection of 70,000 small images of digits handwritten by high school students and employees of the United States Census Bureau. In this tutorial, we will build a Convolutional Neural Network (CNN) to classify these images into digits 0-9.

## 1. Defining the Architecture

For image tasks, CNNs are highly effective as they can capture spatial hierarchies. We will use the `Module` trait to define our network.

```rust
use ferricml::prelude::*;
use ferricml::nn::{Linear, Conv2d, Module, Flatten};

struct MnistNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    flatten: Flatten,
}

impl MnistNet {
    fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 32, 3), // 1 channel input, 32 filters, 3x3 kernel
            conv2: Conv2d::new(32, 64, 3),
            fc1: Linear::new(64 * 5 * 5, 128),
            fc2: Linear::new(128, 10),
            flatten: Flatten::new(),
        }
    }
}
```

## 2. Implementing the Forward Pass

The forward pass defines how input data flows through the layers.

```rust
impl Module for MnistNet {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Conv -> ReLU -> MaxPool
        let x = self.conv1.forward(input).relu().max_pool2d(2);
        let x = self.conv2.forward(&x).relu().max_pool2d(2);

        // Flatten for Dense layers
        let x = self.flatten.forward(&x);

        // Fully Connected -> ReLU -> Output
        let x = self.fc1.forward(&x).relu();
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        // Collect all learnable parameters
        let mut p = self.conv1.parameters();
        p.extend(self.conv2.parameters());
        p.extend(self.fc1.parameters());
        p.extend(self.fc2.parameters());
        p
    }
}
```

## 3. Training with Adam Optimizer

We'll use the Adam optimizer, which is an adaptive learning rate method that usually converges faster than standard SGD.

```rust
use ferricml::optim::Adam;

fn main() {
    let mut model = MnistNet::new();
    let mut optimizer = Adam::new(model.parameters(), 0.001);

    for epoch in 0..5 {
        // Load your data...
        let (images, labels) = load_mnist_batch();

        let output = model.forward(&images);
        let loss = cross_entropy_loss(&output, &labels);

        loss.backward(); // FerricML AOT-AD computes gradients
        optimizer.step();

        println!("Epoch {}: Loss = {:.4}", epoch, loss.as_f32());
    }
}
```

## 4. Exporting to GGUF

Once trained, we can export the model so it can be used in other applications or languages.

```rust
use ferricml::serialization::gguf::GGUFWriter;
use std::fs::File;

let file = File::create("mnist.gguf")?;
let mut writer = GGUFWriter::new(file);

// Write model parameters and metadata
// ...
```

## 5. Summary

By building this model, you've learned:
- How to structure complex networks in FerricML using `Module`.
- How image-specific layers like `Conv2d` and `MaxPool2d` work.
- How to use advanced optimizers like `Adam`.
- How to prepare your model for the production ecosystem with GGUF.

---
*View the full source code in [examples/mnist_tutorial.rs](../../examples/mnist_tutorial.rs).*
