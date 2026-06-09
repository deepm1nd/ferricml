# FerricML User Guide

FerricML is a universal machine learning framework implemented natively in Rust. This guide provides a step-by-step walkthrough for building, training, and exporting an AI model to GGUF format.

## 1. Installation

Add FerricML to your `Cargo.toml`:

```toml
[dependencies]
ferricml = { git = "https://github.com/deepm1nd/ferricml" }
```

## 2. Building a Model

Define your model using the `Module` trait and standard layers.

```rust
use ferricml::prelude::*;

struct MyModel {
    layer1: Linear,
    layer2: Linear,
}

impl Module for MyModel {
    fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.layer1.forward(input);
        let x = relu(&x);
        self.layer2.forward(&x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut p = self.layer1.parameters();
        p.extend(self.layer2.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut p = self.layer1.parameters_mut();
        p.extend(self.layer2.parameters_mut());
        p
    }
}
```

## 3. Training the Model

Set up your training loop with data, loss functions, and optimizers.

```rust
let mut model = MyModel {
    layer1: Linear::new(784, 128),
    layer2: Linear::new(128, 10),
};

let optimizer = SGD { lr: 0.01 };

for epoch in 0..10 {
    for (x, y) in dataloader {
        let output = model.forward(&x);
        let loss = mse_loss(&output, &y);

        // Compute gradients (Autograd)
        // loss.backward();

        // Update weights
        // optimizer.step(model.parameters_mut());
    }
}
```

## 4. Exporting to GGUF

Export your trained model to the popular GGUF format for use in other inference engines.

```rust
use ferricml::serialization::gguf::GGUFWriter;
use std::fs::File;

let file = File::create("model.gguf")?;
let mut writer = GGUFWriter::new(file);

let params = model.parameters();
writer.write_header(params.len() as u32, 0)?;

let mut offset = 0;
for (name, param) in named_params {
    writer.write_tensor(name, param, offset)?;
    offset += (param.shape().numel() * 4) as u64; // for f32
}

for param in params {
    writer.write_tensor_data(param)?;
}
```

## 5. Other Formats

FerricML also supports exporting to SafeTensors and custom binary checkpoints.

```rust
// Export to SafeTensors
// ...
```
