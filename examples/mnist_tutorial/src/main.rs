use ferricml::prelude::*;
use ferricml::nn::{Linear, Conv2d, Module};

struct MnistNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
}

impl MnistNet {
    fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 32, (3, 3), (1, 1), (0, 0)),
            conv2: Conv2d::new(32, 64, (3, 3), (1, 1), (0, 0)),
            fc1: Linear::new(64 * 5 * 5, 128),
            fc2: Linear::new(128, 10),
        }
    }
}

impl Module for MnistNet {
    fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.conv1.forward(input);
        let x = relu(&x);
        let x = ferricml::cpu::ops::max_pool2d(&x, (2, 2), (2, 2));

        let x = self.conv2.forward(&x);
        let x = relu(&x);
        let x = ferricml::cpu::ops::max_pool2d(&x, (2, 2), (2, 2));

        // Simplified flattening
        let x_shape = x.shape().dims();
        let flat_len = x_shape[1] * x_shape[2] * x_shape[3];
        let x_data = match x.storage().as_ref() {
            ferric_core::Storage::Cpu(s) => s.as_slice::<f32>().to_vec()
        };
        let x = Tensor::new(x_data, vec![x_shape[0], flat_len]);

        let x = self.fc1.forward(&x);
        let x = relu(&x);
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut p = Vec::new();
        p.push(&self.conv1.weight);
        if let Some(ref b) = self.conv1.bias { p.push(b); }
        p.push(&self.conv2.weight);
        if let Some(ref b) = self.conv2.bias { p.push(b); }
        p.push(&self.fc1.weight);
        if let Some(ref b) = self.fc1.bias { p.push(b); }
        p.push(&self.fc2.weight);
        if let Some(ref b) = self.fc2.bias { p.push(b); }
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut p = Vec::new();
        p.push(&mut self.conv1.weight);
        if let Some(ref mut b) = self.conv1.bias { p.push(b); }
        p.push(&mut self.conv2.weight);
        if let Some(ref mut b) = self.conv2.bias { p.push(b); }
        p.push(&mut self.fc1.weight);
        if let Some(ref mut b) = self.fc1.bias { p.push(b); }
        p.push(&mut self.fc2.weight);
        if let Some(ref mut b) = self.fc2.bias { p.push(b); }
        p
    }
}

fn main() {
    println!("MNIST Tutorial starting...");
    let model = MnistNet::new();
    let input = Tensor::zeros(vec![1, 1, 28, 28], TypeId::Float32);
    let output = model.forward(&input);
    println!("Output shape: {:?}", output.shape());
    println!("MNIST Tutorial completed successfully!");
}
