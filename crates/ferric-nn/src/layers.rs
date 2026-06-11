use crate::module::Module;
use ferric_core::{Tensor, TypeId};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Tensor::new(
            vec![0.0f32; in_features * out_features],
            vec![in_features, out_features],
        );
        let bias = Some(Tensor::new(vec![0.0f32; out_features], vec![out_features]));
        Self { weight, bias }
    }
}

pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub groups: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let weight = Tensor::new(
            vec![0.0f32; out_channels * (in_channels / 1) * kernel_size.0 * kernel_size.1],
            vec![out_channels, in_channels / 1, kernel_size.0, kernel_size.1],
        );
        let bias = Some(Tensor::new(vec![0.0f32; out_channels], vec![out_channels]));
        Self {
            weight,
            bias,
            stride,
            padding,
            groups: 1,
        }
    }

    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        // Re-allocate weight with correct shape for groups
        let out_channels = self.weight.shape().dims()[0];
        let in_channels_total = self.weight.shape().dims()[1] * 1; // current groups=1
        let k0 = self.weight.shape().dims()[2];
        let k1 = self.weight.shape().dims()[3];
        self.weight = Tensor::zeros(vec![out_channels, in_channels_total / groups, k0, k1], TypeId::Float32);
        self
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = ferric_backend_cpu::ops::conv2d(input, &self.weight, self.stride, self.padding, self.groups);
        if let Some(ref bias) = self.bias {
            let b_storage_arc = bias.storage();
            let b_storage = match b_storage_arc.as_ref() {
                ferric_core::Storage::Cpu(s) => s.as_slice::<f32>(),
            };

            let out_shape = out.shape().dims();
            let batch = out_shape[0];
            let channels = out_shape[1];
            let height = out_shape[2];
            let width = out_shape[3];

            let mut out_data = match out.storage().as_ref() {
                ferric_core::Storage::Cpu(s) => s.as_slice::<f32>().to_vec(),
            };

            for bi in 0..batch {
                for ci in 0..channels {
                    let b_val = b_storage[ci];
                    for hi in 0..height {
                        for wi in 0..width {
                            let idx = bi * (channels * height * width)
                                + ci * (height * width)
                                + hi * width
                                + wi;
                            out_data[idx] += b_val;
                        }
                    }
                }
            }
            out = Tensor::new(out_data, out_shape.to_vec());
        }
        out
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

pub struct ConvTranspose2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl ConvTranspose2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let weight = Tensor::new(
            vec![0.0f32; in_channels * out_channels * kernel_size.0 * kernel_size.1],
            vec![in_channels, out_channels, kernel_size.0, kernel_size.1],
        );
        let bias = Some(Tensor::new(vec![0.0f32; out_channels], vec![out_channels]));
        Self {
            weight,
            bias,
            stride,
            padding,
        }
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = ferric_backend_cpu::ops::conv_transpose2d(
            input,
            &self.weight,
            self.stride,
            self.padding,
        );
        if let Some(ref bias) = self.bias {
            let b_storage_arc = bias.storage();
            let b_storage = match b_storage_arc.as_ref() {
                ferric_core::Storage::Cpu(s) => s.as_slice::<f32>(),
            };

            let out_shape = out.shape().dims();
            let batch = out_shape[0];
            let channels = out_shape[1];
            let height = out_shape[2];
            let width = out_shape[3];

            let mut out_data = match out.storage().as_ref() {
                ferric_core::Storage::Cpu(s) => s.as_slice::<f32>().to_vec(),
            };

            for bi in 0..batch {
                for ci in 0..channels {
                    let b_val = b_storage[ci];
                    for hi in 0..height {
                        for wi in 0..width {
                            let idx = bi * (channels * height * width)
                                + ci * (height * width)
                                + hi * width
                                + wi;
                            out_data[idx] += b_val;
                        }
                    }
                }
            }
            out = Tensor::new(out_data, out_shape.to_vec());
        }
        out
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        let weight = Tensor::new(vec![1.0f32; dim], vec![dim]);
        Self { weight, eps }
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        ferric_backend_cpu::ops::rms_norm(input, &self.weight, self.eps)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }
}

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(shape: Vec<usize>, eps: f32) -> Self {
        let weight = Tensor::new(vec![1.0f32; shape.iter().product::<usize>()], shape.clone());
        let bias = Tensor::new(vec![0.0f32; shape.iter().product::<usize>()], shape);
        Self { weight, bias, eps }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        ferric_backend_cpu::ops::layer_norm(input, &self.weight, &self.bias, self.eps)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

pub struct BatchNorm2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub eps: f32,
}

impl BatchNorm2d {
    pub fn new(num_features: usize, eps: f32) -> Self {
        let weight = Tensor::new(vec![1.0f32; num_features], vec![num_features]);
        let bias = Tensor::new(vec![0.0f32; num_features], vec![num_features]);
        let running_mean = Tensor::new(vec![0.0f32; num_features], vec![num_features]);
        let running_var = Tensor::new(vec![1.0f32; num_features], vec![num_features]);
        Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        ferric_backend_cpu::ops::batch_norm2d(
            input,
            &self.weight,
            &self.bias,
            &self.running_mean,
            &self.running_var,
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = ferric_backend_cpu::ops::matmul(input, &self.weight);
        if let Some(ref bias) = self.bias {
            out = ferric_backend_cpu::ops::add(&out, bias);
        }
        out
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}
