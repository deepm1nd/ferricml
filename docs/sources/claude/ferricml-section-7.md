# Section 7: Neural Network Modules

**FerricML Architecture Specification v3.0**  
**Word Count:** 3,500+ words  
**Implementation Priority:** Phase 3 - High-Level API

---

## Table of Contents

1. [Module System Design](#1-module-system-design)
2. [Core Layers](#2-core-layers)
3. [Activation Functions](#3-activation-functions)
4. [Normalization Layers](#4-normalization-layers)
5. [Transformer Components](#5-transformer-components)
6. [Sequential and Complex Models](#6-sequential-and-complex-models)
7. [Parameter Initialization](#7-parameter-initialization)
8. [Custom Layer Development](#8-custom-layer-development)

---

## 1. Module System Design

### 1.1 Module Trait

```rust
pub trait Module: Send + Sync {
    /// Input type
    type Input;
    
    /// Output type
    type Output;
    
    /// Forward pass
    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
    
    /// Get all parameters that require gradients
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        vec![]
    }
    
    /// Get mutable references to parameters
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![]
    }
    
    /// Get all trainable parameters (name, tensor)
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![]
    }
    
    /// Set training mode
    fn train(&mut self) {
        // Override in subclasses that have different train/eval behavior
    }
    
    /// Set evaluation mode
    fn eval(&mut self) {
        // Override in subclasses
    }
    
    /// Move module to device
    fn to(&mut self, device: Device) -> Result<()> {
        for param in self.parameters_mut() {
            *param = param.to(device.clone())?;
        }
        Ok(())
    }
    
    /// Zero gradients of all parameters
    fn zero_grad(&mut self) {
        for param in self.parameters_mut() {
            if let Some(grad_info) = &param.grad_info {
                grad_info.lock().unwrap().zero_grad();
            }
        }
    }
}

/// Module state (training vs evaluation)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleMode {
    Train,
    Eval,
}

/// Helper macro for implementing Module for simple layers
#[macro_export]
macro_rules! impl_module {
    ($type:ty, $input:ty, $output:ty) => {
        impl Module for $type {
            type Input = $input;
            type Output = $output;
            
            fn forward(&self, input: Self::Input) -> Result<Self::Output> {
                self.forward_impl(input)
            }
        }
    };
}
```

### 1.2 Parameter Management

```rust
pub struct Parameter {
    /// The actual tensor
    data: Tensor<f32>,
    
    /// Whether this parameter requires gradient
    requires_grad: bool,
    
    /// Parameter name
    name: String,
}

impl Parameter {
    pub fn new(data: Tensor<f32>, name: impl Into<String>) -> Self {
        Self {
            data: data.requires_grad(true),
            requires_grad: true,
            name: name.into(),
        }
    }
    
    pub fn from_shape(shape: &[usize], name: impl Into<String>) -> Self {
        let data = Tensor::zeros(shape).requires_grad(true);
        Self {
            data,
            requires_grad: true,
            name: name.into(),
        }
    }
    
    pub fn data(&self) -> &Tensor<f32> {
        &self.data
    }
    
    pub fn data_mut(&mut self) -> &mut Tensor<f32> {
        &mut self.data
    }
    
    pub fn grad(&self) -> Option<Tensor<f32>> {
        self.data.grad()
    }
    
    pub fn zero_grad(&mut self) {
        if let Some(grad_info) = &self.data.grad_info {
            grad_info.lock().unwrap().zero_grad();
        }
    }
}

impl Deref for Parameter {
    type Target = Tensor<f32>;
    
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for Parameter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
```

---

## 2. Core Layers

### 2.1 Linear Layer

```rust
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    weight: Parameter,
    
    /// Bias vector [out_features] (optional)
    bias: Option<Parameter>,
    
    /// Input features
    in_features: usize,
    
    /// Output features
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = Parameter::from_shape(
            &[out_features, in_features],
            "weight"
        );
        
        let bias = if bias {
            Some(Parameter::from_shape(&[out_features], "bias"))
        } else {
            None
        };
        
        let mut layer = Self {
            weight,
            bias,
            in_features,
            out_features,
        };
        
        // Initialize parameters
        layer.reset_parameters();
        
        layer
    }
    
    fn reset_parameters(&mut self) {
        // Kaiming uniform initialization
        let k = (1.0 / self.in_features as f32).sqrt();
        self.weight.uniform_(-k, k);
        
        if let Some(ref mut bias) = self.bias {
            bias.uniform_(-k, k);
        }
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        // input: [batch, in_features]
        // weight: [out_features, in_features]
        // output: [batch, out_features]
        
        let mut output = input.matmul(&self.weight.transpose(-1, -2)?)?;
        
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.unsqueeze(0)?)?;
        }
        
        Ok(output)
    }
}

impl Module for Linear {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_impl(input)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        let mut params = vec![&*self.weight];
        if let Some(ref bias) = self.bias {
            params.push(&**bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        let mut params = vec![&mut *self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(&mut **bias);
        }
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut params = vec![("weight".to_string(), &*self.weight)];
        if let Some(ref bias) = self.bias {
            params.push(("bias".to_string(), &**bias));
        }
        params
    }
}
```

### 2.2 Convolutional Layers

```rust
pub struct Conv2d {
    /// Weight [out_channels, in_channels, kernel_h, kernel_w]
    weight: Parameter,
    
    /// Bias [out_channels] (optional)
    bias: Option<Parameter>,
    
    /// Number of input channels
    in_channels: usize,
    
    /// Number of output channels
    out_channels: usize,
    
    /// Kernel size (height, width)
    kernel_size: (usize, usize),
    
    /// Stride (height, width)
    stride: (usize, usize),
    
    /// Padding (height, width)
    padding: (usize, usize),
    
    /// Dilation (height, width)
    dilation: (usize, usize),
    
    /// Number of groups for grouped convolution
    groups: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self {
        let weight = Parameter::from_shape(
            &[out_channels, in_channels, kernel_size.0, kernel_size.1],
            "weight"
        );
        
        let bias = if bias {
            Some(Parameter::from_shape(&[out_channels], "bias"))
        } else {
            None
        };
        
        let mut layer = Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation: (1, 1),
            groups: 1,
        };
        
        layer.reset_parameters();
        
        layer
    }
    
    fn reset_parameters(&mut self) {
        // Kaiming uniform for conv
        let k = 1.0 / (self.in_channels * self.kernel_size.0 * self.kernel_size.1) as f32;
        let k = k.sqrt();
        
        self.weight.uniform_(-k, k);
        
        if let Some(ref mut bias) = self.bias {
            bias.uniform_(-k, k);
        }
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        // input: [batch, in_channels, height, width]
        // weight: [out_channels, in_channels, kernel_h, kernel_w]
        // output: [batch, out_channels, out_height, out_width]
        
        let mut output = input.conv2d(
            &self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;
        
        if let Some(ref bias) = self.bias {
            // Add bias: [out_channels] -> [1, out_channels, 1, 1]
            let bias_reshaped = bias
                .unsqueeze(0)?
                .unsqueeze(2)?
                .unsqueeze(3)?;
            output = output.add(&bias_reshaped)?;
        }
        
        Ok(output)
    }
}

impl Module for Conv2d {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_impl(input)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        let mut params = vec![&*self.weight];
        if let Some(ref bias) = self.bias {
            params.push(&**bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        let mut params = vec![&mut *self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(&mut **bias);
        }
        params
    }
}

/// Transposed convolution (deconvolution)
pub struct ConvTranspose2d {
    weight: Parameter,
    bias: Option<Parameter>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
}

impl ConvTranspose2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        bias: bool,
    ) -> Self {
        let weight = Parameter::from_shape(
            &[in_channels, out_channels, kernel_size.0, kernel_size.1],
            "weight"
        );
        
        let bias = if bias {
            Some(Parameter::from_shape(&[out_channels], "bias"))
        } else {
            None
        };
        
        let mut layer = Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        };
        
        layer.reset_parameters();
        layer
    }
    
    fn reset_parameters(&mut self) {
        let k = 1.0 / (self.in_channels * self.kernel_size.0 * self.kernel_size.1) as f32;
        let k = k.sqrt();
        
        self.weight.uniform_(-k, k);
        
        if let Some(ref mut bias) = self.bias {
            bias.uniform_(-k, k);
        }
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        input.conv_transpose2d(
            &self.weight,
            self.stride,
            self.padding,
            self.output_padding,
        )
    }
}

impl_module!(ConvTranspose2d, Tensor<f32>, Tensor<f32>);
```

### 2.3 Pooling Layers

```rust
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        input.max_pool2d(self.kernel_size, self.stride, self.padding)
    }
}

impl_module!(MaxPool2d, Tensor<f32>, Tensor<f32>);

pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl AvgPool2d {
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        input.avg_pool2d(self.kernel_size, self.stride, self.padding)
    }
}

impl_module!(AvgPool2d, Tensor<f32>, Tensor<f32>);

pub struct AdaptiveAvgPool2d {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        let input_size = (input.shape()[2], input.shape()[3]);
        
        let stride = (
            input_size.0 / self.output_size.0,
            input_size.1 / self.output_size.1,
        );
        
        let kernel_size = (
            input_size.0 - (self.output_size.0 - 1) * stride.0,
            input_size.1 - (self.output_size.1 - 1) * stride.1,
        );
        
        input.avg_pool2d(kernel_size, stride, (0, 0))
    }
}

impl_module!(AdaptiveAvgPool2d, Tensor<f32>, Tensor<f32>);
```

---

## 3. Activation Functions

### 3.1 Common Activations

```rust
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        input.relu()
    }
}

impl_module!(ReLU, Tensor<f32>, Tensor<f32>);

pub struct LeakyReLU {
    negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        input.leaky_relu(self.negative_slope)
    }
}

impl_module!(LeakyReLU, Tensor<f32>, Tensor<f32>);

pub struct GELU {
    approximate: bool,
}

impl GELU {
    pub fn new(approximate: bool) -> Self {
        Self { approximate }
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        if self.approximate {
            // Approximate: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let x3 = input.pow(3.0)?;
            let inner = input.add(&x3.mul_scalar(0.044715)?)?;
            let inner = inner.mul_scalar((2.0 / std::f32::consts::PI).sqrt())?;
            let tanh_val = inner.tanh()?;
            let one_plus_tanh = tanh_val.add_scalar(1.0)?;
            let half_x = input.mul_scalar(0.5)?;
            
            half_x.mul(&one_plus_tanh)
        } else {
            // Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
            let x_scaled = input.div_scalar(2.0_f32.sqrt())?;
            let erf_val = x_scaled.erf()?;
            let one_plus_erf = erf_val.add_scalar(1.0)?;
            let half_x = input.mul_scalar(0.5)?;
            
            half_x.mul(&one_plus_erf)
        }
    }
}

impl_module!(GELU, Tensor<f32>, Tensor<f32>);

pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        input.sigmoid()
    }
}

impl_module!(Sigmoid, Tensor<f32>, Tensor<f32>);

pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        input.tanh()
    }
}

impl_module!(Tanh, Tensor<f32>, Tensor<f32>);

pub struct Softmax {
    dim: isize,
}

impl Softmax {
    pub fn new(dim: isize) -> Self {
        Self { dim }
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        input.softmax(self.dim)
    }
}

impl_module!(Softmax, Tensor<f32>, Tensor<f32>);
```

---

## 4. Normalization Layers

### 4.1 Batch Normalization

```rust
pub struct BatchNorm2d {
    /// Number of features
    num_features: usize,
    
    /// Learnable scale parameter [num_features]
    gamma: Parameter,
    
    /// Learnable shift parameter [num_features]
    beta: Parameter,
    
    /// Running mean [num_features]
    running_mean: Tensor<f32>,
    
    /// Running variance [num_features]
    running_var: Tensor<f32>,
    
    /// Momentum for running statistics
    momentum: f32,
    
    /// Epsilon for numerical stability
    eps: f32,
    
    /// Current mode
    mode: ModuleMode,
    
    /// Number of batches tracked
    num_batches_tracked: usize,
}

impl BatchNorm2d {
    pub fn new(num_features: usize, eps: f32, momentum: f32) -> Self {
        let gamma = Parameter::from_shape(&[num_features], "gamma");
        let beta = Parameter::from_shape(&[num_features], "beta");
        
        let mut layer = Self {
            num_features,
            gamma,
            beta,
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            momentum,
            eps,
            mode: ModuleMode::Train,
            num_batches_tracked: 0,
        };
        
        // Initialize gamma to 1, beta to 0
        layer.gamma.fill_(1.0);
        layer.beta.fill_(0.0);
        
        layer
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        // input: [batch, channels, height, width]
        
        match self.mode {
            ModuleMode::Train => {
                // Compute batch statistics
                let mean = input.mean(&[0, 2, 3], true)?;  // [1, channels, 1, 1]
                let var = input.var(&[0, 2, 3], true, 0)?;
                
                // Update running statistics
                self.update_running_stats(&mean, &var)?;
                
                // Normalize
                self.normalize(input, &mean, &var)
            }
            ModuleMode::Eval => {
                // Use running statistics
                let mean = self.running_mean.view(&[1, self.num_features, 1, 1])?;
                let var = self.running_var.view(&[1, self.num_features, 1, 1])?;
                
                self.normalize(input, &mean, &var)
            }
        }
    }
    
    fn normalize(&self, input: Tensor<f32>, mean: &Tensor<f32>, var: &Tensor<f32>) -> Result<Tensor<f32>> {
        // (x - mean) / sqrt(var + eps) * gamma + beta
        
        let normalized = input.sub(mean)?;
        let std = var.add_scalar(self.eps)?.sqrt()?;
        let normalized = normalized.div(&std)?;
        
        // Reshape gamma and beta for broadcasting
        let gamma = self.gamma.view(&[1, self.num_features, 1, 1])?;
        let beta = self.beta.view(&[1, self.num_features, 1, 1])?;
        
        let scaled = normalized.mul(&gamma)?;
        scaled.add(&beta)
    }
    
    fn update_running_stats(&self, batch_mean: &Tensor<f32>, batch_var: &Tensor<f32>) -> Result<()> {
        // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        // running_var = (1 - momentum) * running_var + momentum * batch_var
        
        let batch_mean = batch_mean.squeeze(&[0, 2, 3])?;
        let batch_var = batch_var.squeeze(&[0, 2, 3])?;
        
        unsafe {
            // This is mutable access to self in immutable method
            // In production, use interior mutability (RefCell/Mutex)
            let running_mean_ptr = self.running_mean.data_ptr() as *mut f32;
            let running_var_ptr = self.running_var.data_ptr() as *mut f32;
            
            for i in 0..self.num_features {
                let rm = std::ptr::read(running_mean_ptr.add(i));
                let rv = std::ptr::read(running_var_ptr.add(i));
                
                let new_mean = (1.0 - self.momentum) * rm + self.momentum * batch_mean.data()[i];
                let new_var = (1.0 - self.momentum) * rv + self.momentum * batch_var.data()[i];
                
                std::ptr::write(running_mean_ptr.add(i), new_mean);
                std::ptr::write(running_var_ptr.add(i), new_var);
            }
        }
        
        Ok(())
    }
}

impl Module for BatchNorm2d {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_impl(input)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        vec![&*self.gamma, &*self.beta]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![&mut *self.gamma, &mut *self.beta]
    }
    
    fn train(&mut self) {
        self.mode = ModuleMode::Train;
    }
    
    fn eval(&mut self) {
        self.mode = ModuleMode::Eval;
    }
}
```

### 4.2 Layer Normalization

```rust
pub struct LayerNorm {
    /// Normalized shape
    normalized_shape: Vec<usize>,
    
    /// Learnable scale
    gamma: Parameter,
    
    /// Learnable bias
    beta: Parameter,
    
    /// Epsilon
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Self {
        let gamma = Parameter::from_shape(&normalized_shape, "gamma");
        let beta = Parameter::from_shape(&normalized_shape, "beta");
        
        let mut layer = Self {
            normalized_shape,
            gamma,
            beta,
            eps,
        };
        
        layer.gamma.fill_(1.0);
        layer.beta.fill_(0.0);
        
        layer
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        // Normalize over last normalized_shape.len() dimensions
        let ndim = input.ndim();
        let norm_dims = &self.normalized_shape;
        let norm_ndim = norm_dims.len();
        
        let axes: Vec<isize> = ((ndim - norm_ndim) as isize..ndim as isize).collect();
        
        let mean = input.mean(&axes, true)?;
        let var = input.var(&axes, true, 0)?;
        
        let normalized = input.sub(&mean)?;
        let std = var.add_scalar(self.eps)?.sqrt()?;
        let normalized = normalized.div(&std)?;
        
        let scaled = normalized.mul(&self.gamma)?;
        scaled.add(&self.beta)
    }
}

impl Module for LayerNorm {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_impl(input)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        vec![&*self.gamma, &*self.beta]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![&mut *self.gamma, &mut *self.beta]
    }
}
```

---

## 5. Transformer Components

### 5.1 Multi-Head Attention

```rust
pub struct MultiHeadAttention {
    /// Number of attention heads
    num_heads: usize,
    
    /// Embedding dimension
    embed_dim: usize,
    
    /// Dimension per head
    head_dim: usize,
    
    /// Query projection
    q_proj: Linear,
    
    /// Key projection
    k_proj: Linear,
    
    /// Value projection
    v_proj: Linear,
    
    /// Output projection
    out_proj: Linear,
    
    /// Dropout probability
    dropout: f32,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        assert_eq!(embed_dim % num_heads, 0, "embed_dim must be divisible by num_heads");
        
        let head_dim = embed_dim / num_heads;
        
        Self {
            num_heads,
            embed_dim,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim, true),
            k_proj: Linear::new(embed_dim, embed_dim, true),
            v_proj: Linear::new(embed_dim, embed_dim, true),
            out_proj: Linear::new(embed_dim, embed_dim, true),
            dropout,
        }
    }
    
    fn forward_impl(
        &self,
        query: Tensor<f32>,
        key: Tensor<f32>,
        value: Tensor<f32>,
        mask: Option<Tensor<f32>>,
    ) -> Result<Tensor<f32>> {
        // query, key, value: [batch, seq_len, embed_dim]
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];
        
        // Project and reshape to [batch, num_heads, seq_len, head_dim]
        let q = self.q_proj.forward(query)?
            .view(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        
        let k = self.k_proj.forward(key)?
            .view(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        
        let v = self.v_proj.forward(value)?
            .view(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        
        // Scaled dot-product attention
        let attn_output = self.scaled_dot_product_attention(q, k, v, mask)?;
        
        // Reshape back to [batch, seq_len, embed_dim]
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .view(&[batch_size, seq_len, self.embed_dim])?;
        
        // Final projection
        self.out_proj.forward(attn_output)
    }
    
    fn scaled_dot_product_attention(
        &self,
        q: Tensor<f32>,
        k: Tensor<f32>,
        v: Tensor<f32>,
        mask: Option<Tensor<f32>>,
    ) -> Result<Tensor<f32>> {
        // q, k, v: [batch, num_heads, seq_len, head_dim]
        
        // Compute attention scores: Q @ K^T / sqrt(head_dim)
        let scores = q.matmul(&k.transpose(-2, -1)?)?;
        let scale = (self.head_dim as f32).sqrt();
        let scores = scores.div_scalar(scale)?;
        
        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            // mask: [batch, 1, seq_len, seq_len] or [seq_len, seq_len]
            let mask_value = -1e9_f32;
            let mask_expanded = mask.mul_scalar(mask_value)?;
            scores.add(&mask_expanded)?
        } else {
            scores
        };
        
        // Softmax
        let attn_weights = scores.softmax(-1)?;
        
        // Apply dropout (in training mode)
        let attn_weights = if self.dropout > 0.0 {
            attn_weights.dropout(self.dropout)?
        } else {
            attn_weights
        };
        
        // Apply attention to values
        attn_weights.matmul(&v)
    }
}

impl Module for MultiHeadAttention {
    type Input = (Tensor<f32>, Tensor<f32>, Tensor<f32>, Option<Tensor<f32>>);
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (query, key, value, mask) = input;
        self.forward_impl(query, key, value, mask)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters_mut());
        params.extend(self.k_proj.parameters_mut());
        params.extend(self.v_proj.parameters_mut());
        params.extend(self.out_proj.parameters_mut());
        params
    }
}
```

### 5.2 Transformer Encoder Layer

```rust
pub struct TransformerEncoderLayer {
    /// Multi-head self-attention
    self_attn: MultiHeadAttention,
    
    /// Feed-forward network
    ffn: FeedForward,
    
    /// Layer normalization 1
    norm1: LayerNorm,
    
    /// Layer normalization 2
    norm2: LayerNorm,
    
    /// Dropout
    dropout: f32,
}

impl TransformerEncoderLayer {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        dropout: f32,
    ) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(embed_dim, num_heads, dropout),
            ffn: FeedForward::new(embed_dim, ffn_dim, dropout),
            norm1: LayerNorm::new(vec![embed_dim], 1e-5),
            norm2: LayerNorm::new(vec![embed_dim], 1e-5),
            dropout,
        }
    }
    
    fn forward_impl(&self, x: Tensor<f32>, mask: Option<Tensor<f32>>) -> Result<Tensor<f32>> {
        // Self-attention with residual connection
        let attn_output = self.self_attn.forward((
            x.clone(),
            x.clone(),
            x.clone(),
            mask,
        ))?;
        
        let attn_output = if self.dropout > 0.0 {
            attn_output.dropout(self.dropout)?
        } else {
            attn_output
        };
        
        let x = x.add(&attn_output)?;
        let x = self.norm1.forward(x)?;
        
        // Feed-forward with residual connection
        let ffn_output = self.ffn.forward(x.clone())?;
        
        let ffn_output = if self.dropout > 0.0 {
            ffn_output.dropout(self.dropout)?
        } else {
            ffn_output
        };
        
        let x = x.add(&ffn_output)?;
        self.norm2.forward(x)
    }
}

impl Module for TransformerEncoderLayer {
    type Input = (Tensor<f32>, Option<Tensor<f32>>);
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (x, mask) = input;
        self.forward_impl(x, mask)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.ffn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters_mut());
        params.extend(self.ffn.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params
    }
}

pub struct FeedForward {
    fc1: Linear,
    fc2: Linear,
    activation: GELU,
    dropout: f32,
}

impl FeedForward {
    pub fn new(embed_dim: usize, ffn_dim: usize, dropout: f32) -> Self {
        Self {
            fc1: Linear::new(embed_dim, ffn_dim, true),
            fc2: Linear::new(ffn_dim, embed_dim, true),
            activation: GELU::new(true),
            dropout,
        }
    }
    
    fn forward_impl(&self, x: Tensor<f32>) -> Result<Tensor<f32>> {
        let x = self.fc1.forward(x)?;
        let x = self.activation.forward(x)?;
        let x = if self.dropout > 0.0 {
            x.dropout(self.dropout)?
        } else {
            x
        };
        self.fc2.forward(x)
    }
}

impl Module for FeedForward {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_impl(input)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters_mut());
        params.extend(self.fc2.parameters_mut());
        params
    }
}
```

---

## 6. Sequential and Complex Models

### 6.1 Sequential Container

```rust
pub struct Sequential {
    layers: Vec<Box<dyn Module<Input = Tensor<f32>, Output = Tensor<f32>>>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
        }
    }
    
    pub fn add<M>(mut self, module: M) -> Self
    where
        M: Module<Input = Tensor<f32>, Output = Tensor<f32>> + 'static,
    {
        self.layers.push(Box::new(module));
        self
    }
    
    fn forward_impl(&self, mut input: Tensor<f32>) -> Result<Tensor<f32>> {
        for layer in &self.layers {
            input = layer.forward(input)?;
        }
        Ok(input)
    }
}

impl Module for Sequential {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_impl(input)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        self.layers.iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        self.layers.iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }
    
    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }
    
    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}

/// Example: ResNet-style block
pub struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<Sequential>,
}

impl ResidualBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let downsample = if stride != 1 || in_channels != out_channels {
            Some(Sequential::new()
                .add(Conv2d::new(in_channels, out_channels, (1, 1), (stride, stride), (0, 0), false))
                .add(BatchNorm2d::new(out_channels, 1e-5, 0.1)))
        } else {
            None
        };
        
        Self {
            conv1: Conv2d::new(in_channels, out_channels, (3, 3), (stride, stride), (1, 1), false),
            bn1: BatchNorm2d::new(out_channels, 1e-5, 0.1),
            relu: ReLU::new(),
            conv2: Conv2d::new(out_channels, out_channels, (3, 3), (1, 1), (1, 1), false),
            bn2: BatchNorm2d::new(out_channels, 1e-5, 0.1),
            downsample,
        }
    }
    
    fn forward_impl(&self, x: Tensor<f32>) -> Result<Tensor<f32>> {
        let identity = x.clone();
        
        let out = self.conv1.forward(x)?;
        let out = self.bn1.forward(out)?;
        let out = self.relu.forward(out)?;
        
        let out = self.conv2.forward(out)?;
        let out = self.bn2.forward(out)?;
        
        let identity = if let Some(ref downsample) = self.downsample {
            downsample.forward(identity)?
        } else {
            identity
        };
        
        let out = out.add(&identity)?;
        self.relu.forward(out)
    }
}

impl Module for ResidualBlock {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_impl(input)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some(ref ds) = self.downsample {
            params.extend(ds.parameters());
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.bn1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        params.extend(self.bn2.parameters_mut());
        if let Some(ref mut ds) = self.downsample {
            params.extend(ds.parameters_mut());
        }
        params
    }
    
    fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
        if let Some(ref mut ds) = self.downsample {
            ds.train();
        }
    }
    
    fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
        if let Some(ref mut ds) = self.downsample {
            ds.eval();
        }
    }
}
```

### 6.2 Complex Model Example

```rust
/// Complete ResNet-18 implementation
pub struct ResNet18 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,
    layer1: Sequential,
    layer2: Sequential,
    layer3: Sequential,
    layer4: Sequential,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear,
}

impl ResNet18 {
    pub fn new(num_classes: usize) -> Self {
        Self {
            conv1: Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), false),
            bn1: BatchNorm2d::new(64, 1e-5, 0.1),
            relu: ReLU::new(),
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            layer1: Self::make_layer(64, 64, 2, 1),
            layer2: Self::make_layer(64, 128, 2, 2),
            layer3: Self::make_layer(128, 256, 2, 2),
            layer4: Self::make_layer(256, 512, 2, 2),
            avgpool: AdaptiveAvgPool2d::new((1, 1)),
            fc: Linear::new(512, num_classes, true),
        }
    }
    
    fn make_layer(in_channels: usize, out_channels: usize, num_blocks: usize, stride: usize) -> Sequential {
        let mut layers = Sequential::new();
        
        // First block with stride
        layers = layers.add(ResidualBlock::new(in_channels, out_channels, stride));
        
        // Remaining blocks
        for _ in 1..num_blocks {
            layers = layers.add(ResidualBlock::new(out_channels, out_channels, 1));
        }
        
        layers
    }
    
    fn forward_impl(&self, x: Tensor<f32>) -> Result<Tensor<f32>> {
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward(x)?;
        let x = self.relu.forward(x)?;
        let x = self.maxpool.forward(x)?;
        
        let x = self.layer1.forward(x)?;
        let x = self.layer2.forward(x)?;
        let x = self.layer3.forward(x)?;
        let x = self.layer4.forward(x)?;
        
        let x = self.avgpool.forward(x)?;
        let x = x.view(&[x.shape()[0], -1])?;  // Flatten
        
        self.fc.forward(x)
    }
}

impl Module for ResNet18 {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_impl(input)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.layer1.parameters());
        params.extend(self.layer2.parameters());
        params.extend(self.layer3.parameters());
        params.extend(self.layer4.parameters());
        params.extend(self.fc.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.bn1.parameters_mut());
        params.extend(self.layer1.parameters_mut());
        params.extend(self.layer2.parameters_mut());
        params.extend(self.layer3.parameters_mut());
        params.extend(self.layer4.parameters_mut());
        params.extend(self.fc.parameters_mut());
        params
    }
    
    fn train(&mut self) {
        self.bn1.train();
        self.layer1.train();
        self.layer2.train();
        self.layer3.train();
        self.layer4.train();
    }
    
    fn eval(&mut self) {
        self.bn1.eval();
        self.layer1.eval();
        self.layer2.eval();
        self.layer3.eval();
        self.layer4.eval();
    }
}
```

---

## 7. Parameter Initialization

### 7.1 Initialization Strategies

```rust
pub mod init {
    use super::*;
    
    /// Xavier/Glorot uniform initialization
    pub fn xavier_uniform(tensor: &mut Tensor<f32>) {
        let fan_in = tensor.shape()[1];
        let fan_out = tensor.shape()[0];
        let bound = ((6.0 / (fan_in + fan_out) as f32)).sqrt();
        tensor.uniform_(-bound, bound);
    }
    
    /// Xavier/Glorot normal initialization
    pub fn xavier_normal(tensor: &mut Tensor<f32>) {
        let fan_in = tensor.shape()[1];
        let fan_out = tensor.shape()[0];
        let std = ((2.0 / (fan_in + fan_out) as f32)).sqrt();
        tensor.normal_(0.0, std);
    }
    
    /// Kaiming/He uniform initialization
    pub fn kaiming_uniform(tensor: &mut Tensor<f32>, nonlinearity: &str) {
        let fan_in = tensor.shape()[1];
        let gain = calculate_gain(nonlinearity);
        let bound = gain * ((3.0 / fan_in as f32).sqrt());
        tensor.uniform_(-bound, bound);
    }
    
    /// Kaiming/He normal initialization
    pub fn kaiming_normal(tensor: &mut Tensor<f32>, nonlinearity: &str) {
        let fan_in = tensor.shape()[1];
        let gain = calculate_gain(nonlinearity);
        let std = gain / (fan_in as f32).sqrt();
        tensor.normal_(0.0, std);
    }
    
    /// Orthogonal initialization
    pub fn orthogonal(tensor: &mut Tensor<f32>, gain: f32) {
        let shape = tensor.shape();
        let rows = shape[0];
        let cols = shape[1];
        
        // Generate random matrix
        let mut flat = Tensor::randn(&[rows * cols]);
        
        // QR decomposition
        let (q, _r) = qr_decomposition(&flat.view(&[rows, cols]).unwrap());
        
        // Scale by gain
        let q = q.mul_scalar(gain).unwrap();
        
        // Copy to tensor
        tensor.copy_(&q);
    }
    
    fn calculate_gain(nonlinearity: &str) -> f32 {
        match nonlinearity {
            "linear" => 1.0,
            "sigmoid" => 1.0,
            "tanh" => 5.0 / 3.0,
            "relu" => (2.0_f32).sqrt(),
            "leaky_relu" => (2.0 / (1.0 + 0.01_f32.powi(2))).sqrt(),
            _ => 1.0,
        }
    }
    
    fn qr_decomposition(matrix: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
        // Simplified QR using Gram-Schmidt
        // Production code would use LAPACK
        unimplemented!("Use LAPACK's DGEQRF")
    }
}

impl Tensor<f32> {
    pub fn uniform_(&mut self, low: f32, high: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for val in self.data_mut() {
            *val = rng.gen_range(low..high);
        }
    }
    
    pub fn normal_(&mut self, mean: f32, std: f32) {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(mean, std).unwrap();
        let mut rng = rand::thread_rng();
        
        for val in self.data_mut() {
            *val = normal.sample(&mut rng);
        }
    }
    
    pub fn fill_(&mut self, value: f32) {
        for val in self.data_mut() {
            *val = value;
        }
    }
}
```

---

## 8. Custom Layer Development

### 8.1 Custom Layer Template

```rust
/// Template for custom layer development
pub struct CustomLayer {
    // Parameters
    weight: Parameter,
    bias: Option<Parameter>,
    
    // Hyperparameters
    some_config: f32,
    
    // State (non-trainable)
    mode: ModuleMode,
}

impl CustomLayer {
    pub fn new(in_features: usize, out_features: usize, some_config: f32) -> Self {
        let weight = Parameter::from_shape(&[out_features, in_features], "weight");
        let bias = Some(Parameter::from_shape(&[out_features], "bias"));
        
        let mut layer = Self {
            weight,
            bias,
            some_config,
            mode: ModuleMode::Train,
        };
        
        // Initialize
        init::kaiming_uniform(&mut layer.weight, "relu");
        if let Some(ref mut bias) = layer.bias {
            bias.fill_(0.0);
        }
        
        layer
    }
    
    fn forward_impl(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        // Custom forward logic
        let output = input.matmul(&self.weight.transpose(-1, -2)?)?;
        
        if let Some(ref bias) = self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }
}

impl Module for CustomLayer {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_impl(input)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        let mut params = vec![&*self.weight];
        if let Some(ref bias) = self.bias {
            params.push(&**bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        let mut params = vec![&mut *self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(&mut **bias);
        }
        params
    }
    
    fn train(&mut self) {
        self.mode = ModuleMode::Train;
    }
    
    fn eval(&mut self) {
        self.mode = ModuleMode::Eval;
    }
}
```

### 8.2 Usage Example

```rust
fn train_model() -> Result<()> {
    // Create model
    let mut model = Sequential::new()
        .add(Linear::new(784, 256, true))
        .add(ReLU::new())
        .add(Linear::new(256, 128, true))
        .add(ReLU::new())
        .add(Linear::new(128, 10, true));
    
    // Move to GPU
    model.to(Device::cuda(0)?)?;
    
    // Set training mode
    model.train();
    
    // Training loop
    for epoch in 0..10 {
        for (x, y) in dataloader {
            // Forward pass
            let output = model.forward(x)?;
            
            // Compute loss
            let loss = cross_entropy_loss(&output, &y)?;
            
            // Backward pass
            model.zero_grad();
            loss.backward()?;
            
            // Update parameters
            optimizer.step()?;
        }
    }
    
    // Switch to evaluation mode
    model.eval();
    
    Ok(())
}
```

---

## Summary

This section detailed the neural network module system for FerricML:

**Key Components:**
1. **Module Trait:** Unified interface for all layers
2. **Core Layers:** Linear, Conv2d, pooling with proper initialization
3. **Activations:** ReLU, GELU, Sigmoid with optimized implementations
4. **Normalization:** BatchNorm, LayerNorm with train/eval modes
5. **Transformers:** Multi-head attention and encoder layers
6. **Sequential:** Composable model construction
7. **Initialization:** Xavier, Kaiming, orthogonal strategies
8. **Custom Layers:** Template and guidelines

**Design Decisions:**
- Generic Module trait for composability
- Parameter wrapper for gradient tracking
- Separate train/eval modes for normalization
- Builder pattern for Sequential composition
- Proper initialization prevents gradient issues

**Implementation Priority:**
1. Module trait and Parameter wrapper
2. Core layers (Linear, Conv2d, ReLU)
3. Normalization layers
4. Sequential container
5. Transformer components
6. Advanced architectures (ResNet, etc.)

**Integration Points:**
- Section 1 provides Tensor foundation
- Section 5 autograd enables backpropagation
- Section 3/4 backends execute operations
- Section 10 handles multi-GPU distribution

**Next Steps:**
- Implement remaining sections (8, 9, 10)
- Add more layer types (GRU, LSTM, attention variants)
- Optimize common patterns (fused ops)
- Add model zoo with pretrained weights