# Section 5: Automatic Differentiation Engine

**FerricML Architecture Specification v3.0**  
**Word Count:** 3,400+ words  
**Implementation Priority:** Phase 1 - Critical Foundation

---

## Table of Contents

1. [Autograd Architecture](#1-autograd-architecture)
2. [Computation Graph](#2-computation-graph)
3. [Reverse-Mode Differentiation](#3-reverse-mode-differentiation)
4. [Gradient Functions](#4-gradient-functions)
5. [Memory-Efficient Backprop](#5-memory-efficient-backprop)
6. [Higher-Order Derivatives](#6-higher-order-derivatives)
7. [Custom Gradients](#7-custom-gradients)
8. [Static vs Dynamic Graphs](#8-static-vs-dynamic-graphs)

---

## 1. Autograd Architecture

### 1.1 Design Overview

FerricML implements reverse-mode automatic differentiation (backpropagation) through computation graph construction:

```rust
pub struct AutogradEngine {
    /// Current execution mode
    mode: Arc<AtomicCell<GradMode>>,
    
    /// Tape for recording operations (dynamic mode)
    tape: Arc<Mutex<ComputationTape>>,
    
    /// Graph for static mode
    static_graph: Option<Arc<StaticGraph>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradMode {
    /// Gradients computed and tracked
    Enabled,
    
    /// No gradient tracking (inference mode)
    Disabled,
}

impl AutogradEngine {
    pub fn new() -> Self {
        Self {
            mode: Arc::new(AtomicCell::new(GradMode::Enabled)),
            tape: Arc::new(Mutex::new(ComputationTape::new())),
            static_graph: None,
        }
    }
    
    pub fn set_grad_mode(&self, mode: GradMode) {
        self.mode.store(mode);
    }
    
    pub fn is_grad_enabled(&self) -> bool {
        self.mode.load() == GradMode::Enabled
    }
    
    /// Execute forward pass and record operations
    pub fn forward<T: DType>(&self, op: Operation, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        // Execute operation
        let output = self.execute_operation(op, inputs)?;
        
        // Record in tape if gradients enabled
        if self.is_grad_enabled() && inputs.iter().any(|t| t.requires_grad()) {
            let mut tape = self.tape.lock().unwrap();
            tape.record(op, inputs, &output);
        }
        
        Ok(output)
    }
    
    /// Compute gradients via backpropagation
    pub fn backward<T: DType>(&self, loss: &Tensor<T>) -> Result<()> {
        if !self.is_grad_enabled() {
            return Err(Error::GradientsDisabled);
        }
        
        let tape = self.tape.lock().unwrap();
        let grad_tape = tape.build_gradient_tape(loss)?;
        
        // Execute backward pass
        grad_tape.execute()?;
        
        Ok(())
    }
}
```

### 1.2 Gradient Tracking Information

```rust
pub struct GradInfo {
    /// Gradient accumulator
    grad: Arc<Mutex<Option<Tensor<f32>>>>,
    
    /// Gradient function for this operation
    grad_fn: Option<Arc<dyn GradientFunction>>,
    
    /// Reference count for gradient computation
    grad_users: AtomicUsize,
    
    /// Whether this tensor requires gradient
    requires_grad: bool,
}

impl GradInfo {
    pub fn new() -> Self {
        Self {
            grad: Arc::new(Mutex::new(None)),
            grad_fn: None,
            grad_users: AtomicUsize::new(0),
            requires_grad: true,
        }
    }
    
    pub fn accumulate_grad(&self, incoming_grad: Tensor<f32>) -> Result<()> {
        let mut grad = self.grad.lock().unwrap();
        
        if let Some(existing_grad) = grad.as_mut() {
            // Accumulate: grad += incoming_grad
            *existing_grad = existing_grad.add(&incoming_grad)?;
        } else {
            *grad = Some(incoming_grad);
        }
        
        Ok(())
    }
    
    pub fn get_grad(&self) -> Option<Tensor<f32>> {
        self.grad.lock().unwrap().clone()
    }
    
    pub fn zero_grad(&self) {
        *self.grad.lock().unwrap() = None;
    }
}

impl<T: DType> Tensor<T> {
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        if requires_grad {
            self.grad_info = Some(Arc::new(Mutex::new(GradInfo::new())));
        } else {
            self.grad_info = None;
        }
        self
    }
    
    pub fn requires_grad(&self) -> bool {
        self.grad_info.is_some()
    }
    
    pub fn grad(&self) -> Option<Tensor<f32>> {
        self.grad_info.as_ref()?.lock().unwrap().get_grad()
    }
    
    pub fn backward(&self) -> Result<()> {
        let engine = AutogradEngine::global();
        engine.backward(self)
    }
}
```

---

## 2. Computation Graph

### 2.1 Tape-Based Recording

```rust
pub struct ComputationTape {
    /// Recorded operations
    operations: Vec<TapeEntry>,
    
    /// Map from tensor ID to tape index
    tensor_to_op: HashMap<TensorId, usize>,
}

pub struct TapeEntry {
    /// Operation that produced this result
    op: Operation,
    
    /// Input tensor IDs
    inputs: Vec<TensorId>,
    
    /// Output tensor ID
    output: TensorId,
    
    /// Gradient function
    grad_fn: Arc<dyn GradientFunction>,
    
    /// Saved tensors for backward pass
    saved_tensors: Vec<SavedTensor>,
}

pub enum SavedTensor {
    /// Full tensor saved
    Tensor(Tensor<f32>),
    
    /// Only shape saved (recompute in backward)
    Shape(Shape),
    
    /// Scalar value
    Scalar(f64),
}

impl ComputationTape {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            tensor_to_op: HashMap::new(),
        }
    }
    
    pub fn record<T: DType>(
        &mut self,
        op: Operation,
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) {
        let input_ids: Vec<TensorId> = inputs.iter().map(|t| t.id).collect();
        
        let grad_fn = Self::get_gradient_function(&op);
        let saved_tensors = grad_fn.save_for_backward(inputs, output);
        
        let entry = TapeEntry {
            op: op.clone(),
            inputs: input_ids,
            output: output.id,
            grad_fn,
            saved_tensors,
        };
        
        let idx = self.operations.len();
        self.operations.push(entry);
        self.tensor_to_op.insert(output.id, idx);
    }
    
    pub fn build_gradient_tape(&self, loss: &Tensor<f32>) -> Result<GradientTape> {
        // Topological sort for backward pass order
        let backward_order = self.topological_sort(loss.id)?;
        
        Ok(GradientTape {
            entries: backward_order,
            gradients: HashMap::new(),
        })
    }
    
    fn topological_sort(&self, start: TensorId) -> Result<Vec<&TapeEntry>> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_mark = HashSet::new();
        
        self.visit(start, &mut visited, &mut temp_mark, &mut order)?;
        
        order.reverse();  // Reverse for backward pass
        Ok(order)
    }
    
    fn visit(
        &self,
        tensor_id: TensorId,
        visited: &mut HashSet<TensorId>,
        temp_mark: &mut HashSet<TensorId>,
        order: &mut Vec<&TapeEntry>,
    ) -> Result<()> {
        if visited.contains(&tensor_id) {
            return Ok(());
        }
        
        if temp_mark.contains(&tensor_id) {
            return Err(Error::CyclicGraph);
        }
        
        temp_mark.insert(tensor_id);
        
        if let Some(&op_idx) = self.tensor_to_op.get(&tensor_id) {
            let entry = &self.operations[op_idx];
            
            for &input_id in &entry.inputs {
                self.visit(input_id, visited, temp_mark, order)?;
            }
            
            order.push(entry);
        }
        
        temp_mark.remove(&tensor_id);
        visited.insert(tensor_id);
        
        Ok(())
    }
}
```

### 2.2 Gradient Tape Execution

```rust
pub struct GradientTape {
    entries: Vec<TapeEntry>,
    gradients: HashMap<TensorId, Tensor<f32>>,
}

impl GradientTape {
    pub fn execute(&mut self) -> Result<()> {
        // Initialize gradient of loss to 1.0
        if let Some(entry) = self.entries.first() {
            let loss_id = entry.output;
            let loss_shape = entry.saved_tensors[0].shape();
            self.gradients.insert(loss_id, Tensor::ones(&loss_shape));
        }
        
        // Execute backward pass in reverse topological order
        for entry in &self.entries {
            let output_grad = self.gradients.get(&entry.output)
                .ok_or(Error::MissingGradient)?
                .clone();
            
            // Compute input gradients
            let input_grads = entry.grad_fn.backward(
                &output_grad,
                &entry.saved_tensors,
            )?;
            
            // Accumulate gradients for inputs
            for (input_id, grad) in entry.inputs.iter().zip(input_grads) {
                if let Some(grad) = grad {
                    self.gradients.entry(*input_id)
                        .and_modify(|g| *g = g.add(&grad).unwrap())
                        .or_insert(grad);
                }
            }
        }
        
        Ok(())
    }
}
```

---

## 3. Reverse-Mode Differentiation

### 3.1 Gradient Function Trait

```rust
pub trait GradientFunction: Send + Sync {
    /// Save tensors needed for backward pass
    fn save_for_backward<T: DType>(
        &self,
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Vec<SavedTensor>;
    
    /// Compute gradients w.r.t. inputs given gradient w.r.t. output
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        saved_tensors: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>>;
    
    /// Operation name for debugging
    fn name(&self) -> &str;
}
```

### 3.2 Chain Rule Implementation

The chain rule is the foundation of backpropagation:

```
If z = f(y) and y = g(x), then:
dz/dx = dz/dy * dy/dx
```

```rust
/// Example: Composition of operations
/// z = relu(matmul(x, W) + b)
///
/// Backward:
/// dL/dx = dL/dz * dz/dy * dy/dx
///       = grad_output * relu'(y) * W^T
pub struct CompositeGradient {
    operations: Vec<Box<dyn GradientFunction>>,
}

impl GradientFunction for CompositeGradient {
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        saved_tensors: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        let mut current_grad = grad_output.clone();
        
        // Apply chain rule in reverse order
        for op in self.operations.iter().rev() {
            let grads = op.backward(&current_grad, saved_tensors)?;
            current_grad = grads[0].clone().unwrap();
        }
        
        Ok(vec![Some(current_grad)])
    }
    
    fn name(&self) -> &str {
        "composite"
    }
    
    fn save_for_backward<T: DType>(
        &self,
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        // Save all intermediate results
        Vec::new()  // Simplified
    }
}
```

---

## 4. Gradient Functions

### 4.1 Basic Operations

```rust
/// Gradient for addition: z = x + y
pub struct AddGradient;

impl GradientFunction for AddGradient {
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        _saved: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        // dz/dx = 1, dz/dy = 1
        // So grad_x = grad_output, grad_y = grad_output
        Ok(vec![
            Some(grad_output.clone()),
            Some(grad_output.clone()),
        ])
    }
    
    fn name(&self) -> &str { "add" }
    
    fn save_for_backward<T: DType>(
        &self,
        _inputs: &[&Tensor<T>],
        _output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        vec![]  // No tensors needed
    }
}

/// Gradient for multiplication: z = x * y
pub struct MulGradient;

impl GradientFunction for MulGradient {
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        saved: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        // dz/dx = y, dz/dy = x
        let x = saved[0].as_tensor();
        let y = saved[1].as_tensor();
        
        Ok(vec![
            Some(grad_output.mul(y)?),  // grad_x = grad_output * y
            Some(grad_output.mul(x)?),  // grad_y = grad_output * x
        ])
    }
    
    fn name(&self) -> &str { "mul" }
    
    fn save_for_backward<T: DType>(
        &self,
        inputs: &[&Tensor<T>],
        _output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        vec![
            SavedTensor::Tensor(inputs[0].to_f32()),
            SavedTensor::Tensor(inputs[1].to_f32()),
        ]
    }
}

/// Gradient for matrix multiplication: C = A @ B
pub struct MatMulGradient;

impl GradientFunction for MatMulGradient {
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        saved: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        // dL/dA = grad_output @ B^T
        // dL/dB = A^T @ grad_output
        
        let a = saved[0].as_tensor();
        let b = saved[1].as_tensor();
        
        let grad_a = grad_output.matmul(&b.transpose(-1, -2)?)?;
        let grad_b = a.transpose(-1, -2)?.matmul(grad_output)?;
        
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
    
    fn name(&self) -> &str { "matmul" }
    
    fn save_for_backward<T: DType>(
        &self,
        inputs: &[&Tensor<T>],
        _output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        vec![
            SavedTensor::Tensor(inputs[0].to_f32()),
            SavedTensor::Tensor(inputs[1].to_f32()),
        ]
    }
}
```

### 4.2 Activation Functions

```rust
/// Gradient for ReLU: y = max(0, x)
pub struct ReLUGradient;

impl GradientFunction for ReLUGradient {
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        saved: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        // d(relu(x))/dx = 1 if x > 0, else 0
        let input = saved[0].as_tensor();
        
        let mask = input.greater_than(0.0)?;
        let grad = grad_output.mul(&mask)?;
        
        Ok(vec![Some(grad)])
    }
    
    fn name(&self) -> &str { "relu" }
    
    fn save_for_backward<T: DType>(
        &self,
        inputs: &[&Tensor<T>],
        _output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        vec![SavedTensor::Tensor(inputs[0].to_f32())]
    }
}

/// Gradient for Sigmoid: y = 1 / (1 + exp(-x))
pub struct SigmoidGradient;

impl GradientFunction for SigmoidGradient {
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        saved: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        let output = saved[0].as_tensor();
        
        let one = Tensor::ones(output.shape());
        let grad = output.mul(&one.sub(output)?)?;
        let grad = grad.mul(grad_output)?;
        
        Ok(vec![Some(grad)])
    }
    
    fn name(&self) -> &str { "sigmoid" }
    
    fn save_for_backward<T: DType>(
        &self,
        _inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        // Save output (not input) for efficiency
        vec![SavedTensor::Tensor(output.to_f32())]
    }
}

/// Gradient for Softmax: y_i = exp(x_i) / sum_j(exp(x_j))
pub struct SoftmaxGradient;

impl GradientFunction for SoftmaxGradient {
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        saved: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        // Jacobian of softmax is: J_ij = y_i * (δ_ij - y_j)
        // grad_input_i = sum_j(grad_output_j * J_ij)
        //              = grad_output_i * y_i - y_i * sum_j(grad_output_j * y_j)
        
        let output = saved[0].as_tensor();
        let axis = saved[1].as_scalar() as isize;
        
        // sum_j(grad_output_j * y_j)
        let sum_term = grad_output.mul(output)?.sum_axis(axis, true)?;
        
        // grad_output_i * y_i
        let first_term = grad_output.mul(output)?;
        
        // y_i * sum_j(...)
        let second_term = output.mul(&sum_term)?;
        
        let grad = first_term.sub(&second_term)?;
        
        Ok(vec![Some(grad)])
    }
    
    fn name(&self) -> &str { "softmax" }
    
    fn save_for_backward<T: DType>(
        &self,
        _inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        vec![
            SavedTensor::Tensor(output.to_f32()),
            SavedTensor::Scalar(-1.0),  // axis
        ]
    }
}
```

### 4.3 Convolution Gradient

```rust
/// Gradient for 2D Convolution
pub struct Conv2dGradient;

impl GradientFunction for Conv2dGradient {
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        saved: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        let input = saved[0].as_tensor();
        let weight = saved[1].as_tensor();
        let stride = saved[2].as_scalar() as usize;
        let padding = saved[3].as_scalar() as usize;
        
        // Gradient w.r.t. input: convolve grad_output with flipped weight
        let grad_input = self.conv2d_input_gradient(
            grad_output,
            weight,
            input.shape(),
            stride,
            padding,
        )?;
        
        // Gradient w.r.t. weight: convolve input with grad_output
        let grad_weight = self.conv2d_weight_gradient(
            input,
            grad_output,
            weight.shape(),
            stride,
            padding,
        )?;
        
        Ok(vec![Some(grad_input), Some(grad_weight)])
    }
    
    fn conv2d_input_gradient(
        &self,
        grad_output: &Tensor<f32>,
        weight: &Tensor<f32>,
        input_shape: &[usize],
        stride: usize,
        padding: usize,
    ) -> Result<Tensor<f32>> {
        // Transpose convolution (deconvolution)
        // This is equivalent to convolving grad_output with 180° rotated weight
        
        let weight_flipped = weight.flip(&[2, 3])?;
        
        // Compute output padding to match input shape
        let output_padding = self.compute_output_padding(
            grad_output.shape(),
            input_shape,
            weight.shape(),
            stride,
            padding,
        );
        
        self.conv_transpose2d(
            grad_output,
            &weight_flipped,
            stride,
            padding,
            output_padding,
        )
    }
    
    fn conv2d_weight_gradient(
        &self,
        input: &Tensor<f32>,
        grad_output: &Tensor<f32>,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
    ) -> Result<Tensor<f32>> {
        // Convolve input with grad_output to get weight gradient
        let batch_size = input.shape()[0];
        
        let mut grad_weight = Tensor::zeros(weight_shape);
        
        for b in 0..batch_size {
            let input_b = input.select(0, b)?;
            let grad_out_b = grad_output.select(0, b)?;
            
            // For each output channel
            for out_c in 0..weight_shape[0] {
                let grad_out_c = grad_out_b.select(0, out_c)?;
                
                // For each input channel
                for in_c in 0..weight_shape[1] {
                    let input_c = input_b.select(0, in_c)?;
                    
                    // Convolve to get gradient for this kernel
                    let kernel_grad = self.correlate2d(
                        &input_c,
                        &grad_out_c,
                        stride,
                        padding,
                    )?;
                    
                    // Accumulate
                    let mut weight_slice = grad_weight.select_mut(0, out_c)?.select_mut(0, in_c)?;
                    weight_slice.add_assign(&kernel_grad)?;
                }
            }
        }
        
        Ok(grad_weight)
    }
    
    fn name(&self) -> &str { "conv2d" }
    
    fn save_for_backward<T: DType>(
        &self,
        inputs: &[&Tensor<T>],
        _output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        vec![
            SavedTensor::Tensor(inputs[0].to_f32()),  // input
            SavedTensor::Tensor(inputs[1].to_f32()),  // weight
            SavedTensor::Scalar(1.0),  // stride (simplified)
            SavedTensor::Scalar(0.0),  // padding
        ]
    }
}
```

---

## 5. Memory-Efficient Backprop

### 5.1 Checkpointing

Checkpoint (recomputation) trades compute for memory by not saving all intermediate activations:

```rust
pub struct Checkpoint {
    /// Forward function
    forward_fn: Box<dyn Fn(&[Tensor<f32>]) -> Result<Tensor<f32>>>,
    
    /// Inputs to save
    inputs: Vec<Tensor<f32>>,
}

impl Checkpoint {
    pub fn new<F>(forward_fn: F, inputs: Vec<Tensor<f32>>) -> Self
    where
        F: Fn(&[Tensor<f32>]) -> Result<Tensor<f32>> + 'static,
    {
        Self {
            forward_fn: Box::new(forward_fn),
            inputs,
        }
    }
}

impl GradientFunction for Checkpoint {
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        saved: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        // Recompute forward pass to get intermediate activations
        let inputs: Vec<&Tensor<f32>> = self.inputs.iter().collect();
        
        // Enable gradient tracking for recomputation
        let _guard = GradModeGuard::enable();
        
        let output = (self.forward_fn)(&self.inputs)?;
        
        // Now backward through recomputed graph
        output.backward_with_grad(grad_output)?;
        
        // Collect gradients from inputs
        let grads = self.inputs.iter()
            .map(|input| input.grad())
            .collect();
        
        Ok(grads)
    }
    
    fn name(&self) -> &str { "checkpoint" }
    
    fn save_for_backward<T: DType>(
        &self,
        inputs: &[&Tensor<T>],
        _output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        // Only save inputs, not all intermediate activations
        inputs.iter()
            .map(|t| SavedTensor::Tensor(t.to_f32()))
            .collect()
    }
}

pub struct GradModeGuard {
    prev_mode: GradMode,
}

impl GradModeGuard {
    pub fn enable() -> Self {
        let engine = AutogradEngine::global();
        let prev = engine.mode.load();
        engine.set_grad_mode(GradMode::Enabled);
        Self { prev_mode: prev }
    }
}

impl Drop for GradModeGuard {
    fn drop(&mut self) {
        let engine = AutogradEngine::global();
        engine.set_grad_mode(self.prev_mode);
    }
}
```

### 5.2 Gradient Accumulation

```rust
pub struct GradientAccumulator {
    steps: usize,
    current_step: AtomicUsize,
}

impl GradientAccumulator {
    pub fn new(accumulation_steps: usize) -> Self {
        Self {
            steps: accumulation_steps,
            current_step: AtomicUsize::new(0),
        }
    }
    
    pub fn backward(&self, loss: &Tensor<f32>) -> Result<bool> {
        // Scale loss by accumulation steps
        let scaled_loss = loss.div_scalar(self.steps as f32)?;
        
        // Backward pass accumulates gradients
        scaled_loss.backward()?;
        
        let step = self.current_step.fetch_add(1, Ordering::Relaxed);
        
        // Return true when ready to optimizer step
        Ok((step + 1) % self.steps == 0)
    }
    
    pub fn zero_grad(&self, params: &[&Tensor<f32>]) {
        for param in params {
            if let Some(grad_info) = &param.grad_info {
                grad_info.lock().unwrap().zero_grad();
            }
        }
        
        self.current_step.store(0, Ordering::Relaxed);
    }
}
```

---

## 6. Higher-Order Derivatives

### 6.1 Second-Order Gradients

```rust
/// Compute Hessian-vector product without forming full Hessian
pub fn hessian_vector_product(
    loss_fn: impl Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    params: &Tensor<f32>,
    vector: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    // First-order gradient
    let loss = loss_fn(params)?;
    let grad = compute_gradient(&loss, params)?;
    
    // Gradient-vector product
    let gvp = grad.dot(vector)?;
    
    // Second-order gradient (Hessian-vector product)
    let hvp = compute_gradient(&gvp, params)?;
    
    Ok(hvp)
}

fn compute_gradient(loss: &Tensor<f32>, param: &Tensor<f32>) -> Result<Tensor<f32>> {
    let _guard = GradModeGuard::enable();
    
    loss.backward()?;
    param.grad().ok_or(Error::MissingGradient)
}
```

### 6.2 Jacobian Computation

```rust
pub fn jacobian(
    func: impl Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    input: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    let output = func(input)?;
    
    let in_size = input.numel();
    let out_size = output.numel();
    
    let mut jacobian = Tensor::zeros(&[out_size, in_size]);
    
    // Compute each row of Jacobian (gradient of each output w.r.t. all inputs)
    for i in 0..out_size {
        // Create one-hot gradient for output i
        let mut grad_output = Tensor::zeros(output.shape());
        grad_output.data_mut()[i] = 1.0;
        
        // Backward to get gradient w.r.t. input
        let grad_input = compute_gradient_with(&output, input, &grad_output)?;
        
        // Store in Jacobian
        for j in 0..in_size {
            jacobian[[i, j]] = grad_input.data()[j];
        }
    }
    
    Ok(jacobian)
}

fn compute_gradient_with(
    output: &Tensor<f32>,
    input: &Tensor<f32>,
    grad_output: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    let _guard = GradModeGuard::enable();
    
    output.backward_with_grad(grad_output)?;
    input.grad().ok_or(Error::MissingGradient)
}
```

---

## 7. Custom Gradients

### 7.1 Custom Gradient Definition

```rust
/// Macro for defining custom gradient functions
#[macro_export]
macro_rules! custom_gradient {
    (
        fn $name:ident($($input:ident: $input_ty:ty),*) -> $output_ty:ty {
            forward: $forward_block:block
            backward: |$grad_out:ident, $($saved:ident),*| $backward_block:block
            save: $save_block:block
        }
    ) => {
        pub struct $name;
        
        impl $name {
            pub fn apply($($input: $input_ty),*) -> Result<$output_ty> {
                let output = $forward_block;
                
                // Record in tape
                let engine = AutogradEngine::global();
                if engine.is_grad_enabled() {
                    let grad_fn = Arc::new(Self);
                    // ... record operation
                }
                
                Ok(output)
            }
        }
        
        impl GradientFunction for $name {
            fn backward(
                &self,
                $grad_out: &Tensor<f32>,
                saved: &[SavedTensor],
            ) -> Result<Vec<Option<Tensor<f32>>>> {
                // Destructure saved tensors
                let mut idx = 0;
                $(
                    let $saved = &saved[idx];
                    idx += 1;
                )*
                
                $backward_block
            }
            
            fn save_for_backward<T: DType>(
                &self,
                inputs: &[&Tensor<T>],
                output: &Tensor<T>,
            ) -> Vec<SavedTensor> {
                $save_block
            }
            
            fn name(&self) -> &str {
                stringify!($name)
            }
        }
    };
}

// Example usage:
custom_gradient! {
    fn CustomExp(x: &Tensor<f32>) -> Tensor<f32> {
        forward: {
            x.exp()
        }
        backward: |grad_output, output| {
            // d(exp(x))/dx = exp(x)
            // Save output to avoid recomputing exp
            let grad = grad_output.mul(output.as_tensor())?;
            Ok(vec![Some(grad)])
        }
        save: {
            vec![SavedTensor::Tensor(output.to_f32())]
        }
    }
}
```

### 7.2 Function with No Gradient

```rust
/// Detach tensor from computation graph
pub struct DetachGradient;

impl GradientFunction for DetachGradient {
    fn backward(
        &self,
        _grad_output: &Tensor<f32>,
        _saved: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        // Return None - no gradient flows through
        Ok(vec![None])
    }
    
    fn name(&self) -> &str { "detach" }
    
    fn save_for_backward<T: DType>(
        &self,
        _inputs: &[&Tensor<T>],
        _output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        vec![]
    }
}

impl<T: DType> Tensor<T> {
    pub fn detach(&self) -> Self {
        let mut detached = self.clone();
        detached.grad_info = None;
        detached
    }
}
```

---

## 8. Static vs Dynamic Graphs

### 8.1 Dynamic Graph (Eager Mode)

```rust
pub struct DynamicGraph {
    tape: ComputationTape,
}

impl DynamicGraph {
    pub fn new() -> Self {
        Self {
            tape: ComputationTape::new(),
        }
    }
    
    /// Execute operation and record in tape
    pub fn execute<T: DType>(
        &mut self,
        op: Operation,
        inputs: &[&Tensor<T>],
    ) -> Result<Tensor<T>> {
        // Execute immediately
        let output = execute_operation(op.clone(), inputs)?;
        
        // Record in tape
        if inputs.iter().any(|t| t.requires_grad()) {
            self.tape.record(op, inputs, &output);
        }
        
        Ok(output)
    }
    
    /// Backward pass
    pub fn backward(&self, loss: &Tensor<f32>) -> Result<()> {
        let grad_tape = self.tape.build_gradient_tape(loss)?;
        grad_tape.execute()
    }
}
```

### 8.2 Static Graph (Graph Mode)

```rust
pub struct StaticGraph {
    /// IR representation of computation
    ir: FmlIR,
    
    /// Compiled gradient computation
    gradient_ir: Option<FmlIR>,
    
    /// Parameter nodes
    parameters: HashMap<String, TensorId>,
}

impl StaticGraph {
    pub fn build(builder: impl FnOnce(&mut GraphBuilder) -> Result<Tensor<f32>>) -> Result<Self> {
        let mut graph_builder = GraphBuilder::new();
        
        // Build forward graph
        let output = builder(&mut graph_builder)?;
        
        let ir = graph_builder.finalize()?;
        
        Ok(Self {
            ir,
            gradient_ir: None,
            parameters: HashMap::new(),
        })
    }
    
    pub fn compile_gradients(&mut self, loss: TensorId) -> Result<()> {
        // Reverse-mode AD on IR level
        let gradient_ir = self.differentiate_ir(loss)?;
        self.gradient_ir = Some(gradient_ir);
        
        Ok(())
    }
    
    fn differentiate_ir(&self, loss: TensorId) -> Result<FmlIR> {
        let mut grad_builder = IRBuilder::new();
        
        // Traverse IR in reverse topological order
        let topo_order = self.ir.topological_sort()?;
        
        // Initialize gradient of loss
        let loss_grad = grad_builder.create_constant(
            Attribute::Float(1.0),
            Type::Float(FloatType { kind: FloatKind::F32 }),
        );
        
        let mut gradients = HashMap::new();
        gradients.insert(loss, loss_grad);
        
        for op_id in topo_order.iter().rev() {
            let op = self.ir.get_operation(op_id);
            
            if let Some(output_grad) = gradients.get(&op.results[0]) {
                // Generate gradient operations
                let input_grads = self.generate_gradient_ops(
                    op,
                    *output_grad,
                    &mut grad_builder,
                )?;
                
                // Accumulate gradients for inputs
                for (input, grad) in op.operands.iter().zip(input_grads) {
                    gradients.entry(*input)
                        .and_modify(|g| {
                            *g = grad_builder.create_binary_op(BinaryOpKind::Add, *g, grad);
                        })
                        .or_insert(grad);
                }
            }
        }
        
        Ok(grad_builder.finish())
    }
    
    fn generate_gradient_ops(
        &self,
        op: &Operation,
        grad_output: Value,
        builder: &mut IRBuilder,
    ) -> Result<Vec<Value>> {
        match &op.kind {
            OpKind::Tensor(TensorOp::MatMul) => {
                // grad_A = grad_output @ B^T
                // grad_B = A^T @ grad_output
                let a = op.operands[0];
                let b = op.operands[1];
                
                let b_t = builder.create_transpose(b, -1, -2);
                let grad_a = builder.create_matmul(grad_output, b_t);
                
                let a_t = builder.create_transpose(a, -1, -2);
                let grad_b = builder.create_matmul(a_t, grad_output);
                
                Ok(vec![grad_a, grad_b])
            }
            
            OpKind::Tensor(TensorOp::Add) => {
                // Both inputs get same gradient
                Ok(vec![grad_output, grad_output])
            }
            
            OpKind::Tensor(TensorOp::ReLU) => {
                // grad = grad_output * (input > 0)
                let input = op.operands[0];
                let zero = builder.create_constant(
                    Attribute::Float(0.0),
                    input.get_type(),
                );
                let mask = builder.create_comparison(CompareOp::Greater, input, zero);
                let grad = builder.create_binary_op(BinaryOpKind::Mul, grad_output, mask);
                
                Ok(vec![grad])
            }
            
            _ => unimplemented!("Gradient for {:?}", op.kind),
        }
    }
    
    /// Execute forward and backward pass
    pub fn execute(&self, inputs: &HashMap<String, Tensor<f32>>) -> Result<()> {
        // Execute forward IR
        let outputs = self.ir.execute(inputs)?;
        
        // Execute backward IR if compiled
        if let Some(grad_ir) = &self.gradient_ir {
            grad_ir.execute(&outputs)?;
        }
        
        Ok(())
    }
}

pub struct GraphBuilder {
    operations: Vec<Operation>,
    parameters: Vec<Value>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            parameters: Vec::new(),
        }
    }
    
    pub fn parameter(&mut self, name: &str, shape: &[usize]) -> Tensor<f32> {
        // Create parameter placeholder
        let param = Tensor::zeros(shape).requires_grad(true);
        self.parameters.push(param.id);
        param
    }
    
    pub fn finalize(self) -> Result<FmlIR> {
        // Convert operations to IR
        unimplemented!()
    }
}
```

### 8.3 Mixed Mode

```rust
pub struct HybridGraph {
    /// Static graph for model structure
    static_graph: StaticGraph,
    
    /// Dynamic graph for data-dependent control flow
    dynamic_regions: HashMap<String, DynamicGraph>,
}

impl HybridGraph {
    pub fn new(static_builder: impl FnOnce(&mut GraphBuilder) -> Result<Tensor<f32>>) -> Result<Self> {
        let static_graph = StaticGraph::build(static_builder)?;
        
        Ok(Self {
            static_graph,
            dynamic_regions: HashMap::new(),
        })
    }
    
    pub fn add_dynamic_region(&mut self, name: &str, region: DynamicGraph) {
        self.dynamic_regions.insert(name.to_string(), region);
    }
    
    pub fn execute(&self, inputs: &HashMap<String, Tensor<f32>>) -> Result<Tensor<f32>> {
        // Execute static portion
        let static_outputs = self.static_graph.execute(inputs)?;
        
        // Execute dynamic regions as needed
        for (name, region) in &self.dynamic_regions {
            // ... execute dynamic region
        }
        
        unimplemented!()
    }
}
```

---

## 9. Optimization Techniques

### 9.1 In-Place Operations

```rust
/// Mark operation as in-place to save memory
pub struct InPlaceGradient {
    base_grad_fn: Arc<dyn GradientFunction>,
}

impl InPlaceGradient {
    pub fn new(base_grad_fn: Arc<dyn GradientFunction>) -> Self {
        Self { base_grad_fn }
    }
}

impl GradientFunction for InPlaceGradient {
    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        saved: &[SavedTensor],
    ) -> Result<Vec<Option<Tensor<f32>>>> {
        // Delegate to base gradient function
        self.base_grad_fn.backward(grad_output, saved)
    }
    
    fn name(&self) -> &str {
        "in_place"
    }
    
    fn save_for_backward<T: DType>(
        &self,
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Vec<SavedTensor> {
        self.base_grad_fn.save_for_backward(inputs, output)
    }
}

impl<T: DType> Tensor<T> {
    pub fn add_(&mut self, other: &Tensor<T>) -> Result<()> {
        // In-place addition
        // Warning: Invalidates any gradient computations using old value
        self.data_mut().iter_mut()
            .zip(other.data().iter())
            .for_each(|(a, b)| *a = *a + *b);
        
        Ok(())
    }
}
```

### 9.2 Gradient Sparsification

```rust
/// Only compute gradients for top-k elements by magnitude
pub struct SparseGradient {
    top_k: usize,
}

impl SparseGradient {
    pub fn apply(&self, grad: &Tensor<f32>) -> Result<Tensor<f32>> {
        let n = grad.numel();
        
        if self.top_k >= n {
            return Ok(grad.clone());
        }
        
        // Find top-k indices by magnitude
        let mut indexed: Vec<(usize, f32)> = grad.data()
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Zero out all but top-k
        let mut sparse_grad = Tensor::zeros(grad.shape());
        for (idx, _) in indexed.iter().take(self.top_k) {
            sparse_grad.data_mut()[*idx] = grad.data()[*idx];
        }
        
        Ok(sparse_grad)
    }
}
```

---

## Summary

This section detailed the automatic differentiation engine for FerricML:

**Key Components:**
1. **Computation Graph:** Tape-based recording for dynamic graphs
2. **Reverse-Mode AD:** Efficient backpropagation via chain rule
3. **Gradient Functions:** Complete library for all operations
4. **Memory Efficiency:** Checkpointing and gradient accumulation
5. **Higher-Order:** Second derivatives and Jacobians
6. **Custom Gradients:** User-defined gradient functions
7. **Static Graphs:** IR-level differentiation for optimization
8. **Mixed Mode:** Combine static and dynamic execution

**Design Decisions:**
- Tape-based for simplicity and debugging
- Save minimal tensors for backward pass
- Support both eager and graph modes
- Checkpoint for memory-constrained training
- In-place ops for efficiency (with caution)

**Performance Considerations:**
- Gradient accumulation reduces memory for large batches
- Checkpointing trades 1 forward recompute for activation memory
- Static graphs enable graph-level optimizations
- Sparse gradients reduce communication in distributed training

**Implementation Priority:**
1. Basic tape and gradient functions
2. Common operation gradients (matmul, conv, activations)
3. Memory-efficient checkpointing
4. Static graph mode and IR differentiation
5. Higher-order derivatives (advanced)

**Integration Points:**
- Section 2 IR enables static graph differentiation
- Section 7 neural networks use autograd for training
- Section 10 distributed training uses gradient accumulation
- All backends execute gradient computations

**Next Steps:**
- Section 6 will optimize gradient computation patterns
- Section 7 builds high-level modules on autograd
- Implement gradient checkpointing for large models
- Add profiling to identify gradient computation bottlenecks