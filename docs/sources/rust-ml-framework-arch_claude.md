# Ferric ML: Universal Deep Learning Framework Architecture Specification

**Version:** 1.0  
**Date:** October 2025  
**Target Implementation:** Rust  

---

## Executive Summary

Ferric ML is a next-generation deep learning framework combining PyTorch's ergonomics with TensorFlow's production readiness, implemented entirely in Rust with a unified compilation infrastructure inspired by LLVM. The framework features a novel multi-level intermediate representation (MLIR-style) specifically optimized for AI/ML workloads and supports CPU, NVIDIA GPU (CUDA), AMD GPU (ROCm), and TPU backends through a unified abstraction layer.

### Core Design Principles

1. **Safety First**: Leverage Rust's ownership system for memory safety and thread safety
2. **Zero-Cost Abstractions**: Compile-time optimizations with minimal runtime overhead
3. **Hardware Agnostic**: Write once, run efficiently everywhere
4. **Python-First UX**: Ergonomic Python bindings with native Rust performance
5. **Gradual Compilation**: Support both eager execution (PyTorch-style) and graph compilation (TensorFlow-style)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend Layer (Python/Rust)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Eager      │  │   Graph      │  │   Hybrid (JIT)       │  │
│  │  Execution   │  │  Definition  │  │   torch.compile      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼──────────────────┼───────────────────────┼────────────┘
          │                  │                       │
          v                  v                       v
┌─────────────────────────────────────────────────────────────────┐
│                    FML IR (Ferric ML IR)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  High-Level Dialect: Tensor Operations, DL Primitives    │  │
│  └────────────────────┬──────────────────────────────────────┘  │
│  ┌────────────────────┴──────────────────────────────────────┐  │
│  │  Mid-Level Dialect: Loop Optimization, Memory Planning   │  │
│  └────────────────────┬──────────────────────────────────────┘  │
│  ┌────────────────────┴──────────────────────────────────────┐  │
│  │  Low-Level Dialect: Hardware Primitives, Scheduling      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────┬───────────────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────────────┐
│              Optimization Pipeline (Pass Manager)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Operator │  │ Kernel   │  │ Memory   │  │ Auto-tuning  │   │
│  │  Fusion  │  │ Selection│  │ Planning │  │ & Codegen    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────┬───────────────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────────────┐
│                    Hardware Backends                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │   CPU    │  │  CUDA    │  │  ROCm    │  │     TPU      │   │
│  │ (LLVM)   │  │ (NVPTX)  │  │(AMDGPU)  │  │   (XLA)      │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

### 2.1 Tensor Abstraction Layer

**Location:** `ferric_core::tensor`

#### Tensor Structure
```rust
pub struct Tensor<T: Dtype> {
    /// Raw data storage (device-agnostic pointer)
    data: Arc<Storage<T>>,
    
    /// Shape information (static and dynamic)
    shape: Shape,
    
    /// Strides for memory layout
    strides: Strides,
    
    /// Device placement
    device: Device,
    
    /// Gradient tracking information
    grad_info: Option<Arc<GradInfo>>,
    
    /// Unique tensor ID for graph construction
    id: TensorId,
}

pub enum Storage<T> {
    Cpu(CpuStorage<T>),
    Cuda(CudaStorage<T>),
    Rocm(RocmStorage<T>),
    Tpu(TpuStorage<T>),
}

pub enum Device {
    Cpu,
    Cuda(CudaDevice),
    Rocm(RocmDevice),
    Tpu(TpuDevice),
}
```

#### Memory Management
- **Unified Virtual Addressing (UVA)**: Single address space across CPU and accelerators
- **Lazy Allocation**: Defer memory allocation until kernel execution
- **Memory Pooling**: Per-device memory pools with configurable strategies
- **Zero-Copy Transfers**: Where supported by hardware

### 2.2 FML IR (Ferric ML Intermediate Representation)

Inspired by MLIR but specifically designed for ML workloads.

#### Dialect Hierarchy

##### 1. Tensor Dialect (High-Level)
```
// Example: Matrix multiplication with ReLU
%result = fml.matmul %input : tensor<[B, M, K], f32>, 
                      %weight : tensor<[K, N], f32>
         -> tensor<[B, M, N], f32>
%activated = fml.relu %result : tensor<[B, M, N], f32>
```

**Operations:**
- `fml.matmul`, `fml.conv2d`, `fml.attention`
- `fml.relu`, `fml.gelu`, `fml.softmax`
- `fml.layer_norm`, `fml.batch_norm`
- `fml.embedding`, `fml.dropout`

##### 2. Loop Dialect (Mid-Level)
```
// Tiled matrix multiplication
fml.parallel %batch in 0..B {
  fml.tiled_loop (%i, %j, %k) in (0..M, 0..N, 0..K)
    tile_size (128, 128, 128) {
    %tile_a = fml.load %A[%i:%i+128, %k:%k+128]
    %tile_b = fml.load %B[%k:%k+128, %j:%j+128]
    %tile_c = fml.compute_tile %tile_a, %tile_b
    fml.store %C[%i:%i+128, %j:%j+128], %tile_c
  }
}
```

**Features:**
- Polyhedral loop transformations
- Tile size optimization
- Loop fusion and distribution
- Vectorization hints

##### 3. Hardware Dialect (Low-Level)
```
// CUDA-specific operations
%shared_mem = fml.cuda.shared_memory 
  <128x128xf32> alignment 16
%warp_result = fml.cuda.warp_reduce %partial_sum
%tensorcore = fml.cuda.mma_m16n8k16 %a, %b, %c
```

**Backends:**
- `fml.cpu.*`: SIMD instructions, cache hints
- `fml.cuda.*`: Thread blocks, shared memory, tensor cores
- `fml.rocm.*`: Wavefronts, LDS, matrix cores
- `fml.tpu.*`: Systolic array operations, HBM management

### 2.3 Automatic Differentiation Engine

Based on reverse-mode automatic differentiation (backpropagation).

#### Design
```rust
pub struct AutogradEngine {
    /// Computational graph
    graph: ComputeGraph,
    
    /// Gradient computation functions
    grad_fns: HashMap<OpType, GradFn>,
    
    /// Execution order (topological sort)
    execution_order: Vec<NodeId>,
}

pub trait GradientFunction {
    /// Forward pass with gradient recording
    fn forward(&self, inputs: &[Tensor]) -> Tensor;
    
    /// Backward pass gradient computation
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}
```

#### Implementation Strategy
1. **Tape-based Recording**: Build DAG during forward pass
2. **Lazy Gradient Computation**: Only compute gradients when `.backward()` is called
3. **Checkpointing**: For memory-efficient training of large models
4. **Mixed-Precision Support**: FP16/BF16 gradients with FP32 master weights

### 2.4 Compilation Pipeline

#### Stage 1: Frontend Lowering
- Python/Rust API → Tensor Dialect IR
- Shape inference and type checking
- Constant folding and dead code elimination

#### Stage 2: High-Level Optimization
- Operator fusion (vertical and horizontal)
- Common subexpression elimination
- Algebraic simplifications
- Layout optimization (NCHW ↔ NHWC)

#### Stage 3: Loop-Level Optimization
- Polyhedral analysis and transformation
- Tile size auto-tuning
- Memory access pattern optimization
- Parallelization strategy selection

#### Stage 4: Backend-Specific Lowering
- Target selection based on device capabilities
- Hardware-specific instruction selection
- Register allocation and scheduling

#### Stage 5: Code Generation
- **CPU**: LLVM IR → Native code
- **CUDA**: NVPTX → PTX → SASS
- **ROCm**: AMDGPU → GCN/RDNA ISA
- **TPU**: XLA HLO → TPU executable

### 2.5 Execution Modes

#### Eager Execution (PyTorch-style)
```python
# Python API
import ferric as fml

x = fml.randn([64, 784])
W1 = fml.randn([784, 256], requires_grad=True)
b1 = fml.zeros([256], requires_grad=True)

# Operations execute immediately
h = fml.matmul(x, W1) + b1
h = fml.relu(h)
```

**Characteristics:**
- Immediate operation execution
- Dynamic computational graph
- Easy debugging
- Overhead per operation

#### Graph Mode (TensorFlow-style)
```python
@fml.compile
def forward(x, W1, b1):
    h = fml.matmul(x, W1) + b1
    h = fml.relu(h)
    return h

# Graph is traced once, then compiled
result = forward(x, W1, b1)
```

**Characteristics:**
- Static computational graph
- Aggressive optimizations
- Better performance
- Requires static shapes (or bucketing)

#### Hybrid Mode (torch.compile-style)
```python
# Automatically compile hot paths
model = MyModel()
compiled_model = fml.compile(model, mode='reduce-overhead')

# First few iterations trace and compile
# Subsequent iterations use compiled code
for batch in dataloader:
    loss = compiled_model(batch)
    loss.backward()
```

---

## 3. Hardware Backend Architecture

### 3.1 CPU Backend

**Implementation:** LLVM-based code generation

#### Features
- AVX2/AVX-512 vectorization
- OpenMP parallel loops
- Memory prefetching
- Cache-aware tiling

#### Optimizations
- Loop unrolling and vectorization
- SLP (Superword-Level Parallelism) vectorization
- Polyhedral optimization via Polly

### 3.2 NVIDIA GPU Backend (CUDA)

**Implementation:** NVPTX backend + CUDA runtime

#### Memory Hierarchy Utilization
```
Registers (per thread):
  - Fastest (1 cycle latency)
  - Limited (~255 per thread)
  - Automatic allocation

Shared Memory (per thread block):
  - Fast (~20-30 cycle latency)
  - 48-164KB per SM
  - User-managed caching

L1/L2 Cache:
  - Transparent
  - Coalescing optimization critical

Global Memory (DRAM):
  - High bandwidth (up to 2TB/s on H100)
  - High latency (~200-600 cycles)
  - Requires coalesced access
```

#### Kernel Generation Strategy
```rust
// Automatic kernel instantiation
pub struct CudaKernel {
    ptx_code: String,
    block_dim: (u32, u32, u32),
    grid_dim: (u32, u32, u32),
    shared_mem_bytes: usize,
}

impl CudaKernel {
    /// Generate CUDA kernel from FML IR
    pub fn from_ir(ir: &FmlOperation) -> Result<Self> {
        // 1. Tile size selection via auto-tuning
        let tile_config = AutoTuner::find_optimal_tiling(ir);
        
        // 2. Memory access pattern analysis
        let mem_pattern = analyze_memory_pattern(ir);
        
        // 3. Generate kernel code
        let ptx = codegen_nvptx(ir, tile_config, mem_pattern);
        
        Ok(Self { ptx_code: ptx, ... })
    }
}
```

#### Tensor Core Support
- Automatic detection and utilization of Tensor Cores for GEMM operations
- Mixed-precision accumulation (FP16 → FP32)
- Warp-level matrix operations (WMMA API)

### 3.3 AMD GPU Backend (ROCm)

**Implementation:** AMDGPU backend + HIP runtime

#### Key Differences from CUDA
- Wavefront size: 32 (RDNA) or 64 (CDNA) vs 32 (CUDA warp)
- LDS (Local Data Share) instead of shared memory
- GCN/RDNA ISA instead of SASS

#### HIP Abstraction Layer
```rust
/// Unified API for CUDA and ROCm
pub trait GpuRuntime {
    fn launch_kernel(&self, kernel: &CompiledKernel, 
                     grid: GridConfig, args: &[TensorRef]);
    fn synchronize(&self);
    fn get_device_properties(&self) -> DeviceProperties;
}

pub struct RocmRuntime {
    // ROCm-specific implementation
}

pub struct CudaRuntime {
    // CUDA-specific implementation
}
```

### 3.4 TPU Backend

**Implementation:** XLA compiler integration

#### Architecture Considerations
- Systolic array (128×128 MXU)
- High-Bandwidth Memory (HBM)
- Streaming architecture
- Limited branching support

#### XLA Integration
```rust
pub struct TpuBackend {
    xla_builder: XlaBuilder,
}

impl TpuBackend {
    /// Convert FML IR to XLA HLO
    pub fn lower_to_hlo(&self, ir: &FmlIR) -> XlaComputation {
        // Map FML operations to XLA HLO operations
        let hlo = HloBuilder::new();
        
        for op in ir.operations() {
            match op {
                Op::MatMul(a, b) => hlo.dot(a, b),
                Op::Conv2D(input, kernel) => hlo.convolution(input, kernel),
                Op::Add(a, b) => hlo.add(a, b),
                // ... more operations
            }
        }
        
        hlo.build()
    }
}
```

---

## 4. Optimization Infrastructure

### 4.1 Operator Fusion

#### Vertical Fusion (Producer-Consumer)
```
Before:
  %1 = matmul(%input, %weight)
  %2 = add(%1, %bias)
  %3 = relu(%2)

After:
  %result = fused_linear_relu(%input, %weight, %bias)
```

#### Horizontal Fusion (Independent Operations)
```
Before:
  %1 = relu(%x)
  %2 = tanh(%y)
  
After:
  %result = fused_elementwise(%x, %y, [relu, tanh])
```

### 4.2 Auto-Tuning System

Inspired by TVM AutoScheduler and Triton's approach.

```rust
pub struct AutoTuner {
    search_space: SearchSpace,
    cost_model: Box<dyn CostModel>,
    trials: usize,
}

pub trait CostModel {
    /// Predict execution time for configuration
    fn predict(&self, config: &KernelConfig) -> f32;
    
    /// Update model with actual measurements
    fn update(&mut self, config: &KernelConfig, actual_time: f32);
}

impl AutoTuner {
    /// Find optimal kernel configuration
    pub fn tune(&mut self, operation: &Operation) -> KernelConfig {
        let mut best_config = None;
        let mut best_time = f32::MAX;
        
        for _ in 0..self.trials {
            let config = self.search_space.sample();
            let predicted = self.cost_model.predict(&config);
            
            // Only benchmark promising configurations
            if predicted < best_time * 1.5 {
                let actual = benchmark_config(&config, operation);
                self.cost_model.update(&config, actual);
                
                if actual < best_time {
                    best_time = actual;
                    best_config = Some(config);
                }
            }
        }
        
        best_config.unwrap()
    }
}
```

### 4.3 Memory Planning

#### Static Memory Planning
- Analyze tensor lifetimes
- Apply graph coloring algorithm for buffer reuse
- Minimize peak memory usage

```rust
pub struct MemoryPlanner {
    /// Liveness analysis results
    live_ranges: HashMap<TensorId, (usize, usize)>,
}

impl MemoryPlanner {
    pub fn plan(&self, graph: &ComputeGraph) -> MemoryPlan {
        // Interference graph construction
        let interference = self.build_interference_graph();
        
        // Graph coloring for buffer assignment
        let allocation = greedy_coloring(&interference);
        
        MemoryPlan { allocation }
    }
}
```

---

## 5. Python Integration

### 5.1 PyO3 Bindings

```rust
#[pyclass]
pub struct Tensor {
    inner: fml_core::Tensor<f32>,
}

#[pymethods]
impl Tensor {
    #[new]
    fn new(shape: Vec<usize>) -> Self {
        Self {
            inner: fml_core::Tensor::zeros(shape),
        }
    }
    
    fn __add__(&self, other: &Tensor) -> PyResult<Tensor> {
        Ok(Tensor {
            inner: (&self.inner + &other.inner)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        })
    }
    
    fn backward(&self) -> PyResult<()> {
        self.inner.backward()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}
```

### 5.2 NumPy Integration

- Zero-copy conversion where possible
- Support for NumPy array protocol
- Compatible with existing NumPy ecosystem

---

## 6. Distributed Training

### 6.1 Data Parallelism

```rust
pub struct DataParallel<M: Module> {
    model: Arc<M>,
    devices: Vec<Device>,
    strategy: DPStrategy,
}

pub enum DPStrategy {
    /// PyTorch DDP-style
    DistributedDataParallel,
    /// Horovod-style
    AllReduce,
    /// Parameter Server
    ParameterServer,
}
```

### 6.2 Model Parallelism

#### Tensor Parallelism
- Distribute layers across devices
- Communication via collective operations

#### Pipeline Parallelism
- Split model into stages
- Micro-batching for efficiency

### 6.3 Communication Backend

Support for multiple collective communication libraries:
- NCCL (NVIDIA)
- RCCL (AMD)
- Gloo (CPU)
- MPI

---

## 7. Model Serialization

### Format: FlatBuffers + Custom Schema

```
Model {
  version: u32,
  graph: ComputeGraph,
  weights: [Tensor],
  metadata: ModelMetadata,
  target_device: DeviceSpec,
}
```

#### Benefits
- Zero-copy deserialization
- Cross-platform compatibility
- Versioning support
- Efficient storage

---

## 8. Benchmarking & Profiling

### Built-in Profiler

```rust
pub struct Profiler {
    events: Vec<ProfileEvent>,
    overhead: Duration,
}

pub struct ProfileEvent {
    name: String,
    start: Instant,
    duration: Duration,
    device: Device,
    memory_delta: i64,
}
```

### Integration Points
- CUDA Profiling Tools Interface (CUPTI)
- ROCm Profiling Tools (rocProfiler)
- Chrome Trace Event Format output

---

## 9. Safety Guarantees

### Rust Safety Features

1. **Memory Safety**: No use-after-free, no data races
2. **Thread Safety**: `Send` and `Sync` markers for safe concurrency
3. **Type Safety**: Strong type system prevents mismatched operations
4. **No Null Pointer Dereferencing**: `Option<T>` for optional values

### Additional Safety Layers

- **Shape Checking**: Compile-time and runtime shape verification
- **Device Consistency**: Ensure operations on same device
- **Gradient Validity**: Prevent invalid gradient computations

---

## 10. Performance Targets

### Inference (ResNet-50, Batch=1)
- CPU: Within 10% of PyTorch+TorchScript
- CUDA: Match PyTorch+Triton or better
- ROCm: Match PyTorch+HIP
- TPU: Within 15% of JAX+XLA

### Training (Transformer-Large, Batch=32)
- GPU: Match PyTorch 2.0 with torch.compile
- Multi-GPU: Within 5% of PyTorch DDP
- TPU: Competitive with JAX on similar hardware

---

## 11. Development Roadmap

### Phase 1: Core Infrastructure (Months 0-6)
- Basic tensor operations (CPU only)
- Eager execution mode
- Automatic differentiation
- Python bindings

### Phase 2: GPU Support (Months 6-12)
- CUDA backend
- Basic operator fusion
- Memory optimization

### Phase 3: Advanced Features (Months 12-18)
- Graph compilation mode
- ROCm backend
- Distributed training (DDP)

### Phase 4: Production Readiness (Months 18-24)
- TPU support
- Auto-tuning infrastructure
- Model zoo and benchmarks
- Documentation and tutorials

---

## 12. References & Inspiration

### Academic Papers
1. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
2. MLIR: A Compiler Infrastructure for the End of Moore's Law
3. XLA: TensorFlow, Compiled
4. In-datacenter Performance Analysis of a Tensor Processing Unit

### Open Source Projects
- PyTorch: Dynamic graph construction, autograd
- TensorFlow: Static graph optimization, XLA
- Apache TVM: Auto-tuning, compilation
- MLIR: Multi-level IR design
- Triton: GPU programming abstractions

---

## Appendix A: Example End-to-End Flow

```python
import ferric as fml

# Define model
class SimpleNN(fml.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = fml.nn.Linear(784, 256)
        self.fc2 = fml.nn.Linear(256, 10)
    
    def forward(self, x):
        x = fml.relu(self.fc1(x))
        return self.fc2(x)

# Create model and move to GPU
model = SimpleNN().cuda()

# Compile for optimal performance
compiled_model = fml.compile(model, mode='max-autotune')

# Training loop
optimizer = fml.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        # Forward pass
        output = compiled_model(batch_x)
        loss = fml.nn.cross_entropy(output, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

# Save model
fml.save(model, 'model.fml')
```

**What happens internally:**

1. **First Forward Pass:**
   - Operations recorded in computational graph
   - Shapes inferred and validated
   - Device placement determined

2. **Compilation:**
   - Graph converted to FML IR (Tensor Dialect)
   - Operator fusion applied (matmul+relu fused)
   - Memory planning performed
   - Backend-specific lowering (CUDA)
   - Auto-tuning for optimal kernel parameters
   - PTX code generated and cached

3. **Subsequent Iterations:**
   - Use compiled kernel directly
   - Zero Python overhead
   - Optimal GPU utilization

4. **Backward Pass:**
   - Traverse graph in reverse
   - Compute gradients using registered gradient functions
   - Accumulate gradients in parameter tensors

---

## Appendix B: IR Example

### High-Level (Tensor Dialect)
```
func @forward(%input: tensor<64x784xf32>, 
              %w1: tensor<784x256xf32>,
              %b1: tensor<256xf32>) -> tensor<64x10xf32> {
  %0 = fml.matmul %input, %w1 : tensor<64x256xf32>
  %1 = fml.add %0, %b1 : tensor<64x256xf32>
  %2 = fml.relu %1 : tensor<64x256xf32>
  return %2
}
```

### After Fusion
```
func @forward_fused(%input: tensor<64x784xf32>,
                    %w1: tensor<784x256xf32>,
                    %b1: tensor<256xf32>) -> tensor<64x10xf32> {
  %0 = fml.fused_linear_relu %input, %w1, %b1
  return %0
}
```

### Low-Level (CUDA Dialect)
```
fml.cuda.kernel @fused_linear_relu_kernel
  (%in: memref<64x784xf32>, %w: memref<784x256xf32>, 
   %b: memref<256xf32>, %out: memref<64x256xf32>) {
  
  %block_idx = fml.cuda.block_id x
  %thread_idx = fml.cuda.thread_id x
  
  %row = %block_idx * 16 + %thread_idx / 16
  %col = %thread_idx % 16
  
  %smem = fml.cuda.shared_memory <16x16xf32>
  
  // Tiled matrix multiplication with ReLU fusion
  // ... (detailed implementation)
}
```

---

## Conclusion

Ferric ML represents a modern approach to deep learning frameworks, combining the best aspects of existing systems while addressing their limitations through Rust's safety guarantees and a unified compilation infrastructure. The LLVM-inspired multi-level IR design enables both ease of use and peak performance across diverse hardware platforms.