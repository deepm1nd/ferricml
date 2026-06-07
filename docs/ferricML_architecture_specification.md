# ferricML: Universal Machine Learning Framework
## Comprehensive Architecture Specification v3.0

**Date:** November 2025
**Status:** Technical Specification - Implementation Ready
**Target:** Senior Rust Engineers & AI/ML Systems Developers

---

## Master Table of Contents

| Section | Title | Focus Area |
| :--- | :--- | :--- |
| I | **Core Type System & Memory Model** | DType trait, symbolic tensors, memory pools, and UVA. |
| II | **FML Intermediate Representation (FML-IR)** | Multi-level SSA-based dialects (Tensor, Structured, Target). |
| III | **Dynamic Graph Capture & AOT-AD Engine** | Safe Tracing, ShapeGuards, and Ahead-of-Time Autodiff. |
| IV | **CUDA Backend Implementation** | NVVM/PTX pipeline, Tensor Cores, and Fat Binaries. |
| V | **CPU Backend & Vectorization** | LLVM, SIMD (AVX-512/NEON), and NUMA awareness. |
| VI | **ROCm & TPU Backend Integration** | ROCDL/HSACO and StableHLO/XLA pipelines. |
| VII | **Automatic Differentiation Engine** | Tape-based recording, reverse-mode algorithms, and custom grads. |
| VIII | **Optimization Pass Infrastructure** | Fusion, MPA, Bufferization, and Auto-tuning. |
| IX | **Neural Network Modules** | High-level API, standard layers, and Transformer components. |
| X | **Tree-Based Models & Ensemble Methods** | GBDT, Random Forests, EFB, and GOSS optimizations. |
| XI | **Probabilistic Graphical Models** | Bayesian Networks, MRFs, and inference algorithms. |
| XII | **Distributed Training & Serialization** | Data/Model/Pipeline parallelism and fault-tolerant checkpoints. |

---

## I. Core Type System & Memory Model

### I.1. Design Philosophy
ferricML's type system follows three core principles:
1. **Static Safety:** Leverage Rust's type system to prevent runtime errors.
2. **Zero-Cost Abstraction:** No runtime overhead for type information during hot paths.
3. **Hardware Awareness:** Types encode alignment, layout, and device placement.

### I.2. The DType Trait
The `DType` trait is the foundation for all numeric operations in ferricML.

```rust
pub trait DType: Copy + Send + Sync + 'static {
    const SIZE: usize;
    const ALIGNMENT: usize;
    const ZERO: Self;
    const ONE: Self;
    const TYPE_ID: TypeId;
    const SIMD_CAPABLE: bool;
    const SIMD_WIDTH: usize;

    fn from_f64(val: f64) -> Self;
    fn to_f64(self) -> f64;
}
```

**TypeId Enumeration:**
```rust
#[repr(u8)]
pub enum TypeId {
    Float16, BFloat16, Float32, Float64,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Bool, Complex64, Complex128,
    QInt8, QUInt8, QInt32
}
```

### I.3. The Symbolic ferricML Tensor
Tensors in ferricML are primarily symbolic handles linking to the Abstract Semantic Graph (ASG). This ensures a "Define-then-Run" paradigm.

```rust
pub struct Tensor<B: FerricBackend> {
    id: u64, // Unique ID linking to the ASG::Node
    dtype: B::FloatElem,
    shape: Shape,
    device: B::Device,
    grad_fn_ref: Option<GradFnId>,
}
```

### I.4. Memory Management & Storage
ferricML abstracts physical memory through a multi-backend storage system.

- **Unified Virtual Addressing (UVA):** Single address space across CPU and accelerators.
- **Memory Pool Allocator:** Uses a B-Tree based management system to minimize fragmentation.
- **Buffer Reuse Optimization (BReO):** Leverages lifetime analysis to reuse memory blocks between non-interfering tensors.

```rust
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Rocm(RocmStorage),
    Tpu(TpuStorage),
}

pub struct CpuStorage {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
    alignment: usize,
    allocator: Option<Arc<dyn Allocator>>,
}
```

---

## II. FML Intermediate Representation (FML-IR)

FML-IR is a multi-level stack based on Static Single Assignment (SSA) form, allowing progressive lowering from high-level machine learning concepts to low-level hardware instructions.

### II.1. IR Dialect Hierarchy

1.  **ferric-Tensor Dialect (L1):**
    - **Purpose:** High-level semantic graph.
    - **Ops:** `fml.matmul`, `fml.conv2d`, `fml.tree.predict`.
    - **Types:** Uses `tensor<...>` types, supporting dynamic dimensions.
2.  **ferric-Structured Dialect (L2):**
    - **Purpose:** Target-agnostic loop optimization (Linalg-on-Tensors).
    - **Ops:** `linalg.generic`, `scf.for`, `scf.parallel`.
    - **Constraint:** Requires static shapes resolved by the compiler.
3.  **ferric-Target Dialect (L3):**
    - **Purpose:** Direct hardware/memory interface.
    - **Ops:** `gpu.launch`, `tpu.execute`, `llvm.intrinsics`.
    - **Types:** Uses `memref<...>` types with explicit strides and memory spaces.

### II.2. IR Structure in Rust
```rust
pub struct Module {
    name: String,
    functions: Vec<Function>,
    constants: HashMap<SymbolRef, Constant>,
    types: TypeTable,
}

pub struct Operation {
    id: OpId,
    kind: OpKind,
    operands: Vec<Value>,
    results: Vec<Value>,
    attributes: AttributeDict,
}
```

---

## III. Dynamic Graph Capture & AOT-AD Engine

To achieve PyTorch-like flexibility with XLA-like performance, ferricML uses a "Define-by-Run-and-Compile" approach.

### III.1. Safe Tracing via Procedural Macros
The `#[ferric::compile]` macro performs static analysis on the forward function. It instruments control flow and identifies data-dependent branches.

### III.2. Runtime ShapeGuard Mechanism
For every execution path captured, a `ShapeGuard` is inserted at the entry point.
- **Validation:** On subsequent executions, it verifies if input shapes and control flow paths match the compiled trace.
- **Failure Mode:** If validation fails, the system falls back to the safe Eager execution path, preventing incorrect kernel execution.

### III.3. Ahead-of-Time Automatic Differentiation (AOT-AD)
Once traced and guarded, the engine performs reverse traversal of the ASG *before* IR lowering.
- **Unified Forward-Backward DAG:** This generates a single graph containing both forward and backward operations.
- **Cross-Boundary Optimization:** Allows the compiler to optimize the entire training step simultaneously (e.g., fusing weight updates with gradient calculations).

---

## IV. CUDA Backend Implementation

The CUDA backend targets NVIDIA GPUs by lowering FML-Target IR to NVVM IR and subsequently to PTX.

### IV.1. Thread Hierarchy Mapping
ferricML maps IR parallel loops to CUDA's grid/block/warp hierarchy.
- **Warp-Level Primitives:** Uses `__shfl_sync` and `__ballot_sync` for efficient reductions.
- **Tensor Cores:** Targets the WMMA (Warp Matrix Multiply-Accumulate) API for accelerated GEMM and Convolution operations.

### IV.2. Memory Hierarchy Utilization
- **Shared Memory:** Automatically managed via a tiling pass in the compiler.
- **Bank Conflicts:** Padding is inserted into shared memory buffers to ensure maximum throughput.
- **Coalescing:** The Layout Optimization pass ensures global memory accesses are coalesced.

### IV.3. Fat Binary Generation
ferricML embeds architecture-agnostic PTX and multiple AOT-compiled SASS versions (e.g., sm_80, sm_90) into a single "Fat Binary" for deployment robustness.

---

## V. CPU Backend & Vectorization

The CPU backend utilizes LLVM to generate high-performance native machine code.

### V.1. SIMD Abstraction Layer
A unified trait-based abstraction allows the compiler to target AVX2, AVX-512 (Intel/AMD), and NEON (ARM) without changing the optimization logic.

### V.2. Cache-Aware Optimization
- **Blocking/Tiling:** Matrix multiplication and convolution use blocked layouts to maximize L1/L2 cache hits.
- **NUMA Awareness:** The runtime includes a NUMA-aware allocator and thread-binding logic to minimize cross-socket memory traffic on servers.

### V.3. Work-Stealing Thread Pool
ferricML implements a custom work-stealing scheduler to balance parallel workloads across many CPU cores, preventing "tail latency" issues in batch processing.

---

## VI. ROCm & TPU Backend Integration

### VI.1. AMD ROCm Pipeline
- **Lowering:** FML-Target -> ROCDL IR -> HSA Code Object (HSACO).
- **Toolchain:** Seamless integration with `hipcc` and `amdclang++`.
- **Matrix Cores:** Leverages CDNA Matrix Core intrinsics for competitive performance against CUDA.

### VI.2. Google TPU via XLA
- **Strict Static Shapes:** mandatory resolution of all dynamic dimensions before translation.
- **StableHLO:** FML-IR Level 1 is translated directly to StableHLO, the input dialect for the XLA compiler.
- **Systolic Array Tiling:** The compiler automatically tiles data into 128x8 chunks to match the TPU's hardware structure.


---

## VII. Automatic Differentiation Engine

ferricML supports reverse-mode automatic differentiation through both tape-based recording (for Eager mode) and symbolic transformation (for Compiled mode).

### VII.1. Tape-Based Recording
During Eager execution, operations are recorded on a `ComputationTape`.

```rust
pub struct TapeEntry {
    op: Operation,
    inputs: Vec<TensorId>,
    output: TensorId,
    grad_fn: Arc<dyn GradientFunction>,
    saved_tensors: Vec<SavedTensor>,
}
```

### VII.2. Memory-Efficient Backprop
- **Checkpointing:** Users can mark subgraphs for recomputation to trade compute for memory.
- **In-place Gradients:** The engine identifies when a gradient can be accumulated in-place into an existing buffer.

### VII.3. Higher-Order Derivatives
ferricML's symbolic transformation on FML-IR enables the generation of Hessian-vector products and Jacobians without forming full matrices.

---

## VIII. Optimization Pass Infrastructure

The `PassManager` orchestrates target-agnostic and target-specific transformations on FML-IR.

### VIII.1. Operator Fusion
- **Vertical Fusion:** Combines producer-consumer chains (e.g., Matmul -> BatchNorm -> ReLU) into a single optimized kernel.
- **Horizontal Fusion:** Merges independent operations (e.g., multiple element-wise additions) to reduce kernel launch overhead.

### VIII.2. Mixed-Precision Analysis (MPA)
The MPA pass statically analyzes the Unified Forward-Backward DAG.
- **Optimization:** Heavy compute operations (GEMM, Conv) are cast to FP16 or BF16.
- **Stability:** Sensitive operations (Loss calculation, Weight updates) are maintained in FP32 using automatic loss scaling.

### VIII.3. Auto-Tuning System
Inspired by TVM, ferricML uses a cost-model-driven auto-tuner.
- **Search Space:** Different tiling factors and loop unrolling strategies.
- **Benchmark:** Empirical measurement on the target hardware to find the globally optimal configuration for each kernel.

---

## IX. Neural Network Modules

The high-level `nn` API provides ergonomic building blocks for deep learning.

### IX.1. Module Trait
All layers implement a unified `Module` trait.

```rust
pub trait Module: Send + Sync {
    type Input;
    type Output;
    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
}
```

### IX.2. Built-in Layers
- **Convolutional:** `Conv2d`, `ConvTranspose2d`.
- **Normalization:** `BatchNorm`, `LayerNorm`, `GroupNorm`.
- **Transformers:** Multi-head attention, FFN blocks, and complete Encoder/Decoder layers.
- **Containers:** `Sequential` and `ModuleList` for complex composition.


---

## X. Tree-Based Models & Ensemble Methods

ferricML provides first-class support for classical machine learning through optimized IR lowering.

### X.1. Decision Tree Construction
- **Criteria:** Supports Gini Impurity, Entropy, and Variance Reduction.
- **Split Finding:** Implements both Exact and Histogram-based split finding.

### X.2. Ensemble Methods
- **Gradient Boosting (GBDT):** Implements XGBoost-style boosting with L1/L2 regularization.
- **Random Forest:** Parallel tree construction using the CPU work-stealing thread pool.

### X.3. Tree Optimizations
- **GOSS:** Gradient-based One-Side Sampling to reduce sample size during boosting.
- **EFB:** Exclusive Feature Bundling to reduce feature dimensionality.
- **Vectorized Prediction:** Trees are compiled into vectorized IR, enabling batch predictions at millions of samples per second.

---

## XI. Probabilistic Graphical Models

Support for modeling uncertainty through structured probabilistic relationships.

### XI.1. Representation
- **Bayesian Networks:** Directed acyclic graphs with Tabular or Gaussian CPDs.
- **Markov Random Fields:** Undirected models with potential functions.

### XI.2. Inference Algorithms
- **Exact:** Variable Elimination and Junction Tree propagation.
- **Approximate:** Gibbs Sampling and Mean-Field Variational Inference.

### XI.3. Learning
- **Parameter Learning:** Maximum Likelihood Estimation (MLE) and Bayesian estimation.
- **Structure Learning:** Constraint-based (PC algorithm) and Score-based (Hill climbing) discovery.

---

## XII. Distributed Training & Serialization

### XII.1. Distributed Strategies
- **Data Parallelism:** Replicated models with synchronous gradient All-Reduce (via NCCL/RCCL).
- **Model Parallelism:** Tensor sharding (Row/Column-wise) for layers too large for a single device.
- **Pipeline Parallelism:** Microbatch-based execution across multiple stages.

### XII.2. Serialization & Fault Tolerance
- **Format:** Supports binary serialization via Bincode/FlatBuffers for zero-copy deserialization.
- **Fault-Tolerant Trainer:** Periodic checkpointing with automatic rollback and recovery from node failures.
- **Compression:** Optional Gzip/LZ4 compression for large model checkpoints.

---

## XIII. Implementation Roadmap

### Phase 1: Foundation (Months 1-3)
- Core type system and symbolic tensor handles.
- Basic ASG and FML-IR Level 1.
- Initial CPU backend with LLVM.

### Phase 2: Compiler & Acceleration (Months 4-6)
- FML-IR Levels 2 and 3.
- Safe Tracing and ShapeGuards.
- CUDA backend and Tensor Core integration.

### Phase 3: Paradigms (Months 7-9)
- High-level `nn`, `tree`, and `pgm` modules.
- Autograd AOT-AD engine.
- ROCm support.

### Phase 4: Scale & Production (Months 10-12)
- Distributed training infrastructure.
- XLA/TPU lowering path.
- Production hardening and auto-tuning.

---

**Works Cited:**
1.  LLVM Language Reference & MLIR Specification.
2.  PyTorch 2.x (TorchDynamo / Guards) Design.
3.  "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", OSDI 2018.
4.  StableHLO / OpenXLA Documentation.

---

## Appendix A: Implementation-Level Details

### A.1. Core Type System & Memory Model Implementation

#### A.1.1 Scalar Implementations
Each scalar type implements the `DType` trait with specific characteristics:

```rust
// Float32 Implementation
impl DType for f32 {
    const SIZE: usize = 4;
    const ALIGNMENT: usize = 4;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TYPE_ID: TypeId = TypeId::Float32;
    const SIMD_WIDTH: usize = 8; // AVX2 can process 8 f32s

    #[inline]
    fn from_f64(val: f64) -> Self { val as f32 }

    #[inline]
    fn to_f64(self) -> f64 { self as f64 }
}
```

#### A.1.2 Memory Pool Allocator
The memory pool minimizes allocation latency by reusing blocks:

```rust
pub struct MemoryPool {
    /// Free blocks organized by size
    free_blocks: BTreeMap<usize, Vec<*mut u8>>,

    /// Allocated blocks with their sizes
    allocated: HashMap<*mut u8, BlockInfo>,

    /// Total allocated bytes
    total_allocated: AtomicUsize,
}

impl MemoryPool {
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<*mut u8> {
        let size = size.next_power_of_two();
        if let Some(blocks) = self.free_blocks.get_mut(&size) {
            if let Some(ptr) = blocks.pop() {
                return Ok(ptr);
            }
        }
        self.allocate_new(size, alignment)
    }
}
```

### A.2. FML-IR Detailed Operation Definitions

#### A.2.1 Tensor Operations
```rust
pub enum TensorOp {
    /// %result = fml.matmul %lhs, %rhs : tensor<MxK>, tensor<KxN> -> tensor<MxN>
    MatMul,
    /// %result = fml.conv2d %input, %kernel {stride, padding}
    Conv2d,
    /// Element-wise operations
    Add, Sub, Mul, Div,
    /// Reduction operations
    ReduceSum, ReduceMax,
}
```

#### A.2.2 Affine Maps for Layouts
Affine maps express complex memory reindexing for transformations like Transpose or Tiling:
```rust
/// Transpose map: (d0, d1) -> (d1, d0)
pub fn transpose_2d() -> AffineMap {
    AffineMap {
        num_dims: 2,
        num_symbols: 0,
        results: vec![AffineExpr::Dim(1), AffineExpr::Dim(0)],
    }
}
```


### A.3. CUDA Backend Deep Dive

#### A.3.1 Launch Configuration
```rust
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
    pub shared_mem_bytes: usize,
    pub stream: CudaStream,
}

impl LaunchConfig {
    pub fn matrix(m: usize, n: usize, tile_m: u32, tile_n: u32) -> Self {
        Self {
            grid_dim: Dim3::new(((n as u32) + tile_n - 1) / tile_n, ((m as u32) + tile_m - 1) / tile_m, 1),
            block_dim: Dim3::new(tile_n, tile_m, 1),
            shared_mem_bytes: 0,
            stream: CudaStream::default(),
        }
    }
}
```

#### A.3.2 Shared Memory Tiling Example (MatMul)
```cuda
#define TILE_SIZE 16
__global__ void tiled_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

### A.4. Autograd Gradient Function Implementations

#### A.4.1 Matrix Multiplication Gradient
```rust
pub struct MatMulGradient;
impl GradientFunction for MatMulGradient {
    fn backward(&self, grad_output: &Tensor<f32>, saved: &[SavedTensor]) -> Result<Vec<Option<Tensor<f32>>>> {
        // dL/dA = grad_output @ B^T
        // dL/dB = A^T @ grad_output
        let a = saved[0].as_tensor();
        let b = saved[1].as_tensor();
        let grad_a = grad_output.matmul(&b.transpose(-1, -2)?)?;
        let grad_b = a.transpose(-1, -2)?.matmul(grad_output)?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
}
```

### A.5. Optimization Pipeline Details

#### A.5.1 Liveness Analysis for Memory Planning
```rust
fn compute_liveness(ir: &FmlIR) -> HashMap<Value, LivenessInterval> {
    let mut liveness = HashMap::new();
    let exec_order = ir.topological_sort().unwrap();
    for (time, op_id) in exec_order.iter().enumerate() {
        let op = ir.get_operation(*op_id);
        for &operand in &op.operands {
            liveness.entry(operand).and_modify(|i: &mut LivenessInterval| i.end = time);
        }
        for (idx, _) in op.results.iter().enumerate() {
            let result = Value::new_op_result(*op_id, idx);
            liveness.insert(result, LivenessInterval { start: time, end: time });
        }
    }
    liveness
}
```


### A.6. Neural Network Standard Layers

#### A.6.1 Linear Layer Forward
```rust
impl Module for Linear {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut output = input.matmul(&self.weight.transpose(-1, -2)?)?;
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.unsqueeze(0)?)?;
        }
        Ok(output)
    }
}
```

### A.7. Tree-Based Models & GBDT

#### A.7.1 Information Gain for Regression
```rust
fn compute_split_gain(&self, y: &Tensor<f32>, parent_indices: &[usize], left_indices: &[usize], right_indices: &[usize]) -> Result<f64> {
    let parent_impurity = self.compute_variance(y, parent_indices)?;
    let left_impurity = self.compute_variance(y, left_indices)?;
    let right_impurity = self.compute_variance(y, right_indices)?;
    let n_parent = parent_indices.len() as f64;
    let n_left = left_indices.len() as f64;
    let n_right = right_indices.len() as f64;
    let weighted_child_impurity = (n_left / n_parent) * left_impurity + (n_right / n_parent) * right_impurity;
    Ok(parent_impurity - weighted_child_impurity)
}
```

### A.8. Distributed Training Logic

#### A.8.1 Gradient Synchronization
```rust
fn synchronize_gradients(&mut self) -> Result<()> {
    let params_per_replica: Vec<Vec<&Tensor<f32>>> = self.replicas.iter().map(|r| r.parameters()).collect();
    for param_idx in 0..params_per_replica[0].len() {
        let mut grads: Vec<Tensor<f32>> = params_per_replica.iter().filter_map(|params| params[param_idx].grad()).collect();
        if !grads.is_empty() {
            self.comm.all_reduce(&mut grads, ReduceOp::Sum)?;
            let avg_grad = grads[0].div_scalar(self.devices.len() as f32)?;
            for replica in &mut self.replicas {
                let params_mut = replica.parameters_mut();
                params_mut[param_idx].accumulate_grad(avg_grad.clone())?;
            }
        }
    }
    Ok(())
}
```


### A.9. Probabilistic Graphical Models (PGM) Inference

#### A.9.1 Variable Elimination Algorithm
Variable elimination reduces a complex joint distribution by marginalizing out variables sequentially:
1. **Factor Creation:** Create factors from the network's Conditional Probability Distributions (CPDs) and evidence.
2. **Ordering:** Determine an elimination order (e.g., Min-Fill or Min-Degree).
3. **Elimination:** For each variable to eliminate:
    a. Find all factors mentioning the variable.
    b. Multiply these factors.
    c. Sum out the variable from the product to create a new factor.
4. **Normalization:** Multiply remaining factors and normalize to get the posterior.

```rust
fn eliminate_variable(&self, var: NodeIndex, mut factors: Vec<Factor>) -> Result<Vec<Factor>> {
    let (relevant, others): (Vec<_>, Vec<_>) = factors.into_iter().partition(|f| f.scope.contains(&var));
    if relevant.is_empty() { return Ok(others); }
    let product = self.multiply_factors(&relevant)?;
    let marginalized = product.marginalize(var)?;
    let mut result = others;
    result.push(marginalized);
    Ok(result)
}
```

### A.10. Hidden Markov Models (HMM) Algorithms

#### A.10.1 The Viterbi Algorithm
Finds the most likely sequence of hidden states:
- **Initialization:** $\delta_1(s) = \pi_s \cdot b_s(o_1)$
- **Recursion:** $\delta_t(s) = \max_{s'} (\delta_{t-1}(s') \cdot a_{s',s}) \cdot b_s(o_t)$
- **Termination:** $P^* = \max_s \delta_T(s)$

```rust
pub fn viterbi(&self, observations: &[usize]) -> Result<Vec<usize>> {
    let t_max = observations.len();
    let mut delta = vec![vec![0.0; self.n_states]; t_max];
    let mut psi = vec![vec![0; self.n_states]; t_max];
    // Initialization
    for s in 0..self.n_states { delta[0][s] = self.initial[s] * self.emission[s][observations[0]]; }
    // Recursion
    for t in 1..t_max {
        for s in 0..self.n_states {
            let (max_val, max_state) = (0..self.n_states).map(|ps| (delta[t-1][ps] * self.transition[ps][s], ps))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
            delta[t][s] = max_val * self.emission[s][observations[t]];
            psi[t][s] = max_state;
        }
    }
    // Backtrack to find optimal path
    let mut path = vec![0; t_max];
    path[t_max - 1] = delta[t_max - 1].iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap();
    for t in (0..t_max - 1).rev() {
        path[t] = psi[t + 1][path[t + 1]];
    }
    Ok(path)
}
```


### A.11. Advanced Neural Architectures: Transformer Encoder

The Transformer Encoder Layer combines multi-head self-attention with position-wise feed-forward networks:

```rust
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl Module for TransformerEncoderLayer {
    type Input = (Tensor<f32>, Option<Tensor<f32>>);
    type Output = Tensor<f32>;
    fn forward(&self, (x, mask): Self::Input) -> Result<Self::Output> {
        let attn_output = self.self_attn.forward((x.clone(), x.clone(), x.clone(), mask))?;
        let x = self.norm1.forward(x.add(&attn_output)?)?;
        let ffn_output = self.ffn.forward(x.clone())?;
        self.norm2.forward(x.add(&ffn_output)?)
    }
}
```

### A.12. Kernel Methods Implementation: SVM

Support Vector Machines find the optimal hyperplane that maximizes the margin between classes.

#### A.12.1 Quadratic Programming Solver (SMO)
ferricML implements the Sequential Minimal Optimization (SMO) algorithm to solve the dual optimization problem:
$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n y_i y_j \alpha_i \alpha_j K(x_i, x_j)$$
Subject to: $0 \le \alpha_i \le C$ and $\sum_{i=1}^n \alpha_i y_i = 0$.

```rust
pub struct SVM {
    kernel: Box<dyn Kernel>,
    C: f64,
    support_vectors: Option<Tensor>,
    alpha: Option<Tensor>,
    bias: f64,
}

impl SVM {
    pub fn fit(&mut self, X: &Tensor, y: &Tensor) -> Result<()> {
        let K = self.kernel.kernel_matrix(X);
        let alpha = self.smo_solver(&K, y)?;
        let mask = alpha.greater_than(1e-5);
        self.support_vectors = Some(X.masked_select(&mask));
        self.alpha = Some(alpha.masked_select(&mask));
        Ok(())
    }
}
```

### A.13. FML-IR Level 2 (Structured Dialect) Specification

L2 is the primary optimization hub. High-level operations are normalized into `linalg.generic` operations.

#### A.13.1 Linalg Generic Op Definition
A `linalg.generic` operation is defined by:
1. **Indexing Maps:** Affine maps that define how inputs/outputs are accessed relative to the iteration space.
2. **Iterator Types:** Mark loops as "parallel" or "reduction".
3. **Region:** The scalar computation performed at each point in the iteration space.

Example: Fused Matmul-ReLU in L2 IR
```
%res = linalg.generic {
  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                   affine_map<(d0, d1, d2) -> (d2, d1)>,
                   affine_map<(d0, d1, d2) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B) outs(%C) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %prod = arith.mulf %a, %b : f32
    %sum = arith.addf %c, %prod : f32
    %relu = arith.maxf %sum, %zero : f32
    linalg.yield %relu : f32
}
```


### A.14. CUDA Occupancy Optimization

Maximum occupancy is critical for latency hiding on NVIDIA GPUs. ferricML's CUDA backend uses an analytical model to calculate optimal block sizes:

```rust
impl CudaDevice {
    pub fn compute_occupancy(&self, threads_per_block: i32, shared_mem: usize, registers_per_thread: i32) -> f32 {
        let blocks_per_sm_threads = self.max_threads_per_sm / threads_per_block;
        let blocks_per_sm_shared = if shared_mem > 0 { self.shared_memory_per_sm / shared_mem } else { i32::MAX as usize } as i32;
        let blocks_per_sm_regs = if registers_per_thread > 0 { self.registers_per_sm / (threads_per_block * registers_per_thread) } else { i32::MAX };

        let blocks_per_sm = blocks_per_sm_threads.min(blocks_per_sm_shared).min(blocks_per_sm_regs).max(1);
        let active_warps = (blocks_per_sm * threads_per_block) / self.warp_size;
        let max_warps = self.max_threads_per_sm / self.warp_size;

        active_warps as f32 / max_warps as f32
    }
}
```

### A.15. Automatic Mixed Precision (AMP) and Loss Scaling

To prevent gradient underflow in FP16/BF16 training, ferricML implements an automatic `GradScaler`.

#### A.15.1 Scaling Algorithm
1.  **Scale Loss:** Before the backward pass, multiply the loss by a large factor $S$ (e.g., $2^{16}$).
2.  **Backprop:** Gradients are computed on the scaled loss.
3.  **Unscale Gradients:** Before updating weights, divide gradients by $S$.
4.  **Dynamic Update:** If any gradient contains `NaN` or `Inf`, skip the update and decrease $S$. Otherwise, periodically increase $S$.

```rust
pub struct GradScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
}

impl GradScaler {
    pub fn scale(&self, loss: Tensor<f32>) -> Tensor<f32> {
        loss.mul_scalar(self.scale).unwrap()
    }

    pub fn unscale_(&self, optimizer: &mut dyn Optimizer) {
        for param in optimizer.parameters() {
            if let Some(mut grad) = param.grad() {
                grad.div_scalar_(self.scale);
            }
        }
    }
}
```

### A.16. Distributed Data Parallel (DDP) Initialization

ferricML utilizes a hierarchical rendezvous system for multi-node training:

```rust
pub struct ProcessGroup {
    rank: usize,
    world_size: usize,
    backend: Arc<dyn CommunicationBackend>,
}

impl ProcessGroup {
    pub fn new_nccl(rank: usize, world_size: usize, master_addr: &str) -> Result<Self> {
        // 1. Initialize NCCL ID via TCP rendezvous
        let nccl_id = tcp_rendezvous(master_addr, rank, world_size)?;
        // 2. Initialize NCCL Communicator
        let backend = NcclBackend::init(rank, world_size, nccl_id)?;
        Ok(Self { rank, world_size, backend: Arc::new(backend) })
    }
}
```


### A.17. Quantization Formats & Lowering

ferricML supports Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).

#### A.17.1 Quantized DTypes
Quantized types (`QInt8`, `QUInt8`) store a raw integer along with a `scale` and `zero_point`.
$$x_{float} = (x_{quant} - zero\_point) \cdot scale$$

```rust
pub struct QuantizedTensor<T: DType> {
    data: Tensor<T>,
    scale: f32,
    zero_point: i32,
}
```

#### A.17.2 Integer Arithmetic Lowering
The compiler lowers high-level `fml.matmul` on quantized tensors to target-specific integer instructions (e.g., `dp4a` on NVIDIA or `vpdotp` on ARM), ensuring that the final accumulation is performed in 32-bit precision before requantization.

### A.18. Sparse Tensor Storage Formats

For large, sparse datasets (common in recommender systems), ferricML provides optimized sparse formats.

#### A.18.1 Compressed Sparse Row (CSR)
Stores non-zero values in a contiguous array, with `col_indices` and `row_ptr` arrays for indexing.

```rust
pub struct SparseCSR<T: DType> {
    values: Vec<T>,
    col_indices: Vec<usize>,
    row_ptrs: Vec<usize>,
    shape: (usize, usize),
}

impl<T: DType> SparseCSR<T> {
    pub fn get(&self, row: usize, col: usize) -> T {
        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];
        match self.col_indices[start..end].binary_search(&col) {
            Ok(idx) => self.values[start + idx],
            Err(_) => T::ZERO,
        }
    }
}
```

#### A.18.2 Coordinate (COO) Format
A simple list of (row, col, value) tuples, primarily used for building sparse matrices before conversion to CSR/CSC.

### A.19. Compiler Dialect Conversion (L2 to L3)

The "Bufferization" pass converts L2 (Linalg on Tensors) to L3 (Target MemRefs). This is a critical transition from value semantics to memory semantics.

#### A.19.1 In-place Bufferization Rules
1.  **Read-Only Tensors:** Allocated in constant memory or passed by reference.
2.  **Overwrite Tensors:** If an operation's output shape matches an input shape and that input is not used subsequently (as determined by liveness analysis), the input buffer is reused.
3.  **Allocations:** If no reuse is possible, a new buffer is requested from the `MemoryPoolAllocator`.

Example: Bufferized Add in L3 IR
```
// L2: %3 = linalg.add ins(%0, %1) outs(%2)
// L3:
%mem0 = memref.alloc() : memref<128xf32>
gpu.launch blocks(8,1,1) threads(16,1,1) args(%in0, %in1, %mem0) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %res = arith.addf %a, %b : f32
    memref.store %res, %out : memref<128xf32>
}
```
