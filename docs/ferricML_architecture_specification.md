# ferricML Architecture Specification

**Version:** 3.0
**Date:** November 2025
**Status:** Comprehensive Technical Specification - Implementation Ready
**Target:** Senior Rust Engineers & AI/ML Systems Developers

---

## 1. Executive Summary

ferricML is a comprehensive, pure-Rust, end-to-end AI and machine learning development platform. It aims to provide the ergonomic flexibility of PyTorch (Eager execution) with the production-grade performance of TensorFlow (Graph compilation), all while leveraging Rust's safety and performance characteristics. ferricML is designed to be a unified infrastructure for Deep Learning, Classical ML, and Probabilistic Modeling.

### 1.1 Core Design Mandates

1.  **Pure Rust Implementation:** No Python dependency for core execution. Leverages Rust's ownership model for zero-cost memory safety.
2.  **Unified MLIR-based Infrastructure:** A multi-level Intermediate Representation (FML-IR) stack optimized for diverse ML workloads.
3.  **Define-by-Run-and-Compile:** Bridging eager execution and static optimization using Safe Tracing, ShapeGuards, and AOT-AD.
4.  **Hardware Agnostic Backends:** Unified abstraction for CPU (LLVM), NVIDIA GPU (CUDA), AMD GPU (ROCm), and TPU (XLA).
5.  **Multi-Paradigm First-Class Support:** Unified IR dialects for Tensors, Trees, Graphs, and Kernels.

---

## 2. System Architecture Hierarchy

```mermaid
graph TD
    subgraph "Frontend Layer"
        API[Rust API: nn, tree, pgm, kernel]
        Proc[#[ferric::compile] Tracing]
        ASG[Abstract Semantic Graph]
    end

    subgraph "Compilation & Autograd"
        AOT_AD[AOT-AD Engine]
        Dialect_L1[L1: ferric-Tensor Dialect]
        Dialect_L2[L2: ferric-Structured Dialect]
        Dialect_L3[L3: ferric-Target Dialect]
    end

    subgraph "Optimization Passes"
        Fusion[Operator Fusion]
        MPA[Mixed-Precision Analysis]
        BReO[Buffer Reuse Optimization]
        AutoTune[Hardware Auto-Tuning]
    end

    subgraph "Execution Backends"
        HAL[Unified Backend Trait]
        CPU[CPU Backend]
        CUDA[CUDA Backend]
        ROCm[ROCm Backend]
        TPU[TPU Backend]
    end

    API --> Proc
    Proc --> ASG
    ASG --> AOT_AD
    AOT_AD --> Dialect_L1
    Dialect_L1 --> Dialect_L2
    Dialect_L2 --> Dialect_L3
    Dialect_L3 --> Fusion
    Fusion --> MPA
    MPA --> BReO
    BReO --> AutoTune
    AutoTune --> HAL
    HAL --> CPU
    HAL --> CUDA
    HAL --> ROCm
    HAL --> TPU
```

---

## 3. Core Type System & Memory Model

### 3.1 Scalar and DType System

The `DType` trait defines scalar behavior and bit-level layouts.

```rust
pub trait DType: Copy + Send + Sync + 'static {
    const SIZE: usize;
    const ALIGNMENT: usize;
    const ZERO: Self;
    const ONE: Self;
    const TYPE_ID: TypeId;
    const SIMD_WIDTH: usize;

    fn from_f64(val: f64) -> Self;
    fn to_f64(self) -> f64;
}

#[repr(u8)]
pub enum TypeId {
    Float16, BFloat16, Float32, Float64,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Bool, Complex64, Complex128,
    QInt8, QUInt8, QInt32
}
```

### 3.2 Symbolic Tensor and ASG

Tensors in ferricML are primarily symbolic handles.

```rust
pub struct Tensor<B: FerricBackend> {
    id: u64, // Unique ID linking to the ASG::Node
    shape: Shape,
    device: B::Device,
    grad_fn_ref: Option<GradFnId>,
}
```

The **Abstract Semantic Graph (ASG)** tracks computation for both eager execution and subsequent compilation.

```rust
pub struct ASGNode {
    op: FerricOp,
    inputs: Vec<u64>, // Tensor IDs
    requires_grad: bool,
    backward_closure: Option<Box<dyn Fn(Vec<Tensor>) -> Vec<Tensor>>>,
}
```

### 3.3 Memory Model

1.  **Unified Virtual Addressing (UVA):** Where supported, memory is treated as a single address space across CPU and accelerators.
2.  **Memory Pool Allocator:** B-Tree based management of free blocks to minimize fragmentation and allocation latency.
3.  **Buffer Reuse Optimization (BReO):** Static analysis of tensor lifetimes to allow in-place operations and buffer sharing.

---

## 4. FML Intermediate Representation (FML-IR)

FML-IR follows the MLIR design, allowing high-level semantics to be progressively lowered.

### 4.1 L1: ferric-Tensor Dialect (High-Level)
-   **Purpose:** Preserves ML-specific semantics.
-   **Operations:** `fml.matmul`, `fml.conv2d`, `fml.tree.split`, `fml.pgm.infer`.
-   **Types:** Uses `tensor<...>` types, supporting dynamic dimensions.

### 4.2 L2: ferric-Structured Dialect (Mid-Level)
-   **Purpose:** Target-agnostic loop optimization.
-   **Operations:** `linalg.generic`, `scf.for`, `scf.parallel`.
-   **Constraint:** Requires static shapes (resolved by Shape Specialization Pass).

### 4.3 L3: ferric-Target Dialect (Low-Level)
-   **Purpose:** Direct hardware interface.
-   **Operations:** `gpu.launch`, `tpu.execute`, `llvm.intrinsics`.
-   **Types:** Uses `memref<...>` types with explicit strides and memory spaces (Shared, Global, Constant).

---

## 5. Autograd: Tracing & AOT-AD

### 5.1 Safe Tracing via Procedural Macros
The `#[ferric::compile]` macro performs static analysis on the forward function. It instruments control flow and inserts **ShapeGuards**.

### 5.2 Runtime ShapeGuards
Guards verify:
1.  Input tensor shapes match the compiled trace.
2.  Control flow paths (if/while) remain stable.
3.  Data-dependent branches are handled by falling back to Eager execution if they deviate.

### 5.3 AOT Automatic Differentiation (AOT-AD)
Once traced and guarded, the engine performs reverse traversal of the ASG *before* lowering to MLIR. This generates a unified graph containing both forward and backward operations, enabling cross-boundary optimizations like fusing weight updates with gradient calculations.

---

## 6. Hardware Backends

### 6.1 Backend Trait System
All hardware targets implement the `FerricBackend` trait using the **Decorator Pattern**.

```rust
pub trait FerricBackend: Clone + Send + Sync + 'static {
    type Device: Clone + Send + Sync;
    type Elem: DType;

    fn execute(kernel: &CompiledKernel, inputs: &[&Tensor]) -> Result<Tensor>;
}
```

### 6.2 Specific Backend Implementations

-   **NVIDIA CUDA:** Lowering to NVVM IR -> PTX. Generates **Fat Binaries** for runtime architecture dispatch. Exploits **Tensor Cores** via the WMMA API.
-   **AMD ROCm:** Lowering to ROCDL IR -> HSA Code Object (HSACO). Integrated with the **HIP** runtime.
-   **CPU Backend:** LLVM-based JIT/AOT. Implements **Work-Stealing Thread Pools** and target-specific SIMD (AVX-512, NEON).
-   **TPU Backend:** Lowering to **StableHLO**. Leverages the **XLA** compiler for systolic array tiling (128x8 chunks) and HBM management.

---

## 7. Optimization Pipeline

The `PassManager` executes a pipeline of critical transformations:

1.  **Operator Fusion:**
    -   *Vertical:* Producer-consumer fusion (e.g., Matmul + ReLU).
    -   *Horizontal:* Independent parallel op fusion.
2.  **Mixed-Precision Analysis (MPA):** Statically analyzes the graph to insert `fml.cast` operations. Optimizes heavy compute (GEMM/Conv) to FP16/BF16 while keeping sensitive ops (Loss/Update) in FP32.
3.  **Layout Optimization:** Selection of optimal memory layouts (NCHW vs NHWC) based on target device characteristics.
4.  **Auto-Tuning:** Empirical search for optimal tile sizes and block configurations, using evolutionary algorithms and cost models.

---

## 8. Unified Machine Learning Support

### 8.1 Neural Networks (`ferric::nn`)
-   Standard layer implementations (`Linear`, `Conv2d`, `BatchNorm`, `Transformer`).
-   Initialization strategies: Xavier/Kaiming/Orthogonal.
-   Distributed training support: Data Parallel (via NCCL/RCCL) and Pipeline Parallelism.

### 8.2 Tree-Based Models (`ferric::tree`)
-   First-class support for **Histogram-based GBDT**.
-   Optimization via **Exclusive Feature Bundling (EFB)** and **Gradient-based One-Side Sampling (GOSS)**.
-   Direct compilation to vectorized CPU instructions or GPU kernels.

### 8.3 Probabilistic Graphical Models (`ferric::pgm`)
-   Directed (Bayesian) and Undirected (MRF) graph support.
-   Exact inference via **Variable Elimination** and **Junction Tree**.
-   Approximate inference via **Gibbs Sampling** and **Mean-Field VI**.

### 8.4 Kernel Methods (`ferric::kernel`)
-   Optimized Support Vector Machines (SVM) and Gaussian Processes.
-   GPU-accelerated kernel matrix computation.

---

## 9. Distribution & Production

### 9.1 Distributed Training
-   **Data Parallelism:** Synchronous All-Reduce (NCCL).
-   **Model Parallelism:** Row-wise/Column-wise sharding for massive linear layers.
-   **Pipeline Parallelism:** Microbatching to maximize device utilization.

### 9.2 Serialization
-   **Format:** Binary serialization using FlatBuffers/Bincode for zero-copy loading.
-   **Checkpointing:** Fault-tolerant trainers with periodic state-saving and automatic recovery.

---

## 10. Development Roadmap

1.  **Phase 1: Foundation:** Scalar system, Symbolic Tensors, ASG, and Basic CPU Backend.
2.  **Phase 2: Compiler:** FML-IR Dialects, AOT-AD, and CUDA/ROCm pipelines.
3.  **Phase 3: Paradigms:** High-level APIs for NN, Tree, and PGM modules.
4.  **Phase 4: Scale:** Distributed training infrastructure, Auto-tuning, and TPU/XLA integration.

---

**Works Cited:**
- LLVM/MLIR Language Reference
- PyTorch 2.x (TorchDynamo/Guards) Architecture
- XLA/StableHLO Specification
- Burn (Rust DL) Backend Trait Design
