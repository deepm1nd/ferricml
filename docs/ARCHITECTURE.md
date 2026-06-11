# FerricML Architecture Specification (RMLC)

This document provides a comprehensive technical specification of the **Rust Machine Learning Compiler (RMLC)**, the core engine powering FerricML.

## 1. Introduction

RMLC is a multi-stage, hardware-agnostic compiler designed to transform symbolic machine learning graphs into optimized machine code. Unlike interpreted frameworks, FerricML treats the entire model lifecycle as a compilation problem, enabling aggressive optimizations across the forward and backward passes.

## 2. Multi-Level Intermediate Representation (IR)

RMLC utilizes a tiered IR architecture inspired by MLIR, allowing for optimizations at the appropriate level of abstraction.

### 2.1 The Tensor Dialect (High-Level)
The Tensor Dialect operates on high-level mathematical concepts.
- **Operations:** `MatMul`, `Conv2d`, `BatchNorm`, `Softmax`.
- **Optimizations:**
    - **Operator Fusion:** Combining consecutive element-wise operations into a single kernel (e.g., `ReLU(Add(x, y))`).
    - **Dead Code Elimination:** Removing nodes that do not contribute to the final output or gradient.
    - **Constant Folding:** Evaluating subgraphs with constant inputs at compile-time.

### 2.2 The Structured Dialect (Mid-Level)
The Structured Dialect bridges the gap between math and hardware execution. It represents computation as structured loops and memory access patterns.
- **Operations:** `Linalg.generic`, `Tiled.loop`.
- **Optimizations:**
    - **Loop Tiling:** Breaking large operations into cache-sized chunks.
    - **Vectorization:** Mapping operations to SIMD instructions (AVX, NEON).
    - **Memory Layout Transformation:** Switching between Row-Major and Column-Major layouts based on hardware preference.

### 2.3 The Target Dialect (Low-Level)
The Target Dialect is specific to the hardware backend.
- **Backends:**
    - **LLVM:** For CPUs.
    - **NVVM / PTX:** For NVIDIA GPUs.
    - **ROCDL:** For AMD GPUs.
    - **StableHLO:** For TPUs.

## 3. Ahead-of-Time (AOT) Automatic Differentiation

Traditional frameworks use a dynamic "tape" to record operations for differentiation. FerricML uses **AOT-AD**, which offers several critical advantages:

### 3.1 The Differentiation Algorithm
RMLC implements reverse-mode differentiation at the symbolic level. Given a forward graph $F$, RMLC generates a backward graph $B$ such that:
$$ \nabla L = B(F(x), L) $$
Both $F$ and $B$ are then compiled into a single optimized executable.

### 3.2 Advantages of AOT-AD
1. **Zero Runtime Overhead:** No need to manage a dynamic graph data structure during training.
2. **Cross-Pass Optimization:** The compiler can optimize the forward and backward passes together, for example, by reusing intermediate buffers and reducing memory pressure.
3. **Static Memory Planning:** RMLC can pre-calculate the exact memory requirements for the entire training step, eliminating dynamic allocations.

## 4. ShapeGuards: The Type-Safe ML Engine

ShapeGuards are a core innovation in FerricML that bring Rust's "if it compiles, it works" philosophy to machine learning.

- **Static Guards:** Whenever possible, tensor dimensions are tracked as part of the type system, allowing the Rust compiler to catch dimension mismatches (e.g., `[B, 128] * [256, 10]`) at build time.
- **Dynamic Guards:** For cases where shapes are only known at runtime (e.g., variable batch sizes), RMLC injects highly efficient guard checks at the boundaries of the compiled kernels.

## 5. Execution Model

FerricML supports two execution modes:
1. **Eager Mode:** Operations are executed immediately on the selected device. Useful for debugging and research.
2. **Compiled Mode (Graph Mode):** The entire model is traced, optimized by RMLC, and executed as a single unit. This mode is recommended for production and large-scale training.

## 6. Serialization and Interoperability

RMLC generates extensive metadata for every compiled model, which is stored natively in the **GGUF (v3)** format. This ensures that models trained in FerricML are not just safe (via **SafeTensors**) but also highly portable across different inference engines.
