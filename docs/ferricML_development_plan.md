# ferricML Development Plan

This document outlines the step-by-step implementation strategy for ferricML, as defined in the `ferricML Architecture Specification`.

## Phase 1: Core Infrastructure (Months 1-3)
*Goal: Establish the foundational type system, symbolic tensor representation, and basic CPU execution.*

1.  **Fundamental Types & Memory:**
    *   Implement the `DType` trait and core scalar types.
    *   Develop the `Storage` abstraction and `CpuStorage` backend.
    *   Implement basic `Shape` and `Strides` logic.
2.  **Symbolic Core & ASG:**
    *   Implement the symbolic `Tensor` handle.
    *   Develop the `AbstractSemanticGraph (ASG)` data structures and node types.
    *   Create basic tensor operations that populate the ASG.
3.  **Basic CPU Backend:**
    *   Implement a minimal `ferricBackend` for CPU.
    *   Integrate LLVM-based JIT for executing simple element-wise operations.
    *   Implement a basic thread pool for parallel execution.

## Phase 2: FML-IR & AOT-AD (Months 4-6)
*Goal: Build the multi-level compiler stack and the differentiation engine.*

1.  **FML-IR Implementation:**
    *   Define the L1 (Tensor), L2 (Structured), and L3 (Target) dialects using an MLIR-like structure.
    *   Implement the `PassManager` and basic lowering passes (L1 -> L2 -> L3).
2.  **Advanced Autograd:**
    *   Develop the procedural macros for Safe Tracing (`#[ferric::compile]`).
    *   Implement the `ShapeGuard` mechanism for runtime validation.
    *   Build the `AOT-AD` engine for generating the Unified Forward-Backward DAG.
3.  **Primary Optimization Passes:**
    *   Implement vertical and horizontal operator fusion.
    *   Develop the Mixed-Precision Analysis (MPA) pass.
    *   Implement the Bufferization and BReO passes.

## Phase 3: GPU Backends & ML Paradigms (Months 7-9)
*Goal: Enable hardware acceleration and high-level user APIs.*

1.  **CUDA Backend:**
    *   Implement the NVVM/PTX lowering pipeline.
    *   Integrate WMMA API for Tensor Core support.
    *   Implement CUDA-specific memory management and streams.
2.  **ROCm Backend:**
    *   Develop the ROCDL lowering pipeline and HSACO generation.
    *   Integrate with the HIP runtime.
3.  **Neural Network API (`nn`):**
    *   Implement the `Module` trait and standard layers (Linear, Conv2d, Transformer).
    *   Add common activation functions and normalization layers.
4.  **Tree-Based Models (`tree`):**
    *   Implement histogram-based GBDT construction.
    *   Develop vectorized/GPU kernels for tree prediction.

## Phase 4: Scale & Production (Months 10-12)
*Goal: Support distributed workloads, TPUs, and production hardening.*

1.  **Distributed Training:**
    *   Implement the `CommunicationBackend` trait with NCCL/RCCL integration.
    *   Develop Data Parallel and Pipeline Parallel trainers.
    *   Add gradient compression (Top-K/Quantization).
2.  **TPU & XLA Integration:**
    *   Implement the StableHLO lowering path.
    *   Integrate the XLA compiler for TPU execution.
3.  **Classical & Probabilistic ML:**
    *   Implement `pgm` (Bayesian Networks/Inference) and `kernel` (SVM/GP) modules.
4.  **Production Hardening:**
    *   Implement comprehensive serialization and fault-tolerant checkpointing.
    *   Perform extensive benchmarking and hardware auto-tuning.
    *   Finalize documentation and model zoo.
