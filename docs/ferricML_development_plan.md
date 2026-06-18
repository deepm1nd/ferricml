# Development Plan: ferricML
v.0.0.01

## 1. Introduction
- **Purpose:** Provide a detailed, unit-level decomposition for implementing the ferricML platform.
- **Scope:** Covers all phases from core foundation to distributed production deployment.
- **References:** `docs/ferricML_architecture_specification.md`

## 2. Technology Stack
- **Languages:** Rust (Stable/Nightly).
- **Compiler IR:** MLIR (Melior), LLVM.
- **Parallelism:** Rayon, NCCL/RCCL.
- **Serialization:** Bincode, FlatBuffers.

## 3. Project Folder Structure
```
ferricml/
├── ferric-core/           # DType, Shape, Strides, Symbolic Tensor
├── ferric-asg/            # ASG nodes, Tracing engine
├── ferric-compiler/       # FML-IR Dialects, Optimization passes
├── ferric-runtime/        # Storage, Memory Pool, Device drivers
├── ferric-macros/         # #[ferric::compile] proc macros
├── ferric-nn/             # Deep Learning modules
├── ferric-tree/           # GBDT, Random Forest, Histogram building
├── ferric-pgm/            # Bayesian Nets, Factor graphs
├── ferric-kernel/         # SVM, GP implementations
└── ferric-distributed/    # DDP, Pipeline parallelism, RPC
```

## 4. Phases and Milestones

| Phase | Milestone | Duration | Target |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Foundation & CPU Backend | Months 1-3 | Functional Eager CPU execution. |
| **Phase 2** | Compiler Stack & AOT-AD | Months 4-6 | Functional compiled DAG execution. |
| **Phase 3** | GPU Acceleration & Paradigms | Months 7-9 | CUDA/ROCm support + NNs & Trees. |
| **Phase 4** | Scale, TPU & Production | Months 10-12 | Distributed training & XLA integration. |

## 5. Task Decomposition

### Phase 1: Foundation (UNIT-CORE)

- **Task ID:** `CORE-001`
- **Description:** Implement `ferric_core::dtype::DType` trait and f32/f16/i32 implementations.
- **Component(s):** `ferric-core/src/dtype.rs`
- **Dependencies:** None
- **Exit Criteria:** `f32::TYPE_ID` returns `Float32`; alignment matches Rust layout.
- **Testing:** Unit tests verifying bit-level compatibility and `from_f64` accuracy.

- **Task ID:** `CORE-002`
- **Description:** Implement symbolic `ferric_core::tensor::Tensor` handle and `Shape`/`Strides`.
- **Component(s):** `ferric-core/src/tensor.rs`, `ferric-core/src/layout.rs`
- **Dependencies:** `CORE-001`
- **Exit Criteria:** Tensor creation records a unique ID in the global ASG singleton.
- **Testing:** Verified by creating tensors and checking ASG node count.

- **Task ID:** `RUNTIME-001`
- **Description:** Implement `ferric_runtime::memory::MemoryPool` with B-Tree block tracking.
- **Component(s):** `ferric-runtime/src/memory/pool.rs`
- **Dependencies:** None
- **Exit Criteria:** Allocations are reusable; fragmentation is minimized via block merging.
- **Testing:** Benchmark for allocation latency; stress test with thousands of small buffers.

- **Task ID:** `ASG-001`
- **Description:** Implement `ferric_asg::graph::AbstractSemanticGraph` registry.
- **Component(s):** `ferric-asg/src/graph.rs`
- **Dependencies:** `CORE-002`
- **Exit Criteria:** Graph correctly maintains directed edges between operation nodes.
- **Testing:** Visualization of a simple Matmul graph to DOT format.

### Phase 2: Compiler & Autograd (UNIT-COMP)

- **Task ID:** `IR-001`
- **Description:** Define L1 (Tensor) Dialect in Melior.
- **Component(s):** `ferric-compiler/src/dialects/l1.rs`
- **Dependencies:** `ASG-001`
- **Exit Criteria:** ASG can be lowered to L1 IR string representation.
- **Testing:** Round-trip test: ASG -> L1 IR -> Parse -> Verify structure.

- **Task ID:** `AD-001`
- **Description:** Implement AOT-AD reverse traversal on ASG.
- **Component(s):** `ferric-compiler/src/autograd/aot_ad.rs`
- **Dependencies:** `ASG-001`
- **Exit Criteria:** Backward nodes for Matmul and Add are correctly generated.
- **Testing:** Verify gradient value of $y = x^2$ is $2x$ at L1 IR level.

- **Task ID:** `MACRO-001`
- **Description:** Implement `#[ferric::compile]` proc-macro for tracing.
- **Component(s):** `ferric-macros/src/compile.rs`
- **Dependencies:** `ASG-001`
- **Exit Criteria:** Rust control flow is intercepted; `ShapeGuards` are inserted.
- **Testing:** Compile a function with an `if` statement and verify guard failure on path change.

### Phase 3: Hardware Acceleration & Paradigms (UNIT-ACCEL)

- **Task ID:** `CUDA-001`
- **Description:** Implement `CudaBackend` and NVVM lowering pass.
- **Component(s):** `ferric-compiler/src/lowering/cuda.rs`, `ferric-runtime/src/backends/cuda.rs`
- **Dependencies:** `IR-001`
- **Exit Criteria:** Valid PTX code is generated for a Tiled Matmul operation.
- **Testing:** Execution on NVIDIA GPU verifying results against reference CPU implementation.

- **Task ID:** `NN-001`
- **Description:** Implement `Linear` and `Conv2d` modules.
- **Component(s):** `ferric-nn/src/layers/linear.rs`, `ferric-nn/src/layers/conv.rs`
- **Dependencies:** `CORE-002`, `AD-001`
- **Exit Criteria:** Training a small MLP results in decreasing loss on XOR dataset.
- **Testing:** MLP training integration test.

- **Task ID:** `TREE-001`
- **Description:** Implement histogram-based split finding unit.
- **Component(s):** `ferric-tree/src/splits/histogram.rs`
- **Dependencies:** `CORE-002`
- **Exit Criteria:** Continuous features are correctly binned into 256 buckets.
- **Testing:** Precision check of histogram sums against exact sums.

### Phase 4: Scale & Production (UNIT-SCALE)

- **Task ID:** `DIST-001`
- **Description:** Implement NCCL-based All-Reduce communication backend.
- **Component(s):** `ferric-distributed/src/comm/nccl.rs`
- **Dependencies:** `CUDA-001`
- **Exit Criteria:** Gradients are successfully averaged across 2 GPUs.
- **Testing:** Multi-GPU training script check.

- **Task ID:** `TPU-001`
- **Description:** Implement StableHLO translation for L1 IR.
- **Component(s):** `ferric-compiler/src/lowering/tpu.rs`
- **Dependencies:** `IR-001`
- **Exit Criteria:** IR can be serialized to StableHLO bytecode.
- **Testing:** XLA compilation check (dry run).

- **Task ID:** `PGM-001`
- **Description:** Implement Variable Elimination inference unit.
- **Component(s):** `ferric-pgm/src/inference/ve.rs`
- **Dependencies:** `CORE-002`
- **Exit Criteria:** Correct marginal probability calculated for "Student" network.
- **Testing:** Verified against Scikit-PGM or similar reference.

## 6. Test Strategy & Plan
- **Unit Testing:** Mandatory for all `.rs` files in `src/`. Coverage target: 90%.
- **Regression Testing:** Automated suite running on all PRs using GitHub Actions with GPU runners.
- **Performance Testing:** Weekly automated benchmarks tracking FLOPs and memory bandwidth utilization.

## 7. Logging Strategy
- Implementation of mandated 8-level system in `ferric-core/src/utils/logger.rs`.
- Integration of `NOTICE` level as default in `Runtime::init()`.
- Telemetry logging for GPU memory utilization using Task `LOG-001`.

---

## Appendix R - Revision History
| Version | Date | Author | Changes |
|---|---|---|---|
| 0.0.01  | 2025-11-20 | Jules | Unit-level decomposition across all 4 implementation phases. |
