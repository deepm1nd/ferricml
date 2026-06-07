# Development Checklist: ferricML

This checklist tracks the implementation progress of ferricML according to the `docs/ferricML_development_plan.md`.

## Phase 1: Foundation (UNIT-CORE)
- [ ] **CORE-001:** Implement `DType` trait and scalar implementations.
- [ ] **CORE-002:** Implement symbolic `Tensor` handle and `Shape`/`Strides`.
- [ ] **RUNTIME-001:** Implement `MemoryPool` with B-Tree block tracking.
- [ ] **ASG-001:** Implement `AbstractSemanticGraph` registry.
- [ ] **STORAGE-001:** Implement `CpuStorage` unit.

## Phase 2: Compiler & Autograd (UNIT-COMP)
- [ ] **IR-001:** Define Level 1 (ferric-Tensor) MLIR dialect.
- [ ] **IR-002:** Define Level 2 (ferric-Structured) MLIR dialect.
- [ ] **IR-003:** Define Level 3 (ferric-Target) MLIR dialect.
- [ ] **AD-001:** Implement AOT-AD reverse traversal logic.
- [ ] **MACRO-001:** Implement `#[ferric::compile]` procedural macro.
- [ ] **GUARD-001:** Implement `ShapeGuard` runtime validation.
- [ ] **PASS-001:** Implement Vertical and Horizontal Operator Fusion.

## Phase 3: Hardware Acceleration & Paradigms (UNIT-ACCEL)
- [ ] **CUDA-001:** Implement `CudaBackend` and NVVM lowering pass.
- [ ] **CUDA-002:** Integrate WMMA API for Tensor Cores.
- [ ] **ROCM-001:** Implement ROCDL lowering and HSACO generation.
- [ ] **NN-001:** Implement `Linear` and `Conv2d` modules.
- [ ] **TREE-001:** Implement histogram-based split finding unit.
- [ ] **KERNEL-001:** Implement SVM with SMO solver.

## Phase 4: Scale & Production (UNIT-SCALE)
- [ ] **DIST-001:** Implement NCCL-based All-Reduce communication backend.
- [ ] **DIST-002:** Implement Pipeline Parallelism with microbatching.
- [ ] **TPU-001:** Implement StableHLO translation for L1 IR.
- [ ] **SER-001:** Implement Bincode/FlatBuffers model serialization.
- [ ] **FAULT-001:** Implement automatic failure recovery in `Trainer`.

## Documentation & Logging
- [ ] **LOG-001:** Implement 8-level logging system.
- [ ] **DOC-001:** Generate comprehensive API documentation and examples.
