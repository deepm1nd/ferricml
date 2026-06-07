

# **RMLC: A Rust Native Deep Learning Compiler Architecture Specification (V3)**

## **Master Document Table of Contents (TOC)**

| Document No. | Section Title | Focus Area |
| :---- | :---- | :---- |
| I. | **Architectural Overview & Rust Native Frontend (RMLC.rs)** | Core Rust API design, Tensor structure, and modular trait system for Heterogeneous Backends (Burn-style). |
| II. | **Dynamic Graph Capture and AOT-AD Engine** | Technical specification of Safe Tracing via Rust Procedural Macros and Runtime Guards, and Ahead-of-Time Automatic Differentiation (AOT-AD) trace generation. |
| III. | **RMLC Multi-Level Intermediate Representation (RMLC-MLIR)** | Formal specification of RMLC-Tensor, RMLC-Structured (Linalg-based), and RMLC-Target Dialects, including type system and Rust FFI (Melior) requirements. |
| IV. | **Core Optimization Passes and Buffer Management** | Deep dive into critical target-agnostic passes: Shape Specialization, Compiler-Driven Mixed-Precision Analysis (MPA), and the Bufferization Pass (Tensor-to-MemRef conversion). |
| V. | **Heterogeneous Backend Code Generation Abstract Interface** | Specification of the LLVM CodeGen interface: TargetLowering class implementation, SelectionDAG lowering, and the Machine Code (MC) Layer for JIT/AOT dispatch. |
| VI. | **NVIDIA GPU (CUDA) Code Generation Pipeline** | Lowering to NVVM, PTX generation, SASS final assembly, Fat Binary structure, and runtime linking via the CUDA driver. |
| VII. | **AMD GPU (ROCm) Code Generation Pipeline** | Lowering to ROCDL, HSA Code Object (HSACO) generation, hipcc/amdclang++ integration, and specialized AMDGPU metadata usage. |
| VIII. | **TPU and ML Accelerator Integration via XLA** | Requirements for mandatory static shapes, translation to StableHLO IR, and XLA optimization for systolic array tiling and specialized memory spaces (VMEM/SMEM). |

---

## **I. Architectural Overview & Rust Native Frontend (RMLC.rs)**

The RMLC framework must be built around Rust's core principles of ownership, performance, and safety, creating an API that is generic over the execution backend.

### **I.1. RMLC Component Hierarchy and Crates**

The framework is segmented into minimal, highly focused Rust crates, maximizing modularity, similar to the LLVM architecture 1:

1. **rmlc-core:** Defines the core symbolic Tensor structure, DType, and the Device enum.  
2. **rmlc-backend-traits:** Defines the public RMLCBackend trait, the decorator Autodiff\<B\>, and memory management traits (Alloc).  
3. **rmlc-asg:** Implements the runtime **Abstract Semantic Graph (ASG)** data structure.  
4. **rmlc-compiler:** Contains the main compiler driver, the MLIR FFI (via melior), and the optimization passes.  
5. **rmlc-runtime:** The minimal library linked into the final executable, managing memory, kernel launch, and the AOT/JIT dispatcher.

### **I.2. The Symbolic RMLC Tensor**

The rmlc\_core::Tensor\<B: RMLCBackend\> is the core user-facing handle. It is **symbolic**, not data-containing, ensuring that operations build the ASG (Define-then-Run).2

Rust

pub struct Tensor\<B: RMLCBackend\> {  
    id: u64, // Unique ID linking to the ASG::Node  
    dtype: B::FloatElem, // Backend-associated floating point type  
    shape: Shape, // Inferred or known dimension sizes (Vec\<usize\>)  
    device: B::Device, // The target device (CPU, GPU, etc.)  
    // Runtime link for eager execution and differentiation tracing  
    grad\_fn\_ref: Option\<GradFnId\>,   
}

### **I.3. The Modular Backend Trait System**

The core design principle is the **Backend Trait Decorator Pattern** 3:

1. **RMLCBackend Trait:** Must be implemented by all hardware targets:  
   Rust  
   pub trait RMLCBackend: Clone \+ Send \+ Sync \+ 'static {  
       type Device: Clone \+ Send \+ Sync;  
       type Elem: Copy \+ Debug; // Element type (e.g., f32, f64)

       // Core Primitive Operations (to be lowered to kernels)  
       fn matmul(lhs: \&Tensor\<Self\>, rhs: \&Tensor\<Self\>) \-\> Tensor\<Self\>;  
       fn conv2d(...) \-\> Tensor\<Self\>;  
       //... (etc.)  
   }

2. **Autodiff\<B\> Decorator Trait:** This struct **implements** RMLCBackend by wrapping a base backend (B).3 When a tensor operation is called on Autodiff\<B\>, the decorator *intercepts* the call, records the operation and its inputs in the **ASG**, registers the backward closure (grad\_fn), and then may optionally execute the kernel (eager mode) or return a new symbolic tensor (static mode). This cleanly separates the high-performance kernel logic from the differentiation logic.3

---

## **II. Dynamic Graph Capture and AOT-AD Engine**

The goal is to safely translate the Rust program's dynamic execution into a static, optimizable computation graph, overcoming the challenge of dynamic control flow.5

### **II.1. ASG Node Structure for Autodiff**

The rmlc-asg crate manages the AbstractSemanticGraph (ASG), a dynamic DAG of CNodes (computation nodes).7

| Field Name | Type | Purpose in AD/Tracing |
| :---- | :---- | :---- |
| OpId | u64 | Unique operation identifier. |
| OpType | enum RMLCOp | High-level operation (e.g., Matmul, Relu). |
| Inputs | Vec\<TensorId\> | Direct dependencies for the forward pass. |
| RequiresGrad | bool | Flag to denote if the node is a parameter or requires gradient flow.4 |
| **BackwardClosure** | Box\<dyn Fn(...) \-\> Vec\<GradientOp\>\> | The concrete Rust closure capturing the derivative logic (grad\_fn).4 |

### **II.2. Safe Graph Capture via Procedural Macros and Runtime Guards**

The framework employs a **Define-by-Run-and-Compile** paradigm.5 The critical **Eager-to-Static Transition** is handled by:

1. **\#\[rmlc::compile\] Procedural Macro:** Applied to the model's forward function. At Rust compile time, this macro analyzes the function's AST.  
   * It instruments the code to trace the ASG creation during the first execution run (Eager Trace).  
   * It identifies Rust control flow (if/while) that depends on a computed **tensor value** (e.g., if tensor.mean() \> 0.5). This is dynamic and unsafe for static compilation.9  
2. **Runtime ShapeGuard Mechanism:** For every execution path captured, RMLC inserts a runtime rmlc\_runtime::ShapeGuard.  
   * **Capture:** The ShapeGuard records the exact input tensor shapes and the control flow path taken during the *initial* trace.9  
   * **Validation:** On subsequent executions, the ShapeGuard verifies if the new input shapes match the recorded shapes and if the control flow path is identical.  
   * **Failure Mode:** If the validation fails (e.g., input batch size changes, or a conditional branch is taken that was not captured in the trace), the Guard is *invalidated*. The system must then fall back to the slower, safe eager execution path, preventing the use of the statically compiled (and now incorrect) kernel, similar to how TorchDynamo utilizes Guards.9

### **II.3. Ahead-of-Time Automatic Differentiation (AOT-AD)**

Once the forward ASG trace is successfully guarded and validated as static, the AOT-AD engine takes over.9

* It performs a reverse traversal from the loss tensor (root), applying the recorded BackwardClosures.  
* The result is the insertion of concrete gradient operations (e.g., rmlc.add\_grad, rmlc.transpose) directly into the graph **before** any lowering to MLIR.  
* This creates the final **Unified Forward-Backward DAG**, which allows all optimization passes (Section IV) to operate on the entire training step simultaneously.10

---

## **III. RMLC Multi-Level Intermediate Representation (RMLC-MLIR)**

RMLC uses the Multi-Level Intermediate Representation (MLIR) framework, accessed via the Rust binding melior. This MLIR stack defines three critical Dialects, ensuring a modular and target-agnostic optimization pipeline.11

### **III.1. Lowering Pipeline Sequence**

$$\\text{ASG (Rust Dynamic Graph)} \\xrightarrow{\\text{ASG-to-Tensor Pass}} \\text{RMLC-Tensor (L1)} \\xrightarrow{\\text{Shape Specialization}} \\text{RMLC-Structured (L2)} \\xrightarrow{\\text{Bufferization}} \\text{RMLC-Target (L3)} \\xrightarrow{\\text{Target Dispatch}} \\text{Vendor LLVM IR (NVVM/ROCDL/XLA)}$$

### **III.2. RMLC Dialect Specification**

1. **RMLC-Tensor Dialect (Level 1: Semantic Graph)**  
   * **Purpose:** High-level representation of the model structure.13 Retains tensor semantics.  
   * **Key Operations:** rmlc.matmul, rmlc.conv2d, rmlc.collective\_allreduce (for distributed training 14).  
   * **Type System:** Uses MLIR **Tensor Types** (tensor\<...xf32\>), supporting dynamic dimensions (tensor\<4x?xf32\>) via the tensor.dim operation for runtime shape queries.15  
   * **AD Primitives:** Includes dedicated operations for gradient tracking and optimization (e.g., rmlc.param\_update).  
2. **RMLC-Structured Dialect (Level 2: Target-Agnostic Loops)**  
   * **Purpose:** The single optimization hub, equivalent to the **Linalg on Tensors** dialect.13 All high-level ops are normalized into loop nest representations (linalg.generic).  
   * **Constraints:** Requires static, known tensor shapes (tensor\<4x8xf32\>).  
   * **Key Operations:** linalg.generic (defines indexing maps/iterator types), scf.for (structured control flow). This level decouples architectural optimization (tiling, fusion) from instruction selection.16  
3. **RMLC-Target Dialect (Level 3: Memory Interface)**  
   * **Purpose:** Interface to the LLVM backend. Uses memory-centric types and defines kernel execution boundaries.  
   * **Type System:** Uses MLIR **MemRef Types** (memref\<...xf32, strided\<...\>\>), which explicitly encode memory layout, strides, and offsets, the output of the Bufferization Pass.  
   * **Key Operations:** gpu.launch, gpu.func (for kernel definition/outlining), tpu.execute, and direct target intrinsics.

---

## **IV. Core Optimization Passes and Buffer Management**

These passes are applied sequentially to the MLIR stack to prepare the graph for final code generation.

### **IV.1. Shape Specialization Pass (Mandatory)**

This pass operates on the RMLC-Tensor $\\rightarrow$ RMLC-Structured transition.18

* It enforces the crucial constraint required by accelerators (especially **TPU** 16): all dynamic dimensions (? in tensor\<4x?xf32\>) must be resolved to concrete static sizes (e.g., tensor\<4x8xf32\>).  
* The pass utilizes the shape information captured by the Runtime Guards (Section II.2). Failure of this pass (i.e., if a dimension remains truly dynamic) halts compilation for XLA/TPU targets.

### **IV.2. Compiler-Driven Mixed-Precision Analysis (MPA)**

The MPA Pass runs on RMLC-Tensor IR.20

* **Analysis:** It statically analyzes the forward-backward DAG for numerical stability and dynamic range requirements.  
* **Transformation:** It inserts explicit rmlc.cast operations (rmlc.cast\<bf16\>, rmlc.cast\<f8\>) to optimize heavy computations for speed (BF16, FP8 on NVIDIA H100 22).  
* **Gradient Integrity:** It automatically ensures that high-precision (FP32) is used for numerically sensitive steps like the loss calculation and the parameter update itself, implementing **Automatic Loss Scaling and Master Weight** management directly via compiler insertions.21

### **IV.3. Graph & Kernel Optimizations (RMLC-Structured)**

These are performed at the Linalg level to achieve performance portability.23

1. **Operator Fusion:** Combines adjacent operations (e.g., Matmul-Add-ReLU) by fusing their loop nests into a single linalg.generic, reducing memory bandwidth pressure and kernel launch overhead.8  
2. **Tiling and Blocking:** Partitions the computation space (e.g., the iteration domain of a linalg.generic) into smaller blocks to maximize data reuse in cache hierarchies (L1, shared memory).  
   * **TPU Specificity:** When targeting the TPU backend, this pass must adhere to strict hardware constraints, such as tiling into 128x8 chunks to align with the systolic array dimensions.16 The compiler uses MLIR’s Transform dialect to express and verify these transformations.25

### **IV.4. Bufferization Pass (Crucial Memory Optimization)**

This pass transforms the graph from value semantics (Tensor) to memory semantics (MemRef).

* **Purpose:** The pass performs rigorous alias analysis and lifetime analysis to find opportunities for **Buffer Reuse Optimization** (BReO).26  
* **Result:** Converts all tensor types to memref types. The aggressive reuse of memory buffers scheduled by this pass is a core mechanism used by XLA to reduce intermediate storage.26

---

## **V. Heterogeneous Backend Code Generation Abstract Interface**

This layer defines the contract between the RMLC compiler (Rust/MLIR) and the LLVM native code generation components, ensuring target independence until the final selection phase.12

### **V.1. The LLVM TargetLowering Abstraction**

The RMLC-MLIR compiler will rely on the LLVM backend for instruction selection. This requires custom implementations of the TargetLowering class (via FFI):

* **Lowering:** Defines how RMLC-Target Dialect operations (which might represent high-level concepts like rmlc.fused\_op) are translated into LLVM's internal **SelectionDAG** representation.27  
* **Feature Flags:** Each target-specific implementation (e.g., RocmTargetLowering) specifies:  
  * The operations natively supported by the hardware (e.g., fused multiply-add intrinsic).  
  * Target-specific characteristics (e.g., register pressure, cost of division by constant optimization).27

### **V.2. Machine Code (MC) Layer and AOT/JIT Dispatch**

The LLVM **Machine Code (MC) subproject** is utilized for the final assembly/disassembly and AOT binary generation.12

* **AOT Compilation:** The compiler generates LLVM IR (NVVM or ROCDL) which the backend processes to produce a Fat Binary artifact (Section VI.2).  
* **Runtime Dispatch:** The Rust rmlc-runtime contains a small dispatcher that uses the LLVM MC Layer interfaces to decide at execution time:  
  1. Check for the presence of AOT-compiled native code (SASS or HSACO) for the detected device.29  
  2. If found, load and execute the native binary.  
  3. If not found (e.g., new/unknown architecture), invoke the LLVM JIT engine to compile the embedded intermediate code (PTX or equivalent) on the fly.30

---

## **VI. NVIDIA GPU (CUDA) Code Generation Pipeline**

This pipeline is dedicated to lowering RMLC-Target IR to native NVIDIA binaries (cubin/SASS).

### **VI.1. Lowering to NVVM and PTX**

1. **Kernel Outlining:** The RMLC-Target IR is translated to the generic MLIR gpu dialect, with kernels outlined into gpu.func operations.  
2. **NVVM Pipeline:** A specialized MLIR pass pipeline (-gpu-lower-to-nvvm-pipeline 31) converts the gpu dialect into the **NVVM IR** dialect (NVIDIA’s specific LLVM IR).32  
3. **PTX Generation:** The LLVM NVPTX backend consumes the NVVM IR and generates **PTX virtual assembly**. This intermediate code is architecture-agnostic and essential for JIT compilation.30

### **VI.2. Fat Binary Structure**

For optimal performance in deployment (AOT), RMLC creates a **Fat Binary** embedded in the host executable.29

* This binary contains:  
  1. The architecture-agnostic **PTX** virtual assembly.  
  2. Multiple versions of AOT-compiled **SASS** (native machine code, or *cubin*) targeting specific NVIDIA architectures (e.g., sm\_80, sm\_90).29  
* The CUDA driver/runtime (invoked by RMLC) is responsible for inspecting the installed GPU and dispatching the most specific, available SASS binary, falling back to JIT compilation of the PTX only if necessary.30

### **VI.3. Target Specification and Dispatch**

RMLC uses an attribute interface, conceptually similar to MLIR's gpu::TargetAttrInterface, to select the correct lowering path.33 When the Tensor::device is NVIDIA\_GPU, the compiler attaches the NVVMTargetAttr which triggers the NVVM lowering pipeline.

---

## **VII. AMD GPU (ROCm) Code Generation Pipeline**

This pipeline targets the AMD GCN and RDNA architectures via the ROCm compiler stack.

### **VII.1. Lowering to ROCDL and HSACO**

1. **ROCDL Lowering:** The RMLC-Target Dialect is lowered to the **ROCDL** dialect, the LLVM IR representation for AMD GPUs.32 This process is achieved by a sequence of MLIR passes that converts the gpu dialect to the target-specific IR.31  
2. **HSACO Generation:** The LLVM AMDGPU backend compiles the ROCDL IR into the **HSA Code Object (HSACO)** binary format, which contains the device-executable code.34

### **VII.2. Integration with ROCm Toolchain**

The RMLC implementation must integrate with the standard ROCm/HIP toolchain interfaces for deployment.35

* **AOT Compilation:** For offline compilation, RMLC utilizes the amdclang++ or hipcc compilation path to link the ROCm libraries and embed the HSACO device code into the final host object file.34  
* **Metadata Integration:** The RocmTargetLowering implementation must explicitly handle and generate AMD-specific LLVM IR metadata, such as the 'amdgpu.last.use' temporal hint, to ensure optimal performance on AMD architectures.37

### **VII.3. Code Generation Model**

Like the CUDA path, the ROCm compilation supports both **offline (AOT)** compilation for production (generating HSACO) and **runtime (JIT)** compilation, providing flexibility for development and portability across a wide range of AMD hardware.34

---

## **VIII. TPU and ML Accelerator Integration via XLA**

The TPU requires the most rigid compilation process due to its specialized systolic array architecture and dependency on the XLA compiler.38

### **VIII.1. Strict Static Shape Requirement**

The **Shape Specialization Pass** (Section IV.1) is non-negotiable for TPU targeting.18

* TPUs are matrix processors optimized for dense operations.38 The XLA compiler requires all tensor shapes (dimension sizes) to be statically known at the beginning of compilation.  
* **Implementation Requirement:** The RMLC compiler must block the lowering path to StableHLO if any dynamic shape indicator remains in the RMLC-Tensor IR.

### **VIII.2. Translation to StableHLO IR**

For TPU execution, the RMLC-Tensor (Level 1\) IR must be translated into the **StableHLO** dialect.39

* StableHLO is the machine learning operation set designed as a stable interface for XLA.40  
* The translation pass maps RMLC primitives (including AOT-AD operations and collective communication ops 14) to their StableHLO equivalents.

### **VIII.3. XLA Optimization and Systolic Array Tiling**

Once the graph is in StableHLO, it is passed to the XLA compiler backend for specialized optimization.26

1. **Systolic Array Tiling:** XLA applies aggressive, hardware-specific tiling. The compiler must ensure the data organization aligns with the TPU's internal structure, specifically by tiling data into optimal sizes, such as **128x8 chunks**.16  
2. **Specialized Memory Spaces:** The RMLC-Target dialect must be capable of encoding the intent for specialized TPU memory usage, such as **VMEM** (vector SRAM) for dense computation buffers and **SMEM** (scalar SRAM) for scalar loads and stores, allowing XLA to optimize memory placement.41  
3. **Deployment:** XLA compiles the StableHLO into a proprietary TPU executable. The RMLC runtime then manages the execution on the TPU host machine, responsible for loading the executable and handling the communication over the high-bandwidth link.19

#### ---

**Works cited**

1. LLVM Tutorial \- Know LLVM? \- CompilerSutra, accessed October 20, 2025, [https://compilersutra.com/docs/llvm/llvm\_basic/why\_llvm/](https://compilersutra.com/docs/llvm/llvm_basic/why_llvm/)  
2. I Built a Deep Learning Framework in Rust from Scratch. Here's How It Works., accessed October 20, 2025, [https://dev.to/xzdes/i-built-a-deep-learning-framework-in-rust-from-scratch-heres-how-it-works-2984](https://dev.to/xzdes/i-built-a-deep-learning-framework-in-rust-from-scratch-heres-how-it-works-2984)  
3. tracel-ai/burn: Burn is a next generation Deep Learning Framework that doesn't compromise on flexibility, efficiency and portability. \- GitHub, accessed October 20, 2025, [https://github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)  
4. A Gentle Introduction to torch.autograd \- PyTorch, accessed October 20, 2025, [https://docs.pytorch.org/tutorials/beginner/blitz/autograd\_tutorial.html](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)  
5. How does eager execution differ from graph execution? \- LabEx, accessed October 20, 2025, [https://labex.io/questions/how-does-eager-execution-differ-from-graph-execution-625351](https://labex.io/questions/how-does-eager-execution-differ-from-graph-execution-625351)  
6. Graphs and Functions in TensorFlow \- GeeksforGeeks, accessed October 20, 2025, [https://www.geeksforgeeks.org/deep-learning/graphs-and-functions-in-tensorflow/](https://www.geeksforgeeks.org/deep-learning/graphs-and-functions-in-tensorflow/)  
7. MindSpore IR (MindIR), accessed October 20, 2025, [https://www.mindspore.cn/docs/en/r1.9/design/mindir.html](https://www.mindspore.cn/docs/en/r1.9/design/mindir.html)  
8. AI Frameworks \- ML Systems Textbook, accessed October 20, 2025, [https://www.mlsysbook.ai/contents/core/frameworks/frameworks.html](https://www.mlsysbook.ai/contents/core/frameworks/frameworks.html)  
9. PyTorch 2.x, accessed October 20, 2025, [https://pytorch.org/get-started/pytorch-2-x/](https://pytorch.org/get-started/pytorch-2-x/)  
10. Automatic differentiation in ML: Where we are and where we should be going \- arXiv, accessed October 20, 2025, [https://arxiv.org/pdf/1810.11530](https://arxiv.org/pdf/1810.11530)  
11. The Deep Learning Compiler: A Comprehensive Survey \- arXiv, accessed October 20, 2025, [https://arxiv.org/pdf/2002.03794](https://arxiv.org/pdf/2002.03794)  
12. LLVM \- Wikipedia, accessed October 20, 2025, [https://en.wikipedia.org/wiki/LLVM](https://en.wikipedia.org/wiki/LLVM)  
13. Towards a high-performance AI compiler with upstream MLIR \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2404.15204v1](https://arxiv.org/html/2404.15204v1)  
14. PartIR: Composing SPMD Partitioning Strategies for Machine Learning \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2401.11202v2](https://arxiv.org/html/2401.11202v2)  
15. 'tensor' Dialect \- MLIR, accessed October 20, 2025, [https://mlir.llvm.org/docs/Dialects/TensorOps/](https://mlir.llvm.org/docs/Dialects/TensorOps/)  
16. Cloud TPU performance guide, accessed October 20, 2025, [https://cloud.google.com/tpu/docs/performance-guide](https://cloud.google.com/tpu/docs/performance-guide)  
17. An Intermediate Representation for Optimizing Machine Learning Pipelines \- TU Delft Repository, accessed October 20, 2025, [https://repository.tudelft.nl/file/File\_3d63ce07-2938-42c6-bc6a-334acec10088?preview=1](https://repository.tudelft.nl/file/File_3d63ce07-2938-42c6-bc6a-334acec10088?preview=1)  
18. Troubleshooting TensorFlow \- TPU \- Google Cloud, accessed October 20, 2025, [https://cloud.google.com/tpu/docs/troubleshooting/trouble-tf](https://cloud.google.com/tpu/docs/troubleshooting/trouble-tf)  
19. Introduction to Cloud TPU | Google Cloud, accessed October 20, 2025, [https://cloud.google.com/tpu/docs/intro-to-tpu](https://cloud.google.com/tpu/docs/intro-to-tpu)  
20. QuantuneV2: Compiler-Based Local Metric-Driven Mixed Precision Quantization for Practical Embedded AI Applications \- arXiv, accessed October 20, 2025, [https://arxiv.org/html/2501.07161v1](https://arxiv.org/html/2501.07161v1)  
21. Train With Mixed Precision \- NVIDIA Docs, accessed October 20, 2025, [https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)  
22. Mixed precision training \- Amazon SageMaker AI \- AWS Documentation, accessed October 20, 2025, [https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features-v2-mixed-precision.html](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features-v2-mixed-precision.html)  
23. Chapter 0: A Primer on “Structured” Linalg Operations \- MLIR, accessed October 20, 2025, [https://mlir.llvm.org/docs/Tutorials/transform/Ch0/](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/)  
24. Deep Learning Compiler Optimization Techniques \- Aussie AI, accessed October 20, 2025, [https://www.aussieai.com/research/compilers](https://www.aussieai.com/research/compilers)  
25. Chapter 1: Combining Existing Transformations \- MLIR, accessed October 20, 2025, [https://mlir.llvm.org/docs/Tutorials/transform/Ch1/](https://mlir.llvm.org/docs/Tutorials/transform/Ch1/)  
26. XLA architecture | OpenXLA Project, accessed October 20, 2025, [https://openxla.org/xla/architecture](https://openxla.org/xla/architecture)  
27. The LLVM Target-Independent Code Generator, accessed October 20, 2025, [https://llvm.org/docs/CodeGenerator.html](https://llvm.org/docs/CodeGenerator.html)  
28. The LLVM Target-Independent Code Generator \- AMD ROCm documentation, accessed October 20, 2025, [https://rocm.docs.amd.com/projects/llvm-project/en/develop/LLVM/llvm/html/CodeGenerator.html](https://rocm.docs.amd.com/projects/llvm-project/en/develop/LLVM/llvm/html/CodeGenerator.html)  
29. NVIDIA CUDA Compiler Driver Process | by ztex, Tony, Liu | Medium, accessed October 20, 2025, [https://ztex.medium.com/nvidia-cuda-compiler-driver-process-cuda-kernel-deployment-from-code-to-gpu-execution-f94fdc41c8fe](https://ztex.medium.com/nvidia-cuda-compiler-driver-process-cuda-kernel-deployment-from-code-to-gpu-execution-f94fdc41c8fe)  
30. The process of a CUDA program compilation using the NVCC toolchain., accessed October 20, 2025, [https://hpcgpu.mini.pw.edu.pl/cuda-compilation-toolchain/](https://hpcgpu.mini.pw.edu.pl/cuda-compilation-toolchain/)  
31. How to Generate AMDGPU Code from MLIR? Is There a Pipeline Similar to \-gpu-lower-to-nvvm-pipeline? \- LLVM Discussion Forums, accessed October 20, 2025, [https://discourse.llvm.org/t/how-to-generate-amdgpu-code-from-mlir-is-there-a-pipeline-similar-to-gpu-lower-to-nvvm-pipeline/88627](https://discourse.llvm.org/t/how-to-generate-amdgpu-code-from-mlir-is-there-a-pipeline-similar-to-gpu-lower-to-nvvm-pipeline/88627)  
32. Deep Learning Compiler and Optimizer \- Microsoft Research, accessed October 20, 2025, [https://www.microsoft.com/en-us/research/project/deep-learning-compiler-and-optimizer/](https://www.microsoft.com/en-us/research/project/deep-learning-compiler-and-optimizer/)  
33. \[RFC\] Cleaning the GPU dialect \- Page 2 \- MLIR \- LLVM Discussion Forums, accessed October 20, 2025, [https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170?page=2](https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170?page=2)  
34. HIP compilers — HIP 6.2.41134 Documentation \- AMD ROCm documentation, accessed October 20, 2025, [https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.4/understand/compilers.html](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.4/understand/compilers.html)  
35. Compiler reference guide \- AMD ROCm documentation, accessed October 20, 2025, [https://rocm.docs.amd.com/en/docs-6.0.2/reference/rocmcc.html](https://rocm.docs.amd.com/en/docs-6.0.2/reference/rocmcc.html)  
36. HIP compilers \- AMD ROCm documentation, accessed October 20, 2025, [https://rocm.docs.amd.com/projects/HIP/en/docs-develop/understand/compilers.html](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/understand/compilers.html)  
37. User Guide for AMDGPU Backend — LLVM 20.0.0git documentation, accessed October 20, 2025, [https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html)  
38. TPU architecture | Google Cloud, accessed October 20, 2025, [https://cloud.google.com/tpu/docs/system-architecture-tpu-vm](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)  
39. Amazon SageMaker Training Compiler \- AWS Documentation, accessed October 20, 2025, [https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler.html](https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler.html)  
40. XLA \- OpenXLA Project, accessed October 20, 2025, [https://openxla.org/xla](https://openxla.org/xla)  
41. TPU Pipelining \- JAX documentation, accessed October 20, 2025, [https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)