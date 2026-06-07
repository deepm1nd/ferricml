# Section 3: CUDA Backend Implementation

**FerricML Architecture Specification v3.0**  
**Word Count:** 3,600+ words  
**Implementation Priority:** Phase 2 - Primary Backend

---

## Table of Contents

1. [CUDA Architecture Overview](#1-cuda-architecture-overview)
2. [Thread Hierarchy Mapping](#2-thread-hierarchy-mapping)
3. [Memory Hierarchy](#3-memory-hierarchy)
4. [Kernel Generation Pipeline](#4-kernel-generation-pipeline)
5. [Tensor Core Programming](#5-tensor-core-programming)
6. [Optimization Strategies](#6-optimization-strategies)
7. [Stream Management](#7-stream-management)
8. [Multi-GPU Support](#8-multi-gpu-support)

---

## 1. CUDA Architecture Overview

### 1.1 GPU Architecture Model

```rust
pub struct CudaDevice {
    /// Device ID
    pub id: i32,
    
    /// Compute capability (major, minor)
    pub compute_capability: (i32, i32),
    
    /// Number of streaming multiprocessors
    pub sm_count: i32,
    
    /// Maximum threads per SM
    pub max_threads_per_sm: i32,
    
    /// Maximum threads per block
    pub max_threads_per_block: i32,
    
    /// Warp size (typically 32)
    pub warp_size: i32,
    
    /// Shared memory per SM (bytes)
    pub shared_memory_per_sm: usize,
    
    /// Registers per SM
    pub registers_per_sm: i32,
    
    /// Global memory size (bytes)
    pub global_memory_size: usize,
    
    /// L2 cache size (bytes)
    pub l2_cache_size: usize,
    
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f32,
    
    /// Peak FLOPS (single precision)
    pub peak_flops_sp: f64,
    
    /// Peak FLOPS (half precision with tensor cores)
    pub peak_flops_fp16_tc: f64,
}

impl CudaDevice {
    pub fn from_device_id(id: i32) -> Result<Self> {
        unsafe {
            cuda_sys::cudaSetDevice(id)?;
            
            let mut props: cuda_sys::cudaDeviceProp = std::mem::zeroed();
            cuda_sys::cudaGetDeviceProperties(&mut props, id)?;
            
            Ok(Self {
                id,
                compute_capability: (props.major, props.minor),
                sm_count: props.multiProcessorCount,
                max_threads_per_sm: props.maxThreadsPerMultiProcessor,
                max_threads_per_block: props.maxThreadsPerBlock,
                warp_size: props.warpSize,
                shared_memory_per_sm: props.sharedMemPerMultiprocessor,
                registers_per_sm: props.regsPerMultiprocessor,
                global_memory_size: props.totalGlobalMem,
                l2_cache_size: props.l2CacheSize,
                memory_bandwidth: (props.memoryClockRate as f32 * 2.0 * props.memoryBusWidth as f32) / 8.0 / 1e6,
                peak_flops_sp: Self::compute_peak_flops(&props),
                peak_flops_fp16_tc: Self::compute_tensor_core_flops(&props),
            })
        }
    }
    
    /// Check if device supports tensor cores
    pub fn has_tensor_cores(&self) -> bool {
        self.compute_capability >= (7, 0)
    }
    
    /// Check if device supports bf16 tensor cores
    pub fn has_bf16_tensor_cores(&self) -> bool {
        self.compute_capability >= (8, 0)
    }
    
    /// Maximum occupancy for given kernel configuration
    pub fn compute_occupancy(&self, threads_per_block: i32, shared_mem: usize, registers_per_thread: i32) -> f32 {
        // Blocks limited by SM capacity
        let blocks_per_sm_threads = self.max_threads_per_sm / threads_per_block;
        let blocks_per_sm_shared = if shared_mem > 0 {
            self.shared_memory_per_sm / shared_mem
        } else {
            i32::MAX as usize
        } as i32;
        let blocks_per_sm_regs = if registers_per_thread > 0 {
            self.registers_per_sm / (threads_per_block * registers_per_thread)
        } else {
            i32::MAX
        };
        
        let blocks_per_sm = blocks_per_sm_threads
            .min(blocks_per_sm_shared)
            .min(blocks_per_sm_regs)
            .max(1);
        
        let active_warps = (blocks_per_sm * threads_per_block) / self.warp_size;
        let max_warps = self.max_threads_per_sm / self.warp_size;
        
        active_warps as f32 / max_warps as f32
    }
}
```

---

## 2. Thread Hierarchy Mapping

### 2.1 Grid, Block, and Thread Indexing

CUDA organizes threads in a 3-level hierarchy:

```
Grid (device-wide)
  └─> Blocks (scheduled on SMs)
      └─> Warps (groups of 32 threads)
          └─> Threads (individual execution units)
```

```rust
/// Kernel launch configuration
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    /// Grid dimensions (number of blocks)
    pub grid_dim: Dim3,
    
    /// Block dimensions (threads per block)
    pub block_dim: Dim3,
    
    /// Shared memory per block (bytes)
    pub shared_mem_bytes: usize,
    
    /// CUDA stream
    pub stream: CudaStream,
}

#[derive(Debug, Clone, Copy)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
    
    pub fn linear(n: u32) -> Self {
        Self { x: n, y: 1, z: 1 }
    }
    
    pub fn total(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl LaunchConfig {
    /// Compute 1D launch config for n elements
    pub fn linear(n: usize, threads_per_block: u32) -> Self {
        let num_blocks = ((n as u32) + threads_per_block - 1) / threads_per_block;
        
        Self {
            grid_dim: Dim3::linear(num_blocks),
            block_dim: Dim3::linear(threads_per_block),
            shared_mem_bytes: 0,
            stream: CudaStream::default(),
        }
    }
    
    /// Compute 2D launch config for MxN matrix
    pub fn matrix(m: usize, n: usize, tile_m: u32, tile_n: u32) -> Self {
        let grid_m = ((m as u32) + tile_m - 1) / tile_m;
        let grid_n = ((n as u32) + tile_n - 1) / tile_n;
        
        Self {
            grid_dim: Dim3::new(grid_n, grid_m, 1),
            block_dim: Dim3::new(tile_n, tile_m, 1),
            shared_mem_bytes: 0,
            stream: CudaStream::default(),
        }
    }
}
```

### 2.2 Index Computation

```cuda
// CUDA device code for computing global indices
__device__ inline int get_global_id_1d() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline int2 get_global_id_2d() {
    return make_int2(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    );
}

__device__ inline int3 get_global_id_3d() {
    return make_int3(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    );
}

// Warp-level primitives
__device__ inline int get_warp_id() {
    return threadIdx.x / 32;
}

__device__ inline int get_lane_id() {
    return threadIdx.x % 32;
}
```

---

## 3. Memory Hierarchy

### 3.1 Memory Spaces

CUDA provides multiple memory spaces with different characteristics:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    /// Global memory (device DRAM)
    /// - Largest capacity (GBs)
    /// - High latency (~400-800 cycles)
    /// - Cached in L1/L2
    Global = 0,
    
    /// Shared memory (on-chip SRAM)
    /// - Limited capacity (~48-164 KB per SM)
    /// - Low latency (~20 cycles)
    /// - Explicitly managed
    Shared = 1,
    
    /// Constant memory
    /// - Cached, read-only
    /// - Good for broadcast patterns
    Constant = 2,
    
    /// Local memory (per-thread)
    /// - Actually in global memory
    /// - For register spills
    Local = 3,
    
    /// Texture memory
    /// - Cached, 2D locality
    Texture = 4,
}
```

### 3.2 Memory Access Patterns

```cuda
// Coalesced access (GOOD - all threads access contiguous memory)
__global__ void coalesced_access(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = data[tid];  // Each thread accesses consecutive address
        // process val...
    }
}

// Strided access (BAD - non-coalesced, multiple memory transactions)
__global__ void strided_access(float* data, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = data[tid * stride];  // Non-contiguous access
        // process val...
    }
}

// Bank conflict in shared memory (BAD)
__global__ void bank_conflict(float* input) {
    __shared__ float smem[32][32];
    int tid = threadIdx.x;
    
    // Column-major access causes bank conflicts
    float val = smem[0][tid];  // All threads access same bank
}

// Padded shared memory (GOOD - avoids bank conflicts)
__global__ void no_bank_conflict(float* input) {
    __shared__ float smem[32][33];  // Extra column for padding
    int tid = threadIdx.x;
    
    // Padding eliminates bank conflicts
    float val = smem[0][tid];
}
```

### 3.3 Rust Abstractions

```rust
pub struct SharedMemory<T> {
    size_bytes: usize,
    _phantom: PhantomData<T>,
}

impl<T> SharedMemory<T> {
    /// Declare shared memory usage
    pub fn new(num_elements: usize) -> Self {
        let size_bytes = num_elements * std::mem::size_of::<T>();
        Self {
            size_bytes,
            _phantom: PhantomData,
        }
    }
    
    /// Padded size to avoid bank conflicts
    pub fn padded_size(num_cols: usize, num_rows: usize) -> usize {
        // Add one extra column to avoid bank conflicts
        (num_cols + 1) * num_rows
    }
}

/// Generate shared memory declaration in CUDA code
pub fn generate_shared_memory_decl(ty: &Type, size: usize) -> String {
    match ty {
        Type::Float(FloatType { kind: FloatKind::F32 }) => {
            format!("__shared__ float smem[{}];", size)
        }
        Type::Float(FloatType { kind: FloatKind::F16 }) => {
            format!("__shared__ half smem[{}];", size)
        }
        _ => unimplemented!("Shared memory for type {:?}", ty),
    }
}
```

---

## 4. Kernel Generation Pipeline

### 4.1 IR to PTX Lowering

```rust
pub struct CudaCodegen {
    device: CudaDevice,
    type_map: HashMap<Type, String>,
}

impl CudaCodegen {
    pub fn generate_kernel(&self, func: &Function) -> Result<String> {
        let mut code = String::new();
        
        // Generate kernel signature
        code.push_str(&self.generate_signature(func)?);
        code.push_str(" {\n");
        
        // Generate local variable declarations
        code.push_str(&self.generate_locals(func)?);
        
        // Generate kernel body
        for block in &func.blocks {
            code.push_str(&self.generate_block(block)?);
        }
        
        code.push_str("}\n");
        
        Ok(code)
    }
    
    fn generate_signature(&self, func: &Function) -> Result<String> {
        let return_type = if func.signature.outputs.is_empty() {
            "void".to_string()
        } else {
            self.map_type(&func.signature.outputs[0])?
        };
        
        let mut params = Vec::new();
        for (idx, input_type) in func.signature.inputs.iter().enumerate() {
            let cuda_type = self.map_type(input_type)?;
            params.push(format!("{} arg{}", cuda_type, idx));
        }
        
        Ok(format!(
            "__global__ {} {}({})",
            return_type,
            func.name,
            params.join(", ")
        ))
    }
    
    fn generate_operation(&self, op: &Operation) -> Result<String> {
        match &op.kind {
            OpKind::Tensor(TensorOp::Add) => {
                // %result = fml.add %lhs, %rhs : tensor<N>
                let lhs = self.value_name(&op.operands[0]);
                let rhs = self.value_name(&op.operands[1]);
                let result = self.value_name(&op.results[0]);
                
                Ok(format!("{} = {} + {};", result, lhs, rhs))
            }
            
            OpKind::Tensor(TensorOp::MatMul) => {
                // Generate optimized matmul kernel
                self.generate_matmul(op)
            }
            
            OpKind::Tensor(TensorOp::Conv2d) => {
                self.generate_conv2d(op)
            }
            
            _ => unimplemented!("Operation {:?}", op.kind),
        }
    }
    
    fn map_type(&self, ty: &Type) -> Result<String> {
        match ty {
            Type::Float(FloatType { kind: FloatKind::F32 }) => Ok("float".to_string()),
            Type::Float(FloatType { kind: FloatKind::F16 }) => Ok("half".to_string()),
            Type::Float(FloatType { kind: FloatKind::F64 }) => Ok("double".to_string()),
            Type::Integer(IntegerType { width: 32, .. }) => Ok("int".to_string()),
            Type::Integer(IntegerType { width: 64, .. }) => Ok("long long".to_string()),
            Type::Tensor(TensorType { element_type, .. }) => {
                // Tensors are pointers to element type
                let elem = self.map_type(element_type)?;
                Ok(format!("{}*", elem))
            }
            _ => Err(Error::UnsupportedType(ty.clone())),
        }
    }
}
```

### 4.2 Element-wise Operation Codegen

```rust
impl CudaCodegen {
    pub fn generate_elementwise_kernel(&self, op: &Operation) -> Result<String> {
        let element_type = self.get_tensor_element_type(&op.operands[0])?;
        let n = self.get_tensor_numel(&op.operands[0])?;
        
        let kernel_code = format!(r#"
__global__ void elementwise_{}(
    {}* output,
    const {}* input1,
    const {}* input2,
    int n
) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {{
        output[tid] = input1[tid] {} input2[tid];
    }}
}}
"#,
            self.op_name(op),
            element_type,
            element_type,
            element_type,
            self.op_symbol(op)
        );
        
        Ok(kernel_code)
    }
}
```

---

## 5. Tensor Core Programming

### 5.1 WMMA API

Tensor Cores perform matrix multiply-accumulate on small tiles:

```cuda
#include <mma.h>
using namespace nvcuda;

// Tile sizes for tensor cores
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void tensor_core_matmul(
    const half* A,     // M x K matrix
    const half* B,     // K x N matrix
    float* C,          // M x N matrix
    int M, int K, int N
) {
    // Warp and lane identification
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Fragments for WMMA operations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Compute starting positions
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;
    
    // Loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) {
        int aCol = i;
        int bRow = i;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrices from global memory
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform matrix multiply-accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store result
    if (aRow < M && bCol < N) {
        wmma::store_matrix_sync(C + aRow * N + bCol, c_frag, N, wmma::mem_row_major);
    }
}
```

### 5.2 Rust Wrapper

```rust
pub struct TensorCoreMatMul {
    device: CudaDevice,
}

impl TensorCoreMatMul {
    pub fn supports_tensor_cores(&self) -> bool {
        self.device.has_tensor_cores()
    }
    
    pub fn matmul_f16(
        &self,
        a: &Tensor<Float16>,
        b: &Tensor<Float16>,
    ) -> Result<Tensor<f32>> {
        assert!(self.supports_tensor_cores(), "Device doesn't support tensor cores");
        
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        
        // Allocate output
        let mut c = Tensor::<f32>::zeros(&[m, n]).to(a.device())?;
        
        // Configure launch parameters
        const WMMA_M: u32 = 16;
        const WMMA_N: u32 = 16;
        const THREADS_PER_BLOCK: u32 = 256;
        
        let grid_dim = Dim3::new(
            (n as u32 + WMMA_N - 1) / WMMA_N,
            (m as u32 + WMMA_M - 1) / WMMA_M,
            1
        );
        
        let block_dim = Dim3::new(THREADS_PER_BLOCK / 32, 32, 1);  // One warp per row
        
        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
            stream: CudaStream::default(),
        };
        
        // Launch kernel
        unsafe {
            self.launch_tensor_core_kernel(
                config,
                a.data_ptr(),
                b.data_ptr(),
                c.data_ptr_mut(),
                m as i32,
                k as i32,
                n as i32,
            )?;
        }
        
        Ok(c)
    }
}
```

---

## 6. Optimization Strategies

### 6.1 Memory Coalescing

```rust
pub struct CoalescingAnalyzer;

impl CoalescingAnalyzer {
    /// Analyze memory access pattern for coalescing
    pub fn analyze_access(&self, op: &Operation) -> AccessPattern {
        match &op.kind {
            OpKind::Tensor(TensorOp::Add) => {
                // Check if all tensors have same stride
                let stride0 = self.get_stride(&op.operands[0], -1);
                let stride1 = self.get_stride(&op.operands[1], -1);
                
                if stride0 == stride1 && stride0 == 1 {
                    AccessPattern::Coalesced
                } else {
                    AccessPattern::Strided(stride0.max(stride1) as usize)
                }
            }
            _ => AccessPattern::Unknown,
        }
    }
    
    /// Transform operation for better coalescing
    pub fn optimize_for_coalescing(&self, op: &mut Operation) -> Result<()> {
        match self.analyze_access(op) {
            AccessPattern::Strided(stride) if stride > 1 => {
                // Insert transpose operation to make contiguous
                // Or use shared memory to reorder data
                self.insert_coalescing_transformation(op)?;
            }
            _ => {}
        }
        
        Ok(())
    }
}

pub enum AccessPattern {
    Coalesced,
    Strided(usize),
    Random,
    Unknown,
}
```

### 6.2 Tiling for Cache Locality

```rust
pub struct TilingOptimizer {
    device: CudaDevice,
}

impl TilingOptimizer {
    /// Compute optimal tile sizes for matmul
    pub fn compute_tile_sizes(&self, m: usize, n: usize, k: usize) -> (usize, usize, usize) {
        let shared_mem = self.device.shared_memory_per_sm;
        let elem_size = 4; // f32
        
        // Constraints:
        // - (TILE_M * TILE_K + TILE_K * TILE_N) * elem_size <= shared_mem
        // - TILE_M * TILE_N <= max_threads_per_block
        
        // Common tile sizes for different GPUs
        match self.device.compute_capability {
            (7, 0..=5) => (128, 128, 8),  // Volta/Turing
            (8, 0..=9) => (256, 128, 16), // Ampere
            (9, 0) => (256, 256, 16),     // Hopper
            _ => (64, 64, 8),             // Default
        }
    }
    
    /// Generate tiled matmul kernel
    pub fn generate_tiled_matmul(&self, m: usize, n: usize, k: usize) -> String {
        let (tile_m, tile_n, tile_k) = self.compute_tile_sizes(m, n, k);
        
        format!(r#"
#define TILE_M {}
#define TILE_N {}
#define TILE_K {}

__global__ void tiled_matmul(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {{
    // Shared memory for tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Global row/col of C that this thread computes
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles of K dimension
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {{
        // Load tile of A into shared memory
        if (row < M && t * TILE_K + tx < K) {{
            As[ty][tx] = A[row * K + t * TILE_K + tx];
        }} else {{
            As[ty][tx] = 0.0f;
        }}
        
        // Load tile of B into shared memory
        if (t * TILE_K + ty < K && col < N) {{
            Bs[ty][tx] = B[(t * TILE_K + ty) * N + col];
        }} else {{
            Bs[ty][tx] = 0.0f;
        }}
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_K; i++) {{
            sum += As[ty][i] * Bs[i][tx];
        }}
        
        __syncthreads();
    }}
    
    // Write result
    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
"#, tile_m, tile_n, tile_k)
    }
}
```

### 6.3 Occupancy Optimization

```rust
pub struct OccupancyOptimizer {
    device: CudaDevice,
}

impl OccupancyOptimizer {
    /// Find optimal block size for maximum occupancy
    pub fn find_optimal_block_size(&self, kernel_info: &KernelInfo) -> u32 {
        let mut best_occupancy = 0.0f32;
        let mut best_block_size = 128;
        
        // Try powers of 2 from 32 to 1024
        for block_size in [32, 64, 128, 256, 512, 1024] {
            if block_size > self.device.max_threads_per_block as u32 {
                continue;
            }
            
            let occupancy = self.device.compute_occupancy(
                block_size as i32,
                kernel_info.shared_mem_bytes,
                kernel_info.registers_per_thread,
            );
            
            if occupancy > best_occupancy {
                best_occupancy = occupancy;
                best_block_size = block_size;
            }
        }
        
        best_block_size
    }
}

pub struct KernelInfo {
    pub shared_mem_bytes: usize,
    pub registers_per_thread: i32,
}
```

---

## 7. Stream Management

### 7.1 Asynchronous Execution

```rust
pub struct CudaStream {
    handle: cuda_sys::cudaStream_t,
    device_id: i32,
}

impl CudaStream {
    pub fn new(device_id: i32) -> Result<Self> {
        unsafe {
            cuda_sys::cudaSetDevice(device_id)?;
            
            let mut handle: cuda_sys::cudaStream_t = std::ptr::null_mut();
            cuda_sys::cudaStreamCreate(&mut handle)?;
            
            Ok(Self { handle, device_id })
        }
    }
    
    pub fn default() -> Self {
        Self {
            handle: std::ptr::null_mut(), // NULL stream
            device_id: 0,
        }
    }
    
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            cuda_sys::cudaStreamSynchronize(self.handle)?;
        }
        Ok(())
    }
    
    pub fn query(&self) -> Result<bool> {
        unsafe {
            match cuda_sys::cudaStreamQuery(self.handle) {
                Ok(_) => Ok(true),  // Stream is idle
                Err(e) if e == cuda_sys::cudaError::cudaErrorNotReady => Ok(false),
                Err(e) => Err(e.into()),
            }
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                cuda_sys::cudaStreamDestroy(self.handle);
            }
        }
    }
}
```

### 7.2 Stream Pool

```rust
pub struct StreamPool {
    streams: Vec<CudaStream>,
    available: Arc<Mutex<VecDeque<usize>>>,
    device_id: i32,
}

impl StreamPool {
    pub fn new(device_id: i32, size: usize) -> Result<Self> {
        let mut streams = Vec::with_capacity(size);
        for _ in 0..size {
            streams.push(CudaStream::new(device_id)?);
        }
        
        let available: VecDeque<usize> = (0..size).collect();
        
        Ok(Self {
            streams,
            available: Arc::new(Mutex::new(available)),
            device_id,
        })
    }
    
    pub fn acquire(&self) -> Result<StreamGuard> {
        let idx = {
            let mut available = self.available.lock().unwrap();
            available.pop_front().ok_or(Error::NoStreamAvailable)?
        };
        
        Ok(StreamGuard {
            stream: &self.streams[idx],
            idx,
            pool: self.available.clone(),
        })
    }
}

pub struct StreamGuard<'a> {
    stream: &'a CudaStream,
    idx: usize,
    pool: Arc<Mutex<VecDeque<usize>>>,
}

impl<'a> Deref for StreamGuard<'a> {
    type Target = CudaStream;
    
    fn deref(&self) -> &Self::Target {
        self.stream
    }
}

impl<'a> Drop for StreamGuard<'a> {
    fn drop(&mut self) {
        let mut available = self.pool.lock().unwrap();
        available.push_back(self.idx);
    }
}
```

---

## 8. Multi-GPU Support

### 8.1 Device Management

```rust
pub struct MultiGpuManager {
    devices: Vec<CudaDevice>,
    current_device: AtomicI32,
}

impl MultiGpuManager {
    pub fn new() -> Result<Self> {
        let device_count = Self::get_device_count()?;
        
        let mut devices = Vec::new();
        for i in 0..device_count {
            devices.push(CudaDevice::from_device_id(i)?);
        }
        
        Ok(Self {
            devices,
            current_device: AtomicI32::new(0),
        })
    }
    
    fn get_device_count() -> Result<i32> {
        unsafe {
            let mut count: i32 = 0;
            cuda_sys::cudaGetDeviceCount(&mut count)?;
            Ok(count)
        }
    }
    
    pub fn set_device(&self, device_id: i32) -> Result<()> {
        if device_id >= self.devices.len() as i32 {
            return Err(Error::InvalidDeviceId);
        }
        
        unsafe {
            cuda_sys::cudaSetDevice(device_id)?;
        }
        
        self.current_device.store(device_id, Ordering::Relaxed);
        Ok(())
    }
    
    pub fn get_device(&self, device_id: i32) -> Option<&CudaDevice> {
        self.devices.get(device_id as usize)
    }
}
```

### 8.2 Peer-to-Peer Communication

```rust
pub struct P2PManager {
    peer_access_matrix: Vec<Vec<bool>>,
}

impl P2PManager {
    pub fn new(devices: &[CudaDevice]) -> Result<Self> {
        let n = devices.len();
        let mut peer_access_matrix = vec![vec![false; n]; n];
        
        // Check peer access between all device pairs
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    peer_access_matrix[i][j] = Self::can_access_peer(
                        devices[i].id,
                        devices[j].id
                    )?;
                }
            }
        }
        
        // Enable peer access where available
        for i in 0..n {
            for j in 0..n {
                if peer_access_matrix[i][j] {
                    Self::enable_peer_access(devices[i].id, devices[j].id)?;
                }
            }
        }
        
        Ok(Self { peer_access_matrix })
    }
    
    fn can_access_peer(from_device: i32, to_device: i32) -> Result<bool> {
        unsafe {
            let mut can_access: i32 = 0;
            cuda_sys::cudaDeviceCanAccessPeer(&mut can_access, from_device, to_device)?;
            Ok(can_access != 0)
        }
    }
    
    fn enable_peer_access(from_device: i32, to_device: i32) -> Result<()> {
        unsafe {
            cuda_sys::cudaSetDevice(from_device)?;
            cuda_sys::cudaDeviceEnablePeerAccess(to_device, 0)?;
        }
        Ok(())
    }
    
    pub fn copy_peer_to_peer(
        &self,
        src: &CudaStorage,
        dst: &mut CudaStorage,
        size: usize,
    ) -> Result<()> {
        if !self.peer_access_matrix[src.device_id as usize][dst.device_id as usize] {
            return Err(Error::NoPeerAccess);
        }
        
        unsafe {
            cuda_sys::cudaMemcpyPeerAsync(
                dst.device_ptr.as_ptr(),
                dst.device_id,
                src.device_ptr.as_ptr(),
                src.device_id,
                size,
                std::ptr::null_mut(), // Default stream
            )?;
        }
        
        Ok(())
    }
}
```

### 8.3 NCCL Integration

```rust
pub struct NcclCommunicator {
    comm: nccl_sys::ncclComm_t,
    rank: usize,
    world_size: usize,
    device_id: i32,
}

impl NcclCommunicator {
    pub fn init_multi_gpu(device_ids: &[i32]) -> Result<Vec<Self>> {
        let world_size = device_ids.len();
        let mut comms = vec![std::ptr::null_mut(); world_size];
        
        unsafe {
            nccl_sys::ncclCommInitAll(
                comms.as_mut_ptr(),
                world_size as i32,
                device_ids.as_ptr(),
            )?;
        }
        
        let communicators = device_ids.iter().enumerate()
            .map(|(rank, &device_id)| {
                Self {
                    comm: comms[rank],
                    rank,
                    world_size,
                    device_id,
                }
            })
            .collect();
        
        Ok(communicators)
    }
    
    pub fn all_reduce(
        &self,
        send_buf: *const u8,
        recv_buf: *mut u8,
        count: usize,
        datatype: nccl_sys::ncclDataType_t,
        op: nccl_sys::ncclRedOp_t,
        stream: &CudaStream,
    ) -> Result<()> {
        unsafe {
            nccl_sys::ncclAllReduce(
                send_buf as *const std::ffi::c_void,
                recv_buf as *mut std::ffi::c_void,
                count,
                datatype,
                op,
                self.comm,
                stream.handle,
            )?;
        }
        
        Ok(())
    }
    
    pub fn broadcast(
        &self,
        buf: *mut u8,
        count: usize,
        datatype: nccl_sys::ncclDataType_t,
        root: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        unsafe {
            nccl_sys::ncclBroadcast(
                buf as *const std::ffi::c_void,
                buf as *mut std::ffi::c_void,
                count,
                datatype,
                root as i32,
                self.comm,
                stream.handle,
            )?;
        }
        
        Ok(())
    }
}

impl Drop for NcclCommunicator {
    fn drop(&mut self) {
        unsafe {
            nccl_sys::ncclCommDestroy(self.comm);
        }
    }
}
```

---

## 9. Complete Example: Optimized MatMul

### 9.1 Full Implementation

```rust
pub struct CudaMatMul {
    device: CudaDevice,
    use_tensor_cores: bool,
}

impl CudaMatMul {
    pub fn new(device: CudaDevice) -> Self {
        let use_tensor_cores = device.has_tensor_cores();
        Self { device, use_tensor_cores }
    }
    
    pub fn matmul<T: DType>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        
        // Choose implementation based on size and capabilities
        if self.use_tensor_cores && T::TYPE_ID == TypeId::Float16 && m >= 128 && n >= 128 {
            self.matmul_tensor_cores(a, b)
        } else if m * n * k > 1_000_000 {
            self.matmul_tiled(a, b)
        } else {
            self.matmul_naive(a, b)
        }
    }
    
    fn matmul_tiled<T: DType>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        
        // Allocate output
        let mut c = Tensor::<T>::zeros(&[m, n]).to(a.device())?;
        
        // Generate kernel
        let kernel_code = self.generate_tiled_kernel(m, n, k)?;
        let kernel = self.compile_kernel(&kernel_code)?;
        
        // Launch configuration
        const TILE_SIZE: u32 = 16;
        let grid_dim = Dim3::new(
            (n as u32 + TILE_SIZE - 1) / TILE_SIZE,
            (m as u32 + TILE_SIZE - 1) / TILE_SIZE,
            1
        );
        let block_dim = Dim3::new(TILE_SIZE, TILE_SIZE, 1);
        
        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 2 * TILE_SIZE as usize * TILE_SIZE as usize * T::SIZE,
            stream: CudaStream::default(),
        };
        
        // Launch kernel
        unsafe {
            kernel.launch(
                config,
                &[
                    a.data_ptr() as *const std::ffi::c_void,
                    b.data_ptr() as *const std::ffi::c_void,
                    c.data_ptr_mut() as *mut std::ffi::c_void,
                    &m as *const usize as *const std::ffi::c_void,
                    &k as *const usize as *const std::ffi::c_void,
                    &n as *const usize as *const std::ffi::c_void,
                ],
            )?;
        }
        
        Ok(c)
    }
    
    fn compile_kernel(&self, code: &str) -> Result<CudaKernel> {
        // Use NVRTC to compile PTX
        let ptx = self.compile_to_ptx(code)?;
        
        // Load PTX into module
        let module = self.load_ptx(&ptx)?;
        
        Ok(CudaKernel { module })
    }
    
    fn compile_to_ptx(&self, cuda_code: &str) -> Result<String> {
        use nvrtc_sys::*;
        
        unsafe {
            // Create program
            let mut prog: nvrtcProgram = std::ptr::null_mut();
            nvrtcCreateProgram(
                &mut prog,
                cuda_code.as_ptr() as *const i8,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )?;
            
            // Compile with optimization flags
            let arch = format!("--gpu-architecture=compute_{}{}", 
                self.device.compute_capability.0,
                self.device.compute_capability.1
            );
            let options = [
                arch.as_ptr() as *const i8,
                "--use_fast_math\0".as_ptr() as *const i8,
                "--std=c++14\0".as_ptr() as *const i8,
            ];
            
            let result = nvrtcCompileProgram(prog, options.len() as i32, options.as_ptr());
            
            // Get compilation log
            let mut log_size: usize = 0;
            nvrtcGetProgramLogSize(prog, &mut log_size)?;
            
            if log_size > 1 {
                let mut log = vec![0u8; log_size];
                nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut i8)?;
                eprintln!("NVRTC Log: {}", String::from_utf8_lossy(&log));
            }
            
            if result != nvrtcResult::NVRTC_SUCCESS {
                return Err(Error::CompilationFailed);
            }
            
            // Get PTX
            let mut ptx_size: usize = 0;
            nvrtcGetPTXSize(prog, &mut ptx_size)?;
            
            let mut ptx = vec![0u8; ptx_size];
            nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut i8)?;
            
            // Cleanup
            nvrtcDestroyProgram(&mut prog)?;
            
            Ok(String::from_utf8_lossy(&ptx).to_string())
        }
    }
}
```

---

## Summary

This section detailed the CUDA backend implementation for FerricML:

**Key Components:**
1. **Device Management:** Query capabilities, manage multiple GPUs
2. **Thread Hierarchy:** Grid/block/warp organization with proper indexing
3. **Memory Hierarchy:** Efficient use of global, shared, and register memory
4. **Kernel Generation:** IR to PTX/CUDA C++ compilation pipeline
5. **Tensor Cores:** WMMA API for accelerated matrix operations
6. **Optimizations:** Coalescing, tiling, occupancy tuning
7. **Async Execution:** Stream management for overlapping operations
8. **Multi-GPU:** P2P transfers and NCCL integration

**Performance Considerations:**
- Memory coalescing critical for bandwidth
- Shared memory reduces global memory traffic
- Tensor cores provide 8-16x speedup for suitable operations
- Occupancy impacts latency hiding
- Stream pooling enables async execution

**Implementation Priority:**
1. Basic kernel generation and execution
2. Memory management and transfers
3. Tiling and optimization passes
4. Tensor core support
5. Multi-GPU and NCCL

**Integration Points:**
- Section 2 IR lowers to CUDA kernels
- Section 6 optimizations select tile sizes
- Section 10 uses NCCL for distributed training

**Next Steps:**
- Section 4 implements CPU backend with similar optimizations
- Section 6 details optimization passes that target CUDA
- Profile and tune based on actual workloads