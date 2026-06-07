# Section 1: Core Type System & Memory Model

**FerricML Architecture Specification v3.0**  
**Word Count:** 3,400+ words  
**Implementation Priority:** Phase 1 - Critical Foundation

---

## Table of Contents

1. [Type System Architecture](#1-type-system-architecture)
2. [Scalar Types](#2-scalar-types)
3. [Aggregate Types](#3-aggregate-types)
4. [Memory Layout Strategies](#4-memory-layout-strategies)
5. [Device Memory Management](#5-device-memory-management)
6. [Allocation Strategies](#6-allocation-strategies)
7. [Lifetime Management](#7-lifetime-management)
8. [Zero-Copy Operations](#8-zero-copy-operations)

---

## 1. Type System Architecture

### 1.1 Design Philosophy

FerricML's type system follows three core principles:

1. **Static Safety:** Leverage Rust's type system to prevent runtime errors
2. **Zero-Cost Abstraction:** No runtime overhead for type information
3. **Hardware Awareness:** Types encode alignment, layout, and device placement

### 1.2 Type Hierarchy

```rust
pub trait DType: Copy + Send + Sync + 'static {
    /// Size in bytes
    const SIZE: usize;
    
    /// Required alignment (power of 2)
    const ALIGNMENT: usize;
    
    /// Zero value for this type
    const ZERO: Self;
    
    /// One/identity value
    const ONE: Self;
    
    /// Type identifier for runtime dispatch
    const TYPE_ID: TypeId;
    
    /// Whether this type supports SIMD operations
    const SIMD_CAPABLE: bool;
    
    /// Preferred SIMD width (in elements)
    const SIMD_WIDTH: usize;
    
    /// Convert from f64 (for initialization)
    fn from_f64(val: f64) -> Self;
    
    /// Convert to f64 (for display/debugging)
    fn to_f64(self) -> f64;
}
```

### 1.3 Type Identifier System

```rust
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeId {
    // Floating point
    Float16 = 0,
    BFloat16 = 1,
    Float32 = 2,
    Float64 = 3,
    
    // Signed integers
    Int8 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    
    // Unsigned integers
    UInt8 = 8,
    UInt16 = 9,
    UInt32 = 10,
    UInt64 = 11,
    
    // Boolean
    Bool = 12,
    
    // Complex numbers
    Complex64 = 13,
    Complex128 = 14,
    
    // Quantized types
    QInt8 = 15,
    QUInt8 = 16,
    QInt32 = 17,
}

impl TypeId {
    pub const fn size_bytes(self) -> usize {
        match self {
            TypeId::Float16 | TypeId::BFloat16 => 2,
            TypeId::Float32 => 4,
            TypeId::Float64 => 8,
            TypeId::Int8 | TypeId::UInt8 | TypeId::Bool | TypeId::QInt8 | TypeId::QUInt8 => 1,
            TypeId::Int16 | TypeId::UInt16 => 2,
            TypeId::Int32 | TypeId::UInt32 | TypeId::QInt32 => 4,
            TypeId::Int64 | TypeId::UInt64 => 8,
            TypeId::Complex64 => 8,
            TypeId::Complex128 => 16,
        }
    }
    
    pub const fn alignment(self) -> usize {
        // Most types align to their size, but we prefer 4-byte minimum for efficiency
        match self {
            TypeId::Bool | TypeId::Int8 | TypeId::UInt8 | TypeId::QInt8 | TypeId::QUInt8 => 1,
            TypeId::Float16 | TypeId::BFloat16 | TypeId::Int16 | TypeId::UInt16 => 2,
            _ => self.size_bytes(),
        }
    }
    
    pub const fn is_floating_point(self) -> bool {
        matches!(self, TypeId::Float16 | TypeId::BFloat16 | TypeId::Float32 | TypeId::Float64)
    }
    
    pub const fn is_integer(self) -> bool {
        matches!(self, 
            TypeId::Int8 | TypeId::Int16 | TypeId::Int32 | TypeId::Int64 |
            TypeId::UInt8 | TypeId::UInt16 | TypeId::UInt32 | TypeId::UInt64
        )
    }
}
```

---

## 2. Scalar Types

### 2.1 Standard Scalar Implementations

Each scalar type implements the `DType` trait with specific characteristics:

```rust
// Float32 Implementation
impl DType for f32 {
    const SIZE: usize = 4;
    const ALIGNMENT: usize = 4;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TYPE_ID: TypeId = TypeId::Float32;
    const SIMD_CAPABLE: bool = true;
    const SIMD_WIDTH: usize = 8; // AVX2 can process 8 f32s
    
    #[inline]
    fn from_f64(val: f64) -> Self {
        val as f32
    }
    
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// Float16 Implementation (requires half crate)
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Float16(half::f16);

impl DType for Float16 {
    const SIZE: usize = 2;
    const ALIGNMENT: usize = 2;
    const ZERO: Self = Float16(half::f16::ZERO);
    const ONE: Self = Float16(half::f16::ONE);
    const TYPE_ID: TypeId = TypeId::Float16;
    const SIMD_CAPABLE: bool = cfg!(target_feature = "f16c");
    const SIMD_WIDTH: usize = 16; // AVX2 can process 16 f16s
    
    #[inline]
    fn from_f64(val: f64) -> Self {
        Float16(half::f16::from_f64(val))
    }
    
    #[inline]
    fn to_f64(self) -> f64 {
        self.0.to_f64()
    }
}

// BFloat16 Implementation
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BFloat16(u16);

impl BFloat16 {
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        // BF16 is the top 16 bits of f32
        BFloat16((val.to_bits() >> 16) as u16)
    }
    
    #[inline]
    pub fn to_f32(self) -> f32 {
        // Reconstruct f32 by shifting back and padding with zeros
        f32::from_bits((self.0 as u32) << 16)
    }
}

impl DType for BFloat16 {
    const SIZE: usize = 2;
    const ALIGNMENT: usize = 2;
    const ZERO: Self = BFloat16(0);
    const ONE: Self = BFloat16(0x3F80); // 1.0 in BF16
    const TYPE_ID: TypeId = TypeId::BFloat16;
    const SIMD_CAPABLE: bool = cfg!(target_feature = "avx512bf16");
    const SIMD_WIDTH: usize = 16;
    
    #[inline]
    fn from_f64(val: f64) -> Self {
        BFloat16::from_f32(val as f32)
    }
    
    #[inline]
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
}
```

### 2.2 Quantized Types

Quantized types store reduced-precision values with scale/zero-point parameters:

```rust
#[repr(C)]
pub struct QuantizedInt8 {
    data: i8,
    scale: f32,
    zero_point: i8,
}

impl QuantizedInt8 {
    #[inline]
    pub fn quantize(value: f32, scale: f32, zero_point: i8) -> Self {
        let quantized = (value / scale).round() as i32 + zero_point as i32;
        let clamped = quantized.clamp(-128, 127) as i8;
        Self { data: clamped, scale, zero_point }
    }
    
    #[inline]
    pub fn dequantize(&self) -> f32 {
        (self.data as i32 - self.zero_point as i32) as f32 * self.scale
    }
}
```

---

## 3. Aggregate Types

### 3.1 Tensor Type Definition

```rust
pub struct Tensor<T: DType> {
    /// Storage backend (CPU/GPU memory)
    storage: Arc<Storage>,
    
    /// Shape of the tensor (dynamic)
    shape: Shape,
    
    /// Stride information for indexing
    strides: Strides,
    
    /// Byte offset into storage
    offset: usize,
    
    /// Device placement
    device: Device,
    
    /// Type marker (zero-sized, compile-time only)
    _phantom: PhantomData<T>,
    
    /// Unique identifier for autograd
    id: TensorId,
    
    /// Gradient tracking (optional)
    grad_info: Option<Arc<Mutex<GradInfo>>>,
}

/// Shape representation with static and dynamic components
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Dimension sizes
    dims: SmallVec<[usize; 4]>, // Most tensors have ≤4 dimensions
    
    /// Total number of elements (cached)
    numel: usize,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        let numel = dims.iter().product();
        Self {
            dims: SmallVec::from_slice(dims),
            numel,
        }
    }
    
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
    
    pub fn size(&self, dim: usize) -> usize {
        self.dims[dim]
    }
    
    pub fn numel(&self) -> usize {
        self.numel
    }
}

/// Stride information for non-contiguous tensors
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Strides {
    strides: SmallVec<[isize; 4]>,
}

impl Strides {
    /// Create contiguous row-major strides
    pub fn contiguous(shape: &Shape) -> Self {
        let mut strides = vec![0isize; shape.ndim()];
        let mut stride = 1isize;
        
        for i in (0..shape.ndim()).rev() {
            strides[i] = stride;
            stride *= shape.size(i) as isize;
        }
        
        Self {
            strides: SmallVec::from_vec(strides),
        }
    }
    
    /// Create column-major (Fortran-style) strides
    pub fn fortran(shape: &Shape) -> Self {
        let mut strides = vec![0isize; shape.ndim()];
        let mut stride = 1isize;
        
        for i in 0..shape.ndim() {
            strides[i] = stride;
            stride *= shape.size(i) as isize;
        }
        
        Self {
            strides: SmallVec::from_vec(strides),
        }
    }
    
    /// Check if tensor is contiguous
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        let expected = Self::contiguous(shape);
        self.strides == expected.strides
    }
    
    /// Compute linear index from multi-dimensional index
    pub fn index(&self, indices: &[usize]) -> usize {
        indices.iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| (idx as isize * stride) as usize)
            .sum()
    }
}
```

### 3.2 Sparse Tensor Formats

```rust
/// Coordinate (COO) format
pub struct SparseCOO<T: DType> {
    /// Non-zero values
    values: Vec<T>,
    
    /// Coordinates of non-zero values
    /// Shape: [nnz, ndim]
    indices: Vec<Vec<usize>>,
    
    /// Dense shape
    shape: Shape,
}

/// Compressed Sparse Row (CSR) format
pub struct SparseCSR<T: DType> {
    /// Non-zero values
    values: Vec<T>,
    
    /// Column indices of non-zero values
    col_indices: Vec<usize>,
    
    /// Row pointers (length = nrows + 1)
    row_ptrs: Vec<usize>,
    
    /// Shape (nrows, ncols)
    shape: (usize, usize),
}

impl<T: DType> SparseCSR<T> {
    /// Get value at (row, col), returns T::ZERO if not stored
    pub fn get(&self, row: usize, col: usize) -> T {
        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];
        
        // Binary search in column indices
        match self.col_indices[start..end].binary_search(&col) {
            Ok(idx) => self.values[start + idx],
            Err(_) => T::ZERO,
        }
    }
}
```

---

## 4. Memory Layout Strategies

### 4.1 Dense Layout Options

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutType {
    /// Row-major (C-style): rightmost index varies fastest
    RowMajor,
    
    /// Column-major (Fortran-style): leftmost index varies fastest
    ColumnMajor,
    
    /// Blocked layout for cache efficiency
    /// Parameters: (block_size_x, block_size_y)
    Blocked(usize, usize),
    
    /// Custom strided layout
    Custom,
}
```

### 4.2 Blocked Layout for Matrix Operations

For large matrices, blocked layouts improve cache utilization:

```rust
/// Blocked matrix layout for cache-efficient operations
/// Matrix divided into BxB blocks, stored contiguously
pub struct BlockedMatrix<T: DType> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
    block_size: usize,
}

impl<T: DType> BlockedMatrix<T> {
    pub fn from_row_major(data: &[T], nrows: usize, ncols: usize, block_size: usize) -> Self {
        let num_blocks_row = (nrows + block_size - 1) / block_size;
        let num_blocks_col = (ncols + block_size - 1) / block_size;
        let total_size = num_blocks_row * num_blocks_col * block_size * block_size;
        
        let mut blocked = vec![T::ZERO; total_size];
        
        for block_i in 0..num_blocks_row {
            for block_j in 0..num_blocks_col {
                for i in 0..block_size {
                    for j in 0..block_size {
                        let src_row = block_i * block_size + i;
                        let src_col = block_j * block_size + j;
                        
                        if src_row < nrows && src_col < ncols {
                            let src_idx = src_row * ncols + src_col;
                            let dst_block = block_i * num_blocks_col + block_j;
                            let dst_idx = dst_block * block_size * block_size + i * block_size + j;
                            blocked[dst_idx] = data[src_idx];
                        }
                    }
                }
            }
        }
        
        Self { data: blocked, nrows, ncols, block_size }
    }
}
```

### 4.3 SIMD-Aligned Layouts

```rust
/// Ensure tensor data is aligned for SIMD operations
pub const SIMD_ALIGNMENT: usize = 64; // AVX-512 requires 64-byte alignment

impl<T: DType> Tensor<T> {
    /// Check if tensor data is properly aligned for SIMD
    pub fn is_simd_aligned(&self) -> bool {
        let ptr = self.data_ptr() as usize;
        ptr % SIMD_ALIGNMENT == 0
    }
    
    /// Reallocate with SIMD alignment if necessary
    pub fn ensure_simd_aligned(&mut self) -> Result<()> {
        if !self.is_simd_aligned() {
            // Allocate new aligned storage
            let aligned_storage = Storage::new_aligned(
                self.numel() * T::SIZE,
                SIMD_ALIGNMENT,
                self.device.clone()
            )?;
            
            // Copy data
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data_ptr(),
                    aligned_storage.as_mut_ptr() as *mut T,
                    self.numel()
                );
            }
            
            self.storage = Arc::new(aligned_storage);
            self.offset = 0;
        }
        Ok(())
    }
}
```

---

## 5. Device Memory Management

### 5.1 Storage Abstraction

```rust
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Rocm(RocmStorage),
    Tpu(TpuStorage),
}

impl Storage {
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Storage::Cpu(s) => s.ptr,
            Storage::Cuda(s) => s.device_ptr.as_ptr() as *const u8,
            Storage::Rocm(s) => s.device_ptr.as_ptr() as *const u8,
            Storage::Tpu(s) => unimplemented!("TPU direct access not supported"),
        }
    }
    
    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(s) => s.len,
            Storage::Cuda(s) => s.len,
            Storage::Rocm(s) => s.len,
            Storage::Tpu(s) => s.len,
        }
    }
    
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            Storage::Cuda(s) => Device::Cuda(s.device_id),
            Storage::Rocm(s) => Device::Rocm(s.device_id),
            Storage::Tpu(s) => Device::Tpu(s.device_id),
        }
    }
}
```

### 5.2 CPU Storage Implementation

```rust
pub struct CpuStorage {
    /// Raw pointer to allocated memory
    ptr: *mut u8,
    
    /// Size in bytes
    len: usize,
    
    /// Capacity (may be larger than len)
    capacity: usize,
    
    /// Alignment of allocation
    alignment: usize,
    
    /// Custom allocator (optional)
    allocator: Option<Arc<dyn Allocator>>,
}

impl CpuStorage {
    pub fn new(len: usize, alignment: usize) -> Result<Self> {
        let capacity = len;
        let layout = Layout::from_size_align(capacity, alignment)?;
        
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(Error::AllocationFailed);
        }
        
        Ok(Self {
            ptr,
            len,
            capacity,
            alignment,
            allocator: None,
        })
    }
    
    pub fn as_slice<T>(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr as *const T,
                self.len / std::mem::size_of::<T>()
            )
        }
    }
    
    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr as *mut T,
                self.len / std::mem::size_of::<T>()
            )
        }
    }
}

impl Drop for CpuStorage {
    fn drop(&mut self) {
        if let Some(allocator) = &self.allocator {
            allocator.deallocate(self.ptr, self.len);
        } else {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.capacity, self.alignment);
                std::alloc::dealloc(self.ptr, layout);
            }
        }
    }
}

// Thread-safe reference counting
unsafe impl Send for CpuStorage {}
unsafe impl Sync for CpuStorage {}
```

### 5.3 CUDA Storage Implementation

```rust
pub struct CudaStorage {
    /// Device pointer
    device_ptr: CudaDevicePtr,
    
    /// Size in bytes
    len: usize,
    
    /// CUDA device ID
    device_id: i32,
    
    /// Associated CUDA stream
    stream: CudaStream,
}

impl CudaStorage {
    pub fn new(len: usize, device_id: i32) -> Result<Self> {
        // Set device context
        cuda_sys::cudaSetDevice(device_id)?;
        
        // Allocate device memory
        let mut device_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        cuda_sys::cudaMalloc(&mut device_ptr as *mut *mut _, len)?;
        
        Ok(Self {
            device_ptr: CudaDevicePtr(device_ptr),
            len,
            device_id,
            stream: CudaStream::default(),
        })
    }
    
    pub fn copy_from_host(&mut self, host_data: &[u8]) -> Result<()> {
        assert_eq!(host_data.len(), self.len);
        
        cuda_sys::cudaMemcpyAsync(
            self.device_ptr.as_ptr(),
            host_data.as_ptr() as *const std::ffi::c_void,
            self.len,
            cuda_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            self.stream.as_raw()
        )?;
        
        Ok(())
    }
    
    pub fn copy_to_host(&self, host_buffer: &mut [u8]) -> Result<()> {
        assert_eq!(host_buffer.len(), self.len);
        
        cuda_sys::cudaMemcpyAsync(
            host_buffer.as_mut_ptr() as *mut std::ffi::c_void,
            self.device_ptr.as_ptr(),
            self.len,
            cuda_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            self.stream.as_raw()
        )?;
        
        Ok(())
    }
}

impl Drop for CudaStorage {
    fn drop(&mut self) {
        unsafe {
            cuda_sys::cudaFree(self.device_ptr.as_ptr());
        }
    }
}
```

---

## 6. Allocation Strategies

### 6.1 Memory Pool Allocator

```rust
pub struct MemoryPool {
    /// Free blocks organized by size
    free_blocks: BTreeMap<usize, Vec<*mut u8>>,
    
    /// Allocated blocks with their sizes
    allocated: HashMap<*mut u8, BlockInfo>,
    
    /// Total allocated bytes
    total_allocated: AtomicUsize,
    
    /// Peak allocated bytes
    peak_allocated: AtomicUsize,
    
    /// Device for this pool
    device: Device,
}

struct BlockInfo {
    size: usize,
    alignment: usize,
    timestamp: Instant,
}

impl MemoryPool {
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<*mut u8> {
        // Round up size to next power of 2 for better reuse
        let size = size.next_power_of_two();
        
        // Try to find suitable free block
        if let Some(blocks) = self.free_blocks.get_mut(&size) {
            if let Some(ptr) = blocks.pop() {
                self.allocated.insert(ptr, BlockInfo {
                    size,
                    alignment,
                    timestamp: Instant::now(),
                });
                return Ok(ptr);
            }
        }
        
        // Allocate new block
        let ptr = self.allocate_new(size, alignment)?;
        
        self.allocated.insert(ptr, BlockInfo {
            size,
            alignment,
            timestamp: Instant::now(),
        });
        
        let total = self.total_allocated.fetch_add(size, Ordering::Relaxed) + size;
        self.peak_allocated.fetch_max(total, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    pub fn deallocate(&mut self, ptr: *mut u8) {
        if let Some(info) = self.allocated.remove(&ptr) {
            self.free_blocks
                .entry(info.size)
                .or_insert_with(Vec::new)
                .push(ptr);
            
            self.total_allocated.fetch_sub(info.size, Ordering::Relaxed);
        }
    }
    
    /// Garbage collect old unused blocks
    pub fn gc(&mut self, max_age: Duration) {
        let now = Instant::now();
        
        for blocks in self.free_blocks.values_mut() {
            blocks.retain(|&ptr| {
                if let Some(info) = self.allocated.get(&ptr) {
                    now.duration_since(info.timestamp) < max_age
                } else {
                    false
                }
            });
        }
    }
}
```

### 6.2 Arena Allocator for Temporary Buffers

```rust
pub struct Arena {
    /// Current buffer
    buffer: Vec<u8>,
    
    /// Offset into current buffer
    offset: usize,
    
    /// Alignment for allocations
    alignment: usize,
}

impl Arena {
    pub fn new(capacity: usize, alignment: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            offset: 0,
            alignment,
        }
    }
    
    pub fn allocate<T>(&mut self, n: usize) -> &mut [T] {
        let size = n * std::mem::size_of::<T>();
        let align = std::cmp::max(std::mem::align_of::<T>(), self.alignment);
        
        // Align offset
        let offset = (self.offset + align - 1) & !(align - 1);
        let end = offset + size;
        
        if end > self.buffer.capacity() {
            // Allocate larger buffer
            let new_capacity = self.buffer.capacity() * 2;
            self.buffer.reserve(new_capacity);
        }
        
        self.offset = end;
        
        unsafe {
            self.buffer.set_len(end);
            std::slice::from_raw_parts_mut(
                self.buffer.as_mut_ptr().add(offset) as *mut T,
                n
            )
        }
    }
    
    pub fn reset(&mut self) {
        self.offset = 0;
        unsafe { self.buffer.set_len(0); }
    }
}
```

---

## 7. Lifetime Management

### 7.1 Reference Counting Strategy

```rust
impl<T: DType> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        // Shallow copy: share underlying storage
        Self {
            storage: Arc::clone(&self.storage),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            device: self.device.clone(),
            _phantom: PhantomData,
            id: TensorId::new(), // New tensor gets new ID
            grad_info: self.grad_info.clone(),
        }
    }
}
```

### 7.2 Move Semantics for Zero-Copy

```rust
impl<T: DType> Tensor<T> {
    /// Transfer tensor to new device (may involve copy)
    pub fn to(self, device: Device) -> Result<Self> {
        if self.device == device {
            return Ok(self);
        }
        
        // For now, always copy
        // TODO: Implement peer-to-peer GPU transfers
        let mut new_storage = Storage::new(self.numel() * T::SIZE, device.clone())?;
        
        // Copy data (device-specific)
        self.storage.copy_to(&mut new_storage)?;
        
        Ok(Self {
            storage: Arc::new(new_storage),
            device,
            ..self
        })
    }
}
```

---

## 8. Zero-Copy Operations

### 8.1 View Creation

```rust
impl<T: DType> Tensor<T> {
    /// Create view with different shape (no copy)
    pub fn view(&self, new_shape: &[usize]) -> Result<Self> {
        // Check total elements match
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(Error::InvalidShape);
        }
        
        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape: Shape::new(new_shape),
            strides: Strides::contiguous(&Shape::new(new_shape)),
            offset: self.offset,
            device: self.device.clone(),
            _phantom: PhantomData,
            id: TensorId::new(),
            grad_info: None, // Views don't track gradients
        })
    }
    
    /// Transpose without copying data
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        
        new_shape.dims.swap(dim0, dim1);
        new_strides.strides.swap(dim0, dim1);
        
        Self {
            storage: Arc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            device: self.device.clone(),
            _phantom: PhantomData,
            id: TensorId::new(),
            grad_info: None,
        }
    }
    
    /// Slice tensor along dimension (no copy)
    pub fn slice(&self, dim: usize, range: Range<usize>) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(Error::InvalidDimension);
        }
        
        if range.end > self.shape.size(dim) {
            return Err(Error::InvalidRange);
        }
        
        let mut new_shape = self.shape.clone();
        new_shape.dims[dim] = range.end - range.start;
        new_shape.numel = new_shape.dims.iter().product();
        
        let new_offset = self.offset + range.start * self.strides.strides[dim] as usize;
        
        Self {
            storage: Arc::clone(&self.storage),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
            device: self.device.clone(),
            _phantom: PhantomData,
            id: TensorId::new(),
            grad_info: None,
        }
    }
}
```

### 8.2 Contiguous Conversion

Some operations require contiguous memory layout:

```rust
impl<T: DType> Tensor<T> {
    pub fn contiguous(&self) -> Self {
        if self.strides.is_contiguous(&self.shape) {
            return self.clone();
        }
        
        // Allocate new contiguous storage
        let new_storage = Storage::new(
            self.numel() * T::SIZE,
            self.device.clone()
        ).unwrap();
        
        // Copy with stride handling
        self.copy_strided_to_contiguous(&new_storage);
        
        Self {
            storage: Arc::new(new_storage),
            shape: self.shape.clone(),
            strides: Strides::contiguous(&self.shape),
            offset: 0,
            device: self.device.clone(),
            _phantom: PhantomData,
            id: TensorId::new(),
            grad_info: None,
        }
    }
    
    fn copy_strided_to_contiguous(&self, dst: &Storage) {
        // Multi-dimensional iterator over indices
        let mut indices = vec![0usize; self.ndim()];
        let mut dst_offset = 0;
        
        loop {
            // Compute source offset using strides
            let src_offset = self.offset + self.strides.index(&indices);
            
            // Copy single element
            unsafe {
                let src_ptr = self.storage.as_ptr().add(src_offset * T::SIZE) as *const T;
                let dst_ptr = dst.as_mut_ptr().add(dst_offset * T::SIZE) as *mut T;
                *dst_ptr = *src_ptr;
            }
            
            dst_offset += 1;
            
            // Increment indices
            let mut dim = self.ndim() - 1;
            loop {
                indices[dim] += 1;
                if indices[dim] < self.shape.size(dim) {
                    break;
                }
                indices[dim] = 0;
                if dim == 0 {
                    return; // Done
                }
                dim -= 1;
            }
        }
    }
}
```

---

## Summary

This section defined FerricML's foundational type system and memory management:

**Key Accomplishments:**
- Complete scalar type system with SIMD awareness
- Flexible tensor representation supporting views and strides
- Device-agnostic storage abstraction
- Efficient memory allocation strategies (pooling, arenas)
- Zero-copy operations for common transformations

**Implementation Notes:**
1. All types must be `Copy + Send + Sync` for safe parallelism
2. Use `Arc` for shared ownership of storage
3. Alignment is critical for SIMD performance
4. Device transfers may block; consider async alternatives
5. Memory pools significantly reduce allocation overhead

**Next Steps:**
- Section 2 builds the IR on top of this type system
- Section 3 implements CUDA kernels using these storage primitives
- Section 5 extends tensors with autograd tracking

**Performance Considerations:**
- Avoid contiguous conversions in hot paths
- Prefer views over clones when possible
- Pool allocations should be device-specific
- SIMD alignment provides 2-4x speedups for element-wise ops