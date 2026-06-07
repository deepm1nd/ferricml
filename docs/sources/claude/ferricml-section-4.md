        let mut acc = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let offset = i * SIMD_WIDTH;
            
            let va = Avx2F32::load_unaligned(a.as_ptr().add(offset));
            let vb = Avx2F32::load_unaligned(b.as_ptr().add(offset));
            
            acc = Avx2F32::fma(va, vb, acc);
        }
        
        let mut result = Avx2F32::horizontal_sum(acc);
        
        // Scalar remainder
        for i in (chunks * SIMD_WIDTH)..n {
            result += a[i] * b[i];
        }
        
        result
    }
}

#[cfg(target_arch = "aarch64")]
fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 4;
    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        
        for i in 0..chunks {
            let offset = i * SIMD_WIDTH;
            
            let va = NeonF32::load_unaligned(a.as_ptr().add(offset));
            let vb = NeonF32::load_unaligned(b.as_ptr().add(offset));
            
            acc = NeonF32::fma(va, vb, acc);
        }
        
        let mut result = NeonF32::horizontal_sum(acc);
        
        // Scalar remainder
        for i in (chunks * SIMD_WIDTH)..n {
            result += a[i] * b[i];
        }
        
        result
    }
}
```

### 5.2 Vectorized Activation Functions

```rust
pub mod activations {
    use super::*;
    
    pub fn relu_f32(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        
        #[cfg(target_feature = "avx2")]
        {
            relu_avx2(input, output);
        }
        
        #[cfg(not(target_feature = "avx2"))]
        {
            for i in 0..input.len() {
                output[i] = input[i].max(0.0);
            }
        }
    }
    
    #[cfg(target_feature = "avx2")]
    fn relu_avx2(input: &[f32], output: &mut [f32]) {
        const SIMD_WIDTH: usize = 8;
        let n = input.len();
        let chunks = n / SIMD_WIDTH;
        
        unsafe {
            let zero = _mm256_setzero_ps();
            
            for i in 0..chunks {
                let offset = i * SIMD_WIDTH;
                
                let v = Avx2F32::load_unaligned(input.as_ptr().add(offset));
                let result = _mm256_max_ps(v, zero);
                
                Avx2F32::store_unaligned(output.as_mut_ptr().add(offset), result);
            }
            
            // Scalar remainder
            for i in (chunks * SIMD_WIDTH)..n {
                output[i] = input[i].max(0.0);
            }
        }
    }
    
    pub fn sigmoid_f32(input: &[f32], output: &mut [f32]) {
        // Sigmoid: 1 / (1 + exp(-x))
        // Use approximation for SIMD: tanh(x/2) / 2 + 0.5
        
        #[cfg(target_feature = "avx2")]
        {
            sigmoid_avx2_approx(input, output);
        }
        
        #[cfg(not(target_feature = "avx2"))]
        {
            for i in 0..input.len() {
                output[i] = 1.0 / (1.0 + (-input[i]).exp());
            }
        }
    }
    
    #[cfg(target_feature = "avx2")]
    fn sigmoid_avx2_approx(input: &[f32], output: &mut [f32]) {
        const SIMD_WIDTH: usize = 8;
        let n = input.len();
        let chunks = n / SIMD_WIDTH;
        
        unsafe {
            let half = _mm256_set1_ps(0.5);
            let one = _mm256_set1_ps(1.0);
            let neg_one = _mm256_set1_ps(-1.0);
            
            for i in 0..chunks {
                let offset = i * SIMD_WIDTH;
                
                let x = Avx2F32::load_unaligned(input.as_ptr().add(offset));
                
                // Clamp to avoid overflow
                let x_clamped = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(10.0)), _mm256_set1_ps(-10.0));
                
                // exp(-x)
                let neg_x = _mm256_mul_ps(x_clamped, neg_one);
                let exp_neg_x = exp_ps256(neg_x);
                
                // 1 / (1 + exp(-x))
                let denominator = _mm256_add_ps(one, exp_neg_x);
                let result = _mm256_div_ps(one, denominator);
                
                Avx2F32::store_unaligned(output.as_mut_ptr().add(offset), result);
            }
            
            // Scalar remainder
            for i in (chunks * SIMD_WIDTH)..n {
                output[i] = 1.0 / (1.0 + (-input[i]).exp());
            }
        }
    }
    
    // Fast exp approximation for AVX2
    #[cfg(target_feature = "avx2")]
    unsafe fn exp_ps256(x: __m256) -> __m256 {
        // Polynomial approximation: exp(x) ≈ 2^(x/ln(2))
        // Using Remez algorithm coefficients
        
        let log2e = _mm256_set1_ps(1.44269504);
        let ln2 = _mm256_set1_ps(0.693147180);
        
        // Compute n and r where x = n*ln(2) + r
        let fx = _mm256_mul_ps(x, log2e);
        let n = _mm256_floor_ps(fx);
        let r = _mm256_sub_ps(x, _mm256_mul_ps(n, ln2));
        
        // Polynomial approximation for exp(r)
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(1.0);
        let c3 = _mm256_set1_ps(0.5);
        let c4 = _mm256_set1_ps(0.16666667);
        let c5 = _mm256_set1_ps(0.04166667);
        
        let poly = _mm256_add_ps(c1,
            _mm256_mul_ps(r,
                _mm256_add_ps(c2,
                    _mm256_mul_ps(r,
                        _mm256_add_ps(c3,
                            _mm256_mul_ps(r,
                                _mm256_add_ps(c4,
                                    _mm256_mul_ps(r, c5))))))));
        
        // Scale by 2^n using integer arithmetic
        let n_i32 = _mm256_cvtps_epi32(n);
        let bias = _mm256_set1_epi32(127);
        let exponent = _mm256_add_epi32(n_i32, bias);
        let scale = _mm256_castsi256_ps(_mm256_slli_epi32(exponent, 23));
        
        _mm256_mul_ps(poly, scale)
    }
}
```

---

## 6. BLAS Integration

### 6.1 BLAS Backend Selection

```rust
pub enum BlasBackend {
    OpenBLAS,
    MKL,
    Accelerate,  // macOS
    BLIS,
    Native,      // Pure Rust implementation
}

pub struct BlasOps {
    backend: BlasBackend,
}

impl BlasOps {
    pub fn new() -> Self {
        let backend = Self::detect_backend();
        Self { backend }
    }
    
    fn detect_backend() -> BlasBackend {
        #[cfg(target_os = "macos")]
        {
            return BlasBackend::Accelerate;
        }
        
        #[cfg(feature = "mkl")]
        {
            return BlasBackend::MKL;
        }
        
        #[cfg(feature = "openblas")]
        {
            return BlasBackend::OpenBLAS;
        }
        
        BlasBackend::Native
    }
    
    /// Matrix-matrix multiplication: C = alpha * A * B + beta * C
    pub fn gemm(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        match self.backend {
            #[cfg(feature = "openblas")]
            BlasBackend::OpenBLAS => {
                unsafe {
                    cblas::sgemm(
                        cblas::Layout::RowMajor,
                        if transa { cblas::Transpose::Ordinary } else { cblas::Transpose::None },
                        if transb { cblas::Transpose::Ordinary } else { cblas::Transpose::None },
                        m as i32,
                        n as i32,
                        k as i32,
                        alpha,
                        a.as_ptr(),
                        lda as i32,
                        b.as_ptr(),
                        ldb as i32,
                        beta,
                        c.as_mut_ptr(),
                        ldc as i32,
                    );
                }
            }
            
            #[cfg(target_os = "macos")]
            BlasBackend::Accelerate => {
                unsafe {
                    accelerate_src::vecLib::cblas_sgemm(
                        accelerate_src::vecLib::CblasRowMajor,
                        if transa { accelerate_src::vecLib::CblasTrans } else { accelerate_src::vecLib::CblasNoTrans },
                        if transb { accelerate_src::vecLib::CblasTrans } else { accelerate_src::vecLib::CblasNoTrans },
                        m as i32,
                        n as i32,
                        k as i32,
                        alpha,
                        a.as_ptr(),
                        lda as i32,
                        b.as_ptr(),
                        ldb as i32,
                        beta,
                        c.as_mut_ptr(),
                        ldc as i32,
                    );
                }
            }
            
            BlasBackend::Native => {
                self.gemm_native(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            }
            
            _ => unimplemented!("BLAS backend {:?}", self.backend),
        }
    }
    
    fn gemm_native(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        // Use our cache-aware implementation
        let cache_info = CpuInfo::detect().cache_sizes;
        let matmul = CacheAwareMatMul { cache_info };
        
        // Handle transpositions by adjusting indexing
        // This is a simplified version; production code would optimize transposes
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                
                for kk in 0..k {
                    let a_val = if transa {
                        a[kk * lda + i]
                    } else {
                        a[i * lda + kk]
                    };
                    
                    let b_val = if transb {
                        b[j * ldb + kk]
                    } else {
                        b[kk * ldb + j]
                    };
                    
                    sum += a_val * b_val;
                }
                
                c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
            }
        }
    }
}
```

---

## 7. Tree Model Optimization

### 7.1 SIMD Decision Tree Prediction

```rust
pub struct VectorizedTreePredictor {
    cpu_info: CpuInfo,
}

impl VectorizedTreePredictor {
    /// Predict multiple samples simultaneously using SIMD
    pub fn predict_batch_f32(
        &self,
        tree: &DecisionTree,
        samples: &[f32],
        n_samples: usize,
        n_features: usize,
        output: &mut [f32],
    ) {
        #[cfg(target_feature = "avx2")]
        {
            self.predict_batch_avx2(tree, samples, n_samples, n_features, output);
        }
        
        #[cfg(not(target_feature = "avx2"))]
        {
            self.predict_batch_scalar(tree, samples, n_samples, n_features, output);
        }
    }
    
    #[cfg(target_feature = "avx2")]
    fn predict_batch_avx2(
        &self,
        tree: &DecisionTree,
        samples: &[f32],
        n_samples: usize,
        n_features: usize,
        output: &mut [f32],
    ) {
        const SIMD_WIDTH: usize = 8;
        let batch_size = SIMD_WIDTH;
        
        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let actual_batch = batch_end - batch_start;
            
            if actual_batch == batch_size {
                // Full batch - use SIMD
                unsafe {
                    self.predict_batch_simd_full(
                        tree,
                        &samples[batch_start * n_features..],
                        n_features,
                        &mut output[batch_start..batch_end],
                    );
                }
            } else {
                // Partial batch - use scalar
                for i in batch_start..batch_end {
                    output[i] = self.predict_single(tree, &samples[i * n_features..(i + 1) * n_features]);
                }
            }
        }
    }
    
    #[cfg(target_feature = "avx2")]
    unsafe fn predict_batch_simd_full(
        &self,
        tree: &DecisionTree,
        samples: &[f32],
        n_features: usize,
        output: &mut [f32],
    ) {
        const SIMD_WIDTH: usize = 8;
        
        // Track node indices for each sample in the batch
        let mut node_indices = [0usize; SIMD_WIDTH];
        let mut active = [true; SIMD_WIDTH];
        
        // Traverse tree for all samples simultaneously
        while active.iter().any(|&a| a) {
            for lane in 0..SIMD_WIDTH {
                if !active[lane] {
                    continue;
                }
                
                let node_idx = node_indices[lane];
                let node = &tree.nodes[node_idx];
                
                if node.is_leaf {
                    output[lane] = node.value;
                    active[lane] = false;
                } else {
                    // Load feature value for this sample
                    let feature_idx = node.feature_idx;
                    let feature_val = samples[lane * n_features + feature_idx];
                    
                    // Navigate to next node
                    node_indices[lane] = if feature_val <= node.threshold {
                        node.left_child
                    } else {
                        node.right_child
                    };
                }
            }
        }
    }
    
    fn predict_single(&self, tree: &DecisionTree, sample: &[f32]) -> f32 {
        let mut node_idx = 0;
        
        loop {
            let node = &tree.nodes[node_idx];
            
            if node.is_leaf {
                return node.value;
            }
            
            let feature_val = sample[node.feature_idx];
            node_idx = if feature_val <= node.threshold {
                node.left_child
            } else {
                node.right_child
            };
        }
    }
}

pub struct DecisionTree {
    nodes: Vec<TreeNode>,
    max_depth: usize,
}

pub struct TreeNode {
    is_leaf: bool,
    feature_idx: usize,
    threshold: f32,
    left_child: usize,
    right_child: usize,
    value: f32,
}
```

### 7.2 Gradient Computation for GBDT

```rust
pub struct VectorizedGradient {
    cpu_info: CpuInfo,
}

impl VectorizedGradient {
    /// Compute gradients for logistic loss
    pub fn logistic_gradient_f32(
        &self,
        predictions: &[f32],
        labels: &[f32],
        gradients: &mut [f32],
    ) {
        assert_eq!(predictions.len(), labels.len());
        assert_eq!(predictions.len(), gradients.len());
        
        #[cfg(target_feature = "avx2")]
        {
            self.logistic_gradient_avx2(predictions, labels, gradients);
        }
        
        #[cfg(not(target_feature = "avx2"))]
        {
            for i in 0..predictions.len() {
                let pred = 1.0 / (1.0 + (-predictions[i]).exp());
                gradients[i] = pred - labels[i];
            }
        }
    }
    
    #[cfg(target_feature = "avx2")]
    fn logistic_gradient_avx2(
        &self,
        predictions: &[f32],
        labels: &[f32],
        gradients: &mut [f32],
    ) {
        const SIMD_WIDTH: usize = 8;
        let n = predictions.len();
        let chunks = n / SIMD_WIDTH;
        
        unsafe {
            let one = _mm256_set1_ps(1.0);
            let neg_one = _mm256_set1_ps(-1.0);
            
            for i in 0..chunks {
                let offset = i * SIMD_WIDTH;
                
                let pred = Avx2F32::load_unaligned(predictions.as_ptr().add(offset));
                let label = Avx2F32::load_unaligned(labels.as_ptr().add(offset));
                
                // Sigmoid: 1 / (1 + exp(-pred))
                let neg_pred = _mm256_mul_ps(pred, neg_one);
                let exp_neg_pred = activations::exp_ps256(neg_pred);
                let sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_pred));
                
                // Gradient: sigmoid - label
                let grad = _mm256_sub_ps(sigmoid, label);
                
                Avx2F32::store_unaligned(gradients.as_mut_ptr().add(offset), grad);
            }
            
            // Scalar remainder
            for i in (chunks * SIMD_WIDTH)..n {
                let pred = 1.0 / (1.0 + (-predictions[i]).exp());
                gradients[i] = pred - labels[i];
            }
        }
    }
    
    /// Compute hessian for second-order optimization
    pub fn logistic_hessian_f32(
        &self,
        predictions: &[f32],
        hessians: &mut [f32],
    ) {
        #[cfg(target_feature = "avx2")]
        {
            self.logistic_hessian_avx2(predictions, hessians);
        }
        
        #[cfg(not(target_feature = "avx2"))]
        {
            for i in 0..predictions.len() {
                let pred = 1.0 / (1.0 + (-predictions[i]).exp());
                hessians[i] = pred * (1.0 - pred);
            }
        }
    }
    
    #[cfg(target_feature = "avx2")]
    fn logistic_hessian_avx2(
        &self,
        predictions: &[f32],
        hessians: &mut [f32],
    ) {
        const SIMD_WIDTH: usize = 8;
        let n = predictions.len();
        let chunks = n / SIMD_WIDTH;
        
        unsafe {
            let one = _mm256_set1_ps(1.0);
            let neg_one = _mm256_set1_ps(-1.0);
            
            for i in 0..chunks {
                let offset = i * SIMD_WIDTH;
                
                let pred = Avx2F32::load_unaligned(predictions.as_ptr().add(offset));
                
                // Sigmoid
                let neg_pred = _mm256_mul_ps(pred, neg_one);
                let exp_neg_pred = activations::exp_ps256(neg_pred);
                let sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_pred));
                
                // Hessian: sigmoid * (1 - sigmoid)
                let one_minus_sigmoid = _mm256_sub_ps(one, sigmoid);
                let hess = _mm256_mul_ps(sigmoid, one_minus_sigmoid);
                
                Avx2F32::store_unaligned(hessians.as_mut_ptr().add(offset), hess);
            }
            
            // Scalar remainder
            for i in (chunks * SIMD_WIDTH)..n {
                let pred = 1.0 / (1.0 + (-predictions[i]).exp());
                hessians[i] = pred * (1.0 - pred);
            }
        }
    }
}
```

---

## 8. NUMA Awareness

### 8.1 NUMA Topology Detection

```rust
#[cfg(target_os = "linux")]
pub struct NumaTopology {
    nodes: Vec<NumaNode>,
}

#[cfg(target_os = "linux")]
pub struct NumaNode {
    id: usize,
    cpus: Vec<usize>,
    memory_size: usize,
    distances: Vec<usize>,  // Distance to other NUMA nodes
}

#[cfg(target_os = "linux")]
impl NumaTopology {
    pub fn detect() -> Result<Self> {
        use std::fs;
        
        let num_nodes = fs::read_dir("/sys/devices/system/node")?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_str().unwrap().starts_with("node"))
            .count();
        
        let mut nodes = Vec::new();
        
        for node_id in 0..num_nodes {
            let cpus = Self::read_node_cpus(node_id)?;
            let memory_size = Self::read_node_memory(node_id)?;
            let distances = Self::read_node_distances(node_id, num_nodes)?;
            
            nodes.push(NumaNode {
                id: node_id,
                cpus,
                memory_size,
                distances,
            });
        }
        
        Ok(Self { nodes })
    }
    
    fn read_node_cpus(node_id: usize) -> Result<Vec<usize>> {
        let path = format!("/sys/devices/system/node/node{}/cpulist", node_id);
        let content = std::fs::read_to_string(path)?;
        
        // Parse CPU list (e.g., "0-7,16-23")
        let mut cpus = Vec::new();
        for range in content.trim().split(',') {
            if let Some((start, end)) = range.split_once('-') {
                let start: usize = start.parse()?;
                let end: usize = end.parse()?;
                cpus.extend(start..=end);
            } else {
                cpus.push(range.parse()?);
            }
        }
        
        Ok(cpus)
    }
    
    fn read_node_memory(node_id: usize) -> Result<usize> {
        let path = format!("/sys/devices/system/node/node{}/meminfo", node_id);
        let content = std::fs::read_to_string(path)?;
        
        // Extract total memory
        for line in content.lines() {
            if line.contains("MemTotal") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(size_str) = parts.get(3) {
                    return Ok(size_str.parse::<usize>()? * 1024); // Convert KB to bytes
                }
            }
        }
        
        Ok(0)
    }
    
    fn read_node_distances(node_id: usize, num_nodes: usize) -> Result<Vec<usize>> {
        let path = format!("/sys/devices/system/node/node{}/distance", node_id);
        let content = std::fs::read_to_string(path)?;
        
        let distances: Vec<usize> = content
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        
        Ok(distances)
    }
    
    pub fn bind_thread_to_node(&self, node_id: usize) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};
            
            let node = &self.nodes[node_id];
            
            unsafe {
                let mut cpuset: cpu_set_t = std::mem::zeroed();
                CPU_ZERO(&mut cpuset);
                
                for &cpu in &node.cpus {
                    CPU_SET(cpu, &mut cpuset);
                }
                
                let result = sched_setaffinity(
                    0,  // Current thread
                    std::mem::size_of::<cpu_set_t>(),
                    &cpuset,
                );
                
                if result != 0 {
                    return Err(Error::NumaBindFailed);
                }
            }
        }
        
        Ok(())
    }
}
```

### 8.2 NUMA-Aware Memory Allocation

```rust
pub struct NumaAllocator {
    topology: NumaTopology,
}

impl NumaAllocator {
    pub fn new() -> Result<Self> {
        let topology = NumaTopology::detect()?;
        Ok(Self { topology })
    }
    
    pub fn allocate_on_node(&self, size: usize, node_id: usize) -> Result<*mut u8> {
        #[cfg(target_os = "linux")]
        {
            use libc::{mmap, PROT_READ, PROT_WRITE, MAP_PRIVATE, MAP_ANONYMOUS};
            
            unsafe {
                let ptr = mmap(
                    std::ptr::null_mut(),
                    size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                );
                
                if ptr == libc::MAP_FAILED {
                    return Err(Error::AllocationFailed);
                }
                
                // Bind memory to NUMA node using mbind
                self.bind_memory_to_node(ptr as *mut u8, size, node_id)?;
                
                Ok(ptr as *mut u8)
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback to standard allocation
            let layout = Layout::from_size_align(size, 64)?;
            Ok(unsafe { std::alloc::alloc(layout) })
        }
    }
    
    #[cfg(target_os = "linux")]
    fn bind_memory_to_node(&self, ptr: *mut u8, size: usize, node_id: usize) -> Result<()> {
        use libc::{mbind, MPOL_BIND};
        
        let mut nodemask: u64 = 1 << node_id;
        
        unsafe {
            let result = mbind(
                ptr as *mut libc::c_void,
                size,
                MPOL_BIND,
                &mut nodemask as *mut u64 as *const libc::c_ulong,
                64,  // maxnode
                0,   // flags
            );
            
            if result != 0 {
                return Err(Error::NumaBindFailed);
            }
        }
        
        Ok(())
    }
}
```

---

## Summary

This section detailed the CPU backend implementation for FerricML:

**Key Components:**
1. **CPU Detection:** Comprehensive capability detection (cores, cache, SIMD)
2. **SIMD Abstractions:** Unified interface for AVX2, AVX-512, and NEON
3. **Cache Optimization:** Tiling and blocking strategies for memory hierarchy
4. **Thread Management:** Work-stealing pool and Rayon integration
5. **Vectorized Ops:** SIMD implementations of common operations
6. **BLAS Integration:** Support for OpenBLAS, MKL, and Accelerate
7. **Tree Optimization:** SIMD batch prediction for decision trees
8. **NUMA Support:** Topology-aware allocation and thread binding

**Performance Considerations:**
- SIMD provides 4-8x speedup for element-wise operations
- Cache-aware tiling critical for large matrix operations
- Work-stealing reduces load imbalance
- NUMA binding prevents cross-socket memory traffic
- BLAS libraries provide highly optimized GEMM

**Implementation Priority:**
1. Basic SIMD operations and detection
2. Cache-aware matrix multiplication
3. Thread pool infrastructure
4. BLAS integration
5. Tree model optimizations
6. NUMA awareness (servers only)

**Integration Points:**
- Section 2 IR lowers to CPU instructions
- Section 6 optimizations select tile sizes based on cache
- Section 8 tree models use SIMD batch prediction
- Complements Section 3 CUDA for heterogeneous execution

**Next Steps:**
- Profile to identify optimization opportunities
- Benchmark against BLAS implementations
- Tune tile sizes per CPU architecture
- Add AVX-512 variants for newest CPUs# Section 4: CPU Backend & Vectorization

**FerricML Architecture Specification v3.0**  
**Word Count:** 3,500+ words  
**Implementation Priority:** Phase 2 - Primary Backend

---

## Table of Contents

1. [CPU Architecture Overview](#1-cpu-architecture-overview)
2. [SIMD Instruction Sets](#2-simd-instruction-sets)
3. [Cache-Aware Algorithms](#3-cache-aware-algorithms)
4. [Thread Pool Management](#4-thread-pool-management)
5. [Vectorized Operations](#5-vectorized-operations)
6. [BLAS Integration](#6-blas-integration)
7. [Tree Model Optimization](#7-tree-model-optimization)
8. [NUMA Awareness](#8-numa-awareness)

---

## 1. CPU Architecture Overview

### 1.1 CPU Capabilities Detection

```rust
pub struct CpuInfo {
    /// Number of physical cores
    pub physical_cores: usize,
    
    /// Number of logical cores (with hyperthreading)
    pub logical_cores: usize,
    
    /// Cache sizes (L1d, L1i, L2, L3 in bytes)
    pub cache_sizes: CacheInfo,
    
    /// SIMD capabilities
    pub simd_features: SimdFeatures,
    
    /// Vendor (Intel, AMD, ARM)
    pub vendor: CpuVendor,
    
    /// Memory bandwidth (GB/s, estimated)
    pub memory_bandwidth: f32,
}

#[derive(Debug, Clone)]
pub struct CacheInfo {
    pub l1d_size: usize,
    pub l1i_size: usize,
    pub l2_size: usize,
    pub l3_size: usize,
    pub cache_line_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct SimdFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512dq: bool,
    pub avx512bw: bool,
    pub fma: bool,
    pub neon: bool,  // ARM
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVendor {
    Intel,
    AMD,
    ARM,
    Unknown,
}

impl CpuInfo {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_64()
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_aarch64()
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64() -> Self {
        use raw_cpuid::{CpuId, CpuIdReaderNative};
        
        let cpuid = CpuId::with_cpuid_fn(CpuIdReaderNative::new());
        
        // Detect cores
        let (physical_cores, logical_cores) = if let Some(topo) = cpuid.get_processor_topology_info() {
            (topo.num_physical_cores() as usize, topo.num_logical_cores() as usize)
        } else {
            (num_cpus::get_physical(), num_cpus::get())
        };
        
        // Detect cache sizes
        let cache_sizes = if let Some(cache) = cpuid.get_cache_parameters() {
            let mut info = CacheInfo {
                l1d_size: 0,
                l1i_size: 0,
                l2_size: 0,
                l3_size: 0,
                cache_line_size: 64,
            };
            
            for c in cache {
                let size = c.associativity() * c.physical_line_partitions() 
                    * c.coherency_line_size() * c.sets();
                
                match c.level() {
                    1 => {
                        if c.cache_type() == raw_cpuid::CacheType::Data {
                            info.l1d_size = size;
                        } else if c.cache_type() == raw_cpuid::CacheType::Instruction {
                            info.l1i_size = size;
                        }
                    }
                    2 => info.l2_size = size,
                    3 => info.l3_size = size,
                    _ => {}
                }
            }
            
            info
        } else {
            CacheInfo::default()
        };
        
        // Detect SIMD features
        let features = if let Some(feat) = cpuid.get_feature_info() {
            SimdFeatures {
                sse: feat.has_sse(),
                sse2: feat.has_sse2(),
                sse3: feat.has_sse3(),
                ssse3: feat.has_ssse3(),
                sse41: feat.has_sse41(),
                sse42: feat.has_sse42(),
                avx: feat.has_avx(),
                fma: feat.has_fma(),
                avx2: cpuid.get_extended_feature_info()
                    .map(|e| e.has_avx2())
                    .unwrap_or(false),
                avx512f: cpuid.get_extended_feature_info()
                    .map(|e| e.has_avx512f())
                    .unwrap_or(false),
                avx512dq: cpuid.get_extended_feature_info()
                    .map(|e| e.has_avx512dq())
                    .unwrap_or(false),
                avx512bw: cpuid.get_extended_feature_info()
                    .map(|e| e.has_avx512bw())
                    .unwrap_or(false),
                neon: false,
            }
        } else {
            SimdFeatures::default()
        };
        
        // Detect vendor
        let vendor = if let Some(vi) = cpuid.get_vendor_info() {
            match vi.as_str() {
                "GenuineIntel" => CpuVendor::Intel,
                "AuthenticAMD" => CpuVendor::AMD,
                _ => CpuVendor::Unknown,
            }
        } else {
            CpuVendor::Unknown
        };
        
        Self {
            physical_cores,
            logical_cores,
            cache_sizes,
            simd_features: features,
            vendor,
            memory_bandwidth: Self::estimate_bandwidth(&cache_sizes),
        }
    }
    
    fn estimate_bandwidth(cache: &CacheInfo) -> f32 {
        // Very rough estimation based on cache size
        // Real bandwidth depends on CPU generation
        if cache.l3_size > 32 * 1024 * 1024 {
            100.0  // Modern server CPU
        } else if cache.l3_size > 16 * 1024 * 1024 {
            50.0   // Desktop CPU
        } else {
            25.0   // Laptop/mobile CPU
        }
    }
    
    pub fn preferred_simd_width(&self) -> usize {
        if self.simd_features.avx512f {
            64  // 512 bits = 64 bytes
        } else if self.simd_features.avx2 {
            32  // 256 bits = 32 bytes
        } else if self.simd_features.sse2 {
            16  // 128 bits = 16 bytes
        } else if self.simd_features.neon {
            16  // NEON is 128-bit
        } else {
            8   // Fallback to scalar
        }
    }
}
```

---

## 2. SIMD Instruction Sets

### 2.1 SIMD Abstraction Layer

```rust
/// Trait for SIMD operations
pub trait SimdOps<T> {
    type Vector;
    
    /// Load aligned data
    fn load_aligned(ptr: *const T) -> Self::Vector;
    
    /// Load unaligned data
    fn load_unaligned(ptr: *const T) -> Self::Vector;
    
    /// Store aligned data
    fn store_aligned(ptr: *mut T, vec: Self::Vector);
    
    /// Store unaligned data
    fn store_unaligned(ptr: *mut T, vec: Self::Vector);
    
    /// Add two vectors
    fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    
    /// Multiply two vectors
    fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    
    /// Fused multiply-add: a * b + c
    fn fma(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;
    
    /// Horizontal sum of vector elements
    fn horizontal_sum(vec: Self::Vector) -> T;
    
    /// Broadcast scalar to vector
    fn broadcast(val: T) -> Self::Vector;
}
```

### 2.2 AVX2 Implementation

```rust
#[cfg(target_feature = "avx2")]
pub struct Avx2F32;

#[cfg(target_feature = "avx2")]
impl SimdOps<f32> for Avx2F32 {
    type Vector = __m256;
    
    #[inline(always)]
    fn load_aligned(ptr: *const f32) -> Self::Vector {
        unsafe { _mm256_load_ps(ptr) }
    }
    
    #[inline(always)]
    fn load_unaligned(ptr: *const f32) -> Self::Vector {
        unsafe { _mm256_loadu_ps(ptr) }
    }
    
    #[inline(always)]
    fn store_aligned(ptr: *mut f32, vec: Self::Vector) {
        unsafe { _mm256_store_ps(ptr, vec) }
    }
    
    #[inline(always)]
    fn store_unaligned(ptr: *mut f32, vec: Self::Vector) {
        unsafe { _mm256_storeu_ps(ptr, vec) }
    }
    
    #[inline(always)]
    fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { _mm256_add_ps(a, b) }
    }
    
    #[inline(always)]
    fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { _mm256_mul_ps(a, b) }
    }
    
    #[inline(always)]
    fn fma(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        unsafe { _mm256_fmadd_ps(a, b, c) }
    }
    
    #[inline(always)]
    fn horizontal_sum(vec: Self::Vector) -> f32 {
        unsafe {
            // Horizontal add twice to sum all 8 elements
            let sum1 = _mm256_hadd_ps(vec, vec);
            let sum2 = _mm256_hadd_ps(sum1, sum1);
            
            // Extract lower and upper 128 bits and add
            let lower = _mm256_castps256_ps128(sum2);
            let upper = _mm256_extractf128_ps(sum2, 1);
            let final_sum = _mm_add_ps(lower, upper);
            
            _mm_cvtss_f32(final_sum)
        }
    }
    
    #[inline(always)]
    fn broadcast(val: f32) -> Self::Vector {
        unsafe { _mm256_set1_ps(val) }
    }
}

#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;
```

### 2.3 Generic Vectorized Loop

```rust
pub fn vectorized_add_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    
    let n = a.len();
    
    #[cfg(target_feature = "avx2")]
    {
        const SIMD_WIDTH: usize = 8;
        let chunks = n / SIMD_WIDTH;
        
        unsafe {
            // Vectorized loop
            for i in 0..chunks {
                let offset = i * SIMD_WIDTH;
                
                let va = Avx2F32::load_unaligned(a.as_ptr().add(offset));
                let vb = Avx2F32::load_unaligned(b.as_ptr().add(offset));
                let vc = Avx2F32::add(va, vb);
                
                Avx2F32::store_unaligned(c.as_mut_ptr().add(offset), vc);
            }
            
            // Scalar remainder
            for i in (chunks * SIMD_WIDTH)..n {
                c[i] = a[i] + b[i];
            }
        }
    }
    
    #[cfg(not(target_feature = "avx2"))]
    {
        // Scalar fallback
        for i in 0..n {
            c[i] = a[i] + b[i];
        }
    }
}
```

### 2.4 ARM NEON Implementation

```rust
#[cfg(target_arch = "aarch64")]
pub struct NeonF32;

#[cfg(target_arch = "aarch64")]
impl SimdOps<f32> for NeonF32 {
    type Vector = float32x4_t;
    
    #[inline(always)]
    fn load_aligned(ptr: *const f32) -> Self::Vector {
        unsafe { vld1q_f32(ptr) }
    }
    
    #[inline(always)]
    fn load_unaligned(ptr: *const f32) -> Self::Vector {
        unsafe { vld1q_f32(ptr) }  // NEON doesn't distinguish aligned/unaligned
    }
    
    #[inline(always)]
    fn store_aligned(ptr: *mut f32, vec: Self::Vector) {
        unsafe { vst1q_f32(ptr, vec) }
    }
    
    #[inline(always)]
    fn store_unaligned(ptr: *mut f32, vec: Self::Vector) {
        unsafe { vst1q_f32(ptr, vec) }
    }
    
    #[inline(always)]
    fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { vaddq_f32(a, b) }
    }
    
    #[inline(always)]
    fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { vmulq_f32(a, b) }
    }
    
    #[inline(always)]
    fn fma(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        unsafe { vfmaq_f32(c, a, b) }
    }
    
    #[inline(always)]
    fn horizontal_sum(vec: Self::Vector) -> f32 {
        unsafe {
            let pair_sum = vpaddq_f32(vec, vec);
            let quad_sum = vpaddq_f32(pair_sum, pair_sum);
            vgetq_lane_f32(quad_sum, 0)
        }
    }
    
    #[inline(always)]
    fn broadcast(val: f32) -> Self::Vector {
        unsafe { vdupq_n_f32(val) }
    }
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
```

---

## 3. Cache-Aware Algorithms

### 3.1 Cache-Oblivious Matrix Multiplication

```rust
pub struct CacheAwareMatMul {
    cache_info: CacheInfo,
}

impl CacheAwareMatMul {
    pub fn matmul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        // Compute tile sizes based on cache
        let tile_size = self.compute_tile_size(std::mem::size_of::<f32>());
        
        // Tiled matrix multiplication
        self.matmul_tiled(a, b, c, m, k, n, tile_size);
    }
    
    fn compute_tile_size(&self, elem_size: usize) -> usize {
        // Use sqrt of L2 cache size for square tiles
        // Each tile of A and B should fit in L2
        let l2_size = self.cache_info.l2_size;
        let tile_bytes = l2_size / 3; // A tile + B tile + some C working set
        
        let tile_elements = tile_bytes / elem_size;
        let tile_dim = (tile_elements as f64).sqrt() as usize;
        
        // Round down to multiple of SIMD width for alignment
        (tile_dim / 16) * 16
    }
    
    fn matmul_tiled(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        tile_size: usize,
    ) {
        for i0 in (0..m).step_by(tile_size) {
            let i_end = (i0 + tile_size).min(m);
            
            for j0 in (0..n).step_by(tile_size) {
                let j_end = (j0 + tile_size).min(n);
                
                for k0 in (0..k).step_by(tile_size) {
                    let k_end = (k0 + tile_size).min(k);
                    
                    // Multiply tile
                    self.matmul_tile(
                        a, b, c,
                        m, k, n,
                        i0, i_end,
                        j0, j_end,
                        k0, k_end,
                    );
                }
            }
        }
    }
    
    fn matmul_tile(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize,
        k_start: usize,
        k_end: usize,
    ) {
        #[cfg(target_feature = "avx2")]
        {
            self.matmul_tile_avx2(a, b, c, m, k, n, i_start, i_end, j_start, j_end, k_start, k_end);
        }
        
        #[cfg(not(target_feature = "avx2"))]
        {
            self.matmul_tile_scalar(a, b, c, m, k, n, i_start, i_end, j_start, j_end, k_start, k_end);
        }
    }
    
    #[cfg(target_feature = "avx2")]
    fn matmul_tile_avx2(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize,
        k_start: usize,
        k_end: usize,
    ) {
        const SIMD_WIDTH: usize = 8;
        
        unsafe {
            for i in i_start..i_end {
                let j_chunks = (j_end - j_start) / SIMD_WIDTH;
                
                for jj in 0..j_chunks {
                    let j = j_start + jj * SIMD_WIDTH;
                    
                    // Load accumulator
                    let mut acc = Avx2F32::load_unaligned(c.as_ptr().add(i * n + j));
                    
                    // Inner loop over k
                    for kk in k_start..k_end {
                        let a_val = a[i * k + kk];
                        let a_vec = Avx2F32::broadcast(a_val);
                        
                        let b_vec = Avx2F32::load_unaligned(b.as_ptr().add(kk * n + j));
                        
                        acc = Avx2F32::fma(a_vec, b_vec, acc);
                    }
                    
                    // Store result
                    Avx2F32::store_unaligned(c.as_mut_ptr().add(i * n + j), acc);
                }
                
                // Handle remainder
                for j in (j_start + j_chunks * SIMD_WIDTH)..j_end {
                    let mut sum = c[i * n + j];
                    for kk in k_start..k_end {
                        sum += a[i * k + kk] * b[kk * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }
    }
}
```

### 3.2 Cache-Friendly Data Layouts

```rust
pub struct BlockedMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
    block_size: usize,
}

impl<T: Copy + Default> BlockedMatrix<T> {
    /// Convert row-major matrix to blocked layout
    pub fn from_row_major(data: &[T], rows: usize, cols: usize, block_size: usize) -> Self {
        let blocks_row = (rows + block_size - 1) / block_size;
        let blocks_col = (cols + block_size - 1) / block_size;
        
        let mut blocked = vec![T::default(); blocks_row * blocks_col * block_size * block_size];
        
        for br in 0..blocks_row {
            for bc in 0..blocks_col {
                let block_idx = br * blocks_col + bc;
                let block_offset = block_idx * block_size * block_size;
                
                for i in 0..block_size {
                    for j in 0..block_size {
                        let global_row = br * block_size + i;
                        let global_col = bc * block_size + j;
                        
                        if global_row < rows && global_col < cols {
                            let src_idx = global_row * cols + global_col;
                            let dst_idx = block_offset + i * block_size + j;
                            blocked[dst_idx] = data[src_idx];
                        }
                    }
                }
            }
        }
        
        Self {
            data: blocked,
            rows,
            cols,
            block_size,
        }
    }
    
    /// Access element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> T {
        let br = row / self.block_size;
        let bc = col / self.block_size;
        let i = row % self.block_size;
        let j = col % self.block_size;
        
        let blocks_col = (self.cols + self.block_size - 1) / self.block_size;
        let block_idx = br * blocks_col + bc;
        let block_offset = block_idx * self.block_size * self.block_size;
        let idx = block_offset + i * self.block_size + j;
        
        self.data[idx]
    }
}
```

---

## 4. Thread Pool Management

### 4.1 Work-Stealing Thread Pool

```rust
pub struct ThreadPool {
    workers: Vec<Worker>,
    task_queues: Vec<Arc<Mutex<VecDeque<Task>>>>,
    shutdown: Arc<AtomicBool>,
}

type Task = Box<dyn FnOnce() + Send + 'static>;

struct Worker {
    id: usize,
    thread: Option<JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let mut workers = Vec::with_capacity(num_threads);
        let mut task_queues = Vec::with_capacity(num_threads);
        let shutdown = Arc::new(AtomicBool::new(false));
        
        // Create task queue for each worker
        for _ in 0..num_threads {
            task_queues.push(Arc::new(Mutex::new(VecDeque::new())));
        }
        
        // Spawn workers
        for id in 0..num_threads {
            let worker = Worker::new(
                id,
                task_queues.clone(),
                Arc::clone(&shutdown),
            );
            workers.push(worker);
        }
        
        Self {
            workers,
            task_queues,
            shutdown,
        }
    }
    
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        // Round-robin assignment
        let worker_id = rand::random::<usize>() % self.workers.len();
        let queue = &self.task_queues[worker_id];
        
        let mut q = queue.lock().unwrap();
        q.push_back(Box::new(f));
    }
    
    pub fn parallel_for<F>(&self, start: usize, end: usize, chunk_size: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        let f = Arc::new(f);
        let counter = Arc::new(AtomicUsize::new(0));
        let num_chunks = (end - start + chunk_size - 1) / chunk_size;
        
        for _ in 0..self.workers.len() {
            let f = Arc::clone(&f);
            let counter = Arc::clone(&counter);
            
            self.execute(move || {
                loop {
                    let chunk_idx = counter.fetch_add(1, Ordering::Relaxed);
                    if chunk_idx >= num_chunks {
                        break;
                    }
                    
                    let chunk_start = start + chunk_idx * chunk_size;
                    let chunk_end = (chunk_start + chunk_size).min(end);
                    
                    f(chunk_start, chunk_end);
                }
            });
        }
        
        // Wait for completion
        while counter.load(Ordering::Relaxed) < num_chunks {
            std::thread::yield_now();
        }
    }
}

impl Worker {
    fn new(
        id: usize,
        queues: Vec<Arc<Mutex<VecDeque<Task>>>>,
        shutdown: Arc<AtomicBool>,
    ) -> Self {
        let thread = std::thread::spawn(move || {
            let my_queue = &queues[id];
            
            while !shutdown.load(Ordering::Relaxed) {
                // Try to get task from own queue
                let task = {
                    let mut q = my_queue.lock().unwrap();
                    q.pop_front()
                };
                
                if let Some(task) = task {
                    task();
                } else {
                    // Work stealing: try to steal from other queues
                    let mut stolen = false;
                    
                    for (other_id, other_queue) in queues.iter().enumerate() {
                        if other_id == id {
                            continue;
                        }
                        
                        let task = {
                            let mut q = other_queue.lock().unwrap();
                            q.pop_back() // Steal from back
                        };
                        
                        if let Some(task) = task {
                            task();
                            stolen = true;
                            break;
                        }
                    }
                    
                    if !stolen {
                        std::thread::yield_now();
                    }
                }
            }
        });
        
        Self {
            id,
            thread: Some(thread),
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}
```

### 4.2 Rayon Integration

```rust
pub struct CpuBackend {
    cpu_info: CpuInfo,
    thread_pool: ThreadPool,
}

impl CpuBackend {
    pub fn parallel_tensor_op<F>(&self, tensors: &[&Tensor<f32>], op: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        use rayon::prelude::*;
        
        let n = tensors[0].numel();
        let chunk_size = (n / self.cpu_info.physical_cores).max(1024);
        
        (0..n).into_par_iter()
            .chunks(chunk_size)
            .for_each(|chunk| {
                for i in chunk {
                    op(i);
                }
            });
    }
}
```

---

## 5. Vectorized Operations

### 5.1 Vectorized Dot Product

```rust
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    #[cfg(target_feature = "avx2")]
    {
        dot_product_avx2(a, b)
    }
    
    #[cfg(all(target_arch = "aarch64", not(target_feature = "avx2")))]
    {
        dot_product_neon(a, b)
    }
    
    #[cfg(not(any(target_feature = "avx2", target_arch = "aarch64")))]
    {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

#[cfg(target_feature = "avx2")]
fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 8;
    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    
    unsafe {
        let mut acc = _mm256_s