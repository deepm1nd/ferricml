    fn is_one(&self, ir: &FmlIR, value: Value) -> bool {
        if let Some(op) = ir.get_defining_op(value) {
            if let OpKind::Constant = op.kind {
                if let Some(Attribute::Float(f)) = op.attributes.get("value") {
                    return *f == 1.0;
                }
            }
        }
        false
    }
    
    fn is_negation_of(&self, ir: &FmlIR, v1: Value, v2: Value) -> bool {
        // Check if v2 = -v1 or v1 = -v2
        if let Some(op) = ir.get_defining_op(v2) {
            if let OpKind::Unary(UnaryOp::Neg) = op.kind {
                return op.operands[0] == v1;
            }
        }
        
        if let Some(op) = ir.get_defining_op(v1) {
            if let OpKind::Unary(UnaryOp::Neg) = op.kind {
                return op.operands[0] == v2;
            }
        }
        
        false
    }
    
    fn create_zero_like(&self, ir: &FmlIR, ty: &Type) -> Result<Value> {
        let mut builder = IRBuilder::new();
        builder.create_constant(Attribute::Float(0.0), ty.clone())
    }
}

impl Pass for AlgebraicSimplification {
    fn name(&self) -> &str {
        "algebraic-simplification"
    }
    
    fn run(&self, ir: &mut FmlIR) -> Result<bool> {
        let mut modified = false;
        
        for op_id in ir.operation_ids() {
            if let Some(replacement) = self.simplify_operation(ir, op_id)? {
                ir.replace_value_with(op_id, 0, replacement);
                modified = true;
            }
        }
        
        Ok(modified)
    }
}
```

### 4.2 Constant Folding

```rust
pub struct ConstantFolding;

impl ConstantFolding {
    fn fold_operation(&self, ir: &FmlIR, op_id: OpId) -> Result<Option<Value>> {
        let op = ir.get_operation(op_id);
        
        // Check if all operands are constants
        let const_operands: Option<Vec<_>> = op.operands.iter()
            .map(|&v| self.get_constant_value(ir, v))
            .collect();
        
        let const_operands = match const_operands {
            Some(vals) => vals,
            None => return Ok(None),  // Not all operands are constants
        };
        
        // Evaluate at compile time
        let result = match &op.kind {
            OpKind::Tensor(TensorOp::Add) => {
                const_operands[0] + const_operands[1]
            }
            OpKind::Tensor(TensorOp::Mul) => {
                const_operands[0] * const_operands[1]
            }
            OpKind::Tensor(TensorOp::Sub) => {
                const_operands[0] - const_operands[1]
            }
            OpKind::Tensor(TensorOp::Div) => {
                const_operands[0] / const_operands[1]
            }
            _ => return Ok(None),
        };
        
        // Create constant with result
        let mut builder = IRBuilder::new();
        let const_value = builder.create_constant(
            Attribute::Float(result),
            op.result_types[0].clone(),
        )?;
        
        Ok(Some(const_value))
    }
    
    fn get_constant_value(&self, ir: &FmlIR, value: Value) -> Option<f64> {
        let op = ir.get_defining_op(value)?;
        
        if let OpKind::Constant = op.kind {
            if let Some(Attribute::Float(f)) = op.attributes.get("value") {
                return Some(*f);
            }
        }
        
        None
    }
}

impl Pass for ConstantFolding {
    fn name(&self) -> &str {
        "constant-folding"
    }
    
    fn run(&self, ir: &mut FmlIR) -> Result<bool> {
        let mut modified = false;
        
        for op_id in ir.operation_ids() {
            if let Some(replacement) = self.fold_operation(ir, op_id)? {
                ir.replace_value_with(op_id, 0, replacement);
                ir.remove_operation(op_id);
                modified = true;
            }
        }
        
        Ok(modified)
    }
}
```

---

## 5. Layout Optimization

### 5.1 Layout Transformation

```rust
pub struct LayoutOptimization {
    target_device: Device,
}

impl LayoutOptimization {
    pub fn new(target_device: Device) -> Self {
        Self { target_device }
    }
    
    fn analyze_layouts(&self, ir: &FmlIR) -> HashMap<Value, TensorLayout> {
        let mut layouts = HashMap::new();
        
        for op_id in ir.operation_ids() {
            let op = ir.get_operation(op_id);
            
            // Determine preferred layout for this operation
            let preferred = self.preferred_layout_for_op(&op.kind);
            
            for &operand in &op.operands {
                layouts.entry(operand)
                    .and_modify(|l| *l = self.resolve_conflict(*l, preferred))
                    .or_insert(preferred);
            }
        }
        
        layouts
    }
    
    fn preferred_layout_for_op(&self, op_kind: &OpKind) -> TensorLayout {
        match (op_kind, &self.target_device) {
            // Conv2d prefers NHWC on CPU for cache locality
            (OpKind::Tensor(TensorOp::Conv2d), Device::Cpu) => TensorLayout::NHWC,
            
            // Conv2d prefers NCHW on GPU for memory coalescing
            (OpKind::Tensor(TensorOp::Conv2d), Device::Cuda(_)) => TensorLayout::NCHW,
            
            // MatMul prefers row-major
            (OpKind::Tensor(TensorOp::MatMul), _) => TensorLayout::RowMajor,
            
            // Default
            _ => TensorLayout::RowMajor,
        }
    }
    
    fn resolve_conflict(&self, layout1: TensorLayout, layout2: TensorLayout) -> TensorLayout {
        if layout1 == layout2 {
            layout1
        } else {
            // Choose based on priority
            // For now, prefer GPU-friendly layouts
            match &self.target_device {
                Device::Cuda(_) | Device::Rocm(_) => TensorLayout::NCHW,
                Device::Cpu => TensorLayout::NHWC,
                _ => layout1,
            }
        }
    }
    
    fn insert_layout_conversions(
        &self,
        ir: &mut FmlIR,
        target_layouts: &HashMap<Value, TensorLayout>,
    ) -> Result<bool> {
        let mut modified = false;
        
        for op_id in ir.operation_ids() {
            let op = ir.get_operation(op_id);
            
            for (idx, &operand) in op.operands.iter().enumerate() {
                let current_layout = self.get_current_layout(ir, operand);
                let target_layout = target_layouts.get(&operand).copied()
                    .unwrap_or(TensorLayout::RowMajor);
                
                if current_layout != target_layout {
                    // Insert layout conversion
                    let converted = self.create_layout_conversion(
                        ir,
                        operand,
                        current_layout,
                        target_layout,
                    )?;
                    
                    ir.replace_operand(op_id, idx, converted);
                    modified = true;
                }
            }
        }
        
        Ok(modified)
    }
    
    fn create_layout_conversion(
        &self,
        ir: &mut FmlIR,
        value: Value,
        from: TensorLayout,
        to: TensorLayout,
    ) -> Result<Value> {
        let mut builder = IRBuilder::new();
        
        let conversion = builder.create_op(
            OpKind::LayoutConversion { from, to },
            vec![value],
            vec![value.get_type()],
        )?;
        
        Ok(Value::new_op_result(conversion, 0))
    }
    
    fn get_current_layout(&self, ir: &FmlIR, value: Value) -> TensorLayout {
        // Determine current layout from defining operation
        if let Some(op) = ir.get_defining_op(value) {
            if let OpKind::LayoutConversion { to, .. } = op.kind {
                return to;
            }
        }
        
        TensorLayout::RowMajor  // Default
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    RowMajor,
    ColumnMajor,
    NCHW,  // Batch, Channel, Height, Width
    NHWC,  // Batch, Height, Width, Channel
    Blocked { block_size: usize },
}

impl Pass for LayoutOptimization {
    fn name(&self) -> &str {
        "layout-optimization"
    }
    
    fn run(&self, ir: &mut FmlIR) -> Result<bool> {
        // Analyze preferred layouts
        let target_layouts = self.analyze_layouts(ir);
        
        // Insert conversions where needed
        self.insert_layout_conversions(ir, &target_layouts)
    }
}
```

---

## 6. Memory Planning

### 6.1 Liveness Analysis

```rust
pub struct MemoryPlanning;

impl MemoryPlanning {
    fn compute_liveness(&self, ir: &FmlIR) -> HashMap<Value, LivenessInterval> {
        let mut liveness = HashMap::new();
        
        // Compute execution order
        let exec_order = ir.topological_sort().unwrap();
        
        for (time, op_id) in exec_order.iter().enumerate() {
            let op = ir.get_operation(*op_id);
            
            // Update liveness for inputs (extend to current time)
            for &operand in &op.operands {
                liveness.entry(operand)
                    .and_modify(|interval: &mut LivenessInterval| interval.end = time)
                    .or_insert(LivenessInterval { start: time, end: time });
            }
            
            // Create liveness for outputs
            for (idx, _) in op.results.iter().enumerate() {
                let result = Value::new_op_result(*op_id, idx);
                liveness.insert(result, LivenessInterval { start: time, end: time });
            }
        }
        
        liveness
    }
    
    fn build_interference_graph(&self, liveness: &HashMap<Value, LivenessInterval>) -> InterferenceGraph {
        let mut graph = InterferenceGraph::new();
        
        for (&v1, interval1) in liveness {
            graph.add_node(v1);
            
            for (&v2, interval2) in liveness {
                if v1 != v2 && interval1.overlaps(interval2) {
                    graph.add_edge(v1, v2);
                }
            }
        }
        
        graph
    }
    
    fn allocate_buffers(&self, graph: &InterferenceGraph, ir: &FmlIR) -> BufferAllocation {
        // Graph coloring for buffer assignment
        let mut allocation = BufferAllocation::new();
        let mut coloring = HashMap::new();
        
        // Sort nodes by degree (most constrained first)
        let mut nodes: Vec<_> = graph.nodes().collect();
        nodes.sort_by_key(|&n| std::cmp::Reverse(graph.degree(n)));
        
        for &node in &nodes {
            // Find first available color (buffer ID)
            let used_colors: HashSet<_> = graph.neighbors(node)
                .filter_map(|neighbor| coloring.get(&neighbor))
                .copied()
                .collect();
            
            let color = (0..)
                .find(|c| !used_colors.contains(c))
                .unwrap();
            
            coloring.insert(node, color);
            
            // Determine buffer size
            let size = self.compute_tensor_size(ir, node);
            allocation.assign(node, color, size);
        }
        
        allocation
    }
    
    fn compute_tensor_size(&self, ir: &FmlIR, value: Value) -> usize {
        let ty = value.get_type();
        
        if let Type::Tensor(tensor_ty) = ty {
            let numel: usize = tensor_ty.shape.iter()
                .map(|&dim| dim.unwrap_or(1))
                .product();
            
            let elem_size = match tensor_ty.element_type.as_ref() {
                Type::Float(FloatType { kind: FloatKind::F32 }) => 4,
                Type::Float(FloatType { kind: FloatKind::F16 }) => 2,
                _ => 4,
            };
            
            numel * elem_size
        } else {
            0
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LivenessInterval {
    start: usize,
    end: usize,
}

impl LivenessInterval {
    fn overlaps(&self, other: &Self) -> bool {
        !(self.end < other.start || other.end < self.start)
    }
}

pub struct InterferenceGraph {
    nodes: HashSet<Value>,
    edges: HashSet<(Value, Value)>,
}

impl InterferenceGraph {
    fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: HashSet::new(),
        }
    }
    
    fn add_node(&mut self, node: Value) {
        self.nodes.insert(node);
    }
    
    fn add_edge(&mut self, v1: Value, v2: Value) {
        let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        self.edges.insert(edge);
    }
    
    fn nodes(&self) -> impl Iterator<Item = &Value> {
        self.nodes.iter()
    }
    
    fn neighbors(&self, node: Value) -> impl Iterator<Item = Value> + '_ {
        self.edges.iter()
            .filter_map(move |&(v1, v2)| {
                if v1 == node { Some(v2) }
                else if v2 == node { Some(v1) }
                else { None }
            })
    }
    
    fn degree(&self, node: Value) -> usize {
        self.neighbors(node).count()
    }
}

pub struct BufferAllocation {
    assignments: HashMap<Value, (usize, usize)>,  // (buffer_id, size)
    buffer_sizes: HashMap<usize, usize>,
}

impl BufferAllocation {
    fn new() -> Self {
        Self {
            assignments: HashMap::new(),
            buffer_sizes: HashMap::new(),
        }
    }
    
    fn assign(&mut self, value: Value, buffer_id: usize, size: usize) {
        self.assignments.insert(value, (buffer_id, size));
        self.buffer_sizes.entry(buffer_id)
            .and_modify(|s| *s = (*s).max(size))
            .or_insert(size);
    }
    
    fn total_memory(&self) -> usize {
        self.buffer_sizes.values().sum()
    }
}

impl Pass for MemoryPlanning {
    fn name(&self) -> &str {
        "memory-planning"
    }
    
    fn run(&self, ir: &mut FmlIR) -> Result<bool> {
        // Compute liveness intervals
        let liveness = self.compute_liveness(ir);
        
        // Build interference graph
        let interference_graph = self.build_interference_graph(&liveness);
        
        // Allocate buffers
        let allocation = self.allocate_buffers(&interference_graph, ir);
        
        // Store allocation in IR metadata
        ir.set_buffer_allocation(allocation);
        
        Ok(true)
    }
}
```

---

## 7. Auto-Tuning

### 7.1 Auto-Tuning Infrastructure

```rust
pub struct AutoTuning {
    target_device: Device,
    tuning_cache: Arc<Mutex<TuningCache>>,
    max_trials: usize,
}

pub struct TuningCache {
    configs: HashMap<OpSignature, KernelConfig>,
}

#[derive(Hash, Eq, PartialEq)]
pub struct OpSignature {
    op_kind: OpKind,
    input_shapes: Vec<Vec<usize>>,
    device: String,
}

#[derive(Clone)]
pub struct KernelConfig {
    pub tile_sizes: Vec<usize>,
    pub block_size: (u32, u32, u32),
    pub unroll_factor: usize,
    pub vectorization_width: usize,
}

impl AutoTuning {
    pub fn new(target_device: Device) -> Self {
        Self {
            target_device,
            tuning_cache: Arc::new(Mutex::new(TuningCache::new())),
            max_trials: 100,
        }
    }
    
    fn tune_operation(&self, ir: &FmlIR, op: &Operation) -> Result<Option<KernelConfig>> {
        let signature = self.create_signature(op);
        
        // Check cache first
        {
            let cache = self.tuning_cache.lock().unwrap();
            if let Some(config) = cache.get(&signature) {
                return Ok(Some(config.clone()));
            }
        }
        
        // Generate search space
        let search_space = self.generate_search_space(op)?;
        
        // Search for best configuration
        let best_config = self.search_best_config(ir, op, &search_space)?;
        
        // Cache result
        {
            let mut cache = self.tuning_cache.lock().unwrap();
            cache.insert(signature, best_config.clone());
        }
        
        Ok(Some(best_config))
    }
    
    fn generate_search_space(&self, op: &Operation) -> Result<Vec<KernelConfig>> {
        let mut configs = Vec::new();
        
        match &op.kind {
            OpKind::Tensor(TensorOp::MatMul) => {
                // Common tile sizes for matmul
                let tile_sizes = vec![
                    vec![16, 16, 16],
                    vec![32, 32, 8],
                    vec![64, 64, 8],
                    vec![128, 128, 8],
                    vec![256, 128, 16],
                ];
                
                for tiles in tile_sizes {
                    configs.push(KernelConfig {
                        tile_sizes: tiles,
                        block_size: (16, 16, 1),
                        unroll_factor: 4,
                        vectorization_width: 4,
                    });
                }
            }
            
            OpKind::Tensor(TensorOp::Conv2d) => {
                // Conv2d configurations
                let configs_list = vec![
                    (16, 16, 1),
                    (16, 16, 4),
                    (32, 32, 1),
                ];
                
                for (tile_h, tile_w, unroll) in configs_list {
                    configs.push(KernelConfig {
                        tile_sizes: vec![tile_h, tile_w],
                        block_size: (16, 16, 1),
                        unroll_factor: unroll,
                        vectorization_width: 4,
                    });
                }
            }
            
            _ => return Ok(vec![]),
        }
        
        Ok(configs)
    }
    
    fn search_best_config(
        &self,
        ir: &FmlIR,
        op: &Operation,
        search_space: &[KernelConfig],
    ) -> Result<KernelConfig> {
        let mut best_config = search_space[0].clone();
        let mut best_time = f64::MAX;
        
        for config in search_space {
            let time = self.benchmark_config(ir, op, config)?;
            
            if time < best_time {
                best_time = time;
                best_config = config.clone();
            }
        }
        
        Ok(best_config)
    }
    
    fn benchmark_config(
        &self,
        ir: &FmlIR,
        op: &Operation,
        config: &KernelConfig,
    ) -> Result<f64> {
        // Compile kernel with config
        let kernel = self.compile_with_config(ir, op, config)?;
        
        // Warm-up
        for _ in 0..5 {
            kernel.execute()?;
        }
        
        // Benchmark
        let mut times = Vec::new();
        for _ in 0..20 {
            let start = Instant::now();
            kernel.execute()?;
            self.target_device.synchronize()?;
            times.push(start.elapsed().as_secs_f64());
        }
        
        // Return median time
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(times[times.len() / 2])
    }
    
    fn compile_with_config(
        &self,
        ir: &FmlIR,
        op: &Operation,
        config: &KernelConfig,
    ) -> Result<CompiledKernel> {
        // Generate kernel code with configuration
        // This would interface with CUDA/ROCm backend
        unimplemented!()
    }
    
    fn create_signature(&self, op: &Operation) -> OpSignature {
        let input_shapes = op.operands.iter()
            .map(|v| {
                if let Type::Tensor(t) = v.get_type() {
                    t.shape.iter().map(|d| d.unwrap_or(0)).collect()
                } else {
                    vec![]
                }
            })
            .collect();
        
        OpSignature {
            op_kind: op.kind.clone(),
            input_shapes,
            device: format!("{:?}", self.target_device),
        }
    }
}

impl Pass for AutoTuning {
    fn name(&self) -> &str {
        "auto-tuning"
    }
    
    fn run(&self, ir: &mut FmlIR) -> Result<bool> {
        let mut modified = false;
        
        for op_id in ir.operation_ids() {
            let op = ir.get_operation(op_id);
            
            if let Some(config) = self.tune_operation(ir, op)? {
                // Attach config to operation as attribute
                let mut op_mut = ir.get_operation_mut(op_id);
                op_mut.attributes.insert(
                    "kernel_config".to_string(),
                    Attribute::Custom(Box::new(config)),
                );
                modified = true;
            }
        }
        
        Ok(modified)
    }
}
```

---

## 8. Cost Modeling

### 8.1 Analytical Cost Model

```rust
pub trait CostModel: Send + Sync {
    fn estimate_cost(&self, op: &Operation, device: &Device) -> f64;
}

pub struct AnalyticalCostModel {
    device_info: DeviceInfo,
}

pub struct DeviceInfo {
    peak_flops: f64,
    memory_bandwidth: f64,  // GB/s
    cache_size: usize,
}

impl AnalyticalCostModel {
    pub fn new(device: &Device) -> Self {
        let device_info = match device {
            Device::Cuda(gpu) => DeviceInfo {
                peak_flops: gpu.peak_flops_sp,
                memory_bandwidth: gpu.memory_bandwidth as f64,
                cache_size: gpu.l2_cache_size,
            }
            Device::Cpu => DeviceInfo {
                peak_flops: 1000.0e9,  // Estimated
                memory_bandwidth: 50.0,
                cache_size: 32 * 1024 * 1024,
            }
            _ => DeviceInfo {
                peak_flops: 1000.0e9,
                memory_bandwidth: 100.0,
                cache_size: 0,
            }
        };
        
        Self { device_info }
    }
}

impl CostModel for AnalyticalCostModel {
    fn estimate_cost(&self, op: &Operation, _device: &Device) -> f64 {
        match &op.kind {
            OpKind::Tensor(TensorOp::MatMul) => {
                self.estimate_matmul_cost(op)
            }
            
            OpKind::Tensor(TensorOp::Conv2d) => {
                self.estimate_conv2d_cost(op)
            }
            
            OpKind::Tensor(TensorOp::Add) => {
                self.estimate_elementwise_cost(op)
            }
            
            _ => 0.0,  // Unknown operation
        }
    }
}

impl AnalyticalCostModel {
    fn estimate_matmul_cost(&self, op: &Operation) -> f64 {
        // Extract dimensions
        let (m, k, n) = self.extract_matmul_dims(op);
        
        // FLOPs: 2*M*N*K (multiply-add)
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        
        // Memory traffic: M*K + K*N + M*N elements
        let memory_traffic = (m * k + k * n + m * n) as f64 * 4.0;  // 4 bytes per f32
        
        // Time = max(compute_time, memory_time)
        let compute_time = flops / self.device_info.peak_flops;
        let memory_time = memory_traffic / (self.device_info.memory_bandwidth * 1e9);
        
        compute_time.max(memory_time)
    }
    
    fn estimate_conv2d_cost(&self, op: &Operation) -> f64 {
        // Extract conv parameters
        let (batch, out_channels, out_h, out_w, kernel_h, kernel_w, in_channels) = 
            self.extract_conv2d_dims(op);
        
        // FLOPs per output: 2*kernel_h*kernel_w*in_channels
        let flops_per_output = 2.0 * (kernel_h * kernel_w * in_channels) as f64;
        let total_outputs = (batch * out_channels * out_h * out_w) as f64;
        let total_flops = flops_per_output * total_outputs;
        
        // Memory traffic (simplified)
        let input_size = (batch * in_channels * out_h * out_w) as f64 * 4.0;
        let weight_size = (out_channels * in_channels * kernel_h * kernel_w) as f64 * 4.0;
        let output_size = (batch * out_channels * out_h * out_w) as f64 * 4.0;
        let memory_traffic = input_size + weight_size + output_size;
        
        let compute_time = total_flops / self.device_info.peak_flops;
        let memory_time = memory_traffic / (self.device_info.memory_bandwidth * 1e9);
        
        compute_time.max(memory_time)
    }
    
    fn estimate_elementwise_cost(&self, op: &Operation) -> f64 {
        let numel = self.get_tensor_numel(&op.operands[0]);
        let memory_traffic = numel as f64 * 4.0 * 3.0;  // Read 2, write 1
        
        memory_traffic / (self.device_info.memory_bandwidth * 1e9)
    }
    
    fn extract_matmul_dims(&self, op: &Operation) -> (usize, usize, usize) {
        // Simplified - extract from types
        (128, 256, 512)  // Placeholder
    }
    
    fn extract_conv2d_dims(&self, op: &Operation) -> (usize, usize, usize, usize, usize, usize, usize) {
        // Simplified
        (1, 64, 56, 56, 3, 3, 64)  // Placeholder
    }
    
    fn get_tensor_numel(&self, value: &Value) -> usize {
        if let Type::Tensor(t) = value.get_type() {
            t.shape.iter().map(|d| d.unwrap_or(1)).product()
        } else {
            0
        }
    }
}
```

---

## Summary

This section detailed the optimization pass infrastructure for FerricML:

**Key Components:**
1. **Pass Manager:** Framework for composable optimization passes
2. **Pattern Matching:** Declarative pattern specification and rewriting
3. **Operator Fusion:** Vertical and horizontal fusion strategies
4. **Algebraic Simplification:** Identity/zero rules and constant folding
5. **Layout Optimization:** Device-specific memory layout selection
6. **Memory Planning:** Liveness analysis and buffer allocation
7. **Auto-Tuning:** Empirical search for optimal configurations
8. **Cost Modeling:** Analytical performance prediction

**Design Decisions:**
- Pass-based architecture for modularity
- Pattern matching for declarative rewrites
- Cost models guide optimization decisions
- Auto-tuning for device-specific tuning
- Verification after each pass

**Performance Impact:**
- Fusion reduces kernel launch overhead (2-5x speedup)
- Layout optimization improves memory bandwidth (1.5-3x)
- Memory planning reduces peak usage (30-50%)
- Auto-tuning finds 10-30% better configs than defaults

**Implementation Priority:**
1. Basic passes (constant folding, DCE, CSE)
2. Operator fusion
3. Memory planning
4. Layout optimization
5. Auto-tuning (advanced)

**Integration Points:**
- All backends benefit from IR optimizations
- Section 3 CUDA uses tuned kernel configs
- Section 5 autograd respects fusion boundaries
- Cost models inform compilation decisions# Section 6: Optimization Pass Infrastructure

**FerricML Architecture Specification v3.0**  
**Word Count:** 3,600+ words  
**Implementation Priority:** Phase 2 - Performance Critical

---

## Table of Contents

1. [Pass Manager Architecture](#1-pass-manager-architecture)
2. [Pattern Matching and Rewriting](#2-pattern-matching-and-rewriting)
3. [Operator Fusion](#3-operator-fusion)
4. [Algebraic Simplification](#4-algebraic-simplification)
5. [Layout Optimization](#5-layout-optimization)
6. [Memory Planning](#6-memory-planning)
7. [Auto-Tuning](#7-auto-tuning)
8. [Cost Modeling](#8-cost-modeling)

---

## 1. Pass Manager Architecture

### 1.1 Pass Interface

```rust
pub trait Pass: Send + Sync {
    /// Pass name for logging and debugging
    fn name(&self) -> &str;
    
    /// Run pass on IR, return true if modified
    fn run(&self, ir: &mut FmlIR) -> Result<bool>;
    
    /// Dependencies (passes that must run before this one)
    fn dependencies(&self) -> Vec<&str> {
        vec![]
    }
    
    /// Whether this pass preserves IR structure
    fn preserves_structure(&self) -> bool {
        false
    }
}

pub struct PassManager {
    /// Registered passes
    passes: Vec<Box<dyn Pass>>,
    
    /// Configuration
    config: PassConfig,
    
    /// Pass statistics
    stats: Arc<Mutex<PassStatistics>>,
}

pub struct PassConfig {
    /// Optimization level
    pub level: OptLevel,
    
    /// Maximum iterations
    pub max_iterations: usize,
    
    /// Target device for optimization
    pub target_device: Device,
    
    /// Enable/disable specific passes
    pub enabled_passes: HashSet<String>,
    
    /// Verification after each pass
    pub verify_ir: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    O0,  // No optimization
    O1,  // Basic optimization
    O2,  // Aggressive optimization
    O3,  // Maximum optimization with auto-tuning
}

pub struct PassStatistics {
    /// Time spent in each pass
    pass_times: HashMap<String, Duration>,
    
    /// Number of times each pass ran
    pass_runs: HashMap<String, usize>,
    
    /// Number of modifications by each pass
    pass_modifications: HashMap<String, usize>,
}

impl PassManager {
    pub fn new(config: PassConfig) -> Self {
        let mut passes: Vec<Box<dyn Pass>> = Vec::new();
        
        // Add passes based on optimization level
        match config.level {
            OptLevel::O0 => {
                // No optimization passes
            }
            OptLevel::O1 => {
                passes.push(Box::new(ConstantFolding));
                passes.push(Box::new(DeadCodeElimination));
                passes.push(Box::new(CommonSubexpressionElimination));
            }
            OptLevel::O2 => {
                passes.push(Box::new(ConstantFolding));
                passes.push(Box::new(AlgebraicSimplification));
                passes.push(Box::new(CommonSubexpressionElimination));
                passes.push(Box::new(OperatorFusion));
                passes.push(Box::new(LayoutOptimization::new(config.target_device.clone())));
                passes.push(Box::new(DeadCodeElimination));
                passes.push(Box::new(MemoryPlanning));
            }
            OptLevel::O3 => {
                passes.push(Box::new(ConstantFolding));
                passes.push(Box::new(AlgebraicSimplification));
                passes.push(Box::new(CommonSubexpressionElimination));
                passes.push(Box::new(OperatorFusion));
                passes.push(Box::new(LoopOptimization));
                passes.push(Box::new(LayoutOptimization::new(config.target_device.clone())));
                passes.push(Box::new(MemoryPlanning));
                passes.push(Box::new(AutoTuning::new(config.target_device.clone())));
                passes.push(Box::new(DeadCodeElimination));
            }
        }
        
        Self {
            passes,
            config,
            stats: Arc::new(Mutex::new(PassStatistics::new())),
        }
    }
    
    pub fn run(&self, ir: &mut FmlIR) -> Result<()> {
        let mut iteration = 0;
        
        loop {
            let mut modified = false;
            
            for pass in &self.passes {
                if !self.is_pass_enabled(pass.name()) {
                    continue;
                }
                
                let start = Instant::now();
                let pass_modified = pass.run(ir)?;
                let elapsed = start.elapsed();
                
                // Update statistics
                {
                    let mut stats = self.stats.lock().unwrap();
                    *stats.pass_times.entry(pass.name().to_string()).or_insert(Duration::ZERO) += elapsed;
                    *stats.pass_runs.entry(pass.name().to_string()).or_insert(0) += 1;
                    if pass_modified {
                        *stats.pass_modifications.entry(pass.name().to_string()).or_insert(0) += 1;
                    }
                }
                
                if pass_modified {
                    modified = true;
                    
                    // Verify IR if configured
                    if self.config.verify_ir {
                        ir.verify()?;
                    }
                }
            }
            
            iteration += 1;
            
            // Stop if no modifications or max iterations reached
            if !modified || iteration >= self.config.max_iterations {
                break;
            }
        }
        
        Ok(())
    }
    
    fn is_pass_enabled(&self, pass_name: &str) -> bool {
        self.config.enabled_passes.is_empty() ||
            self.config.enabled_passes.contains(pass_name)
    }
    
    pub fn print_statistics(&self) {
        let stats = self.stats.lock().unwrap();
        
        println!("Pass Statistics:");
        println!("{:<30} {:>10} {:>10} {:>15}", "Pass", "Runs", "Mods", "Time (ms)");
        println!("{}", "-".repeat(70));
        
        let mut pass_names: Vec<_> = stats.pass_times.keys().collect();
        pass_names.sort();
        
        for name in pass_names {
            let runs = stats.pass_runs.get(name).unwrap_or(&0);
            let mods = stats.pass_modifications.get(name).unwrap_or(&0);
            let time = stats.pass_times.get(name).unwrap().as_millis();
            
            println!("{:<30} {:>10} {:>10} {:>15}", name, runs, mods, time);
        }
    }
}
```

---

## 2. Pattern Matching and Rewriting

### 2.1 Pattern Specification

```rust
pub struct Pattern {
    /// Root operation to match
    root_op: OpKind,
    
    /// Operand patterns
    operands: Vec<PatternNode>,
    
    /// Constraints on matched values
    constraints: Vec<Box<dyn Constraint>>,
}

pub enum PatternNode {
    /// Match any operation
    Any,
    
    /// Match specific operation
    Op(OpKind),
    
    /// Match constant
    Constant,
    
    /// Match parameter
    Parameter,
    
    /// Recursive pattern
    Pattern(Box<Pattern>),
    
    /// Match and bind to name
    Bind(String, Box<PatternNode>),
}

pub trait Constraint: Send + Sync {
    fn check(&self, matched: &MatchedValues) -> bool;
}

pub struct MatchedValues {
    bindings: HashMap<String, Value>,
    operations: HashMap<String, Operation>,
}

impl Pattern {
    pub fn match_at(&self, ir: &FmlIR, op_id: OpId) -> Option<MatchedValues> {
        let op = ir.get_operation(op_id);
        
        if !self.matches_op_kind(&op.kind) {
            return None;
        }
        
        let mut matched = MatchedValues {
            bindings: HashMap::new(),
            operations: HashMap::new(),
        };
        
        // Match operands recursively
        if !self.match_operands(ir, op, &mut matched) {
            return None;
        }
        
        // Check constraints
        if !self.check_constraints(&matched) {
            return None;
        }
        
        Some(matched)
    }
    
    fn matches_op_kind(&self, kind: &OpKind) -> bool {
        match &self.root_op {
            OpKind::Any => true,
            k => k == kind,
        }
    }
    
    fn match_operands(
        &self,
        ir: &FmlIR,
        op: &Operation,
        matched: &mut MatchedValues,
    ) -> bool {
        if op.operands.len() != self.operands.len() {
            return false;
        }
        
        for (operand, pattern_node) in op.operands.iter().zip(&self.operands) {
            if !self.match_node(ir, *operand, pattern_node, matched) {
                return false;
            }
        }
        
        true
    }
    
    fn match_node(
        &self,
        ir: &FmlIR,
        value: Value,
        pattern: &PatternNode,
        matched: &mut MatchedValues,
    ) -> bool {
        match pattern {
            PatternNode::Any => true,
            
            PatternNode::Op(kind) => {
                if let Some(def_op) = ir.get_defining_op(value) {
                    def_op.kind == *kind
                } else {
                    false
                }
            }
            
            PatternNode::Constant => {
                if let Some(def_op) = ir.get_defining_op(value) {
                    matches!(def_op.kind, OpKind::Constant)
                } else {
                    false
                }
            }
            
            PatternNode::Bind(name, inner) => {
                if self.match_node(ir, value, inner, matched) {
                    matched.bindings.insert(name.clone(), value);
                    true
                } else {
                    false
                }
            }
            
            PatternNode::Pattern(nested) => {
                if let Some(def_op_id) = ir.get_defining_op_id(value) {
                    nested.match_at(ir, def_op_id).is_some()
                } else {
                    false
                }
            }
            
            _ => unimplemented!(),
        }
    }
    
    fn check_constraints(&self, matched: &MatchedValues) -> bool {
        self.constraints.iter().all(|c| c.check(matched))
    }
}
```

### 2.2 Rewrite Rules

```rust
pub struct RewriteRule {
    /// Pattern to match
    pattern: Pattern,
    
    /// Replacement function
    replacement: Box<dyn ReplacementFn>,
    
    /// Benefit estimate (for choosing among multiple matches)
    benefit: i32,
}

pub trait ReplacementFn: Send + Sync {
    fn apply(&self, ir: &mut FmlIR, matched: &MatchedValues, builder: &mut IRBuilder) -> Result<Value>;
}

impl RewriteRule {
    pub fn new(
        pattern: Pattern,
        replacement: Box<dyn ReplacementFn>,
        benefit: i32,
    ) -> Self {
        Self {
            pattern,
            replacement,
            benefit,
        }
    }
    
    pub fn try_apply(&self, ir: &mut FmlIR, op_id: OpId) -> Result<Option<Value>> {
        if let Some(matched) = self.pattern.match_at(ir, op_id) {
            let mut builder = IRBuilder::new();
            let replacement = self.replacement.apply(ir, &matched, &mut builder)?;
            Ok(Some(replacement))
        } else {
            Ok(None)
        }
    }
}

/// Pattern rewriting pass
pub struct PatternRewriter {
    rules: Vec<RewriteRule>,
}

impl PatternRewriter {
    pub fn new() -> Self {
        let mut rules = Vec::new();
        
        // Add common rewrite rules
        rules.extend(Self::create_algebraic_rules());
        rules.extend(Self::create_fusion_rules());
        
        Self { rules }
    }
    
    fn create_algebraic_rules() -> Vec<RewriteRule> {
        vec![
            // x + 0 => x
            Self::rule_add_zero(),
            
            // x * 1 => x
            Self::rule_mul_one(),
            
            // x * 0 => 0
            Self::rule_mul_zero(),
            
            // x - x => 0
            Self::rule_sub_self(),
        ]
    }
    
    fn rule_add_zero() -> RewriteRule {
        let pattern = Pattern {
            root_op: OpKind::Tensor(TensorOp::Add),
            operands: vec![
                PatternNode::Bind("x".to_string(), Box::new(PatternNode::Any)),
                PatternNode::Constant,
            ],
            constraints: vec![
                Box::new(IsZeroConstraint { binding: "operand_1".to_string() }),
            ],
        };
        
        let replacement = Box::new(|_ir: &mut FmlIR, matched: &MatchedValues, _builder: &mut IRBuilder| {
            Ok(matched.bindings["x"])
        });
        
        RewriteRule::new(pattern, replacement, 1)
    }
}

struct IsZeroConstraint {
    binding: String,
}

impl Constraint for IsZeroConstraint {
    fn check(&self, matched: &MatchedValues) -> bool {
        // Check if bound value is constant zero
        unimplemented!()
    }
}

impl Pass for PatternRewriter {
    fn name(&self) -> &str {
        "pattern-rewriter"
    }
    
    fn run(&self, ir: &mut FmlIR) -> Result<bool> {
        let mut modified = false;
        
        // Try each rule on each operation
        for op_id in ir.operation_ids() {
            for rule in &self.rules {
                if let Some(replacement) = rule.try_apply(ir, op_id)? {
                    ir.replace_value(op_id, replacement);
                    modified = true;
                    break;  // Only apply one rule per operation
                }
            }
        }
        
        Ok(modified)
    }
}
```

---

## 3. Operator Fusion

### 3.1 Vertical Fusion

Combine producer-consumer operation chains:

```rust
pub struct OperatorFusion {
    target_device: Device,
}

impl OperatorFusion {
    pub fn new(target_device: Device) -> Self {
        Self { target_device }
    }
    
    fn can_fuse(&self, producer: &Operation, consumer: &Operation) -> bool {
        // Check if operations can be fused
        match (&producer.kind, &consumer.kind) {
            // MatMul + Add (linear layer)
            (OpKind::Tensor(TensorOp::MatMul), OpKind::Tensor(TensorOp::Add)) => true,
            
            // Conv + BatchNorm
            (OpKind::Tensor(TensorOp::Conv2d), OpKind::Tensor(TensorOp::BatchNorm)) => true,
            
            // Any + ReLU
            (_, OpKind::Tensor(TensorOp::ReLU)) => true,
            
            // Any + Sigmoid
            (_, OpKind::Tensor(TensorOp::Sigmoid)) => true,
            
            _ => false,
        }
    }
    
    fn fuse_ops(&self, ir: &mut FmlIR, producer_id: OpId, consumer_id: OpId) -> Result<OpId> {
        let producer = ir.get_operation(producer_id);
        let consumer = ir.get_operation(consumer_id);
        
        match (&producer.kind, &consumer.kind) {
            (OpKind::Tensor(TensorOp::MatMul), OpKind::Tensor(TensorOp::Add)) => {
                self.fuse_matmul_add(ir, producer, consumer)
            }
            
            (OpKind::Tensor(TensorOp::Conv2d), OpKind::Tensor(TensorOp::BatchNorm)) => {
                self.fuse_conv_batchnorm(ir, producer, consumer)
            }
            
            (_, OpKind::Tensor(TensorOp::ReLU)) => {
                self.fuse_with_relu(ir, producer, consumer)
            }
            
            _ => Err(Error::CannotFuse),
        }
    }
    
    fn fuse_matmul_add(
        &self,
        ir: &mut FmlIR,
        matmul: &Operation,
        add: &Operation,
    ) -> Result<OpId> {
        // matmul(X, W) + b => linear(X, W, b)
        let x = matmul.operands[0];
        let w = matmul.operands[1];
        let b = add.operands[1];  // Assuming second operand is bias
        
        let mut builder = IRBuilder::new();
        let linear_op = builder.create_op(
            OpKind::Tensor(TensorOp::Linear),
            vec![x, w, b],
            vec![add.result_types[0].clone()],
        )?;
        
        Ok(linear_op)
    }
    
    fn fuse_conv_batchnorm(
        &self,
        ir: &mut FmlIR,
        conv: &Operation,
        bn: &Operation,
    ) -> Result<OpId> {
        // Fuse by folding BatchNorm into Conv weights and bias
        // new_weight = weight * gamma / sqrt(var + eps)
        // new_bias = beta + (bias - mean) * gamma / sqrt(var + eps)
        
        let input = conv.operands[0];
        let weight = conv.operands[1];
        let conv_bias = if conv.operands.len() > 2 {
            Some(conv.operands[2])
        } else {
            None
        };
        
        let gamma = bn.operands[1];
        let beta = bn.operands[2];
        let mean = bn.operands[3];
        let var = bn.operands[4];
        
        let mut builder = IRBuilder::new();
        
        // Compute scale: gamma / sqrt(var + eps)
        let eps = builder.create_constant(Attribute::Float(1e-5), var.get_type());
        let var_eps = builder.create_binary_op(BinaryOpKind::Add, var, eps);
        let std = builder.create_unary_op(UnaryOpKind::Sqrt, var_eps);
        let scale = builder.create_binary_op(BinaryOpKind::Div, gamma, std);
        
        // New weight: weight * scale (broadcast)
        let new_weight = builder.create_binary_op(BinaryOpKind::Mul, weight, scale);
        
        // New bias
        let new_bias = if let Some(bias) = conv_bias {
            // (bias - mean) * scale + beta
            let bias_centered = builder.create_binary_op(BinaryOpKind::Sub, bias, mean);
            let bias_scaled = builder.create_binary_op(BinaryOpKind::Mul, bias_centered, scale);
            builder.create_binary_op(BinaryOpKind::Add, bias_scaled, beta)
        } else {
            // -mean * scale + beta
            let mean_scaled = builder.create_binary_op(BinaryOpKind::Mul, mean, scale);
            let neg_mean = builder.create_unary_op(UnaryOpKind::Neg, mean_scaled);
            builder.create_binary_op(BinaryOpKind::Add, neg_mean, beta)
        };
        
        // Create fused conv
        let fused_conv = builder.create_op(
            OpKind::Tensor(TensorOp::Conv2d),
            vec![input, new_weight, new_bias],
            vec![bn.result_types[0].clone()],
        )?;
        
        Ok(fused_conv)
    }
}

impl Pass for OperatorFusion {
    fn name(&self) -> &str {
        "operator-fusion"
    }
    
    fn run(&self, ir: &mut FmlIR) -> Result<bool> {
        let mut modified = false;
        
        // Find fusion opportunities
        for consumer_id in ir.operation_ids() {
            let consumer = ir.get_operation(consumer_id);
            
            for &operand in &consumer.operands {
                if let Some(producer_id) = ir.get_defining_op_id(operand) {
                    let producer = ir.get_operation(producer_id);
                    
                    // Check if producer has only one use (consumer)
                    let uses = ir.get_uses(operand);
                    if uses.len() == 1 && self.can_fuse(producer, consumer) {
                        let fused_id = self.fuse_ops(ir, producer_id, consumer_id)?;
                        ir.replace_operation(consumer_id, fused_id);
                        ir.remove_operation(producer_id);
                        modified = true;
                        break;
                    }
                }
            }
        }
        
        Ok(modified)
    }
}
```

### 3.2 Horizontal Fusion

Merge independent operations into single kernel:

```rust
pub struct HorizontalFusion {
    max_fused_ops: usize,
}

impl HorizontalFusion {
    pub fn new() -> Self {
        Self {
            max_fused_ops: 8,  // Limit fusion group size
        }
    }
    
    fn find_fusible_groups(&self, ir: &FmlIR) -> Vec<Vec<OpId>> {
        let mut groups = Vec::new();
        let mut visited = HashSet::new();
        
        for op_id in ir.operation_ids() {
            if visited.contains(&op_id) {
                continue;
            }
            
            let op = ir.get_operation(op_id);
            
            // Only fuse element-wise operations
            if !self.is_elementwise(op) {
                continue;
            }
            
            // Find compatible operations
            let mut group = vec![op_id];
            visited.insert(op_id);
            
            for other_id in ir.operation_ids() {
                if visited.contains(&other_id) || group.len() >= self.max_fused_ops {
                    break;
                }
                
                let other = ir.get_operation(other_id);
                
                if self.is_elementwise(other) && self.can_fuse_horizontally(ir, op, other) {
                    group.push(other_id);
                    visited.insert(other_id);
                }
            }
            
            if group.len() > 1 {
                groups.push(group);
            }
        }
        
        groups
    }
    
    fn is_elementwise(&self, op: &Operation) -> bool {
        matches!(op.kind,
            OpKind::Tensor(TensorOp::Add) |
            OpKind::Tensor(TensorOp::Mul) |
            OpKind::Tensor(TensorOp::Sub) |
            OpKind::Tensor(TensorOp::Div) |
            OpKind::Tensor(TensorOp::ReLU) |
            OpKind::Tensor(TensorOp::Sigmoid) |
            OpKind::Tensor(TensorOp::Tanh)
        )
    }
    
    fn can_fuse_horizontally(&self, ir: &FmlIR, op1: &Operation, op2: &Operation) -> bool {
        // Operations must have same shape
        op1.result_types[0] == op2.result_types[0] &&
        // Must not depend on each other
        !self.has_dependency(ir, op1, op2)
    }
    
    fn has_dependency(&self, ir: &FmlIR, op1: &Operation, op2: &Operation) -> bool {
        // Check if op2 depends on op1 or vice versa
        // Simplified - real implementation would do full dataflow analysis
        op1.operands.iter().any(|&v| ir.is_produced_by(v, op2.id)) ||
        op2.operands.iter().any(|&v| ir.is_produced_by(v, op1.id))
    }
    
    fn fuse_group(&self, ir: &mut FmlIR, group: &[OpId]) -> Result<OpId> {
        // Create fused elementwise kernel
        let mut builder = IRBuilder::new();
        
        // Collect all inputs
        let mut inputs = HashSet::new();
        for &op_id in group {
            let op = ir.get_operation(op_id);
            for &operand in &op.operands {
                // Only add if not produced by operation in group
                if !group.iter().any(|&id| ir.is_produced_by(operand, id)) {
                    inputs.insert(operand);
                }
            }
        }
        
        let inputs: Vec<Value> = inputs.into_iter().collect();
        
        // Create fused operation
        let fused = builder.create_op(
            OpKind::FusedElementwise(group.len()),
            inputs,
            group.iter().map(|&id| ir.get_operation(id).result_types[0].clone()).collect(),
        )?;
        
        Ok(fused)
    }
}

impl Pass for HorizontalFusion {
    fn name(&self) -> &str {
        "horizontal-fusion"
    }
    
    fn run(&self, ir: &mut FmlIR) -> Result<bool> {
        let groups = self.find_fusible_groups(ir);
        
        if groups.is_empty() {
            return Ok(false);
        }
        
        for group in groups {
            let fused_id = self.fuse_group(ir, &group)?;
            
            // Replace all operations in group
            for (idx, &op_id) in group.iter().enumerate() {
                ir.replace_result(op_id, 0, Value::new_op_result(fused_id, idx));
            }
        }
        
        Ok(true)
    }
}
```

---

## 4. Algebraic Simplification

### 4.1 Identity and Zero Rules

```rust
pub struct AlgebraicSimplification;

impl AlgebraicSimplification {
    fn simplify_operation(&self, ir: &mut FmlIR, op_id: OpId) -> Result<Option<Value>> {
        let op = ir.get_operation(op_id);
        
        match &op.kind {
            OpKind::Tensor(TensorOp::Add) => self.simplify_add(ir, op),
            OpKind::Tensor(TensorOp::Mul) => self.simplify_mul(ir, op),
            OpKind::Tensor(TensorOp::Sub) => self.simplify_sub(ir, op),
            OpKind::Tensor(TensorOp::Div) => self.simplify_div(ir, op),
            _ => Ok(None),
        }
    }
    
    fn simplify_add(&self, ir: &FmlIR, op: &Operation) -> Result<Option<Value>> {
        let lhs = op.operands[0];
        let rhs = op.operands[1];
        
        // x + 0 => x
        if self.is_zero(ir, rhs) {
            return Ok(Some(lhs));
        }
        
        // 0 + x => x
        if self.is_zero(ir, lhs) {
            return Ok(Some(rhs));
        }
        
        // x + (-x) => 0
        if self.is_negation_of(ir, lhs, rhs) {
            let zero = self.create_zero_like(ir, &op.result_types[0])?;
            return Ok(Some(zero));
        }
        
        Ok(None)
    }
    
    fn simplify_mul(&self, ir: &FmlIR, op: &Operation) -> Result<Option<Value>> {
        let lhs = op.operands[0];
        let rhs = op.operands[1];
        
        // x * 0 => 0
        if self.is_zero(ir, rhs) {
            let zero = self.create_zero_like(ir, &op.result_types[0])?;
            return Ok(Some(zero));
        }
        
        // 0 * x => 0
        if self.is_zero(ir, lhs) {
            let zero = self.create_zero_like(ir, &op.result_types[0])?;
            return Ok(Some(zero));
        }
        
        // x * 1 => x
        if self.is_one(ir, rhs) {
            return Ok(Some(lhs));
        }
        
        // 1 * x => x
        if self.is_one(ir, lhs) {
            return Ok(Some(rhs));
        }
        
        Ok(None)
    }
    
    fn simplify_sub(&self, ir: &FmlIR, op: &Operation) -> Result<Option<Value>> {
        let lhs = op.operands[0];
        let rhs = op.operands[1];
        
        // x - 0 => x
        if self.is_zero(ir, rhs) {
            return Ok(Some(lhs));
        }
        
        // x - x => 0
        if lhs == rhs {
            let zero = self.create_zero_like(ir, &op.result_types[0])?;
            return Ok(Some(zero));
        }
        
        Ok(None)
    }
    
    fn is_zero(&self, ir: &FmlIR, value: Value) -> bool {
        if let Some(op) = ir.get_defining_op(value) {
            if let OpKind::Constant = op.kind {
                if let Some(Attribute::Float(f)) = op.attributes.get("value") {
                    return *f == 0.0;
                }
            }
        }
        false
    }
    
    fn is_one(&self, ir: &FmlIR, value: Value) -> bool {
        if let Some(