# Section 2: FML Intermediate Representation (IR) Specification

**FerricML Architecture Specification v3.0**  
**Word Count:** 3,800+ words  
**Implementation Priority:** Phase 1 - Critical Foundation

---

## Table of Contents

1. [IR Design Philosophy](#1-ir-design-philosophy)
2. [SSA Form and Basic Blocks](#2-ssa-form-and-basic-blocks)
3. [Type System Integration](#3-type-system-integration)
4. [Operation Definitions](#4-operation-definitions)
5. [Dialect System](#5-dialect-system)
6. [Control Flow](#6-control-flow)
7. [Attributes and Metadata](#7-attributes-and-metadata)
8. [IR Validation](#8-ir-validation)
9. [Serialization Formats](#9-serialization-formats)

---

## 1. IR Design Philosophy

### 1.1 Core Principles

FML IR follows the MLIR philosophy with domain-specific adaptations:

1. **Multi-Level:** Support high-level ML operations and low-level hardware instructions
2. **Extensible:** Dialect system allows custom operations per ML paradigm
3. **SSA-based:** Static Single Assignment for optimization
4. **Type-Safe:** Explicit type checking at IR level
5. **Hardware-Agnostic:** Abstract hardware details until backend lowering

### 1.2 IR Hierarchy

```
High-Level Dialect (fml.*)
    ↓ Progressive Lowering
Mid-Level Dialect (fml.linalg.*, fml.scf.*)
    ↓ Target-Specific Lowering
Low-Level Dialect (llvm.*, nvvm.*, rocdl.*)
    ↓ Code Generation
Machine Code (PTX, SASS, x86-64, etc.)
```

### 1.3 Rust Representation

```rust
/// Top-level IR module
pub struct Module {
    /// Module name
    name: String,
    
    /// Global functions
    functions: Vec<Function>,
    
    /// Global constants
    constants: HashMap<SymbolRef, Constant>,
    
    /// Type definitions
    types: TypeTable,
    
    /// Attributes
    attributes: AttributeDict,
}

/// Function definition
pub struct Function {
    /// Function name
    name: SymbolRef,
    
    /// Function signature
    signature: FunctionType,
    
    /// Basic blocks (empty for declarations)
    blocks: Vec<BasicBlock>,
    
    /// Attributes
    attributes: AttributeDict,
}

/// Basic block in SSA form
pub struct BasicBlock {
    /// Block label
    label: BlockLabel,
    
    /// Arguments (phi nodes)
    arguments: Vec<BlockArgument>,
    
    /// Operations
    operations: Vec<Operation>,
    
    /// Terminator (branch, return, etc.)
    terminator: Terminator,
}

/// Single operation in SSA form
pub struct Operation {
    /// Unique operation ID
    id: OpId,
    
    /// Operation kind (dialect + opcode)
    kind: OpKind,
    
    /// Input operands (SSA values)
    operands: Vec<Value>,
    
    /// Output results (SSA values)
    results: Vec<Value>,
    
    /// Type information
    result_types: Vec<Type>,
    
    /// Attributes
    attributes: AttributeDict,
    
    /// Source location (for debugging)
    location: Location,
}

/// SSA value reference
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value {
    /// Parent operation or block argument
    def: ValueDef,
    
    /// Result index (for multi-result operations)
    result_idx: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueDef {
    OpResult(OpId),
    BlockArgument(BlockLabel, usize),
}
```

---

## 2. SSA Form and Basic Blocks

### 2.1 SSA Invariants

Every value in FML IR obeys SSA properties:

1. **Single Definition:** Each value defined exactly once
2. **Dominance:** Definition dominates all uses
3. **Phi Nodes:** Block arguments represent phi nodes at control flow merge points

### 2.2 Block Argument Example

```rust
// High-level loop in pseudocode:
// x = 0
// for i in 0..10:
//     x = x + i
// return x

// FML IR representation:
func @loop_example() -> i32 {
  %c0 = fml.constant 0 : i32
  %c10 = fml.constant 10 : i32
  br ^loop_header(%c0, %c0 : i32, i32)

^loop_header(%i: i32, %x: i32):  // Block arguments (phi nodes)
  %cmp = fml.cmp_lt %i, %c10 : i32
  cond_br %cmp, ^loop_body, ^exit

^loop_body:
  %next_x = fml.add %x, %i : i32
  %next_i = fml.add %i, 1 : i32
  br ^loop_header(%next_i, %next_x : i32, i32)

^exit:
  return %x : i32
}
```

### 2.3 Use-Def Chain Implementation

```rust
/// Tracks all uses of SSA values
pub struct UseDefChain {
    /// Map from value to its uses
    uses: HashMap<Value, Vec<Use>>,
    
    /// Map from value to its definition
    defs: HashMap<Value, OpId>,
}

pub struct Use {
    /// Operation using the value
    user: OpId,
    
    /// Operand index in user operation
    operand_idx: usize,
}

impl UseDefChain {
    /// Get all operations using a value
    pub fn get_users(&self, value: Value) -> &[Use] {
        self.uses.get(&value).map(|v| v.as_slice()).unwrap_or(&[])
    }
    
    /// Get defining operation for a value
    pub fn get_def(&self, value: Value) -> Option<OpId> {
        self.defs.get(&value).copied()
    }
    
    /// Replace all uses of old_value with new_value
    pub fn replace_all_uses(&mut self, old_value: Value, new_value: Value) {
        if let Some(uses) = self.uses.remove(&old_value) {
            for use_info in uses {
                // Update operand in user operation
                // This requires mutable access to operations
            }
        
        Ok(())
    }
    
    fn validate_dominance(&mut self, function: &Function) -> Result<()> {
        let cfg = CFG::build(function);
        let dom_tree = cfg.dominance_tree();
        
        for block in &function.blocks {
            for op in &block.operations {
                for operand in &op.operands {
                    let def_block = self.get_defining_block(operand, function);
                    
                    // Check that definition dominates use
                    if !dom_tree.dominates(def_block, block.label) {
                        self.errors.push(ValidationError::DominanceViolation {
                            value: *operand,
                            use_location: op.location,
                        });
                    }
                }
            }
        }
        
        Ok(())
    }
}
```

---

## 9. Serialization Formats

### 9.1 Textual Format

FML IR uses a human-readable text format inspired by MLIR:

```
module @example {
  // Function declaration
  func.func @matmul(%arg0: tensor<128x256xf32>, 
                    %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
    // Allocate output tensor
    %0 = tensor.empty() : tensor<128x512xf32>
    
    // Matrix multiplication
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
                       outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    
    // Return result
    return %1 : tensor<128x512xf32>
  }
  
  // Tree prediction function
  func.func @tree_predict(%forest: !fml.forest<100>, 
                          %data: tensor<1000x20xf32>) -> tensor<1000xf32> {
    %result = fml.tree.predict %forest, %data : !fml.forest<100>, tensor<1000x20xf32> -> tensor<1000xf32>
    return %result : tensor<1000xf32>
  }
}
```

### 9.2 Parser Implementation

```rust
pub struct IRParser {
    lexer: Lexer,
    current_token: Token,
}

impl IRParser {
    pub fn parse_module(&mut self) -> Result<Module> {
        self.expect_keyword("module")?;
        
        let name = if self.current_token == Token::At {
            self.advance();
            self.parse_symbol()?
        } else {
            SymbolRef::anonymous()
        };
        
        self.expect(Token::LBrace)?;
        
        let mut functions = Vec::new();
        while self.current_token != Token::RBrace {
            functions.push(self.parse_function()?);
        }
        
        self.expect(Token::RBrace)?;
        
        Ok(Module {
            name: name.to_string(),
            functions,
            constants: HashMap::new(),
            types: TypeTable::new(),
            attributes: AttributeDict::new(),
        })
    }
    
    pub fn parse_function(&mut self) -> Result<Function> {
        self.expect_keyword("func")?;
        self.expect(Token::Dot)?;
        self.expect_keyword("func")?;
        
        // Parse name
        self.expect(Token::At)?;
        let name = self.parse_symbol()?;
        
        // Parse signature
        self.expect(Token::LParen)?;
        let inputs = self.parse_argument_list()?;
        self.expect(Token::RParen)?;
        
        let outputs = if self.current_token == Token::Arrow {
            self.advance();
            self.parse_type_list()?
        } else {
            vec![]
        };
        
        let signature = FunctionType { inputs, outputs };
        
        // Parse body
        self.expect(Token::LBrace)?;
        let blocks = self.parse_block_list()?;
        self.expect(Token::RBrace)?;
        
        Ok(Function {
            name,
            signature,
            blocks,
            attributes: AttributeDict::new(),
        })
    }
    
    pub fn parse_operation(&mut self) -> Result<Operation> {
        // Parse result names
        let results = if self.peek_is_percent() {
            self.parse_value_list()?
        } else {
            vec![]
        };
        
        if !results.is_empty() {
            self.expect(Token::Equal)?;
        }
        
        // Parse operation name (dialect.opname)
        let op_name = self.parse_operation_name()?;
        
        // Parse operands
        let operands = if self.current_token == Token::LParen {
            self.advance();
            let ops = self.parse_value_list()?;
            self.expect(Token::RParen)?;
            ops
        } else {
            self.parse_value_list()?
        };
        
        // Parse attributes
        let attributes = if self.current_token == Token::LBrace {
            self.parse_attribute_dict()?
        } else {
            AttributeDict::new()
        };
        
        // Parse type signature
        self.expect(Token::Colon)?;
        let operand_types = self.parse_type_list()?;
        
        let result_types = if self.current_token == Token::Arrow {
            self.advance();
            self.parse_type_list()?
        } else {
            vec![]
        };
        
        Ok(Operation {
            id: OpId::new(),
            kind: self.resolve_op_name(&op_name)?,
            operands,
            results,
            result_types,
            attributes,
            location: self.current_location(),
        })
    }
}
```

### 9.3 Binary Format

For efficient serialization, use FlatBuffers:

```rust
// FlatBuffers schema (saved as ir.fbs)
/*
namespace FML.IR;

table Module {
  name: string;
  functions: [Function];
}

table Function {
  name: string;
  signature: FunctionType;
  blocks: [BasicBlock];
}

table BasicBlock {
  label: uint;
  arguments: [BlockArgument];
  operations: [Operation];
  terminator: Terminator;
}

table Operation {
  opcode: uint;
  operands: [uint];  // Value IDs
  result_types: [Type];
  attributes: [Attribute];
}

union Type {
  IntegerType,
  FloatType,
  TensorType,
}

table TensorType {
  shape: [long];
  element_type: Type;
}
*/

// Serialization implementation
pub struct IRSerializer;

impl IRSerializer {
    pub fn serialize_module(module: &Module) -> Vec<u8> {
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        
        // Serialize functions
        let functions: Vec<_> = module.functions.iter()
            .map(|f| Self::serialize_function(f, &mut builder))
            .collect();
        
        let functions_vec = builder.create_vector(&functions);
        
        // Create module
        let name = builder.create_string(&module.name);
        let module_offset = fml_ir::Module::create(&mut builder, &fml_ir::ModuleArgs {
            name: Some(name),
            functions: Some(functions_vec),
        });
        
        builder.finish(module_offset, None);
        builder.finished_data().to_vec()
    }
    
    pub fn deserialize_module(data: &[u8]) -> Result<Module> {
        let module_fb = fml_ir::root_as_module(data)?;
        
        let name = module_fb.name().unwrap_or("").to_string();
        
        let functions = module_fb.functions()
            .map(|funcs| {
                funcs.iter()
                    .map(|f| Self::deserialize_function(&f))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        
        Ok(Module {
            name,
            functions,
            constants: HashMap::new(),
            types: TypeTable::new(),
            attributes: AttributeDict::new(),
        })
    }
}
```

---

## 10. IR Builder API

### 10.1 Builder Pattern

```rust
pub struct IRBuilder {
    /// Current insertion point
    insertion_point: InsertionPoint,
    
    /// Symbol table for name resolution
    symbol_table: SymbolTable,
    
    /// Current module being built
    module: Module,
}

pub enum InsertionPoint {
    BlockEnd(BlockLabel),
    BeforeOp(OpId),
    AfterOp(OpId),
}

impl IRBuilder {
    pub fn new() -> Self {
        Self {
            insertion_point: InsertionPoint::BlockEnd(BlockLabel::entry()),
            symbol_table: SymbolTable::new(),
            module: Module::new(""),
        }
    }
    
    /// Create a new basic block
    pub fn create_block(&mut self) -> BlockLabel {
        let label = BlockLabel::new();
        let block = BasicBlock {
            label,
            arguments: vec![],
            operations: vec![],
            terminator: Terminator::Unreachable,
        };
        
        self.get_current_function_mut().blocks.push(block);
        label
    }
    
    /// Set insertion point
    pub fn set_insertion_point(&mut self, point: InsertionPoint) {
        self.insertion_point = point;
    }
    
    /// Create constant operation
    pub fn create_constant(&mut self, value: Attribute, ty: Type) -> Value {
        let op = Operation {
            id: OpId::new(),
            kind: OpKind::Constant,
            operands: vec![],
            results: vec![Value::new_op_result(OpId::new(), 0)],
            result_types: vec![ty],
            attributes: {
                let mut attrs = AttributeDict::new();
                attrs.insert("value".to_string(), value);
                attrs
            },
            location: Location::unknown(),
        };
        
        let result = op.results[0];
        self.insert_op(op);
        result
    }
    
    /// Create binary operation
    pub fn create_binary_op(
        &mut self,
        kind: BinaryOpKind,
        lhs: Value,
        rhs: Value,
    ) -> Value {
        let result_type = lhs.get_type(); // Assume same type
        
        let op = Operation {
            id: OpId::new(),
            kind: OpKind::Binary(kind),
            operands: vec![lhs, rhs],
            results: vec![Value::new_op_result(OpId::new(), 0)],
            result_types: vec![result_type],
            attributes: AttributeDict::new(),
            location: Location::unknown(),
        };
        
        let result = op.results[0];
        self.insert_op(op);
        result
    }
    
    /// Create matrix multiplication
    pub fn create_matmul(&mut self, lhs: Value, rhs: Value) -> Value {
        // Infer result type from operands
        let lhs_type = lhs.get_type();
        let rhs_type = rhs.get_type();
        
        let result_type = match (lhs_type, rhs_type) {
            (Type::Tensor(l), Type::Tensor(r)) => {
                // Extract dimensions
                let m = l.shape[l.shape.len() - 2];
                let n = r.shape[r.shape.len() - 1];
                
                Type::Tensor(TensorType {
                    shape: vec![m, n],
                    element_type: l.element_type.clone(),
                    encoding: TensorEncoding::Dense,
                })
            }
            _ => panic!("Invalid types for matmul"),
        };
        
        let op = Operation {
            id: OpId::new(),
            kind: OpKind::Tensor(TensorOp::MatMul),
            operands: vec![lhs, rhs],
            results: vec![Value::new_op_result(OpId::new(), 0)],
            result_types: vec![result_type],
            attributes: AttributeDict::new(),
            location: Location::unknown(),
        };
        
        let result = op.results[0];
        self.insert_op(op);
        result
    }
    
    /// Create for loop
    pub fn create_for(
        &mut self,
        lower: Value,
        upper: Value,
        step: Value,
        init_args: Vec<Value>,
        body_builder: impl FnOnce(&mut IRBuilder, Value, Vec<Value>) -> Vec<Value>,
    ) -> Vec<Value> {
        // Create region for loop body
        let body_block = self.create_block();
        
        // Save current insertion point
        let saved_ip = self.insertion_point;
        self.set_insertion_point(InsertionPoint::BlockEnd(body_block));
        
        // Create block arguments for IV and iter args
        let iv = self.create_block_argument(Type::Integer(IntegerType::i64()));
        let iter_args: Vec<_> = init_args.iter()
            .map(|v| self.create_block_argument(v.get_type()))
            .collect();
        
        // Build body
        let yield_values = body_builder(self, iv, iter_args);
        
        // Create yield terminator
        self.create_yield(yield_values);
        
        // Restore insertion point
        self.set_insertion_point(saved_ip);
        
        // Create for operation
        let op = Operation {
            id: OpId::new(),
            kind: OpKind::SCF(SCFOp::For),
            operands: {
                let mut ops = vec![lower, upper, step];
                ops.extend(init_args.clone());
                ops
            },
            results: init_args.iter().map(|_| Value::new_op_result(OpId::new(), 0)).collect(),
            result_types: init_args.iter().map(|v| v.get_type()).collect(),
            attributes: AttributeDict::new(),
            location: Location::unknown(),
        };
        
        let results = op.results.clone();
        self.insert_op(op);
        results
    }
    
    fn insert_op(&mut self, op: Operation) {
        match self.insertion_point {
            InsertionPoint::BlockEnd(label) => {
                let block = self.get_block_mut(label);
                block.operations.push(op);
            }
            InsertionPoint::BeforeOp(op_id) => {
                let block = self.find_block_containing_op_mut(op_id);
                let idx = block.operations.iter().position(|o| o.id == op_id).unwrap();
                block.operations.insert(idx, op);
            }
            InsertionPoint::AfterOp(op_id) => {
                let block = self.find_block_containing_op_mut(op_id);
                let idx = block.operations.iter().position(|o| o.id == op_id).unwrap();
                block.operations.insert(idx + 1, op);
            }
        }
    }
}
```

### 10.2 Example Usage

```rust
fn build_simple_matmul_ir() -> Module {
    let mut builder = IRBuilder::new();
    
    // Create function
    let func = builder.create_function(
        "matmul",
        FunctionType {
            inputs: vec![
                Type::Tensor(TensorType {
                    shape: vec![Some(128), Some(256)],
                    element_type: Box::new(Type::Float(FloatType { kind: FloatKind::F32 })),
                    encoding: TensorEncoding::Dense,
                }),
                Type::Tensor(TensorType {
                    shape: vec![Some(256), Some(512)],
                    element_type: Box::new(Type::Float(FloatType { kind: FloatKind::F32 })),
                    encoding: TensorEncoding::Dense,
                }),
            ],
            outputs: vec![
                Type::Tensor(TensorType {
                    shape: vec![Some(128), Some(512)],
                    element_type: Box::new(Type::Float(FloatType { kind: FloatKind::F32 })),
                    encoding: TensorEncoding::Dense,
                }),
            ],
        },
    );
    
    // Create entry block
    let entry = builder.create_block();
    builder.set_insertion_point(InsertionPoint::BlockEnd(entry));
    
    // Get function arguments
    let arg0 = builder.get_function_argument(0);
    let arg1 = builder.get_function_argument(1);
    
    // Create matmul
    let result = builder.create_matmul(arg0, arg1);
    
    // Return result
    builder.create_return(vec![result]);
    
    builder.finish_module()
}
```

---

## Summary

This section specified the FML Intermediate Representation in detail:

**Key Components:**
1. **SSA Form:** Single static assignment for optimization
2. **Type System:** Comprehensive type hierarchy with tensors, functions, and custom types
3. **Operations:** Defined operations for tensors, trees, and probabilistic models
4. **Dialects:** Extensible dialect system for domain-specific operations
5. **Control Flow:** Structured control flow with regions and terminators
6. **Validation:** Complete IR validation rules
7. **Serialization:** Text and binary formats
8. **Builder API:** Ergonomic IR construction

**Design Decisions:**
- SSA form enables powerful optimizations
- Multi-level IR supports progressive lowering
- Type system ensures correctness at IR level
- Dialects provide extensibility without core modifications
- Affine maps express complex memory layouts

**Integration Points:**
- Section 1's type system maps directly to IR types
- Section 3 will lower IR to CUDA kernels
- Section 6 will implement optimization passes on this IR

**Implementation Priority:**
1. Core IR data structures (Operation, BasicBlock, Function)
2. Standard dialects (FML, Linalg, SCF)
3. Type system and validation
4. Builder API for ergonomic construction
5. Parser and serializer

**Performance Notes:**
- Use `SmallVec` for common cases (≤4 dims, ≤4 operands)
- Intern symbols and types to reduce memory
- Use `Arc` for shared immutable data
- Cache type queries to avoid recomputation    self.uses.insert(new_value, uses);
        }
    }
}
```

---

## 3. Type System Integration

### 3.1 IR Type Hierarchy

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// Void type (no value)
    Void,
    
    /// Integer types
    Integer(IntegerType),
    
    /// Floating point types
    Float(FloatType),
    
    /// Tensor types
    Tensor(TensorType),
    
    /// Function types
    Function(FunctionType),
    
    /// Memory reference types
    MemRef(MemRefType),
    
    /// Tree types (for decision trees)
    Tree(TreeType),
    
    /// Graph types (for probabilistic models)
    Graph(GraphType),
    
    /// Custom dialect types
    Custom(CustomType),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntegerType {
    /// Bit width
    width: u32,
    
    /// Signedness
    signedness: Signedness,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Signedness {
    Signed,
    Unsigned,
    Signless, // For operations where sign doesn't matter
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FloatType {
    pub kind: FloatKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FloatKind {
    F16,
    BF16,
    F32,
    F64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorType {
    /// Shape (None for dynamic dimensions)
    shape: Vec<Option<usize>>,
    
    /// Element type
    element_type: Box<Type>,
    
    /// Encoding (dense, sparse, etc.)
    encoding: TensorEncoding,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TensorEncoding {
    Dense,
    SparseCOO,
    SparseCSR,
    Blocked { block_size: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionType {
    /// Input types
    inputs: Vec<Type>,
    
    /// Output types
    outputs: Vec<Type>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemRefType {
    /// Shape
    shape: Vec<Option<usize>>,
    
    /// Element type
    element_type: Box<Type>,
    
    /// Memory space (0=default, 1=shared, 3=constant for CUDA)
    memory_space: u32,
    
    /// Layout (optional affine map)
    layout: Option<AffineMap>,
}
```

### 3.2 Type Checking

```rust
pub trait TypeChecker {
    /// Verify operation has correct types
    fn verify_op(&self, op: &Operation) -> Result<(), TypeError>;
}

pub struct StandardTypeChecker;

impl TypeChecker for StandardTypeChecker {
    fn verify_op(&self, op: &Operation) -> Result<(), TypeError> {
        match &op.kind {
            OpKind::Tensor(TensorOp::MatMul) => {
                // %result = fml.matmul %lhs, %rhs : tensor<MxK>, tensor<KxN> -> tensor<MxN>
                if op.operands.len() != 2 {
                    return Err(TypeError::WrongArity);
                }
                
                let lhs_type = op.operands[0].get_type();
                let rhs_type = op.operands[1].get_type();
                
                let (lhs_tensor, rhs_tensor) = match (lhs_type, rhs_type) {
                    (Type::Tensor(l), Type::Tensor(r)) => (l, r),
                    _ => return Err(TypeError::ExpectedTensor),
                };
                
                // Check dimensions
                if lhs_tensor.shape.len() < 2 || rhs_tensor.shape.len() < 2 {
                    return Err(TypeError::InvalidMatMulShape);
                }
                
                // Check inner dimensions match (K)
                let lhs_k = lhs_tensor.shape[lhs_tensor.shape.len() - 1];
                let rhs_k = rhs_tensor.shape[rhs_tensor.shape.len() - 2];
                
                if lhs_k != rhs_k {
                    return Err(TypeError::DimensionMismatch);
                }
                
                // Verify result type
                let expected_result = self.infer_matmul_type(lhs_tensor, rhs_tensor);
                if op.result_types[0] != expected_result {
                    return Err(TypeError::IncorrectResultType);
                }
                
                Ok(())
            }
            _ => {
                // Check other operations
                unimplemented!()
            }
        }
    }
}
```

---

## 4. Operation Definitions

### 4.1 Tensor Operations

```rust
pub enum TensorOp {
    /// Matrix multiplication
    /// %result = fml.matmul %lhs, %rhs : tensor<MxK>, tensor<KxN> -> tensor<MxN>
    MatMul,
    
    /// Convolution
    /// %result = fml.conv2d %input, %kernel {stride, padding, dilation}
    Conv2d,
    
    /// Element-wise operations
    /// %result = fml.add %lhs, %rhs : tensor<Shape>
    Add,
    Sub,
    Mul,
    Div,
    
    /// Reduction operations
    /// %result = fml.reduce_sum %input {axes} : tensor<...> -> tensor<...>
    ReduceSum,
    ReduceMax,
    ReduceMean,
    
    /// Shape operations
    /// %result = fml.reshape %input, %shape : tensor<...> -> tensor<...>
    Reshape,
    Transpose,
    Slice,
    Concat,
    
    /// Activation functions
    /// %result = fml.relu %input : tensor<Shape>
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    
    /// Normalization
    /// %result = fml.batch_norm %input, %scale, %bias, %mean, %var
    BatchNorm,
    LayerNorm,
}
```

### 4.2 Operation Attributes

```rust
/// Convolution attributes
pub struct Conv2dAttrs {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Operation {
    /// Create conv2d operation
    pub fn conv2d(
        input: Value,
        kernel: Value,
        attrs: Conv2dAttrs,
        result_type: TensorType,
    ) -> Self {
        let mut attributes = AttributeDict::new();
        attributes.insert("stride", Attribute::IntArray(vec![attrs.stride.0 as i64, attrs.stride.1 as i64]));
        attributes.insert("padding", Attribute::IntArray(vec![attrs.padding.0 as i64, attrs.padding.1 as i64]));
        attributes.insert("dilation", Attribute::IntArray(vec![attrs.dilation.0 as i64, attrs.dilation.1 as i64]));
        attributes.insert("groups", Attribute::Int(attrs.groups as i64));
        
        Self {
            id: OpId::new(),
            kind: OpKind::Tensor(TensorOp::Conv2d),
            operands: vec![input, kernel],
            results: vec![Value::new_op_result(OpId::new(), 0)],
            result_types: vec![Type::Tensor(result_type)],
            attributes,
            location: Location::unknown(),
        }
    }
}
```

### 4.3 Tree Operations

```rust
pub enum TreeOp {
    /// Split data at a node
    /// (%left, %right) = fml.tree.split %data, %feature_idx, %threshold
    Split,
    
    /// Predict with tree ensemble
    /// %result = fml.tree.predict %forest, %data : forest<100>, dataframe -> tensor<N>
    Predict,
    
    /// Build tree from data
    /// %tree = fml.tree.build %data, %labels, %config
    Build,
    
    /// Evaluate split quality
    /// %gain = fml.tree.gain %left_stats, %right_stats, %criterion
    Gain,
}

/// Tree ensemble type
pub struct ForestType {
    pub num_trees: usize,
    pub max_depth: usize,
    pub feature_type: Box<Type>,
}
```

### 4.4 Probabilistic Graph Operations

```rust
pub enum PGMOp {
    /// Create Bayesian network
    /// %network = fml.pgm.network %nodes, %edges, %cpds
    Network,
    
    /// Perform inference
    /// %posterior = fml.pgm.infer %network, %query, %evidence
    Infer,
    
    /// Sample from distribution
    /// %samples = fml.pgm.sample %network, %evidence, %num_samples
    Sample,
    
    /// Learn parameters
    /// %learned = fml.pgm.learn %network, %data
    Learn,
}
```

---

## 5. Dialect System

### 5.1 Dialect Registration

```rust
pub trait Dialect: Send + Sync {
    /// Dialect name (e.g., "fml", "tensor", "linalg")
    fn name(&self) -> &str;
    
    /// Register operations
    fn operations(&self) -> Vec<&dyn OperationTrait>;
    
    /// Register types
    fn types(&self) -> Vec<TypeKind>;
    
    /// Register attributes
    fn attributes(&self) -> Vec<AttributeKind>;
}

pub struct DialectRegistry {
    dialects: HashMap<String, Box<dyn Dialect>>,
}

impl DialectRegistry {
    pub fn register(&mut self, dialect: Box<dyn Dialect>) {
        self.dialects.insert(dialect.name().to_string(), dialect);
    }
    
    pub fn get(&self, name: &str) -> Option<&dyn Dialect> {
        self.dialects.get(name).map(|d| d.as_ref())
    }
}
```

### 5.2 Standard Dialects

```rust
/// FML high-level dialect
pub struct FMLDialect;

impl Dialect for FMLDialect {
    fn name(&self) -> &str { "fml" }
    
    fn operations(&self) -> Vec<&dyn OperationTrait> {
        vec![
            &MatMulOp,
            &Conv2dOp,
            &AddOp,
            // ... all tensor ops
        ]
    }
}

/// Linalg dialect (structured operations)
pub struct LinalgDialect;

impl Dialect for LinalgDialect {
    fn name(&self) -> &str { "linalg" }
    
    fn operations(&self) -> Vec<&dyn OperationTrait> {
        vec![
            &LinalgMatMulOp,    // Generic matmul with indexing maps
            &LinalgConvOp,      // Generic convolution
            &LinalgGenericOp,   // Fully generic operation
        ]
    }
}

/// SCF dialect (structured control flow)
pub struct SCFDialect;

impl Dialect for SCFDialect {
    fn name(&self) -> &str { "scf" }
    
    fn operations(&self) -> Vec<&dyn OperationTrait> {
        vec![
            &SCFForOp,          // Affine for loops
            &SCFWhileOp,        // While loops
            &SCFIfOp,           // Conditional
            &SCFParallelOp,     // Parallel loops
        ]
    }
}
```

### 5.3 Dialect Lowering

```rust
pub trait DialectLowering {
    /// Source dialect
    fn source(&self) -> &str;
    
    /// Target dialect
    fn target(&self) -> &str;
    
    /// Lower operation from source to target
    fn lower_op(&self, op: &Operation, builder: &mut IRBuilder) -> Result<()>;
}

/// Example: Lower fml.matmul to linalg.matmul
pub struct MatMulLowering;

impl DialectLowering for MatMulLowering {
    fn source(&self) -> &str { "fml" }
    fn target(&self) -> &str { "linalg" }
    
    fn lower_op(&self, op: &Operation, builder: &mut IRBuilder) -> Result<()> {
        if !matches!(op.kind, OpKind::Tensor(TensorOp::MatMul)) {
            return Ok(()); // Not our operation
        }
        
        // fml.matmul %A, %B : tensor<MxK>, tensor<KxN> -> tensor<MxN>
        // becomes:
        // linalg.matmul ins(%A, %B : tensor<MxK>, tensor<KxN>)
        //               outs(%C : tensor<MxN>)
        
        let lhs = op.operands[0];
        let rhs = op.operands[1];
        let result_type = &op.result_types[0];
        
        // Allocate output buffer
        let output = builder.create_alloc(result_type)?;
        
        // Create linalg.matmul
        let linalg_op = builder.create_op(
            OpKind::Linalg(LinalgOp::MatMul),
            vec![lhs, rhs, output],
            vec![result_type.clone()],
        )?;
        
        builder.replace_op(op.id, linalg_op);
        
        Ok(())
    }
}
```

---

## 6. Control Flow

### 6.1 Terminators

```rust
pub enum Terminator {
    /// Unconditional branch
    /// br ^block(%arg1, %arg2 : type1, type2)
    Branch {
        target: BlockLabel,
        arguments: Vec<Value>,
    },
    
    /// Conditional branch
    /// cond_br %cond, ^true_block, ^false_block
    CondBranch {
        condition: Value,
        true_target: BlockLabel,
        true_args: Vec<Value>,
        false_target: BlockLabel,
        false_args: Vec<Value>,
    },
    
    /// Return from function
    /// return %value : type
    Return {
        values: Vec<Value>,
    },
    
    /// Unreachable code
    /// unreachable
    Unreachable,
}
```

### 6.2 Structured Control Flow

```rust
/// For loop operation (scf.for)
pub struct ForOp {
    /// Lower bound
    pub lower: Value,
    
    /// Upper bound
    pub upper: Value,
    
    /// Step
    pub step: Value,
    
    /// Initial values for loop-carried variables
    pub init_args: Vec<Value>,
    
    /// Loop body (region with block)
    pub body: Region,
}

/// Region containing basic blocks
pub struct Region {
    /// Blocks in this region
    pub blocks: Vec<BasicBlock>,
}

impl Operation {
    /// Create for loop
    pub fn for_loop(
        lower: Value,
        upper: Value,
        step: Value,
        init_args: Vec<Value>,
        body_builder: impl FnOnce(&mut IRBuilder, Value, Vec<Value>) -> Vec<Value>,
    ) -> Self {
        let mut builder = IRBuilder::new();
        
        // Create induction variable and loop-carried arguments
        let iv = builder.create_block_arg(Type::Integer(IntegerType::i64()));
        let iter_args = init_args.iter()
            .map(|v| builder.create_block_arg(v.get_type()))
            .collect::<Vec<_>>();
        
        // Build body
        let result_values = body_builder(&mut builder, iv, iter_args);
        
        // Create yield terminator
        builder.create_yield(result_values);
        
        Self {
            id: OpId::new(),
            kind: OpKind::SCF(SCFOp::For),
            operands: vec![lower, upper, step],
            results: init_args.iter().map(|v| Value::new_op_result(OpId::new(), 0)).collect(),
            result_types: init_args.iter().map(|v| v.get_type()).collect(),
            attributes: AttributeDict::new(),
            location: Location::unknown(),
        }
    }
}
```

### 6.3 Control Flow Graph

```rust
pub struct CFG {
    /// Map from block to its predecessors
    predecessors: HashMap<BlockLabel, Vec<BlockLabel>>,
    
    /// Map from block to its successors
    successors: HashMap<BlockLabel, Vec<BlockLabel>>,
}

impl CFG {
    pub fn build(function: &Function) -> Self {
        let mut cfg = CFG {
            predecessors: HashMap::new(),
            successors: HashMap::new(),
        };
        
        for block in &function.blocks {
            let succs = match &block.terminator {
                Terminator::Branch { target, .. } => vec![*target],
                Terminator::CondBranch { true_target, false_target, .. } => {
                    vec![*true_target, *false_target]
                }
                Terminator::Return { .. } | Terminator::Unreachable => vec![],
            };
            
            cfg.successors.insert(block.label, succs.clone());
            
            for succ in succs {
                cfg.predecessors.entry(succ)
                    .or_insert_with(Vec::new)
                    .push(block.label);
            }
        }
        
        cfg
    }
    
    pub fn dominance_tree(&self) -> DominanceTree {
        // Compute dominator tree using Lengauer-Tarjan algorithm
        unimplemented!()
    }
}
```

---

## 7. Attributes and Metadata

### 7.1 Attribute Types

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
    /// Integer constant
    Int(i64),
    
    /// Floating point constant
    Float(f64),
    
    /// String
    String(String),
    
    /// Boolean
    Bool(bool),
    
    /// Array of integers
    IntArray(Vec<i64>),
    
    /// Array of floats
    FloatArray(Vec<f64>),
    
    /// Type attribute
    Type(Type),
    
    /// Symbol reference (e.g., function name)
    SymbolRef(SymbolRef),
    
    /// Affine map (for layout transformations)
    AffineMap(AffineMap),
    
    /// Custom attribute
    Custom(Box<dyn CustomAttribute>),
}

pub type AttributeDict = HashMap<String, Attribute>;
```

### 7.2 Affine Maps

Affine maps express layout transformations:

```rust
/// Affine map: (d0, d1, ...) -> (expr0, expr1, ...)
pub struct AffineMap {
    /// Number of dimensions
    pub num_dims: usize,
    
    /// Number of symbols
    pub num_symbols: usize,
    
    /// Result expressions
    pub results: Vec<AffineExpr>,
}

pub enum AffineExpr {
    /// Dimension (e.g., d0, d1)
    Dim(usize),
    
    /// Symbol (e.g., s0, s1)
    Symbol(usize),
    
    /// Constant
    Constant(i64),
    
    /// Addition
    Add(Box<AffineExpr>, Box<AffineExpr>),
    
    /// Multiplication
    Mul(Box<AffineExpr>, Box<AffineExpr>),
    
    /// Floor division
    FloorDiv(Box<AffineExpr>, Box<AffineExpr>),
    
    /// Ceiling division
    CeilDiv(Box<AffineExpr>, Box<AffineExpr>),
    
    /// Modulo
    Mod(Box<AffineExpr>, Box<AffineExpr>),
}

impl AffineMap {
    /// Identity map: (d0, d1, ..., dn) -> (d0, d1, ..., dn)
    pub fn identity(n: usize) -> Self {
        Self {
            num_dims: n,
            num_symbols: 0,
            results: (0..n).map(|i| AffineExpr::Dim(i)).collect(),
        }
    }
    
    /// Transpose map: (d0, d1) -> (d1, d0)
    pub fn transpose_2d() -> Self {
        Self {
            num_dims: 2,
            num_symbols: 0,
            results: vec![AffineExpr::Dim(1), AffineExpr::Dim(0)],
        }
    }
    
    /// Blocked layout: (d0, d1) -> (d0 floordiv B, d1 floordiv B, d0 mod B, d1 mod B)
    pub fn blocked_2d(block_size: i64) -> Self {
        let d0 = AffineExpr::Dim(0);
        let d1 = AffineExpr::Dim(1);
        let b = AffineExpr::Constant(block_size);
        
        Self {
            num_dims: 2,
            num_symbols: 0,
            results: vec![
                AffineExpr::FloorDiv(Box::new(d0.clone()), Box::new(b.clone())),
                AffineExpr::FloorDiv(Box::new(d1.clone()), Box::new(b.clone())),
                AffineExpr::Mod(Box::new(d0), Box::new(b.clone())),
                AffineExpr::Mod(Box::new(d1), Box::new(b)),
            ],
        }
    }
}
```

---

## 8. IR Validation

### 8.1 Well-Formedness Rules

```rust
pub struct IRValidator {
    errors: Vec<ValidationError>,
}

pub enum ValidationError {
    UndefinedValue(Value),
    TypeMismatch { expected: Type, actual: Type },
    DominanceViolation { value: Value, use_location: Location },
    InvalidCFG(String),
    MissingTerminator(BlockLabel),
}

impl IRValidator {
    pub fn validate_function(&mut self, function: &Function) -> Result<(), Vec<ValidationError>> {
        // 1. Check SSA form
        self.validate_ssa(function)?;
        
        // 2. Check type correctness
        self.validate_types(function)?;
        
        // 3. Check CFG well-formedness
        self.validate_cfg(function)?;
        
        // 4. Check dominance
        self.validate_dominance(function)?;
        
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }
    
    fn validate_ssa(&mut self, function: &Function) -> Result<()> {
        let mut defined = HashSet::new();
        
        for block in &function.blocks {
            // Block arguments are defined
            for (idx, _) in block.arguments.iter().enumerate() {
                let value = Value {
                    def: ValueDef::BlockArgument(block.label, idx),
                    result_idx: 0,
                };
                defined.insert(value);
            }
            
            // Check operations
            for op in &block.operations {
                // All operands must be defined
                for operand in &op.operands {
                    if !defined.contains(operand) {
                        self.errors.push(ValidationError::UndefinedValue(*operand));
                    }
                }
                
                // Results are now defined
                for (idx, _) in op.results.iter().enumerate() {
                    let value = Value {
                        def: ValueDef::OpResult(op.id),
                        result_idx: idx,
                    };
                    if !defined.insert(value) {
                        // Value defined twice!
                        self.errors.push(ValidationError::SSAViolation(value));
                    }
                }
            }
        }
        