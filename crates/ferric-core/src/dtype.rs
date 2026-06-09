use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeId {
    Float16,
    BFloat16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
}

pub trait DType: Copy + Send + Sync + Debug + 'static {
    const TYPE_ID: TypeId;
    const SIZE: usize;

    fn type_id() -> TypeId {
        Self::TYPE_ID
    }
}

impl DType for f32 {
    const TYPE_ID: TypeId = TypeId::Float32;
    const SIZE: usize = 4;
}

impl DType for f64 {
    const TYPE_ID: TypeId = TypeId::Float64;
    const SIZE: usize = 8;
}

impl DType for i32 {
    const TYPE_ID: TypeId = TypeId::Int32;
    const SIZE: usize = 4;
}

impl DType for i64 {
    const TYPE_ID: TypeId = TypeId::Int64;
    const SIZE: usize = 8;
}

impl DType for u8 {
    const TYPE_ID: TypeId = TypeId::UInt8;
    const SIZE: usize = 1;
}

impl DType for bool {
    const TYPE_ID: TypeId = TypeId::Bool;
    const SIZE: usize = 1;
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Float16(pub half::f16);

impl DType for Float16 {
    const TYPE_ID: TypeId = TypeId::Float16;
    const SIZE: usize = 2;
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct BFloat16(pub half::bf16);

impl DType for BFloat16 {
    const TYPE_ID: TypeId = TypeId::BFloat16;
    const SIZE: usize = 2;
}
