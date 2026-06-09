use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    Void,
    Integer(u32),
    Float(u32),
    Tensor(Vec<Option<usize>>, Box<Type>),
    Function(Vec<Type>, Vec<Type>),
}

impl Type {
    pub fn f32() -> Self {
        Type::Float(32)
    }
}
