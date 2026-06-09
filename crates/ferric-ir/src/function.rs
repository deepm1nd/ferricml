use crate::block::BasicBlock;
use crate::types::Type;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub inputs: Vec<Type>,
    pub outputs: Vec<Type>,
    pub blocks: Vec<BasicBlock>,
}
