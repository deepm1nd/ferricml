use crate::operation::Operation;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BasicBlock {
    pub operations: Vec<Operation>,
}
