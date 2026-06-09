use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Attribute {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}
