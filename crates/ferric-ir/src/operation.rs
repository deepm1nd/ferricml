use crate::attributes::Attribute;
use crate::types::Type;
use crate::value::ValueId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Operation {
    pub name: String,
    pub operands: Vec<ValueId>,
    pub results: Vec<ValueId>,
    pub result_types: Vec<Type>,
    pub attributes: HashMap<String, Attribute>,
}
