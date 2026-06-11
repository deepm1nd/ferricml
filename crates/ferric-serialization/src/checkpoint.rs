use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: String,
    pub model_name: String,
    // In a real implementation, we'd serialize tensor data here
}
