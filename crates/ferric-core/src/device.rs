use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Rocm(usize),
    Tpu(usize),
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}
