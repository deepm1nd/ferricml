use ferric_core::Tensor;

pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn step(&self, _parameters: Vec<&mut Tensor>) {
        // Update parameters
    }
}
