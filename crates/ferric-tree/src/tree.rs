use ferric_core::Tensor;

pub struct DecisionTree {
    pub max_depth: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    pub fn fit(&mut self, _x: &Tensor, _y: &Tensor) {
        // Build tree
    }

    pub fn predict(&self, _x: &Tensor) -> Tensor {
        unimplemented!("Tree predict")
    }
}
