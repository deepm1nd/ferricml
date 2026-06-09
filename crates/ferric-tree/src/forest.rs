use crate::tree::DecisionTree;
use ferric_core::Tensor;

pub struct RandomForest {
    pub trees: Vec<DecisionTree>,
}

impl RandomForest {
    pub fn new(n_estimators: usize, max_depth: usize) -> Self {
        let trees = (0..n_estimators)
            .map(|_| DecisionTree::new(max_depth))
            .collect();
        Self { trees }
    }

    pub fn fit(&mut self, _x: &Tensor, _y: &Tensor) {
        // Fit trees
    }
}
