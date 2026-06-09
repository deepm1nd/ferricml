use crate::tree::DecisionTree;

pub struct GBDT {
    pub trees: Vec<DecisionTree>,
    pub lr: f32,
}

impl GBDT {
    pub fn new(n_estimators: usize, max_depth: usize, lr: f32) -> Self {
        let trees = (0..n_estimators)
            .map(|_| DecisionTree::new(max_depth))
            .collect();
        Self { trees, lr }
    }
}
