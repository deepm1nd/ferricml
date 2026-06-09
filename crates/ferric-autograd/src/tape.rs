use ferric_core::Tensor;
use std::sync::{Arc, Mutex};

pub struct Tape {
    pub operations: Mutex<Vec<TapeOp>>,
}

pub enum TapeOp {
    MatMul {
        lhs: Arc<Tensor>,
        rhs: Arc<Tensor>,
        output: Arc<Tensor>,
    },
    Add {
        lhs: Arc<Tensor>,
        rhs: Arc<Tensor>,
        output: Arc<Tensor>,
    },
    // Add more ops as needed
}

impl Tape {
    pub fn new() -> Self {
        Self {
            operations: Mutex::new(Vec::new()),
        }
    }

    pub fn record(&self, op: TapeOp) {
        self.operations.lock().unwrap().push(op);
    }
}
