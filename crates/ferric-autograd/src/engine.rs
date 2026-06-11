use crate::tape::{Tape, TapeOp};
use ferric_core::Tensor;
use std::sync::Arc;

pub struct Engine {
    pub tape: Arc<Tape>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            tape: Arc::new(Tape::new()),
        }
    }

    pub fn backward(&self, _loss: &Tensor) {
        let ops = self.tape.operations.lock().unwrap();
        for op in ops.iter().rev() {
            match op {
                TapeOp::MatMul {
                    lhs: _,
                    rhs: _,
                    output: _,
                } => {
                    // Compute gradients for MatMul
                    // grad_lhs = grad_output * rhs.T
                    // grad_rhs = lhs.T * grad_output
                }
                TapeOp::Add {
                    lhs: _,
                    rhs: _,
                    output: _,
                } => {
                    // Compute gradients for Add
                    // grad_lhs = grad_output
                    // grad_rhs = grad_output
                }
            }
        }
    }
}
