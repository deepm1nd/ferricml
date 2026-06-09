use ferric_core::Tensor;

use ferric_core::{Storage, TypeId};

pub fn mse_loss(output: &Tensor, target: &Tensor) -> Tensor {
    let binding_out = output.storage();
    let out_storage = match binding_out.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let binding_target = target.storage();
    let target_storage = match binding_target.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut sum = 0.0f32;
    for i in 0..out_storage.len() {
        let diff = out_storage[i] - target_storage[i];
        sum += diff * diff;
    }
    let loss_val = sum / out_storage.len() as f32;
    Tensor::new(vec![loss_val], vec![1])
}
