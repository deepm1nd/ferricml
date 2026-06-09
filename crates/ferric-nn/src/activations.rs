use ferric_core::Tensor;

use ferric_core::Storage;

pub fn relu(input: &Tensor) -> Tensor {
    let binding = input.storage();
    let in_storage = match binding.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let mut out_data = vec![0.0f32; in_storage.len()];
    for (i, &val) in in_storage.iter().enumerate() {
        out_data[i] = val.max(0.0);
    }
    Tensor::new(out_data, input.shape().dims().to_vec())
}

pub fn sigmoid(input: &Tensor) -> Tensor {
    let binding = input.storage();
    let in_storage = match binding.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let mut out_data = vec![0.0f32; in_storage.len()];
    for (i, &val) in in_storage.iter().enumerate() {
        out_data[i] = 1.0 / (1.0 + (-val).exp());
    }
    Tensor::new(out_data, input.shape().dims().to_vec())
}
