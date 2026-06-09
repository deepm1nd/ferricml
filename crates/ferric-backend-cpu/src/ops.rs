use ferric_core::{Storage, Tensor, TypeId};
use rayon::prelude::*;

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // Basic matmul for f32
    if a.dtype() != TypeId::Float32 || b.dtype() != TypeId::Float32 {
        panic!("Only f32 matmul implemented for now");
    }

    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    let a_storage_arc = a.storage();
    let b_storage_arc = b.storage();

    let a_storage = match a_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let b_storage = match b_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut c_data = vec![0.0f32; m * n];

    c_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a_storage[i * k + kk] * b_storage[kk * n + j];
            }
            row[j] = sum;
        }
    });

    Tensor::new(c_data, vec![m, n])
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    if a.dtype() != TypeId::Float32 || b.dtype() != TypeId::Float32 {
        panic!("Only f32 add implemented for now");
    }

    let a_storage_arc = a.storage();
    let b_storage_arc = b.storage();

    let a_storage = match a_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let b_storage = match b_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut c_data = vec![0.0f32; a_storage.len()];
    c_data.par_iter_mut().enumerate().for_each(|(i, val)| {
        *val = a_storage[i] + b_storage[i];
    });

    Tensor::new(c_data, a.shape().dims().to_vec())
}
