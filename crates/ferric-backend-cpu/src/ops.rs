use ferric_core::{Storage, Tensor, TypeId};
use rayon::prelude::*;

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // Basic matmul for f32
    if a.dtype() != TypeId::Float32 || b.dtype() != TypeId::Float32 {
        panic!("Only f32 matmul implemented for now");
    }

    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();

    if a_ndim == 2 && b_ndim == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_storage_arc = a.storage();
        let a_storage = match a_storage_arc.as_ref() {
            Storage::Cpu(s) => s.as_slice::<f32>(),
        };
        let b_storage_arc = b.storage();
        let b_storage = match b_storage_arc.as_ref() {
            Storage::Cpu(s) => s.as_slice::<f32>(),
        };

        let mut c_data = vec![0.0f32; m * n];

        c_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let a_row_offset = i * k;
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a_storage[a_row_offset + kk] * b_storage[kk * n + j];
                }
                row[j] = sum;
            }
        });

        return Tensor::new(c_data, vec![m, n]);
    }

    // High-dimensional matmul (simplified)
    if a_ndim >= 2 && b_ndim >= 2 {
        let m = a_shape[a_ndim - 2];
        let k = a_shape[a_ndim - 1];
        let n = b_shape[b_ndim - 1];
        assert_eq!(k, b_shape[b_ndim - 2]);

        let a_numel = a.shape().numel();
        let b_numel = b.shape().numel();
        let c_numel = (a_numel / k) * n;

        let mut c_data = vec![0.0f32; c_numel];
        let a_storage_arc = a.storage();
        let a_storage = match a_storage_arc.as_ref() {
            Storage::Cpu(s) => s.as_slice::<f32>(),
        };
        let b_storage_arc = b.storage();
        let b_storage = match b_storage_arc.as_ref() {
            Storage::Cpu(s) => s.as_slice::<f32>(),
        };

        c_data
            .par_chunks_mut(m * n)
            .enumerate()
            .for_each(|(mat_idx, mat_chunk)| {
                let a_mat_offset = mat_idx * (m * k);
                let b_mat_offset = if b_numel == k * n { 0 } else { mat_idx * (k * n) };

                for i in 0..m {
                    let a_row_offset = a_mat_offset + i * k;
                    for j in 0..n {
                        let mut sum = 0.0;
                        for kk in 0..k {
                            sum += a_storage[a_row_offset + kk]
                                * b_storage[b_mat_offset + kk * n + j];
                        }
                        mat_chunk[i * n + j] = sum;
                    }
                }
            });

        let mut c_shape = a_shape[..a_ndim - 2].to_vec();
        c_shape.push(m);
        c_shape.push(n);
        return Tensor::new(c_data, c_shape);
    }

    unimplemented!("Matmul for shapes {:?} and {:?}", a_shape, b_shape);
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

    let a_ndim = a.shape().ndim();
    let a_shape = a.shape().dims();
    let a_len = a_storage.len();
    let b_len = b_storage.len();

    let mut c_data = vec![0.0f32; a_len];
    c_data.par_iter_mut().enumerate().for_each(|(i, val)| {
        let b_val = if b_len == a_len {
            b_storage[i]
        } else if b_len == a_shape[a_ndim - 1] {
            // Broadcasting for (..., N) + (N)
            b_storage[i % b_len]
        } else {
            panic!(
                "Unsupported broadcast shapes for add: {:?} and {:?}",
                a.shape(),
                b.shape()
            );
        };
        *val = a_storage[i] + b_val;
    });

    Tensor::new(c_data, a.shape().dims().to_vec())
}

pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
) -> Tensor {
    let in_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    let b = in_shape[0];
    let c_in = in_shape[1];
    let h_in = in_shape[2];
    let w_in = in_shape[3];

    let c_out = weight_shape[0];
    let kh = weight_shape[2];
    let kw = weight_shape[3];

    let h_out = (h_in + 2 * padding.0 - kh) / stride.0 + 1;
    let w_out = (w_in + 2 * padding.1 - kw) / stride.1 + 1;

    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let weight_storage_arc = weight.storage();
    let weight_storage = match weight_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut out_data = vec![0.0f32; b * c_out * h_out * w_out];

    let c_out_per_group = c_out / groups;
    let c_in_per_group = c_in / groups;

    for bi in 0..b {
        for co in 0..c_out {
            let g = co / c_out_per_group;
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut sum = 0.0;
                    for ci in 0..c_in_per_group {
                        let actual_ci = g * c_in_per_group + ci;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = ho * stride.0 + ki;
                                let wi = wo * stride.1 + kj;

                                if hi >= padding.0
                                    && hi < h_in + padding.0
                                    && wi >= padding.1
                                    && wi < w_in + padding.1
                                {
                                    let actual_hi = hi - padding.0;
                                    let actual_wi = wi - padding.1;
                                    let in_idx = bi * (c_in * h_in * w_in)
                                        + actual_ci * (h_in * w_in)
                                        + actual_hi * w_in
                                        + actual_wi;
                                    let w_idx = co * (c_in_per_group * kh * kw) + ci * (kh * kw) + ki * kw + kj;
                                    sum += in_storage[in_idx] * weight_storage[w_idx];
                                }
                            }
                        }
                    }
                    let out_idx =
                        bi * (c_out * h_out * w_out) + co * (h_out * w_out) + ho * w_out + wo;
                    out_data[out_idx] = sum;
                }
            }
        }
    }

    Tensor::new(out_data, vec![b, c_out, h_out, w_out])
}

pub fn max_pool2d(input: &Tensor, kernel_size: (usize, usize), stride: (usize, usize)) -> Tensor {
    let in_shape = input.shape().dims();
    let b = in_shape[0];
    let c = in_shape[1];
    let h_in = in_shape[2];
    let w_in = in_shape[3];

    let h_out = (h_in - kernel_size.0) / stride.0 + 1;
    let w_out = (w_in - kernel_size.1) / stride.1 + 1;

    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut out_data = vec![0.0f32; b * c * h_out * w_out];

    for bi in 0..b {
        for ci in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;
                    for ki in 0..kernel_size.0 {
                        for kj in 0..kernel_size.1 {
                            let hi = ho * stride.0 + ki;
                            let wi = wo * stride.1 + kj;
                            let in_idx =
                                bi * (c * h_in * w_in) + ci * (h_in * w_in) + hi * w_in + wi;
                            max_val = max_val.max(in_storage[in_idx]);
                        }
                    }
                    let out_idx = bi * (c * h_out * w_out) + ci * (h_out * w_out) + ho * w_out + wo;
                    out_data[out_idx] = max_val;
                }
            }
        }
    }

    Tensor::new(out_data, vec![b, c, h_out, w_out])
}

pub fn layer_norm(input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let in_shape = input.shape().dims();
    let n = in_shape[in_shape.len() - 1];
    let numel = input.shape().numel();

    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let w_arc = weight.storage();
    let weight_storage = match w_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let b_arc = bias.storage();
    let bias_storage = match b_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut out_data = vec![0.0f32; numel];

    out_data.par_chunks_mut(n).enumerate().for_each(|(i, chunk)| {
        let in_offset = i * n;
        let in_chunk = &in_storage[in_offset..in_offset + n];

        let mut mean = 0.0;
        for &x in in_chunk {
            mean += x;
        }
        mean /= n as f32;

        let mut var = 0.0;
        for &x in in_chunk {
            let diff = x - mean;
            var += diff * diff;
        }
        var /= n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for j in 0..n {
            chunk[j] = (in_chunk[j] - mean) * inv_std * weight_storage[j] + bias_storage[j];
        }
    });

    Tensor::new(out_data, in_shape.to_vec())
}

pub fn batch_norm2d(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    eps: f32,
) -> Tensor {
    let in_shape = input.shape().dims();
    let b = in_shape[0];
    let c = in_shape[1];
    let h = in_shape[2];
    let w = in_shape[3];

    let numel = input.shape().numel();
    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let w_arc = weight.storage();
    let weight_storage = match w_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let b_arc = bias.storage();
    let bias_storage = match b_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let m_arc = mean.storage();
    let mean_storage = match m_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let v_arc = var.storage();
    let var_storage = match v_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut out_data = vec![0.0f32; numel];

    for bi in 0..b {
        for ci in 0..c {
            let m = mean_storage[ci];
            let v = var_storage[ci];
            let w_val = weight_storage[ci];
            let b_val = bias_storage[ci];
            let inv_std = 1.0 / (v + eps).sqrt();

            for hi in 0..h {
                for wi in 0..w {
                    let idx = bi * (c * h * w) + ci * (h * w) + hi * w + wi;
                    out_data[idx] = (in_storage[idx] - m) * inv_std * w_val + b_val;
                }
            }
        }
    }

    Tensor::new(out_data, in_shape.to_vec())
}

pub fn softmax(input: &Tensor, dim: isize) -> Tensor {
    let in_shape = input.shape().dims();
    let ndim = in_shape.len();
    let actual_dim = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    let n = in_shape[actual_dim];
    let numel = input.shape().numel();

    if actual_dim != ndim - 1 {
        unimplemented!("Softmax only implemented for last dimension");
    }

    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut out_data = vec![0.0f32; numel];

    out_data.par_chunks_mut(n).enumerate().for_each(|(i, chunk)| {
        let in_offset = i * n;
        let in_chunk = &in_storage[in_offset..in_offset + n];

        let mut max_val = f32::NEG_INFINITY;
        for &x in in_chunk {
            max_val = max_val.max(x);
        }

        let mut sum_exp = 0.0;
        for j in 0..n {
            let e = (in_chunk[j] - max_val).exp();
            chunk[j] = e;
            sum_exp += e;
        }

        let inv_sum = 1.0 / sum_exp;
        for j in 0..n {
            chunk[j] *= inv_sum;
        }
    });

    Tensor::new(out_data, in_shape.to_vec())
}

pub fn transpose(input: &Tensor, dim0: usize, dim1: usize) -> Tensor {
    let in_shape = input.shape().dims();
    let ndim = in_shape.len();
    assert!(dim0 < ndim && dim1 < ndim);

    let mut out_shape = in_shape.to_vec();
    out_shape.swap(dim0, dim1);

    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut out_data = vec![0.0f32; input.shape().numel()];

    let mut in_idx_coords = vec![0; ndim];
    let numel = input.shape().numel();

    for i in 0..numel {
        let mut temp = i;
        for d in (0..ndim).rev() {
            in_idx_coords[d] = temp % in_shape[d];
            temp /= in_shape[d];
        }

        let mut out_idx_coords = in_idx_coords.clone();
        out_idx_coords.swap(dim0, dim1);

        let mut out_idx = 0;
        let mut stride = 1;
        for d in (0..ndim).rev() {
            out_idx += out_idx_coords[d] * stride;
            stride *= out_shape[d];
        }

        out_data[out_idx] = in_storage[i];
    }

    Tensor::new(out_data, out_shape)
}

pub fn exp(input: &Tensor) -> Tensor {
    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let mut out_data = vec![0.0f32; in_storage.len()];
    for (i, &x) in in_storage.iter().enumerate() {
        out_data[i] = x.exp();
    }
    Tensor::new(out_data, input.shape().dims().to_vec())
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_storage_arc = a.storage();
    let a_storage = match a_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let b_storage_arc = b.storage();
    let b_storage = match b_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let mut out_data = vec![0.0f32; a_storage.len()];
    for i in 0..a_storage.len() {
        out_data[i] = a_storage[i] * b_storage[i];
    }
    Tensor::new(out_data, a.shape().dims().to_vec())
}

pub fn tanh(input: &Tensor) -> Tensor {
    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let mut out_data = vec![0.0f32; in_storage.len()];
    for (i, &x) in in_storage.iter().enumerate() {
        out_data[i] = x.tanh();
    }
    Tensor::new(out_data, input.shape().dims().to_vec())
}

pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor {
    // weight shape: (C_in, C_out, kH, kW)
    let in_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    let b = in_shape[0];
    let c_in = in_shape[1];
    let h_in = in_shape[2];
    let w_in = in_shape[3];

    let c_out = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];

    let h_out = (h_in - 1) * stride.0 - 2 * padding.0 + kh;
    let w_out = (w_in - 1) * stride.1 - 2 * padding.1 + kw;

    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let w_arc = weight.storage();
    let weight_storage = match w_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut out_data = vec![0.0f32; b * c_out * h_out * w_out];

    for bi in 0..b {
        for ci in 0..c_in {
            for hi in 0..h_in {
                for wi in 0..w_in {
                    let in_val = in_storage[bi * (c_in * h_in * w_in) + ci * (h_in * w_in) + hi * w_in + wi];
                    for co in 0..c_out {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ho = hi * stride.0 + ki;
                                let wo = wi * stride.1 + kj;

                                if ho >= padding.0 && ho < h_out + padding.0 && wo >= padding.1 && wo < w_out + padding.1 {
                                    let actual_ho = ho - padding.0;
                                    let actual_wo = wo - padding.1;
                                    if actual_ho < h_out && actual_wo < w_out {
                                        let out_idx = bi * (c_out * h_out * w_out) + co * (h_out * w_out) + actual_ho * w_out + actual_wo;
                                        let w_idx = ci * (c_out * kh * kw) + co * (kh * kw) + ki * kw + kj;
                                        out_data[out_idx] += in_val * weight_storage[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::new(out_data, vec![b, c_out, h_out, w_out])
}

pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    let in_shape = input.shape().dims();
    let n = in_shape[in_shape.len() - 1];
    let numel = input.shape().numel();

    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let w_arc = weight.storage();
    let weight_storage = match w_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };

    let mut out_data = vec![0.0f32; numel];

    out_data.par_chunks_mut(n).enumerate().for_each(|(i, chunk)| {
        let in_offset = i * n;
        let in_chunk = &in_storage[in_offset..in_offset + n];

        let mut ms = 0.0;
        for &x in in_chunk {
            ms += x * x;
        }
        ms /= n as f32;
        let inv_std = 1.0 / (ms + eps).sqrt();

        for j in 0..n {
            chunk[j] = in_chunk[j] * inv_std * weight_storage[j];
        }
    });

    Tensor::new(out_data, in_shape.to_vec())
}

pub fn silu(input: &Tensor) -> Tensor {
    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let mut out_data = vec![0.0f32; in_storage.len()];
    for (i, &x) in in_storage.iter().enumerate() {
        // x * sigmoid(x)
        let sig = 1.0 / (1.0 + (-x).exp());
        out_data[i] = x * sig;
    }
    Tensor::new(out_data, input.shape().dims().to_vec())
}

pub fn gelu(input: &Tensor) -> Tensor {
    let in_storage_arc = input.storage();
    let in_storage = match in_storage_arc.as_ref() {
        Storage::Cpu(s) => s.as_slice::<f32>(),
    };
    let mut out_data = vec![0.0f32; in_storage.len()];
    for (i, &x) in in_storage.iter().enumerate() {
        let x3 = x * x * x;
        let inner = 0.79788456 * (x + 0.044715 * x3);
        out_data[i] = 0.5 * x * (1.0 + inner.tanh());
    }
    Tensor::new(out_data, input.shape().dims().to_vec())
}
