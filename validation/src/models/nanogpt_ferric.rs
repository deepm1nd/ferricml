use ferricml::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use serde_json::Value;

fn load_tensor_any(v: &Value) -> Tensor {
    if v.as_array().unwrap().len() > 0 && v.as_array().unwrap()[0].is_array() {
        let dims = get_dims(v);
        let data = flatten_json_array(v);
        Tensor::new(data, dims)
    } else {
        let data: Vec<f32> = v.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let len = data.len();
        Tensor::new(data, vec![len])
    }
}

fn get_dims(v: &Value) -> Vec<usize> {
    let mut dims = Vec::new();
    let mut curr = v;
    while curr.is_array() {
        dims.push(curr.as_array().unwrap().len());
        curr = &curr[0];
    }
    dims
}

fn flatten_json_array(v: &Value) -> Vec<f32> {
    let mut result = Vec::new();
    flatten_helper(v, &mut result);
    result
}

fn flatten_helper(v: &Value, result: &mut Vec<f32>) {
    if let Some(arr) = v.as_array() {
        for item in arr {
            flatten_helper(item, result);
        }
    } else if let Some(f) = v.as_f64() {
        result.push(f as f32);
    }
}

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
}

impl CausalSelfAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_shape = x.shape().dims();
        let b_dim = x_shape[0];
        let t_dim = x_shape[1];
        let c_dim = x_shape[2];
        let qkv = self.c_attn.forward(x);
        let qkv_data = match qkv.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
        let q = Tensor::new(qkv_data[0..b_dim*t_dim*c_dim].to_vec(), vec![b_dim, t_dim, c_dim]);
        let k = Tensor::new(qkv_data[b_dim*t_dim*c_dim..2*b_dim*t_dim*c_dim].to_vec(), vec![b_dim, t_dim, c_dim]);
        let v = Tensor::new(qkv_data[2*b_dim*t_dim*c_dim..3*b_dim*t_dim*c_dim].to_vec(), vec![b_dim, t_dim, c_dim]);
        let hs = c_dim / self.n_head;
        let q = q.reshape(vec![b_dim, t_dim, self.n_head, hs]);
        let q = ferricml::cpu::ops::transpose(&q, 1, 2);
        let k = k.reshape(vec![b_dim, t_dim, self.n_head, hs]);
        let k = ferricml::cpu::ops::transpose(&k, 1, 2);
        let v = v.reshape(vec![b_dim, t_dim, self.n_head, hs]);
        let v = ferricml::cpu::ops::transpose(&v, 1, 2);
        let kt = ferricml::cpu::ops::transpose(&k, 2, 3);
        let att = ferricml::cpu::ops::matmul(&q, &kt);
        let scale = 1.0 / (hs as f32).sqrt();
        let mut att_data = match att.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
        for val in att_data.iter_mut() { *val *= scale; }
        for b in 0..b_dim {
            for h in 0..self.n_head {
                for i in 0..t_dim {
                    for j in (i+1)..t_dim {
                        att_data[b * self.n_head * t_dim * t_dim + h * t_dim * t_dim + i * t_dim + j] = -1e9;
                    }
                }
            }
        }
        let att = softmax(&Tensor::new(att_data, att.shape().dims().to_vec()), -1);
        let y = ferricml::cpu::ops::matmul(&att, &v);
        let y = ferricml::cpu::ops::transpose(&y, 1, 2);
        let y = y.reshape(vec![b_dim, t_dim, c_dim]);
        self.c_proj.forward(&y)
    }
}

fn main() -> anyhow::Result<()> {
    let mut file = match File::open("validation/nanogpt/native_output.json") {
        Ok(f) => f,
        Err(_) => {
            println!("Native output not found. Run nanogpt_native.py first.");
            return Ok(());
        }
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: Value = serde_json::from_str(&content)?;
    let b_dim = 1; let t_dim = 8; let c_dim = 32;
    let mut rng_input = vec![0.0f32; b_dim * t_dim * c_dim];
    for i in 0..rng_input.len() { rng_input[i] = (i as f32).sin(); }
    let input_tensor = Tensor::new(rng_input, vec![b_dim, t_dim, c_dim]);
    let params = &v["params"];
    fn load_linear_pt(weight_v: &Value, bias_v: &Value) -> Linear {
        let pt_w = load_tensor_any(weight_v);
        let pt_w_shape = pt_w.shape().dims();
        let out_f = pt_w_shape[0];
        let in_f = pt_w_shape[1];
        let mut weight_ferric = vec![0.0f32; in_f * out_f];
        let binding = pt_w.storage();
        let data = match binding.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
        for i in 0..out_f { for j in 0..in_f { weight_ferric[j * out_f + i] = data[i * in_f + j]; } }
        let mut l = Linear::new(in_f, out_f);
        l.weight = Tensor::new(weight_ferric, vec![in_f, out_f]);
        if !bias_v.is_null() { l.bias = Some(load_tensor_any(bias_v)); } else { l.bias = None; }
        l
    }
    let c_attn = load_linear_pt(&params["transformer.h.0.attn.c_attn.weight"], &params["transformer.h.0.attn.c_attn.bias"]);
    let c_proj = load_linear_pt(&params["transformer.h.0.attn.c_proj.weight"], &params["transformer.h.0.attn.c_proj.bias"]);
    let attn = CausalSelfAttention { c_attn, c_proj, n_head: 2 };
    let output = attn.forward(&input_tensor);
    println!("Attention output shape: {:?}", output.shape());
    Ok(())
}
