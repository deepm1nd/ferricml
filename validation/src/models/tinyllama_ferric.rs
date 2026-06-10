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

struct TinyLlama {
    tok_embeddings: Tensor,
    ffn_norm: RMSNorm,
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl TinyLlama {
    fn forward(&self, x_idx: &[usize]) -> Tensor {
        let b = 1; let t = x_idx.len(); let d = 32;
        let mut h_data = vec![0.0f32; b * t * d];
        let binding = self.tok_embeddings.storage();
        let emb_data = match binding.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
        for i in 0..t {
            let idx = x_idx[i];
            for j in 0..d {
                h_data[i * d + j] = emb_data[idx * d + j];
            }
        }
        let h = Tensor::new(h_data, vec![b, t, d]);
        let norm_h = self.ffn_norm.forward(&h);
        let w1_out = self.w1.forward(&norm_h);
        let w2_out = self.w2.forward(&norm_h);
        let silu_w1 = silu(&w1_out);
        let swiglu = ferricml::cpu::ops::mul(&silu_w1, &w2_out);
        let ffn = self.w3.forward(&swiglu);
        let ffn_data = match ffn.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
        let mut final_data = match h.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
        for i in 0..final_data.len() { final_data[i] += ffn_data[i]; }
        Tensor::new(final_data, vec![b, t, d])
    }
}

fn main() -> anyhow::Result<()> {
    let mut file = match File::open("validation/tinyllama/native_output.json") {
        Ok(f) => f,
        Err(_) => {
            println!("Native output not found. Run tinyllama_native.py first.");
            return Ok(());
        }
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: Value = serde_json::from_str(&content)?;
    let params = &v["params"];
    let tok_embeddings = load_tensor_any(&params["tok_embeddings.weight"]);
    let mut ffn_norm = RMSNorm::new(32, 1e-6);
    ffn_norm.weight = load_tensor_any(&params["ffn_norm.weight"]);
    fn load_linear_pt_nobias(weight_v: &Value) -> Linear {
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
        l.bias = None;
        l
    }
    let w1 = load_linear_pt_nobias(&params["w1.weight"]);
    let w2 = load_linear_pt_nobias(&params["w2.weight"]);
    let w3 = load_linear_pt_nobias(&params["w3.weight"]);
    let model = TinyLlama { tok_embeddings, ffn_norm, w1, w2, w3 };
    let input_idx = vec![102, 51, 92, 14, 106, 71, 60, 20];
    let output = model.forward(&input_idx);
    let output_vec = match output.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let native_output: Vec<f32> = flatten_json_array(&v["output"]);
    let mut max_diff = 0.0f32;
    for i in 0..output_vec.len() {
        let diff = (output_vec[i] - native_output[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("Max difference for TinyLlama: {}", max_diff);
    Ok(())
}
