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

fn main() -> anyhow::Result<()> {
    let mut file = match File::open("validation/nanogpt/native_output.json") {
        Ok(f) => f,
        Err(_) => return Ok(()),
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: Value = serde_json::from_str(&content)?;
    let params = &v["params"];

    fn load_linear_pt(weight_v: &Value, bias_v: &Value) -> Linear {
        let pt_w = load_tensor_any(weight_v);
        let pt_w_shape = pt_w.shape().dims();
        let out_f = pt_w_shape[0];
        let in_f = pt_w_shape[1];
        let mut weight_ferric = vec![0.0f32; in_f * out_f];
        let b = pt_w.storage();
        let data = match b.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
        for i in 0..out_f { for j in 0..in_f { weight_ferric[j * out_f + i] = data[i * in_f + j]; } }
        let mut l = Linear::new(in_f, out_f);
        l.weight = Tensor::new(weight_ferric, vec![in_f, out_f]);
        if !bias_v.is_null() { l.bias = Some(load_tensor_any(bias_v)); } else { l.bias = None; }
        l
    }

    let b_dim = 1; let t_dim = 8; let c_dim = 32; let vocab_size = 128;
    let input_indices: Vec<usize> = v["input"][0].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();

    let wte_pt = load_tensor_any(&params["transformer.wte.weight"]);
    let wpe_pt = load_tensor_any(&params["transformer.wpe.weight"]);
    let ln_f_w = load_tensor_any(&params["transformer.ln_f.weight"]);
    let ln_f_b = load_tensor_any(&params["transformer.ln_f.bias"]);
    let lm_head_w_pt = load_tensor_any(&params["lm_head.weight"]);

    let ln_1_w = load_tensor_any(&params["transformer.h.0.ln_1.weight"]);
    let ln_1_b = load_tensor_any(&params["transformer.h.0.ln_1.bias"]);
    let ln_2_w = load_tensor_any(&params["transformer.h.0.ln_2.weight"]);
    let ln_2_b = load_tensor_any(&params["transformer.h.0.ln_2.bias"]);

    let c_attn = load_linear_pt(&params["transformer.h.0.attn.c_attn.weight"], &params["transformer.h.0.attn.c_attn.bias"]);
    let c_proj = load_linear_pt(&params["transformer.h.0.attn.c_proj.weight"], &params["transformer.h.0.attn.c_proj.bias"]);
    let c_fc = load_linear_pt(&params["transformer.h.0.mlp.c_fc.weight"], &params["transformer.h.0.mlp.c_fc.bias"]);
    let c_mlp_proj = load_linear_pt(&params["transformer.h.0.mlp.c_proj.weight"], &params["transformer.h.0.mlp.c_proj.bias"]);

    let mut x_data = vec![0.0f32; t_dim * c_dim];
    {
        let b1 = wte_pt.storage();
        let wte_data = match b1.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
        let b2 = wpe_pt.storage();
        let wpe_data = match b2.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
        for i in 0..t_dim {
            let idx = input_indices[i];
            for j in 0..c_dim { x_data[i * c_dim + j] = wte_data[idx * c_dim + j] + wpe_data[i * c_dim + j]; }
        }
    }
    let x = Tensor::new(x_data, vec![b_dim, t_dim, c_dim]);

    let residual = x.reshape(vec![b_dim, t_dim, c_dim]);
    let norm1 = (LayerNorm { weight: ln_1_w, bias: ln_1_b, eps: 1e-5 }).forward(&residual);

    let qkv = c_attn.forward(&norm1);
    let qkv_data = match qkv.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let q = Tensor::new(qkv_data[0..c_dim*t_dim].to_vec(), vec![b_dim, t_dim, c_dim]);
    let k = Tensor::new(qkv_data[c_dim*t_dim..2*c_dim*t_dim].to_vec(), vec![b_dim, t_dim, c_dim]);
    let v_t = Tensor::new(qkv_data[2*c_dim*t_dim..3*c_dim*t_dim].to_vec(), vec![b_dim, t_dim, c_dim]);
    let n_head = 2; let hs = c_dim / n_head;
    let q = ferricml::cpu::ops::transpose(&q.reshape(vec![b_dim, t_dim, n_head, hs]), 1, 2);
    let k = ferricml::cpu::ops::transpose(&k.reshape(vec![b_dim, t_dim, n_head, hs]), 1, 2);
    let v_t = ferricml::cpu::ops::transpose(&v_t.reshape(vec![b_dim, t_dim, n_head, hs]), 1, 2);
    let att = ferricml::cpu::ops::matmul(&q, &ferricml::cpu::ops::transpose(&k, 2, 3));
    let mut att_data = match att.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let scale = 1.0 / (hs as f32).sqrt();
    for val in att_data.iter_mut() { *val *= scale; }
    for b in 0..b_dim { for h in 0..n_head { for i in 0..t_dim { for j in (i+1)..t_dim {
        att_data[b * n_head * t_dim * t_dim + h * t_dim * t_dim + i * t_dim + j] = -1e9;
    } } } }
    let att_sm = softmax(&Tensor::new(att_data, att.shape().dims().to_vec()), -1);
    let attn_out = c_proj.forward(&ferricml::cpu::ops::transpose(&ferricml::cpu::ops::matmul(&att_sm, &v_t), 1, 2).reshape(vec![b_dim, t_dim, c_dim]));

    let mut x_post_attn = match residual.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    {
        let b_attn = attn_out.storage();
        let attn_out_data = match b_attn.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
        for i in 0..x_post_attn.len() { x_post_attn[i] += attn_out_data[i]; }
    }
    let x_post_attn = Tensor::new(x_post_attn, vec![b_dim, t_dim, c_dim]);

    let norm2 = (LayerNorm { weight: ln_2_w, bias: ln_2_b, eps: 1e-5 }).forward(&x_post_attn);
    let mlp_out = c_mlp_proj.forward(&gelu(&c_fc.forward(&norm2)));

    let mut x_final = match x_post_attn.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    {
        let b_mlp = mlp_out.storage();
        let mlp_out_data = match b_mlp.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
        for i in 0..x_final.len() { x_final[i] += mlp_out_data[i]; }
    }
    let x_final = Tensor::new(x_final, vec![b_dim, t_dim, c_dim]);

    let ln_f = (LayerNorm { weight: ln_f_w, bias: ln_f_b, eps: 1e-5 }).forward(&x_final);

    let mut lm_head_ferric = vec![0.0f32; c_dim * vocab_size];
    {
        let b_lm = lm_head_w_pt.storage();
        let lm_head_data = match b_lm.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
        for i in 0..vocab_size { for j in 0..c_dim { lm_head_ferric[j * vocab_size + i] = lm_head_data[i * c_dim + j]; } }
    }
    let lm_head = Linear { weight: Tensor::new(lm_head_ferric, vec![c_dim, vocab_size]), bias: None };

    let logits = lm_head.forward(&ln_f);

    let output_vec = match logits.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let mut out_file = File::create("validation/nanogpt/ferric_output.json")?;
    let out_json = serde_json::json!({ "output": output_vec });
    write!(out_file, "{}", out_json.to_string())?;
    Ok(())
}
