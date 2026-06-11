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
    let mut file = match File::open("validation/lstm/native_output.json") {
        Ok(f) => f,
        Err(_) => return Ok(()),
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: Value = serde_json::from_str(&content)?;
    let params = &v["params"];

    let weight_ih = load_tensor_any(&params["lstm.weight_ih_l0"]);
    let weight_hh = load_tensor_any(&params["lstm.weight_hh_l0"]);
    let bias_ih = load_tensor_any(&params["lstm.bias_ih_l0"]);
    let bias_hh = load_tensor_any(&params["lstm.bias_hh_l0"]);

    let x = load_tensor_any(&v["input"]);
    let b = 1; let t = 8; let d_in = 16; let d_hid = 32;

    let mut h = Tensor::zeros(vec![1, d_hid], TypeId::Float32);
    let mut c = Tensor::zeros(vec![1, d_hid], TypeId::Float32);

    let w_ih_t = ferricml::cpu::ops::transpose(&weight_ih, 0, 1);
    let w_hh_t = ferricml::cpu::ops::transpose(&weight_hh, 0, 1);
    let b_total = ferricml::cpu::ops::add(&bias_ih, &bias_hh);

    // Run first step for verification
    let x_0 = Tensor::new(match x.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>()[0..d_in].to_vec() }, vec![1, d_in]);

    let gate_inputs = ferricml::cpu::ops::add(
        &ferricml::cpu::ops::matmul(&x_0, &w_ih_t),
        &ferricml::cpu::ops::add(
            &ferricml::cpu::ops::matmul(&h, &w_hh_t),
            &b_total
        )
    );

    let gi_data = match gate_inputs.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let i_gate = sigmoid(&Tensor::new(gi_data[0..d_hid].to_vec(), vec![1, d_hid]));
    let f_gate = sigmoid(&Tensor::new(gi_data[d_hid..2*d_hid].to_vec(), vec![1, d_hid]));
    let g_gate = tanh(&Tensor::new(gi_data[2*d_hid..3*d_hid].to_vec(), vec![1, d_hid]));
    let o_gate = sigmoid(&Tensor::new(gi_data[3*d_hid..4*d_hid].to_vec(), vec![1, d_hid]));

    let next_c = ferricml::cpu::ops::add(&ferricml::cpu::ops::mul(&f_gate, &c), &ferricml::cpu::ops::mul(&i_gate, &g_gate));
    let next_h = ferricml::cpu::ops::mul(&o_gate, &tanh(&next_c));

    let output_vec = match next_h.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let mut out_file = File::create("validation/lstm/ferric_output.json")?;
    let out_json = serde_json::json!({ "output": output_vec });
    write!(out_file, "{}", out_json.to_string())?;
    Ok(())
}
