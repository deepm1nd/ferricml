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
    let mut file = match File::open("validation/lenet5/native_output.json") {
        Ok(f) => f,
        Err(_) => return Ok(()),
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: Value = serde_json::from_str(&content)?;
    let input_tensor = load_tensor_any(&v["input"]);
    let params = &v["params"];
    let mut conv1 = Conv2d::new(1, 6, (5, 5), (1, 1), (2, 2));
    conv1.weight = load_tensor_any(&params["conv1.weight"]);
    conv1.bias = Some(load_tensor_any(&params["conv1.bias"]));
    let mut conv2 = Conv2d::new(6, 16, (5, 5), (1, 1), (0, 0));
    conv2.weight = load_tensor_any(&params["conv2.weight"]);
    conv2.bias = Some(load_tensor_any(&params["conv2.bias"]));
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
        l.bias = Some(load_tensor_any(bias_v));
        l
    }
    let fc1 = load_linear_pt(&params["fc1.weight"], &params["fc1.bias"]);
    let fc2 = load_linear_pt(&params["fc2.weight"], &params["fc2.bias"]);
    let fc3 = load_linear_pt(&params["fc3.weight"], &params["fc3.bias"]);

    // LeNet5 forward logic
    let x = conv1.forward(&input_tensor);
    let x = relu(&x);
    let x = ferricml::cpu::ops::max_pool2d(&x, (2, 2), (2, 2));
    let x = conv2.forward(&x);
    let x = relu(&x);
    let x = ferricml::cpu::ops::max_pool2d(&x, (2, 2), (2, 2));
    let x_shape = x.shape().dims();
    let flat_len = x_shape[1] * x_shape[2] * x_shape[3];
    let x_data = match x.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let x = Tensor::new(x_data, vec![x_shape[0], flat_len]);
    let x = fc1.forward(&x);
    let x = relu(&x);
    let x = fc2.forward(&x);
    let x = relu(&x);
    let output = fc3.forward(&x);

    let output_vec = match output.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let mut out_file = File::create("validation/lenet5/ferric_output.json")?;
    let out_json = serde_json::json!({ "output": output_vec });
    write!(out_file, "{}", out_json.to_string())?;
    Ok(())
}
