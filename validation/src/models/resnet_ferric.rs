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
    let mut file = match File::open("validation/resnet/native_output.json") {
        Ok(f) => f,
        Err(_) => return Ok(()),
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: Value = serde_json::from_str(&content)?;
    let input_tensor = load_tensor_any(&v["input"]);
    let params = &v["params"];

    let mut conv1 = Conv2d::new(16, 16, (3, 3), (1, 1), (1, 1));
    conv1.weight = load_tensor_any(&params["conv1.weight"]);
    conv1.bias = None;
    let mut bn1 = BatchNorm2d::new(16, 1e-5);
    bn1.weight = load_tensor_any(&params["bn1.weight"]);
    bn1.bias = load_tensor_any(&params["bn1.bias"]);
    bn1.running_mean = load_tensor_any(&params["bn1.running_mean"]);
    bn1.running_var = load_tensor_any(&params["bn1.running_var"]);
    let mut conv2 = Conv2d::new(16, 16, (3, 3), (1, 1), (1, 1));
    conv2.weight = load_tensor_any(&params["conv2.weight"]);
    conv2.bias = None;
    let mut bn2 = BatchNorm2d::new(16, 1e-5);
    bn2.weight = load_tensor_any(&params["bn2.weight"]);
    bn2.bias = load_tensor_any(&params["bn2.bias"]);
    bn2.running_mean = load_tensor_any(&params["bn2.running_mean"]);
    bn2.running_var = load_tensor_any(&params["bn2.running_var"]);

    let identity = &input_tensor;
    let mut out = conv1.forward(&input_tensor);
    out = bn1.forward(&out);
    out = relu(&out);
    out = conv2.forward(&out);
    out = bn2.forward(&out);
    let out_shape = out.shape().dims().to_vec();
    let mut out_data = match out.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let binding = identity.storage();
    let id_data = match binding.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
    for i in 0..out_data.len() { out_data[i] += id_data[i]; }
    let output = relu(&Tensor::new(out_data, out_shape));

    let output_vec = match output.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let mut out_file = File::create("validation/resnet/ferric_output.json")?;
    let out_json = serde_json::json!({ "output": output_vec });
    write!(out_file, "{}", out_json.to_string())?;
    Ok(())
}
