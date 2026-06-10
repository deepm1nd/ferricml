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

struct MobileNetBlock {
    pw1: Conv2d,
    bn1: BatchNorm2d,
    dw: Conv2d,
    bn2: BatchNorm2d,
    pw2: Conv2d,
    bn3: BatchNorm2d,
}

impl MobileNetBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        let identity = x;
        let mut out = self.pw1.forward(x);
        out = self.bn1.forward(&out);
        out = relu6(&out);

        out = self.dw.forward(&out);
        out = self.bn2.forward(&out);
        out = relu6(&out);

        out = self.pw2.forward(&out);
        out = self.bn3.forward(&out);

        let out_shape = out.shape().dims().to_vec();
        let mut out_data = match out.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
        let binding = identity.storage();
        let identity_data = match binding.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
        for i in 0..out_data.len() { out_data[i] += identity_data[i]; }
        Tensor::new(out_data, out_shape)
    }
}

fn main() -> anyhow::Result<()> {
    let mut file = match File::open("validation/mobilenet/native_output.json") {
        Ok(f) => f,
        Err(_) => {
            println!("Native output not found. Run mobilenet_native.py first.");
            return Ok(());
        }
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: Value = serde_json::from_str(&content)?;
    let input_tensor = load_tensor_any(&v["input"]);
    let params = &v["params"];

    let mut pw1 = Conv2d::new(16, 96, (1, 1), (1, 1), (0, 0));
    pw1.weight = load_tensor_any(&params["conv.0.weight"]); pw1.bias = None;
    let mut bn1 = BatchNorm2d::new(96, 1e-5);
    bn1.weight = load_tensor_any(&params["conv.1.weight"]); bn1.bias = load_tensor_any(&params["conv.1.bias"]);
    bn1.running_mean = load_tensor_any(&params["conv.1.running_mean"]); bn1.running_var = load_tensor_any(&params["conv.1.running_var"]);

    let mut dw = Conv2d::new(96, 96, (3, 3), (1, 1), (1, 1)).with_groups(96);
    dw.weight = load_tensor_any(&params["conv.3.weight"]); dw.bias = None;
    let mut bn2 = BatchNorm2d::new(96, 1e-5);
    bn2.weight = load_tensor_any(&params["conv.4.weight"]); bn2.bias = load_tensor_any(&params["conv.4.bias"]);
    bn2.running_mean = load_tensor_any(&params["conv.4.running_mean"]); bn2.running_var = load_tensor_any(&params["conv.4.running_var"]);

    let mut pw2 = Conv2d::new(96, 16, (1, 1), (1, 1), (0, 0));
    pw2.weight = load_tensor_any(&params["conv.6.weight"]); pw2.bias = None;
    let mut bn3 = BatchNorm2d::new(16, 1e-5);
    bn3.weight = load_tensor_any(&params["conv.7.weight"]); bn3.bias = load_tensor_any(&params["conv.7.bias"]);
    bn3.running_mean = load_tensor_any(&params["conv.7.running_mean"]); bn3.running_var = load_tensor_any(&params["conv.7.running_var"]);

    let model = MobileNetBlock { pw1, bn1, dw, bn2, pw2, bn3 };
    let output = model.forward(&input_tensor);
    let output_vec = match output.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let native_output: Vec<f32> = flatten_json_array(&v["output"]);
    let mut max_diff = 0.0f32;
    for i in 0..output_vec.len() {
        let diff = (output_vec[i] - native_output[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("Max difference for MobileNet block: {}", max_diff);
    Ok(())
}
