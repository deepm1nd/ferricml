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
    let mut file = match File::open("validation/unet/native_output.json") {
        Ok(f) => f,
        Err(_) => return Ok(()),
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: Value = serde_json::from_str(&content)?;
    let params = &v["params"];

    let mut down = Conv2d::new(3, 16, (3, 3), (2, 2), (1, 1));
    down.weight = load_tensor_any(&params["down.weight"]);
    down.bias = Some(load_tensor_any(&params["down.bias"]));

    let mut up = ConvTranspose2d::new(16, 3, (4, 4), (2, 2), (1, 1));
    up.weight = load_tensor_any(&params["up.weight"]);
    up.bias = Some(load_tensor_any(&params["up.bias"]));

    let input_tensor = load_tensor_any(&v["input"]);
    let out_down = relu(&down.forward(&input_tensor));
    let out_up = up.forward(&out_down);
    let final_out = sigmoid(&out_up);

    let output_vec = match final_out.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let mut out_file = File::create("validation/unet/ferric_output.json")?;
    let out_json = serde_json::json!({ "output": output_vec });
    write!(out_file, "{}", out_json.to_string())?;
    Ok(())
}
