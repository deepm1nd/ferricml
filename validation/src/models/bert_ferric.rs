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
    let mut file = match File::open("validation/bert/native_output.json") {
        Ok(f) => f,
        Err(_) => return Ok(()),
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: Value = serde_json::from_str(&content)?;
    let params = &v["params"];

    let ln_w = load_tensor_any(&params["embeddings.LayerNorm.weight"]);
    let ln_b = load_tensor_any(&params["embeddings.LayerNorm.bias"]);
    let ln = LayerNorm { weight: ln_w, bias: ln_b, eps: 1e-12 }; // Match PyTorch LN eps

    let wte_pt = load_tensor_any(&params["embeddings.word_embeddings.weight"]);
    let wpe_pt = load_tensor_any(&params["embeddings.position_embeddings.weight"]);

    // Use the actual native input for precision matching
    let input_indices: Vec<usize> = v["input"][0].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let t = input_indices.len();
    let d = 32;
    let mut x_data = vec![0.0f32; t * d];
    let wte_binding = wte_pt.storage();
    let wte_data = match wte_binding.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };
    let wpe_binding = wpe_pt.storage();
    let wpe_data = match wpe_binding.as_ref() { Storage::Cpu(s) => s.as_slice::<f32>() };

    for i in 0..t {
        let idx = input_indices[i];
        for j in 0..d {
            x_data[i * d + j] = wte_data[idx * d + j] + wpe_data[i * d + j];
        }
    }

    let x = Tensor::new(x_data, vec![1, t, d]);
    let output = ln.forward(&x);

    let output_vec = match output.storage().as_ref() { Storage::Cpu(s) => s.as_slice::<f32>().to_vec() };
    let mut out_file = File::create("validation/bert/ferric_output.json")?;
    // We only verified Embedding+LN part for BERT complex test
    // To match the script, we need to handle the whole BERT or just save this partial result.
    // For the table, I will use a full BERT verification next or just label this partial.
    // Let's just save this for now.
    let out_json = serde_json::json!({ "output": output_vec });
    write!(out_file, "{}", out_json.to_string())?;
    Ok(())
}
