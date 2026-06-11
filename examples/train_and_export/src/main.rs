use ferricml::prelude::*;
use ferricml::serialization::gguf::GGUFWriter;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    println!("Initializing FerricML...");

    // 1. Define a simple model
    let model = Linear::new(10, 5);

    // 2. Create some dummy data
    let _input = Tensor::new(vec![1.0f32; 10], vec![1, 10]);
    let _target = Tensor::new(vec![0.0f32; 5], vec![1, 5]);

    // 3. Simple training loop (placeholder)
    println!("Training model...");
    for epoch in 0..5 {
        // let output = model.forward(&input);
        // let loss = mse_loss(&output, &target);
        // loss.backward();
        println!("Epoch {}: Loss = ...", epoch);
    }

    // 4. Save model in GGUF format
    println!("Exporting model to GGUF...");
    let file = File::create("model.gguf")?;
    let mut writer = GGUFWriter::new(file);

    let params = model.parameters();
    writer.write_header(params.len() as u32, 0)?;

    let mut offset = 0;
    writer.write_tensor("weight", params[0], offset)?;
    offset += (params[0].shape().numel() * 4) as u64;

    if params.len() > 1 {
        writer.write_tensor("bias", params[1], offset)?;
    }

    writer.write_tensor_data(params[0])?;
    if params.len() > 1 {
        writer.write_tensor_data(params[1])?;
    }

    println!("Model saved to model.gguf");
    Ok(())
}
