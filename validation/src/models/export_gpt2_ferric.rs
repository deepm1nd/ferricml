use ferricml::prelude::*;
use ferricml::serialization::gguf::{GGUFWriter, GGUFValue};
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let vocab_size = 128;
    let n_embd = 32;
    let n_head = 2;
    let n_layer = 1;
    let block_size = 64;
    let n_ff = 128;

    // Tensors in PyTorch shape
    let tensors = vec![
        ("token_embd.weight", Tensor::zeros(vec![vocab_size, n_embd], TypeId::Float32)),
        ("position_embd.weight", Tensor::zeros(vec![block_size, n_embd], TypeId::Float32)),
        ("blk.0.attn_norm.weight", Tensor::new(vec![1.0f32; n_embd], vec![n_embd])),
        ("blk.0.attn_norm.bias", Tensor::zeros(vec![n_embd], TypeId::Float32)),
        ("blk.0.attn_qkv.weight", Tensor::zeros(vec![3 * n_embd, n_embd], TypeId::Float32)),
        ("blk.0.attn_qkv.bias", Tensor::zeros(vec![3 * n_embd], TypeId::Float32)),
        ("blk.0.attn_output.weight", Tensor::zeros(vec![n_embd, n_embd], TypeId::Float32)),
        ("blk.0.attn_output.bias", Tensor::zeros(vec![n_embd], TypeId::Float32)),
        ("blk.0.ffn_norm.weight", Tensor::new(vec![1.0f32; n_embd], vec![n_embd])),
        ("blk.0.ffn_norm.bias", Tensor::zeros(vec![n_embd], TypeId::Float32)),
        ("blk.0.ffn_up.weight", Tensor::zeros(vec![n_ff, n_embd], TypeId::Float32)),
        ("blk.0.ffn_up.bias", Tensor::zeros(vec![n_ff], TypeId::Float32)),
        ("blk.0.ffn_down.weight", Tensor::zeros(vec![n_embd, n_ff], TypeId::Float32)),
        ("blk.0.ffn_down.bias", Tensor::zeros(vec![n_embd], TypeId::Float32)),
        ("output_norm.weight", Tensor::new(vec![1.0f32; n_embd], vec![n_embd])),
        ("output_norm.bias", Tensor::zeros(vec![n_embd], TypeId::Float32)),
        ("output.weight", Tensor::zeros(vec![vocab_size, n_embd], TypeId::Float32)),
    ];

    let file = File::create("validation/gguf/tiny_gpt2_ferric.gguf")?;
    let mut writer = GGUFWriter::new(file);

    let mut tokens = Vec::new();
    let mut scores = Vec::new();
    let mut toktypes = Vec::new();
    for i in 0..vocab_size {
        tokens.push(GGUFValue::String(format!("token_{}", i)));
        scores.push(GGUFValue::Float32(0.0));
        toktypes.push(GGUFValue::Int32(1));
    }

    writer.write_header(tensors.len() as u32, 14)?;
    writer.write_kv("general.architecture", GGUFValue::String("gpt2".to_string()))?;
    writer.write_kv("gpt2.block_count", GGUFValue::Uint32(n_layer as u32))?;
    writer.write_kv("gpt2.embedding_length", GGUFValue::Uint32(n_embd as u32))?;
    writer.write_kv("gpt2.head_count", GGUFValue::Uint32(n_head as u32))?;
    writer.write_kv("gpt2.feed_forward_length", GGUFValue::Uint32(n_ff as u32))?;
    writer.write_kv("gpt2.attention.head_count", GGUFValue::Uint32(n_head as u32))?;
    writer.write_kv("gpt2.attention.layer_norm_epsilon", GGUFValue::Float32(1e-5))?;
    writer.write_kv("gpt2.context_length", GGUFValue::Uint32(block_size as u32))?;
    writer.write_kv("tokenizer.ggml.model", GGUFValue::String("gpt2".to_string()))?;
    writer.write_kv("tokenizer.ggml.pre", GGUFValue::String("gpt-2".to_string()))?;
    writer.write_kv("tokenizer.ggml.tokens", GGUFValue::Array(tokens))?;
    writer.write_kv("tokenizer.ggml.scores", GGUFValue::Array(scores))?;
    writer.write_kv("tokenizer.ggml.token_type", GGUFValue::Array(toktypes))?;
    writer.write_kv("tokenizer.ggml.merges", GGUFValue::Array(vec![GGUFValue::String("a b".to_string())]))?;

    let mut data_offsets = Vec::new();
    let mut current_offset = 0;
    for (_, tensor) in &tensors {
        data_offsets.push(current_offset);
        let size = (tensor.shape().numel() * 4) as u64;
        let padding = (32 - (size % 32)) % 32;
        current_offset += size + padding;
    }

    for (i, (name, tensor)) in tensors.iter().enumerate() {
        writer.write_tensor(name, tensor, data_offsets[i])?;
    }
    writer.align(32)?;
    for (_, tensor) in tensors {
        writer.write_tensor_data(&tensor)?;
    }
    println!("FerricML GGUF exported.");
    Ok(())
}
