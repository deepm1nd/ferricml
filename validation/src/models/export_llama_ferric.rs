use ferricml::prelude::*;
use ferricml::serialization::gguf::{GGUFWriter, GGUFValue};
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let vocab_size = 128;
    let n_embd = 32;
    let n_layer = 1;
    let n_ctx = 64;
    let n_ff = 128;

    // Llama tensors
    let tensors = vec![
        ("token_embd.weight", Tensor::zeros(vec![vocab_size, n_embd], TypeId::Float32)),
        ("blk.0.attn_norm.weight", Tensor::new(vec![1.0f32; n_embd], vec![n_embd])),
        ("blk.0.attn_q.weight", Tensor::zeros(vec![n_embd, n_embd], TypeId::Float32)),
        ("blk.0.attn_k.weight", Tensor::zeros(vec![n_embd, n_embd], TypeId::Float32)),
        ("blk.0.attn_v.weight", Tensor::zeros(vec![n_embd, n_embd], TypeId::Float32)),
        ("blk.0.attn_output.weight", Tensor::zeros(vec![n_embd, n_embd], TypeId::Float32)),
        ("blk.0.ffn_norm.weight", Tensor::new(vec![1.0f32; n_embd], vec![n_embd])),
        ("blk.0.ffn_gate.weight", Tensor::zeros(vec![n_ff, n_embd], TypeId::Float32)),
        ("blk.0.ffn_up.weight", Tensor::zeros(vec![n_ff, n_embd], TypeId::Float32)),
        ("blk.0.ffn_down.weight", Tensor::zeros(vec![n_embd, n_ff], TypeId::Float32)),
        ("output_norm.weight", Tensor::new(vec![1.0f32; n_embd], vec![n_embd])),
        ("output.weight", Tensor::zeros(vec![vocab_size, n_embd], TypeId::Float32)),
    ];

    let file = File::create("validation/gguf/tiny_llama_ferric.gguf")?;
    let mut writer = GGUFWriter::new(file);

    let mut tokens = Vec::new();
    for i in 0..vocab_size { tokens.push(GGUFValue::String(format!("token_{}", i))); }

    writer.write_header(tensors.len() as u32, 10)?;
    writer.write_kv("general.architecture", GGUFValue::String("llama".to_string()))?;
    writer.write_kv("llama.block_count", GGUFValue::Uint32(n_layer as u32))?;
    writer.write_kv("llama.embedding_length", GGUFValue::Uint32(n_embd as u32))?;
    writer.write_kv("llama.head_count", GGUFValue::Uint32(2))?;
    writer.write_kv("llama.feed_forward_length", GGUFValue::Uint32(n_ff as u32))?;
    writer.write_kv("llama.attention.layer_norm_rms_epsilon", GGUFValue::Float32(1e-6))?;
    writer.write_kv("llama.context_length", GGUFValue::Uint32(n_ctx as u32))?;
    writer.write_kv("tokenizer.ggml.model", GGUFValue::String("llama".to_string()))?;
    writer.write_kv("tokenizer.ggml.pre", GGUFValue::String("llama-bpe".to_string()))?;
    writer.write_kv("tokenizer.ggml.tokens", GGUFValue::Array(tokens))?;

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
    println!("FerricML Llama GGUF exported.");
    Ok(())
}
