# Model Serialization: GGUF and SafeTensors

FerricML prioritizes safe, fast, and portable model serialization. We support the industry-standard GGUF format for inference and SafeTensors for secure weight storage.

## 1. GGUF (v3) Support

GGUF is a binary format designed for fast loading and saving of models, and for ease of reading. It is the primary format for FerricML models intended for production deployment.

### 1.1 Advantages
- **Fast Loading:** Designed to be mapped directly into memory.
- **Extensive Metadata:** Store everything from model hyperparameters to tokenizer configurations.
- **Quantization Friendly:** Built-in support for various quantization levels (Q4_K, Q8_0, etc.).

### 1.2 Saving to GGUF
```rust
use ferricml::serialization::gguf::GGUFWriter;
use std::fs::File;

let file = File::create("model.gguf")?;
let mut writer = GGUFWriter::new(file);

// Write header and tensors
writer.write_header(num_tensors, num_metadata)?;
writer.write_tensor("fc1.weight", &weight_tensor, offset)?;
```

## 2. SafeTensors

SafeTensors is a simple format for storing tensors that is specifically designed to be **safe**. Unlike Python's Pickle format, SafeTensors does not allow for arbitrary code execution during loading.

### 2.1 Why SafeTensors?
- **Security:** Zero risk of "Pickle-bombs" or remote code execution.
- **Zero-copy:** Data can be loaded directly into GPU or CPU memory without extra copies.

### 2.2 Usage
```rust
use ferricml::serialization::safetensors::save_file;

let mut tensors = HashMap::new();
tensors.insert("weight".to_string(), weight_tensor);

save_file(tensors, "model.safetensors")?;
```

## 3. Checkpointing in FerricML

During training, you can use the built-in Checkpointer to save the state of your model and optimizer.

```rust
let mut checkpointer = Checkpointer::new("checkpoints/");

// Save every 1000 steps
if step % 1000 == 0 {
    checkpointer.save(&model, &optimizer, step)?;
}
```

## 4. Interoperability

FerricML models can be converted to and from other framework formats using our CLI tools:

- `ferric-convert pytorch model.pth model.gguf`
- `ferric-convert tf saved_model/ model.safetensors`

## 5. Metadata Standards

FerricML enforces a strict metadata schema for GGUF files to ensure they are compatible with the broader ecosystem (e.g., `llama.cpp`):
- `general.architecture`: e.g., "llama", "resnet".
- `general.name`: Model name.
- `general.author`: Creator of the model.
- `general.license`: License terms.

---
*Back to [Getting Started](GETTING_STARTED.md).*
