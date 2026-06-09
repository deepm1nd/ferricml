use ferric_core::{Storage, Tensor, TypeId};
use std::io::{Result, Write};

pub struct GGUFWriter<W: Write> {
    writer: W,
}

impl<W: Write> GGUFWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn write_header(&mut self, tensor_count: u32, kv_count: u32) -> Result<()> {
        self.writer.write_all(b"GGUF")?; // Magic
        self.writer.write_all(&3u32.to_le_bytes())?; // Version
        self.writer
            .write_all(&(tensor_count as u64).to_le_bytes())?;
        self.writer.write_all(&(kv_count as u64).to_le_bytes())?;
        Ok(())
    }

    pub fn write_tensor(&mut self, name: &str, tensor: &Tensor, offset: u64) -> Result<()> {
        let name_len = name.len() as u64;
        self.writer.write_all(&name_len.to_le_bytes())?;
        self.writer.write_all(name.as_bytes())?;

        let ndim = tensor.shape().ndim() as u32;
        self.writer.write_all(&ndim.to_le_bytes())?;
        for &dim in tensor.shape().dims() {
            self.writer.write_all(&(dim as u64).to_le_bytes())?;
        }

        let dtype = match tensor.dtype() {
            TypeId::Float32 => 0u32, // GGUF_TYPE_F32
            TypeId::Float16 => 1u32, // GGUF_TYPE_F16
            _ => 2u32,               // Others
        };
        self.writer.write_all(&dtype.to_le_bytes())?;

        self.writer.write_all(&offset.to_le_bytes())?;

        Ok(())
    }

    pub fn write_tensor_data(&mut self, tensor: &Tensor) -> Result<()> {
        let binding = tensor.storage();
        let data = match binding.as_ref() {
            Storage::Cpu(s) => &s.data[..s.len],
        };
        self.writer.write_all(data)?;
        Ok(())
    }
}
