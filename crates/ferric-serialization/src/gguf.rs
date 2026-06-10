use ferric_core::{Storage, Tensor, TypeId};
use std::io::{Result, Write};

pub enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GGUFValue {
    fn type_id(&self) -> u32 {
        match self {
            GGUFValue::Uint8(_) => 0,
            GGUFValue::Int8(_) => 1,
            GGUFValue::Uint16(_) => 2,
            GGUFValue::Int16(_) => 3,
            GGUFValue::Uint32(_) => 4,
            GGUFValue::Int32(_) => 5,
            GGUFValue::Float32(_) => 6,
            GGUFValue::Bool(_) => 7,
            GGUFValue::String(_) => 8,
            GGUFValue::Array(_) => 9,
            GGUFValue::Uint64(_) => 10,
            GGUFValue::Int64(_) => 11,
            GGUFValue::Float64(_) => 12,
        }
    }

    fn write<W: Write>(&self, writer: &mut GGUFWriter<W>) -> Result<()> {
        match self {
            GGUFValue::Uint8(v) => writer.write_all(&v.to_le_bytes()),
            GGUFValue::Int8(v) => writer.write_all(&v.to_le_bytes()),
            GGUFValue::Uint16(v) => writer.write_all(&v.to_le_bytes()),
            GGUFValue::Int16(v) => writer.write_all(&v.to_le_bytes()),
            GGUFValue::Uint32(v) => writer.write_all(&v.to_le_bytes()),
            GGUFValue::Int32(v) => writer.write_all(&v.to_le_bytes()),
            GGUFValue::Float32(v) => writer.write_all(&v.to_le_bytes()),
            GGUFValue::Bool(v) => writer.write_all(&[*v as u8]),
            GGUFValue::String(v) => {
                let len = v.len() as u64;
                writer.write_all(&len.to_le_bytes())?;
                writer.write_all(v.as_bytes())
            }
            GGUFValue::Array(v) => {
                let type_id = if v.is_empty() { 0 } else { v[0].type_id() };
                writer.write_all(&type_id.to_le_bytes())?;
                let len = v.len() as u64;
                writer.write_all(&len.to_le_bytes())?;
                for item in v {
                    item.write(writer)?;
                }
                Ok(())
            }
            GGUFValue::Uint64(v) => writer.write_all(&v.to_le_bytes()),
            GGUFValue::Int64(v) => writer.write_all(&v.to_le_bytes()),
            GGUFValue::Float64(v) => writer.write_all(&v.to_le_bytes()),
        }
    }
}

pub struct GGUFWriter<W: Write> {
    writer: W,
    current_pos: u64,
}

impl<W: Write> GGUFWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer, current_pos: 0 }
    }

    pub fn write_all(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data)?;
        self.current_pos += data.len() as u64;
        Ok(())
    }

    pub fn write_header(&mut self, tensor_count: u32, kv_count: u32) -> Result<()> {
        self.write_all(b"GGUF")?; // Magic
        self.write_all(&3u32.to_le_bytes())?; // Version
        self.write_all(&(tensor_count as u64).to_le_bytes())?;
        self.write_all(&(kv_count as u64).to_le_bytes())?;
        Ok(())
    }

    pub fn write_kv(&mut self, key: &str, value: GGUFValue) -> Result<()> {
        let key_len = key.len() as u64;
        self.write_all(&key_len.to_le_bytes())?;
        self.write_all(key.as_bytes())?;
        self.write_all(&value.type_id().to_le_bytes())?;
        value.write(self)
    }

    pub fn write_tensor(&mut self, name: &str, tensor: &Tensor, offset: u64) -> Result<()> {
        let name_len = name.len() as u64;
        self.write_all(&name_len.to_le_bytes())?;
        self.write_all(name.as_bytes())?;

        let ndim = tensor.shape().ndim() as u32;
        self.write_all(&ndim.to_le_bytes())?;
        for &dim in tensor.shape().dims() {
            self.write_all(&(dim as u64).to_le_bytes())?;
        }

        let dtype = match tensor.dtype() {
            TypeId::Float32 => 0u32, // GGUF_TYPE_F32
            TypeId::Float16 => 1u32, // GGUF_TYPE_F16
            _ => 2u32,               // Others
        };
        self.write_all(&dtype.to_le_bytes())?;
        self.write_all(&offset.to_le_bytes())?;

        Ok(())
    }

    pub fn align(&mut self, alignment: u64) -> Result<()> {
        let padding = (alignment - (self.current_pos % alignment)) % alignment;
        if padding > 0 {
            self.write_all(&vec![0u8; padding as usize])?;
        }
        Ok(())
    }

    pub fn write_tensor_data(&mut self, tensor: &Tensor) -> Result<()> {
        let binding = tensor.storage();
        let data = match binding.as_ref() {
            Storage::Cpu(s) => &s.data[..s.len],
        };
        self.write_all(data)?;
        self.align(32)?;
        Ok(())
    }
}
