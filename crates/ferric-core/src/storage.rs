use crate::device::Device;

pub enum Storage {
    Cpu(CpuStorage),
}

pub struct CpuStorage {
    pub data: Vec<u8>,
    pub len: usize,
}

impl CpuStorage {
    pub fn from_vec<T: bytemuck::NoUninit + bytemuck::AnyBitPattern>(data: Vec<T>) -> Self {
        let len = data.len() * std::mem::size_of::<T>();
        let data = bytemuck::pod_collect_to_vec(&data);
        Self { data, len }
    }

    pub fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        bytemuck::cast_slice(&self.data[..self.len])
    }

    pub fn as_mut_slice<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        bytemuck::cast_slice_mut(&mut self.data[..self.len])
    }
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
        }
    }
}
