use crate::device::Device;
use crate::dtype::{DType, TypeId};
use crate::shape::{Shape, Strides};
use crate::storage::{CpuStorage, Storage};
use std::sync::{Arc, Mutex};

pub struct Tensor {
    storage: Arc<Storage>,
    shape: Shape,
    strides: Strides,
    offset: usize,
    dtype: TypeId,
    pub grad: Option<Arc<Mutex<Tensor>>>,
}

impl Tensor {
    pub fn new<T: DType + bytemuck::Pod>(data: Vec<T>, shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let strides = Strides::contiguous(&shape);
        let storage = Arc::new(Storage::Cpu(CpuStorage::from_vec(data)));
        Self {
            storage,
            shape,
            strides,
            offset: 0,
            dtype: T::TYPE_ID,
            grad: None,
        }
    }

    pub fn zeros(shape: impl Into<Shape>, dtype: TypeId) -> Self {
        let shape = shape.into();
        let numel = shape.numel();
        let data = match dtype {
            TypeId::Float32 => bytemuck::allocation::cast_vec(vec![0.0f32; numel]),
            _ => unimplemented!("Zeros for {:?}", dtype),
        };
        let storage = Arc::new(Storage::Cpu(CpuStorage { data, len: numel * 4 }));
        Self {
            storage,
            shape: shape.clone(),
            strides: Strides::contiguous(&shape),
            offset: 0,
            dtype,
            grad: None,
        }
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if requires_grad {
            if self.grad.is_none() {
                self.grad = Some(Arc::new(Mutex::new(Self::zeros(self.shape.clone(), self.dtype))));
            }
        } else {
            self.grad = None;
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> TypeId {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn storage(&self) -> Arc<Storage> {
        self.storage.clone()
    }
}
