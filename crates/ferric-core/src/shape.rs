use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    dims: SmallVec<[usize; 4]>,
}

impl Shape {
    pub fn new(dims: impl Into<SmallVec<[usize; 4]>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(SmallVec::from_slice(dims))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(SmallVec::from_vec(dims))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Strides {
    strides: SmallVec<[isize; 4]>,
}

impl Strides {
    pub fn contiguous(shape: &Shape) -> Self {
        let mut strides = vec![0isize; shape.ndim()];
        let mut stride = 1isize;
        for i in (0..shape.ndim()).rev() {
            strides[i] = stride;
            stride *= shape.dims()[i] as isize;
        }
        Self {
            strides: SmallVec::from_vec(strides),
        }
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }
}
