use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Shape mismatch: expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("Type mismatch: expected {expected:?}, found {found:?}")]
    TypeMismatch { expected: String, found: String },
    #[error("Device mismatch: expected {expected:?}, found {found:?}")]
    DeviceMismatch { expected: String, found: String },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
