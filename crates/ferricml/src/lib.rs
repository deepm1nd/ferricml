pub use ferric_autograd as autograd;
pub use ferric_backend_cpu as cpu;
pub use ferric_core as core;
pub use ferric_ir as ir;
pub use ferric_nn as nn;
pub use ferric_serialization as serialization;

pub mod prelude {
    pub use ferric_core::*;
    pub use ferric_nn::*;
}
