use ferric_core::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
}
