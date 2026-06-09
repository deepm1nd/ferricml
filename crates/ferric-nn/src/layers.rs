use crate::module::Module;
use ferric_core::Tensor;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Tensor::new(
            vec![0.0f32; in_features * out_features],
            vec![in_features, out_features],
        );
        let bias = Some(Tensor::new(vec![0.0f32; out_features], vec![out_features]));
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = ferric_backend_cpu::ops::matmul(input, &self.weight);
        if let Some(ref bias) = self.bias {
            out = ferric_backend_cpu::ops::add(&out, bias);
        }
        out
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}
