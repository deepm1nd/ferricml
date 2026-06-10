# Validation and Numerical Parity

To ensure that FerricML is a reliable and accurate machine learning platform, we maintain a rigorous validation suite that compares FerricML's outputs against industry-standard implementations in PyTorch.

## 1. Methodology: The Parity Test

Our validation methodology is designed to eliminate any ambiguity regarding the correctness of our mathematical kernels and autograd engine.

### Step 1: Model Mirroring
For every supported architecture (e.g., ResNet, BERT, LeNet), we build two identical models:
1.  **Native Reference:** A model built using official PyTorch modules.
2.  **FerricML Implementation:** A model built using FerricML's `nn` modules and kernels.

### Step 2: Weight and Input Synchronization
To ensure a fair comparison, we do not initialize models randomly. Instead:
-   We generate a fixed set of weights and a "dummy" input tensor in Python.
-   These are exported to a standardized JSON format.
-   FerricML loads the exact same weights into its modules and processes the exact same input tensor.

### Step 3: Forward Pass Comparison
We execute the forward pass in both frameworks and calculate the **Maximum Absolute Error (MAE)** between the output tensors.
$$ \text{MAE} = \max(|Y_{pytorch} - Y_{ferricml}|) $$

### Step 4: Gradient Parity (Autograd Validation)
For models requiring training validation, we perform a backward pass in both frameworks and compare the resulting gradients for every learnable parameter.

## 2. Validation Coverage

The following models are part of our continuous validation suite:

| Model Category | Architectures Covered |
|----------------|-----------------------|
| **Computer Vision** | LeNet-5, ResNet-18, MobileNet-V2, UNet |
| **Natural Language** | BERT, NanoGPT, TinyLlama, LSTM |
| **Generative AI** | VAE (Variational Autoencoder) |
| **Graph ML** | GCN (Graph Convolutional Network) |

## 3. Running the Validation Suite

You can run the entire validation suite locally using the provided script:

```bash
./validate_all.sh
```

This script will:
1.  Execute the Python scripts in `validation/models/` to generate PyTorch reference data.
2.  Compile and run the FerricML validation bins in `crates/ferric-validation`.
3.  Report the maximum difference found for each model.

## 4. Current Results

The following table summarizes the logit-level verification results. All models were verified against native PyTorch implementations to ensure mathematical equivalence.

| Model Name | Architecture Type | Key Components Verified | Max Absolute Difference | Status |
|------------|-------------------|-------------------------|-------------------------|--------|
| **LeNet-5** | CNN | Conv2d, MaxPool2d, Linear, ReLU | 3.35e-08 | ✅ PASSED |
| **ResNet-18 Block** | CNN (Residual) | Residual Connections, BatchNorm2d | 2.38e-07 | ✅ PASSED |
| **MobileNetV2 Block** | CNN (Efficiency) | Depthwise Separable Conv, ReLU6 | 2.38e-07 | ✅ PASSED |
| **NanoGPT** | Transformer (Causal) | Causal Self-Attention, Masking, Softmax | < 1.0e-07* | ✅ PASSED |
| **TinyLlama** | Transformer (Modern) | RMSNorm, SwiGLU, GQA Logic | 2.38e-07 | ✅ PASSED |
| **BERT-Base** | Transformer (Encoder) | Bidirectional Attention, LayerNorm | < 1.0e-07* | ✅ PASSED |
| **Stable Diffusion U-Net** | Generative (Spatial) | Transposed Conv2d, Upsampling | 1.19e-07 | ✅ PASSED |
| **VAE** | Generative (Stochastic) | Reparameterization Trick, Exp, Mul | Verified | ✅ PASSED |
| **LSTM** | Recurrent | Gate Logic (i, f, g, o), Tanh, Sigmoid | Verified | ✅ PASSED |
| **GCN** | Graph | Sparse MatMul Equivalence, Adj Matrix | Verified | ✅ PASSED |

*\* Verified at the component level with high-precision logit matching.*

## 5. GGUF Binary Bridge Verification

FerricML includes a "Binary Bridge" for GGUF compatibility, which has been verified for:
- **Metadata:** Correct support for all standard KV pairs (arch, block_count, context_length, etc.).
- **Alignment:** Strict 32-byte alignment for tensor data to ensure zero-copy loading.
- **GGML Compatibility:** Automatic dimension reversing (e.g., PyTorch `[out, in]` correctly becomes GGML `[in, out]`).
