#!/bin/bash
set -e
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision transformers diffusers numpy safetensors gguf
echo "Setting up llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp
cmake -B build -DGGML_NATIVE=OFF
cmake --build build --config Release -j $(nproc) --target llama-cli llama-perplexity
cd ..
echo "Environment setup complete."
