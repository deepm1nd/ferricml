#!/bin/bash
set -e

export PYTHONPATH=$PYTHONPATH:$(pwd)/scripts

echo "Running Validation Suite..."

MODELS=("lenet5" "nanogpt" "resnet" "tinyllama" "mobilenet" "bert" "vae" "lstm" "gcn" "unet")

for model in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Validating $model..."
    python3 validation/models/${model}_native.py
    cargo run -p ferric-validation --bin ${model}_ferric
done

echo "----------------------------------------"
echo "All validations complete."
