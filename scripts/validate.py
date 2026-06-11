import torch
import numpy as np
import os
import subprocess
import json
import argparse
from typing import Dict, Any

class Validator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_dir = f"validation/{model_name}"
        os.makedirs(self.base_dir, exist_ok=True)
        self.native_output_path = f"{self.base_dir}/native_output.json"
        self.ferric_output_path = f"{self.base_dir}/ferric_output.json"
        self.gguf_dir = "validation/gguf"
        os.makedirs(self.gguf_dir, exist_ok=True)

    def save_native_output(self, data: Dict[str, Any]):
        with open(self.native_output_path, 'w') as f:
            json.dump(self.convert_to_serializable(data), f)

    def convert_to_serializable(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_to_serializable(i) for i in obj]
        return obj

    def run_llama_perplexity(self, model_path: str, data_path: str):
        cmd = ["./llama.cpp/build/bin/llama-perplexity", "-m", model_path, "-f", data_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    def compare_outputs(self, native: Any, ferric: Any, tolerance: float = 1e-5):
        native_arr = np.array(native)
        ferric_arr = np.array(ferric)
        diff = np.abs(native_arr - ferric_arr)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        return {
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "passed": bool(max_diff < tolerance)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--native_path", type=str, required=True)
    parser.add_argument("--ferric_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.native_path, 'r') as f:
        native_data = json.load(f)
    with open(args.ferric_path, 'r') as f:
        ferric_data = json.load(f)

    native_out = np.array(native_data["output"]).flatten()
    ferric_out = np.array(ferric_data["output"]).flatten()

    diff = np.abs(native_out - ferric_out)
    max_diff = np.max(diff)
    mae = np.mean(diff)

    print(f"RESULTS_START:{args.model}")
    print(f"MAE: {mae:.12f}")
    print(f"MAX_DIFF: {max_diff:.12f}")
    print(f"RESULTS_END:{args.model}")
