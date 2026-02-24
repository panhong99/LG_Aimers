import argparse
import json
import os
import re
import struct
import sys


def read_safetensors_keys(model_path: str) -> list[str]:
    with open(model_path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError("invalid safetensors file: missing header length")

        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError("invalid safetensors file: incomplete header")

    header = json.loads(header_bytes.decode("utf-8"))
    return [k for k in header.keys() if k != "__metadata__"]


def detect_layers(model_path: str) -> tuple[int, list[int]]:
    max_layer = -1
    layer_set = set()

    for key in read_safetensors_keys(model_path):
        match = re.search(r"model\.layers\.(\d+)\.", key)
        if match:
            idx = int(match.group(1))
            layer_set.add(idx)
            if idx > max_layer:
                max_layer = idx

    return max_layer, sorted(layer_set)


def detect_config_layers(model_dir: str) -> int:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        return int(cfg.num_hidden_layers)
    except Exception:
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg["num_hidden_layers"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check layer indices in a model.safetensors file."
    )
    parser.add_argument(
        "--model-dir",
        default=".",
        help="Directory containing model.safetensors (default: current directory)",
    )
    parser.add_argument(
        "--model-file",
        default="model.safetensors",
        help="Safetensors filename (default: model.safetensors)",
    )
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model_file)

    print("checking:", model_path)

    if not os.path.exists(model_path):
        print("error: model file not found")
        return 1

    file_size_mb = os.path.getsize(model_path) / 1024 / 1024
    print("file size (MB):", round(file_size_mb, 2))

    max_layer, layer_indices = detect_layers(model_path)
    detected_layer_count = len(layer_indices)
    config_layer_count = detect_config_layers(args.model_dir)

    print("max layer index:", max_layer)
    print("detected layer count:", detected_layer_count)
    print("layer indices:", layer_indices)
    print("config num_hidden_layers:", config_layer_count)

    if detected_layer_count == config_layer_count:
        print("match: config and safetensors layer counts are identical")
    else:
        print("mismatch: config and safetensors layer counts are different")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
