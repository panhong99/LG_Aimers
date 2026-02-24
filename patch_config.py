import argparse
import json
import os
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="directory containing config.json"
    )
    args = parser.parse_args()

    cfg_path = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)

    bak_path = cfg_path + ".bak"
    if not os.path.exists(bak_path):
        shutil.copy2(cfg_path, bak_path)
        print(f"[INFO] backup written: {bak_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config")
    if qc is None:
        raise ValueError("quantization_config not found in config.json")

    try:
        weights = qc["config_groups"]["group_0"]["weights"]
        for bad in ("scale_dtype", "zp_dtype"):
            if bad in weights:
                weights.pop(bad)
                print(f"[INFO] removed weights.{bad}")
    except Exception as e:
        print("[WARN] couldn't clean weights keys:", e)

    group0 = qc["config_groups"]["group_0"]
    new_group0 = {
        "targets": group0.get("targets", ["Linear"]),
        "format": group0.get("format", qc.get("format", "pack-quantized")),
        "weights": group0.get("weights", {}),
        "input_activations": group0.get("input_activations"),
        "output_activations": group0.get("output_activations"),
    }

    if isinstance(new_group0["weights"], dict):
        new_group0["weights"].pop("scale_dtype", None)
        new_group0["weights"].pop("zp_dtype", None)

    qc["config_groups"] = {"group_0": new_group0}
    cfg["quantization_config"] = qc

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[OK] patched: {cfg_path}")


if __name__ == "__main__":
    main()
