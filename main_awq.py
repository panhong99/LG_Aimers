from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

import os
import torch
import shutil
from pathlib import Path
import json

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
import argparse

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# Quantization
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE  = ["embed_tokens", "lm_head"]

def main(args):

    print("[INFO] 모델 로드 중...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )

    print("[INFO] 모델/토크나이저 로드 완료")

    print("[INFO] 캘리브레이션 데이터 로드 중...")

    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    )

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False)
        }

    ds = ds.map(preprocess)

    print("[INFO] 데이터 전처리 완료")

    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            scheme="W4A16",
            targets=["Linear"],
            duo_scaling=True,
        ),
    ]
    print(recipe)

    # Apply algorithms.
    print("[INFO] AWQ W4A16 양자화 진행 중...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    model.save_pretrained(args.out_dir, save_compressed=True)
    tokenizer.save_pretrained(args.out_dir)
    
    # 양자화 정보 명시적 저장 (Marlin 커널이 감지하기 위함)
    import json
    config_path = Path(args.out_dir) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 양자화 config 추가 (없으면)
        if 'quantization_config' not in config:
            config['quantization_config'] = {
                'quant_method': 'awq',
                'w_bit': 4,
                'a_bit': 16,
                'scheme': 'W4A16'
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print("[INFO] 양자화 정보 저장 완료")

    print(f"[INFO] 모델 저장 완료: {args.out_dir}")

    zip_name = args.out_dir
    print(f"[INFO] {zip_name}.zip 생성 중...")

    shutil.make_archive(
        base_name=zip_name,
        format="zip",
        root_dir=".",
        base_dir=args.out_dir,
    )

    print(f"[INFO] 생성 완료: {zip_name}.zip")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWQ W4A16 양자화")
    parser.add_argument("--model_id", type=str, required=True, help="양자화할 모델 경로")
    parser.add_argument("--out_dir", type=str, default="./KD_AWQ_W4A16_model", help="양자화된 모델 저장 경로")
    args = parser.parse_args()
    main(args)  