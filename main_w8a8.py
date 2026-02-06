from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot

import os
import torch
import shutil
from pathlib import Path
import argparse

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
        args.dataset_id,
        split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
    )

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False)
        }

    ds = ds.map(preprocess)

    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
    )

    # 3) Apply quantization and save in compressed-tensors format.
    oneshot(
        model=model,
        recipe=recipe,
        tokenizer=tokenizer,
    )
    os.makedirs(args.out_dir, exist_ok=True)

    model.save_pretrained(args.out_dir, save_compressed=True)
    tokenizer.save_pretrained(args.out_dir)

    print(f"[INFO] 모델 저장 완료: {args.out_dir}")

    zip_name = "W8A8_submit"
    print(f"[INFO] {zip_name}.zip 생성 중...")

    shutil.make_archive(
        base_name=zip_name,
        format="zip",
        root_dir=".",
        base_dir=args.out_dir,
    )

    print(f"[INFO] 생성 완료: {zip_name}.zip")
    
if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('--model_id', type=str, default="./base_model", help='Path to the base model')
    parse.add_argument('--out_dir', type=str, default="./W8A8_model", help='Output directory for the quantized model')
    parse.add_argument('--dataset_id', type=str, default="LGAI-EXAONE/MANTA-1M", help='Dataset identifier')
    parse.add_argument('--dataset_split', type=str, default="train", help='Dataset split to use')
    parse.add_argument('--num_calibration_samples', type=int, default=256, help='Number of calibration samples')
    parse.add_argument('--max_sequence_length', type=int, default=512, help='Maximum sequence length for tokenization')
    args = parse.parse_args()

    main(args)
