from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

import os
import torch
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "./KD_student_model/checkpoint-768"     
OUT_DIR  = f"./KD_AWQ_W4A16_model"          

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# Quantization
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE  = ["embed_tokens", "lm_head"]

def main():

    print("[INFO] 모델 로드 중...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
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

    os.makedirs(OUT_DIR, exist_ok=True)

    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

    zip_name = "KD_AWQ_W4A16_submit"
    print(f"[INFO] {zip_name}.zip 생성 중...")

    shutil.make_archive(
        base_name=zip_name,
        format="zip",
        root_dir=".",
        base_dir=OUT_DIR,
    )

    print(f"[INFO] 생성 완료: {zip_name}.zip")
    
if __name__ == "__main__":
    main()
