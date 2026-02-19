import os
import torch
import shutil
import argparse
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from safetensors.torch import load_file, save_file

# =============================================================================
# 1. 설정 및 상수
# =============================================================================
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

def main(args):
    print(f"[INFO] 모델 로드 중: {args.model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" # 양자화 속도를 위해 GPU 자동 할당
    )

    print("[INFO] 캘리브레이션 데이터 준비 중...")
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

    def preprocess(example):
        # 학습 시 사용한 포맷과 100% 일치시켜야 양자화 오차가 줄어듭니다.
        messages = example["conversations"]
        prompt = f"### User: {messages[0]['content']}\n### Assistant: {messages[1]['content']}"
        return {"text": prompt}

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    # =============================================================================
    # 2. AWQ 양자화 설정 (Recipe)
    # =============================================================================
    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            scheme="W4A16",
            targets=["Linear"],
            duo_scaling=False, # 프루닝된 모델의 안정성을 위해 False 설정
        ),
    ]

    print("[INFO] AWQ W4A16 양자화 진행 중 (One-shot)...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # =============================================================================
    # 3. 저장 및 vLLM 호환성 보정 (Key Remapping)
    # =============================================================================
    # 출력 경로 설정 (v6_awq 형태)
    base_name = Path(args.model_id).name
    out_path = Path(args.out_dir) / f"awq_{base_name}"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 모델 저장 중 (Compressed): {out_path}")
    # save_compressed=True를 통해 weight_packed 생성을 유도합니다.
    model.save_pretrained(out_path, save_compressed=True)
    tokenizer.save_pretrained(out_path)

    # [핵심] vLLM KeyError 방지를 위한 가중치 키 이름 보정
    print("[INFO] vLLM 호환성을 위한 가중치 키 보정 시작...")
    for safetensors_file in out_path.glob("*.safetensors"):
        weights = load_file(str(safetensors_file), device="cpu")
        fixed_weights = {}
        for key, value in weights.items():
            # 'model.' 접두어가 있으면 제거, 없으면 유지
            new_key = key.replace("model.", "") if key.startswith("model.") else key
            fixed_weights[new_key] = value
        
        # 보정된 키로 파일 덮어쓰기
        save_file(fixed_weights, str(safetensors_file))
        print(f"  - {safetensors_file.name}: 접두어 제거 및 재저장 완료")

    # [중요] config.json에 양자화 정보 명시
    config_path = out_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config['quantization_config'] = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "version": "gemm" # vLLM Marlin 에러 방지용
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("[INFO] config.json 양자화 정보 업데이트 완료")

    # =============================================================================
    # 4. 압축 및 마무리
    # =============================================================================
    print(f"[INFO] 모든 작업 완료: {out_path}")
    
    # Zip 아카이브 생성
    shutil.make_archive(str(out_path), 'zip', root_dir=out_path.parent, base_dir=out_path.name)
    print(f"[INFO] 압축 파일 생성 완료: {out_path}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXAONE-Pruned AWQ Quantization for vLLM")
    parser.add_argument("--model_id", type=str, required=True, help="프루닝된 모델 경로 (e.g., ./trainer_output_v6)")
    parser.add_argument("--out_dir", type=str, default="./quantized_models", help="최종 모델 저장 루트 폴더")
    args = parser.parse_args()
    main(args)