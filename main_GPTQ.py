import os
import torch
import shutil
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =============================================================================
# 1. 설정 및 경로
# =============================================================================
MODEL_ID = "./models/trainer_output_v6"     
OUT_DIR  = "./model_KD_v6_GPTQ_v2"          

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# [보완] 샘플 수를 256 -> 512로 늘려 정밀도 향상
NUM_CALIBRATION_SAMPLES = 256 
# [보완] 학습 시 설정했던 max_seq_length와 동일하게 맞춤
MAX_SEQUENCE_LENGTH = 512

print("[INFO] 모델 및 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

# =============================================================================
# 2. 전처리: 학습 시 사용했던 포맷 강제 적용
# =============================================================================
print("[INFO] 캘리브레이션 데이터 전처리 중...")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    # ★ 핵심: 학습(KD) 시 사용했던 문자열 포맷과 토씨 하나 안 틀리고 똑같이 맞춰야 합니다.
    messages = example["conversations"]
    user_content = messages[0]['content']
    # 학습 때 300자 트림을 했다면 여기서도 동일하게 적용하는 것이 좋습니다.
    assistant_content = messages[1]['content'][:300] if len(messages[1]['content']) > 300 else messages[1]['content']
    
    full_prompt = f"### User: {user_content}\n### Assistant: {assistant_content}{tokenizer.eos_token}\n"
    
    return {"text": full_prompt}

ds = ds.map(preprocess, remove_columns=ds.column_names)

# =============================================================================
# 3. GPTQ 실행 설정
# =============================================================================
print(f"[INFO] GPTQ 진행 (Samples: {NUM_CALIBRATION_SAMPLES}, Max Len: {MAX_SEQUENCE_LENGTH})...")

recipe = [
    GPTQModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        # [추가] 가중치 업데이트 시의 감쇠율을 조절하여 급격한 변화 방지
        dampening_frac=0.01, 
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# =============================================================================
# 4. 저장 및 vLLM 호환성 보정
# =============================================================================
print("[INFO] 양자화 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# [보완] 가중치 키 정규화: vLLM과 Transformers 호환성 보장
def normalize_safetensors_keys(output_dir):
    """
    safetensors 파일의 가중치 키를 정규화:
    - "model.model." 중복 제거
    - 최종 형식: "model.layers.*" 등으로 통일
    """
    from safetensors.torch import load_file, save_file
    
    for safetensors_file in Path(output_dir).glob("*.safetensors"):
        print(f"  정규화 중: {safetensors_file.name}")
        weights = load_file(str(safetensors_file), device="cpu")
        fixed_weights = {}
        
        for k, v in weights.items():
            # "model.model.*" 형태면 "model." 하나만 남기기
            if k.startswith("model.model."):
                new_key = k.replace("model.model.", "model.", 1)
            else:
                new_key = k
            fixed_weights[new_key] = v
        
        save_file(fixed_weights, str(safetensors_file))
        print(f"    ✓ 완료 (총 {len(fixed_weights)} 가중치)")

normalize_safetensors_keys(OUT_DIR)
print(f"[INFO] 모델 저장 및 가중치 키 정규화 완료: {OUT_DIR}")