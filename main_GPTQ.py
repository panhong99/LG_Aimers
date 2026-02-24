import os
import torch
import shutil
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
# =============================================================================
# 1. 설정 및 경로
# =============================================================================
<<<<<<< HEAD
MODEL_ID = "./models/trainer_output_v6"     
OUT_DIR  = "./model_KD_v6_GPTQ_NVFP4"          
=======
MODEL_ID = "./trainer_output_v6"
OUT_DIR  = "./model_KD_v6_GPTQ_NVFP4"
>>>>>>> 4108062 (add NVFP4)

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

print("[INFO] 모델 및 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

# =============================================================================
# [추가] 2. 양자화 전에 weight tying 복구/강제
# =============================================================================
print("[INFO] (필수) 양자화 전 weight tying 복구 중...")

# 1) tie_weights 호출 (config tie_word_embeddings=true 기반)
#    - 일부 모델은 이 호출로 lm_head와 embed_tokens 공유를 복구할 수 있음
if hasattr(model, "tie_weights"):
    model.tie_weights()

# =============================================================================
# 3. 전처리: 학습 시 사용했던 포맷 강제 적용
# =============================================================================
print("[INFO] 캘리브레이션 데이터 전처리 중...")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    messages = example["conversations"]
    user_content = messages[0]["content"]
    assistant_content = (
        messages[1]["content"][:300] if len(messages[1]["content"]) > 300 else messages[1]["content"]
    )

    full_prompt = f"### User: {user_content}\n### Assistant: {assistant_content}{tokenizer.eos_token}\n"
    return {"text": full_prompt}

ds = ds.map(preprocess, remove_columns=ds.column_names)

# =============================================================================
# 4. GPTQ 양자화 실행
# =============================================================================
print(f"[INFO] GPTQ W4A16 진행 (Samples: {NUM_CALIBRATION_SAMPLES}, Max Len: {MAX_SEQUENCE_LENGTH})...")

<<<<<<< HEAD
recipe = [
    GPTQModifier(
        scheme="NVFP4",
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        # [추가] 가중치 업데이트 시의 감쇠율을 조절하여 급격한 변화 방지
        dampening_frac=0.01, 
    )
]
=======
# YAML 포맷 recipe
recipe_yaml = """
default_stage:
  default_modifiers:
    - GPTQModifier:
        targets: [Linear]
        ignore: [embed_tokens, lm_head]
        scheme: W4A16
        block_size: 128
        dampening_frac: 0.01
        actorder: static
        offload_hessians: false
"""
>>>>>>> 4108062 (add NVFP4)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe_yaml,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# =============================================================================
# 5. 저장 및 정규화
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
    - tie_word_embeddings 설정 확인
    """
    from safetensors.torch import load_file, save_file
    
    config_path = Path(output_dir) / "config.json"
    tie_embeddings = False
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        tie_embeddings = cfg.get("tie_word_embeddings", False)

    for safetensors_file in Path(output_dir).glob("*.safetensors"):
        print(f"  정규화 중: {safetensors_file.name}")
        weights = load_file(str(safetensors_file), device="cpu")
        fixed_weights = {}

        for k, v in weights.items():
            # tie_word_embeddings=True면 lm_head.weight 제거
            if tie_embeddings and k == "lm_head.weight":
                print(f"    ✗ 제거: {k} (tie_word_embeddings=True)")
                continue
            # "model.model.*" 형태면 "model." 하나만 남기기
            if k.startswith("model.model."):
                new_key = k.replace("model.model.", "model.", 1)
            else:
                new_key = k
            fixed_weights[new_key] = v

        save_file(fixed_weights, str(safetensors_file))
        print(f"    ✓ 완료 (총 {len(fixed_weights)} 가중치)")
    
    # config.json ignore 확인
    if tie_embeddings and config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config", {})
        ignore_list = qc.get("ignore", [])
        if "lm_head" not in ignore_list:
            ignore_list.append("lm_head")
            qc["ignore"] = ignore_list
            cfg["quantization_config"] = qc
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print("    ✓ config.json ignore에 lm_head 추가")

normalize_safetensors_keys(OUT_DIR)
print(f"[INFO] 모델 저장 및 가중치 키 정규화 완료: {OUT_DIR}")