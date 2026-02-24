import json
from pathlib import Path

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# 1. 설정 및 경로
# =============================================================================
MODEL_ID = "./base_model/base_model"
OUT_DIR = "./model_KD_v6_NVFP4"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512


def validate_nvfp4_artifacts(output_dir: str) -> None:
    out_path = Path(output_dir)
    config_path = out_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json이 없습니다: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "compressed-tensors":
        raise RuntimeError(
            "quantization_config.quant_method가 compressed-tensors가 아닙니다."
        )

    config_groups = qc.get("config_groups")
    if not isinstance(config_groups, dict) or len(config_groups) == 0:
        raise RuntimeError(
            "quantization_config.config_groups가 비어 있습니다. "
            "NVFP4 압축이 실제로 적용되지 않은 모델입니다."
        )

    safetensor_files = sorted(out_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError("safetensors 가중치 파일을 찾지 못했습니다.")

    has_weight_packed = False
    for st_file in safetensor_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as sf:
            if any(name.endswith("weight_packed") for name in sf.keys()):
                has_weight_packed = True
                break

    if not has_weight_packed:
        raise RuntimeError(
            "압축 텐서(weight_packed)가 없습니다. 양자화가 실패했거나 저장 방식이 잘못되었습니다."
        )

    print("[INFO] ✓ NVFP4 산출물 검증 완료")


print("[INFO] 모델 및 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# =============================================================================
# 2. Weight tying 복구
# =============================================================================
print("[INFO] Weight tying 복구 중...")
if hasattr(model, "tie_weights"):
    model.tie_weights()

# =============================================================================
# 3. 캘리브레이션 데이터 준비
# =============================================================================
print("[INFO] 캘리브레이션 데이터 전처리 중...")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess(example):
    messages = example["conversations"]
    user_content = messages[0]["content"]
    assistant_content = (
        messages[1]["content"][:300]
        if len(messages[1]["content"]) > 300
        else messages[1]["content"]
    )
    full_prompt = (
        f"### User: {user_content}\n"
        f"### Assistant: {assistant_content}{tokenizer.eos_token}\n"
    )
    return {"text": full_prompt}


ds = ds.map(preprocess, remove_columns=ds.column_names)

# =============================================================================
# 4. NVFP4 양자화
# =============================================================================
print(
    f"[INFO] NVFP4 양자화 진행 (Samples: {NUM_CALIBRATION_SAMPLES}, "
    f"Max Len: {MAX_SEQUENCE_LENGTH})..."
)

recipe = QuantizationModifier(
    targets=["Linear"],
    ignore=["embed_tokens", "lm_head"],
    scheme="NVFP4",
)

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=ds,
    recipe=recipe,
    output_dir=OUT_DIR,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
)

# 환경/버전별로 processor 저장이 누락되는 경우가 있어 tokenizer를 한 번 더 저장
tokenizer.save_pretrained(OUT_DIR)

# =============================================================================
# 5. 결과 검증
# =============================================================================
validate_nvfp4_artifacts(OUT_DIR)
print(f"[INFO] 모델 저장 및 검증 완료: {OUT_DIR}")
