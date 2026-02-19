from safetensors.torch import load_file, save_file
from pathlib import Path
import os

model_path = './model_KD_v6_GPTQ'
safetensors_file = os.path.join(model_path, "model.safetensors")

# 다시 'model.' 접두어를 붙여서 Transformers 호환용으로 변경
weights = load_file(safetensors_file, device="cpu")
fixed_weights = {}

for k, v in weights.items():
    # 이미 'layers'로 시작한다면 'model.'을 붙여줌
    if k.startswith("layers") or k.startswith("embed_tokens") or k.startswith("norm"):
        new_key = f"model.{k}"
    else:
        new_key = k
    fixed_weights[new_key] = v

save_file(fixed_weights, safetensors_file)
print("✅ Transformers 로딩용으로 가중치 키 복구 완료")