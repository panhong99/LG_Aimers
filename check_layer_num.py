import torch
from safetensors.torch import load_file
from pathlib import Path

# 양자화된 모델 폴더 직접 지정
model_dir = Path("./awq_trainer_output_v6")

print(f"[INFO] 폴더 체크: {model_dir}")
print(f"[INFO] 폴더 존재: {model_dir.exists()}")

if not model_dir.exists():
    print(f"❌ 폴더를 찾을 수 없습니다!")
    exit(1)

# 폴더 내 모든 .safetensors 파일 찾기
safetensors_files = list(model_dir.glob("*.safetensors"))
print(f"\n폴더 내 .safetensors 파일:")
for f in safetensors_files:
    print(f"  - {f.name}")

if not safetensors_files:
    print(f"\n❌ 파일을 찾을 수 없습니다!")
    print(f"폴더 내용:")
    for f in model_dir.iterdir():
        print(f"  - {f.name}")
    exit(1)

# 첫 번째 safetensors 파일 로드
file_path = safetensors_files[0]
print(f"\n✅ 파일 로드 중: {file_path}")

try:
    weights = load_file(str(file_path))
    print(f"✅ 로드 성공! 총 {len(weights)} 개의 키")
except Exception as e:
    print(f"❌ 로드 실패: {e}")
    exit(1)

# 레이어 0번과 관련된 키값들이 어떻게 생겼는지 출력
print("\n--- Layer 0 Keys (처음 5개) ---")
layer0_keys = [k for k in weights.keys() if "layers.0." in k]
for key in layer0_keys[:5]:
    print(f"  {key}")

# 마지막 레이어 번호가 몇 번으로 되어 있는지 확인
try:
    # model.layers.X.mlp... 형식이므로 [2]가 레이어 번호
    all_layers = sorted(set([int(k.split('.')[2]) for k in weights.keys() if "layers." in k]))
    print(f"\n--- Layer Numbers: {all_layers} ---")
    print(f"✅ 총 레이어: {len(all_layers)}개")
    print(f"✅ 첫 번째 레이어: {min(all_layers)}, 마지막 레이어: {max(all_layers)}")
    
    # 양자화 상태 확인
    print(f"\n--- AWQ 양자화 확인 ---")
    has_packed = any("weight_packed" in k for k in weights.keys())
    has_scale = any("weight_scale" in k for k in weights.keys())
    print(f"  weight_packed: {'✅' if has_packed else '❌'}")
    print(f"  weight_scale: {'✅' if has_scale else '❌'}")
    
except Exception as e:
    print(f"⚠️ 레이어 번호 파싱 실패: {e}")
