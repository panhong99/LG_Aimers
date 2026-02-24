import os
from collections import Counter
from safetensors import safe_open

MODEL_DIR="./models/model_KD_v6_GPTQ_v2"# KD+GPTQ 모델 폴더로 수정
MODEL_PATH=os.path.join(MODEL_DIR,"model.safetensors")

dtypes=Counter()
count=0

with safe_open(MODEL_PATH,framework="pt",device="cpu") as f:
    keys = f.keys()
    for k in keys:
        t = f.get_tensor(k)
        dtypes[str(t.dtype)] += 1
        count += 1

print("="*60)
print("[1] DTYPE 분포 확인")
print("="*60)
print("file MB:",round(os.path.getsize(MODEL_PATH)/1024/1024,2))
print("num_tensors:",count)
print("dtype_counts:",dict(dtypes))

# bf16/fp16 텐서가 큰 비중인지 빠르게 보기
bf16_count = dtypes.get("torch.bfloat16", 0)
fp16_count = dtypes.get("torch.float16", 0)
print(f"bf16+fp16 ratio: {(bf16_count + fp16_count) / max(count, 1):.2%}")
print(f"  - bf16: {bf16_count}, fp16: {fp16_count}")

# ============================================
# 상위 텐서 크기 기준 분석
# ============================================
print("\n" + "="*60)
print("[2] 메모리 기여도 상위 텐서 (상위 20개)")
print("="*60)

tensor_info = []
with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        num_elements = t.numel()
        # dtype별 바이트 계산
        dtype_str = str(t.dtype)
        if "bfloat16" in dtype_str or "float16" in dtype_str:
            bytes_per_elem = 2
        elif "float32" in dtype_str:
            bytes_per_elem = 4
        elif "int32" in dtype_str:
            bytes_per_elem = 4
        elif "int8" in dtype_str:
            bytes_per_elem = 1
        else:
            bytes_per_elem = 1  # 기본값
        
        total_bytes = num_elements * bytes_per_elem
        tensor_info.append({
            "name": k,
            "dtype": dtype_str,
            "elements": num_elements,
            "bytes": total_bytes,
            "mb": total_bytes / 1024 / 1024
        })

# 크기순 정렬
tensor_info.sort(key=lambda x: x["bytes"], reverse=True)

print(f"{'Rank':<5} {'Name':<40} {'DType':<20} {'MB':<10} {'Elements':<15}")
print("-" * 90)
for i, info in enumerate(tensor_info[:20], 1):
    print(f"{i:<5} {info['name']:<40} {info['dtype']:<20} {info['mb']:<10.2f} {info['elements']:<15}")

# 통계
total_bf16_mb = sum(info["mb"] for info in tensor_info if "bfloat16" in info["dtype"])
total_fp16_mb = sum(info["mb"] for info in tensor_info if "float16" in info["dtype"])
total_mb = sum(info["mb"] for info in tensor_info)

print("\n" + "="*60)
print("[3] 메모리 통계")
print("="*60)
print(f"Total model size: {total_mb:.2f} MB")
print(f"BF16/FP32 텐서들: {total_bf16_mb:.2f} MB ({total_bf16_mb/total_mb*100:.1f}%)")
print(f"FP16 텐서들: {total_fp16_mb:.2f} MB ({total_fp16_mb/total_mb*100:.1f}%)")

# 양자화 가능성 진단
if total_bf16_mb + total_fp16_mb > total_mb * 0.1:
    print("\n⚠️  경고: BF16/FP16이 10% 이상 → 부분 양자화/복원된 모델일 가능성 높음")
else:
    print("\n✓ BF16/FP16 비중 정상 → 대부분 양자화됨")