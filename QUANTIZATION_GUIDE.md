# 양자화 모델 제출 가이드

## 개요

KD(Knowledge Distillation) + GPTQ(Quantization) + vLLM 환경에서 모델 로딩/추론 시 발생하는 문제를 해결하기 위한 완전한 파이프라인입니다.

### 문제 2가지

1. **Weight Tying 미복구**: embedding과 lm_head가 독립 텐서로 저장 → 중복 메모리 사용 (800MB+) → vLLM 혼동 → OOM/타임아웃
2. **generation_config.json 부재**: 기본 샘플링 파라미터로 인한 점수 급락

### 해결책

- ✅ Weight tying 자동 복구 (`ensure_weight_tying()`)
- ✅ generation_config.json 자동 생성
- ✅ 저장 후 검증 (`validate_model_structure()`)

---

## 파일 구성

### 1. `model_utils.py` (공용 유틸리티)

제출용 모델 준비를 위한 함수 모음:

#### 함수 목록

| 함수 | 용도 | 위치 |
|------|------|------|
| `ensure_weight_tying()` | embedding ↔ lm_head tying 복구 | KD 학습 후, GPTQ 후 |
| `remove_lm_head_from_state_dict()` | lm_head 텐서 제거 (optional) | save 전 |
| `create_generation_config()` | generation_config.json 생성 | save 후 |
| `normalize_safetensors_keys()` | 키 정규화 (model.model.* 중복 제거) | save 후 |
| `validate_model_structure()` | dtype/중복 텐서 검증 | save 후 |
| `prepare_model_for_submission()` | 위 모든 함수를 한 번에 실행 | save 대신 사용 가능 |

---

## 파이프라인 수정 사항

### 2. `main_kd_gemini.py` (KD 학습)

**변경 항목:**
- Import 추가 (model_utils 함수들)
- trainer.train() 직후 weight tying 복구
- generation_config.json 자동 생성

**수정된 코드:**

```python
# ★ 추가된 부분
from model_utils import ensure_weight_tying, create_generation_config, normalize_safetensors_keys

# (중간에 trainer.train() 실행)

# [5] ★ 저장 전 Weight Tying 복구 및 generation_config 생성
print("\n" + "="*70)
print("[제출용 모델 준비]")
print("="*70)

# Step 1: Weight tying 복구
print("\n[Step 1] Weight tying 복구...")
ensure_weight_tying(student_model, config=student_model.config, verbose=True)

# Step 2: 모델 저장
print(f"\n[Step 2] 모델 저장 중: {training_args.output_dir}")
student_model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

# Step 3: generation_config 생성
print(f"\n[Step 3] generation_config 생성...")
create_generation_config(
    student_model, 
    tokenizer=tokenizer,
    output_dir=training_args.output_dir,
    verbose=True
)

# Step 4: safetensors 정규화
print(f"\n[Step 4] safetensors 키 정규화...")
normalize_safetensors_keys(training_args.output_dir, verbose=True)
```

---

### 3. `main_GPTQ.py` (양자화)

**변경 항목:**
- Import 추가 (model_utils 함수들)
- GPTQ 완료 후 weight tying 복구
- generation_config.json 자동 생성
- normalize_safetensors_keys로 키 정규화 (기존 로직 대체)
- validate_model_structure로 최종 검증

**수정된 코드:**

```python
# ★ 추가된 부분
from model_utils import (
    ensure_weight_tying, 
    remove_lm_head_from_state_dict,
    create_generation_config, 
    normalize_safetensors_keys,
    validate_model_structure
)

# (중간에 GPTQ 실행 - oneshot())

# [4] 제출용 모델 준비
print("\n" + "="*70)
print("[제출용 모델 준비]")
print("="*70)

# Step 1: GPTQ 후 Weight Tying 복구
print("\n[Step 1] GPTQ 후 weight tying 복구...")
ensure_weight_tying(model, config=model.config, verbose=True)

# Step 2: lm_head 제거 (선택사항)
# remove_lm_head_option = False
# if remove_lm_head_option:
#     print("\n[Step 2] lm_head 제거 (optional)...")
#     remove_lm_head_from_state_dict(model, verbose=True)
print("\n[Step 2] lm_head 제거 스킵 (fully tied 상태 유지)")

# Step 3: 모델 저장
print(f"\n[Step 3] 양자화 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# Step 4: generation_config 생성
print(f"\n[Step 4] generation_config 생성...")
create_generation_config(
    model, 
    tokenizer=tokenizer,
    output_dir=OUT_DIR,
    verbose=True
)

# Step 5: safetensors 정규화
print(f"\n[Step 5] safetensors 키 정규화...")
normalize_safetensors_keys(OUT_DIR, verbose=True)

# Step 6: 최종 검증
print(f"\n[Step 6] 최종 검증...")
validate_model_structure(OUT_DIR, verbose=True)

print("\n" + "="*70)
print("✓ 제출용 모델 준비 완료!")
print("="*70)
```

---

### 4. `validate_model_structure.py` (검증 스크립트)

독립적으로 저장된 모델을 검증합니다.

**사용법:**

```bash
python3 validate_model_structure.py ./models/model_KD_v6_GPTQ_v2
```

**출력 예시:**

```
[검증 결과] 저장된 모델 구조 분석

[1] 기본 통계
  파일 크기: 1221.01 MB
  총 텐서 개수: 603
  dtype 분포: {'torch.bfloat16': 267, 'torch.int32': 168, 'torch.int64': 168}

[2] BF16/FP16 분석
  BF16/FP16 비중: 44.28%
  BF16/FP16 용량: 812.95 MB
  ⚠️  경고: BF16/FP16이 10% 이상 → 부분 양자화/복원 모델 가능성

[3] Embedding/lm_head 중복 체크
  embed_tokens 존재: ✓
  lm_head 존재: ✓
  ⚠️  경고: 둘 다 존재 → 중복 저장 가능성 (weight tying 미복구)

[4] 용량 기여도 상위 10개 텐서
  Rank  Name                                     DType            MB
  ────────────────────────────────────────────────────────────────
  1     lm_head.weight                           torch.bfloat16   400.00
  2     model.embed_tokens.weight                torch.bfloat16   400.00
  ...
```

---

## 실행 순서

### 시나리오 1: 새로 학습 (권장)

```bash
# 1. KD 학습 (main_kd_gemini.py)
python3 main_kd_gemini.py

# 2. GPTQ 양자화 (main_GPTQ.py)
python3 main_GPTQ.py

# 3. (선택) 최종 검증
python3 validate_model_structure.py ./model_KD_v6_GPTQ_v2
```

### 시나리오 2: 기존 모델 검증/복구

```bash
# 1. 현재 모델 상태 확인
python3 validate_model_structure.py ./models/model_KD_v6_GPTQ_v2

# 2. 문제 발견 → 재학습이 가장 확실한 해결책
python3 main_kd_gemini.py
python3 main_GPTQ.py
```

---

## Generation Config 내용

`generation_config.json`에 자동으로 저장되는 내용:

```json
{
  "bos_token_id": 1,
  "eos_token_id": 361,
  "pad_token_id": 0,
  "temperature": 0.1,
  "top_p": 1.0,
  "do_sample": false,
  "max_new_tokens": 512,
  "top_k": 50,
  "repetition_penalty": 1.0,
  "length_penalty": 1.0
}
```

**설명:**
- `do_sample=false`: greedy decoding (결정적 출력)
- `temperature=0.1`: 매우 낮은 온도 (안정적 평가용)
- `top_p=1.0`: top-p 샘플링 비활성화
- `bos/eos/pad`: EXAONE 모델 기본값 (tokenizer 우선)

---

## 문제 해결 가이드

### Q1: "embedding/lm_head 중복" 경고가 나오면?

**원인:**
- weight tying이 제대로 복구되지 않음
- 모델 저장 전에 `ensure_weight_tying()` 미호출

**해결:**
1. `main_kd_gemini.py` 또는 `main_GPTQ.py` 재확인
2. 함수가 호출되었는지 로그 확인
3. 필요시 model_utils.py의 `ensure_weight_tying()` 함수 수동 호출

```python
from model_utils import ensure_weight_tying
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./models/problematic_model")
ensure_weight_tying(model, verbose=True)
model.save_pretrained("./models/fixed_model")
```

### Q2: "BF16/FP16이 10% 이상" 경고?

**원인:**
- GPTQ 실패 또는 부분 양자화
- embedding/lm_head 제거 안 됨

**해결:**
- 재학습 권장 (가장 확실)
- 또는 `remove_lm_head_from_state_dict()` 사용 시도

### Q3: 제출 후 점수가 낮으면?

**원인:**
- generation_config.json 부재 또는 파라미터 이상
- weight tying 미복구 (모델 혼동)

**해결:**
1. generation_config.json 확인:
   ```bash
   cat model_KD_v6_GPTQ_v2/generation_config.json
   ```

2. weight tying 상태 확인:
   ```bash
   python3 validate_model_structure.py model_KD_v6_GPTQ_v2
   ```

---

## 주의사항

1. **weight tying은 필수**: 
   - tie_word_embeddings=True인 모델이라도 저장 전에 `ensure_weight_tying()` 호출 필수
   - GPTQ 후에도 다시 호출 필수

2. **generation_config.json**:
   - 리더보드 평가용이므로 do_sample=False, temperature=0.1 권장
   - 다른 파라미터 필요시 직접 수정 가능

3. **재학습 vs 복구**:
   - 이미 문제 있는 모델 수정보다는 **새로 학습하는 게 더 확실**
   - 모든 함수가 이미 통합되어 있으므로 새로 실행하면 자동 적용됨

---

## 개발 노트

### 구조적 개선

- ✅ 공용 유틸리티 함수 분리 (model_utils.py)
- ✅ KD 파이프라인에 weight tying 통합
- ✅ GPTQ 파이프라인에 weight tying + generation_config 통합
- ✅ 독립 검증 스크립트 추가
- ✅ Edge case 처리 (모듈 경로 다름, lm_head 없음 등)

### 추가 기능 (future)

- [ ] lm_head 완전 제거 옵션 (현재 주석 상태)
- [ ] dtype 변환 자동화 (필요시)
- [ ] multi-device 로딩 테스트
- [ ] vLLM 호환성 자동 검증

---

## 참고 자료

**EXAONE 모델 특성:**
- tie_word_embeddings=True (embedding과 lm_head 공유)
- bos_token_id=1, eos_token_id=361, pad_token_id=0

**vLLM 요구사항:**
- 명시적 generation_config.json
- 일관된 dtype 분포
- 명확한 weight 구조 (tied 또는 independent)
