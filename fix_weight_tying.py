#!/usr/bin/env python3
"""
기존 저장된 모델의 weight tying 문제 복구 스크립트

사용법:
  python3 fix_weight_tying.py ./models/model_KD_v6_GPTQ_v2

역할:
  1. 저장된 모델 로드
  2. weight tying 강제 복구 (embedding ↔ lm_head)
  3. lm_head 제거 (state_dict에서)
  4. generation_config 생성
  5. 재저장
"""

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import (
    ensure_weight_tying,
    remove_lm_head_from_state_dict,
    create_generation_config,
    normalize_safetensors_keys,
    validate_model_structure
)


def fix_model_weight_tying(model_dir: str, remove_lm_head: bool = True):
    """
    기존 저장된 모델의 weight tying 복구 및 lm_head 제거
    
    Args:
        model_dir: 모델 폴더 경로 (예: ./models/model_KD_v6_GPTQ_v2)
        remove_lm_head: lm_head를 state_dict에서 제거할지 여부
    """
    
    print("="*70)
    print("[Weight Tying 복구 스크립트]")
    print("="*70)
    
    if not os.path.exists(model_dir):
        print(f"❌ 오류: {model_dir} 폴더를 찾을 수 없습니다")
        sys.exit(1)
    
    # Step 1: 모델 로드
    print(f"\n[Step 1] 모델 로드 중: {model_dir}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu"  # CPU에서 처리 (safe)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"✓ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        sys.exit(1)
    
    # Step 2: Weight tying 복구
    print(f"\n[Step 2] Weight tying 강제 복구...")
    ensure_weight_tying(model, config=model.config, verbose=True)
    
    # Step 3: lm_head 제거 (선택사항)
    if remove_lm_head:
        print(f"\n[Step 3] lm_head 제거 (state_dict에서)...")
        remove_lm_head_from_state_dict(model, verbose=True)
    else:
        print(f"\n[Step 3] lm_head 제거 스킵 (tied 상태 유지)")
    
    # Step 4: 재저장
    print(f"\n[Step 4] 수정된 모델 재저장 중: {model_dir}")
    try:
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        print(f"✓ 재저장 완료")
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        sys.exit(1)
    
    # Step 5: generation_config 생성
    print(f"\n[Step 5] generation_config.json 생성...")
    create_generation_config(
        model,
        tokenizer=tokenizer,
        output_dir=model_dir,
        verbose=True
    )
    
    # Step 6: safetensors 정규화
    print(f"\n[Step 6] safetensors 키 정규화...")
    normalize_safetensors_keys(model_dir, verbose=True)
    
    # Step 7: 검증
    print(f"\n[Step 7] 최종 검증...")
    report = validate_model_structure(model_dir, verbose=True)
    
    if report is None:
        print("\n❌ 검증 실패")
        sys.exit(1)
    
    # 문제 체크
    if report['embedding_lm_head_duplicate']:
        print("\n❌ CRITICAL: 여전히 embedding/lm_head 중복 존재!")
        print("   → weight tying 복구 재시도 필요")
        sys.exit(1)
    
    if report['bf16_fp16_ratio'] > 0.15:
        print("\n⚠️  경고: BF16/FP16이 15% 이상 존재")
        print("   → 부분 양자화 상태 가능성")
    
    # 성공
    print("\n" + "="*70)
    print("✅ Weight tying 복구 완료!")
    print("="*70)
    print(f"\n모델 폴더: {model_dir}")
    print("\n다음 단계:")
    print("  1. test.py로 성능 검증")
    print("  2. 리더보드에 제출")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 fix_weight_tying.py <model_dir>")
        print("예시: python3 fix_weight_tying.py ./models/model_KD_v6_GPTQ_v2")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    
    # lm_head 제거 여부 (기본: 제거)
    remove_lm_head = True
    if len(sys.argv) > 2 and sys.argv[2] == "--keep-lm-head":
        remove_lm_head = False
    
    fix_model_weight_tying(model_dir, remove_lm_head=remove_lm_head)
