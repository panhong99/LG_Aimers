"""
Model utility functions for weight tying, generation config, and validation.
사용 위치:
- main_kd_gemini.py: 저장 전
- main_GPTQ.py: 양자화 후, 저장 전
- validate_model_structure.py: 저장 후 검증
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict, Counter
from safetensors import safe_open
from safetensors.torch import load_file, save_file


# =============================================================================
# [1] Weight Tying 복구/강제
# =============================================================================

def ensure_weight_tying(model, config=None, verbose=True):
    """
    모델의 embedding - lm_head weight tying을 강제로 복구합니다.
    
    사용 위치:
    - main_kd_gemini.py: trainer.train() 직후, save_pretrained 직전
    - main_GPTQ.py: 양자화 완료 후, save_pretrained 직전
    
    Args:
        model: AutoModelForCausalLM 모델 객체
        config: (optional) model.config 또는 별도 config
        verbose: 로깅 여부
    
    Returns:
        수정된 모델
    """
    if config is None:
        config = model.config
    
    # tie_word_embeddings 설정 강제
    config.tie_word_embeddings = True
    
    # embedding 가중치 찾기
    embed_tokens = None
    embed_name = None
    
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_tokens = model.model.embed_tokens
        embed_name = "model.model.embed_tokens"
    elif hasattr(model, "embed_tokens"):
        embed_tokens = model.embed_tokens
        embed_name = "model.embed_tokens"
    
    if embed_tokens is None:
        if verbose:
            print("⚠️  Warning: Could not find embed_tokens layer")
        return model
    
    # lm_head 가중치 찾기
    lm_head = None
    
    if hasattr(model, "lm_head") and model.lm_head is not None:
        lm_head = model.lm_head
    
    if lm_head is None:
        if verbose:
            print("⚠️  Warning: Could not find lm_head layer")
        return model
    
    # Weight tying: lm_head.weight를 embed_tokens.weight로 지정
    if hasattr(lm_head, "weight"):
        original_shape = lm_head.weight.shape
        lm_head.weight = embed_tokens.weight
        
        if verbose:
            print(f"✓ Weight tying 복구 완료:")
            print(f"  lm_head.weight → {embed_name}.weight (tied reference)")
            print(f"  shape: {original_shape}")
    
    # 혹시나 해서 모델 자체의 tie_weights 메서드도 호출 (EXAONE 특화)
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
            if verbose:
                print(f"✓ model.tie_weights() 추가 호출 완료")
        except Exception as e:
            if verbose:
                print(f"  (note: model.tie_weights() 호출 시도했으나 {type(e).__name__}: {e})")
    
    return model


# =============================================================================
# [2] lm_head 제거 (선택사항) - vLLM 호환성 강화
# =============================================================================

def remove_lm_head_from_state_dict(model, verbose=True):
    """
    state_dict에서 lm_head 관련 키를 제거합니다.
    
    tie_word_embeddings=True 기반이므로, 제거해도 generate 시 embed_tokens에서 복원됩니다.
    
    사용 위치:
    - main_GPTQ.py: save_pretrained 이전, remove_lm_head_option=True일 때
    
    Args:
        model: 모델 객체
        verbose: 로깅 여부
    
    Returns:
        수정된 모델 (state_dict에서 lm_head 제거됨)
    """
    state_dict = model.state_dict()
    
    lm_head_keys = [k for k in state_dict.keys() if k.startswith("lm_head")]
    
    if not lm_head_keys:
        if verbose:
            print("✓ lm_head 키 없음 (이미 제거되었거나 tied 상태)")
        return model
    
    for k in lm_head_keys:
        del state_dict[k]
        if verbose:
            print(f"  삭제: {k}")
    
    model.load_state_dict(state_dict, strict=False)
    
    if verbose:
        print(f"✓ lm_head 제거 완료 ({len(lm_head_keys)}개 키 삭제)")
    
    return model


# =============================================================================
# [3] Generation Config 자동 생성/저장
# =============================================================================

def create_generation_config(model, tokenizer=None, output_dir=None, verbose=True):
    """
    모델/토크나이저에서 generation_config.json을 자동 생성합니다.
    
    사용 위치:
    - main_kd_gemini.py: save_pretrained 직후
    - main_GPTQ.py: save_pretrained 직후
    
    Args:
        model: 모델 객체
        tokenizer: 토크나이저 객체 (optional)
        output_dir: 저장 디렉토리 (None이면 생성 안 함, 딕셔너리만 반환)
        verbose: 로깅 여부
    
    Returns:
        generation_config 딕셔너리
    """
    
    # config에서 special token ID 추출
    config = model.config
    
    bos_token_id = getattr(config, "bos_token_id", 1)
    eos_token_id = getattr(config, "eos_token_id", 361)  # EXAONE 기본값
    pad_token_id = getattr(config, "pad_token_id", 0)
    
    # tokenizer가 있으면 우선순위로 사용
    if tokenizer is not None:
        if tokenizer.bos_token_id is not None:
            bos_token_id = tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            eos_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
    
    generation_config = {
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        # 리더보드 평가용 설정 (안정성 우선)
        "temperature": 0.1,
        "top_p": 1.0,
        "do_sample": False,
        "max_new_tokens": 512,
        # 추가 안정성 설정
        "top_k": 50,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
    }
    
    # output_dir이 주어지면 저장
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "generation_config.json")
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(generation_config, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"✓ generation_config.json 저장 완료: {config_path}")
            print(f"  bos_token_id: {bos_token_id}")
            print(f"  eos_token_id: {eos_token_id}")
            print(f"  pad_token_id: {pad_token_id}")
            print(f"  do_sample: False, temperature: 0.1")
    
    return generation_config


# =============================================================================
# [4] 저장 후 검증
# =============================================================================

def validate_model_structure(model_dir, verbose=True):
    """
    저장된 모델의 양자화 상태와 weight tying 상태를 검증합니다.
    
    사용 위치:
    - main_GPTQ.py: save_pretrained 직후
    - validate_model_structure.py: 독립 실행
    
    Args:
        model_dir: 모델 디렉토리 경로
        verbose: 로깅 여부
    
    Returns:
        validation_report (dict)
    """
    
    model_path = os.path.join(model_dir, "model.safetensors")
    
    if not os.path.exists(model_path):
        if verbose:
            print(f"❌ Error: {model_path} not found")
        return None
    
    # dtype 분포 분석
    dtype_counts = Counter()
    dtype_by_module = defaultdict(lambda: defaultdict(int))
    tensor_info = []
    total_bytes = 0
    
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            num_elements = t.numel()
            
            # dtype별 바이트 계산
            if "bfloat16" in str(t.dtype):
                bytes_per_elem = 2
            elif "float16" in str(t.dtype):
                bytes_per_elem = 2
            elif "float32" in str(t.dtype):
                bytes_per_elem = 4
            else:
                bytes_per_elem = 4
            
            item_bytes = num_elements * bytes_per_elem
            dtype_counts[str(t.dtype)] += 1
            total_bytes += item_bytes
            
            # 모듈별 분류
            if "embed_tokens" in k:
                module_type = "embed_tokens"
            elif "lm_head" in k:
                module_type = "lm_head"
            else:
                module_type = "other"
            
            dtype_by_module[module_type][str(t.dtype)] += 1
            
            tensor_info.append({
                "name": k,
                "dtype": str(t.dtype),
                "elements": num_elements,
                "bytes": item_bytes,
                "mb": item_bytes / 1024 / 1024
            })
    
    # 상위 텐서 (용량 기준)
    tensor_info.sort(key=lambda x: x["bytes"], reverse=True)
    
    # 중복 체크
    has_embed_tokens = any("embed_tokens" in info["name"] and "weight" in info["name"] for info in tensor_info)
    has_lm_head = any("lm_head" in info["name"] and "weight" in info["name"] for info in tensor_info)
    
    # 통계
    bf16_fp16_bytes = sum(
        info["bytes"] for info in tensor_info 
        if "bfloat16" in info["dtype"] or "float16" in info["dtype"]
    )
    bf16_fp16_ratio = bf16_fp16_bytes / max(total_bytes, 1)
    
    # 보고서 작성
    report = {
        "file_size_mb": os.path.getsize(model_path) / 1024 / 1024,
        "total_tensors": len(tensor_info),
        "dtype_counts": dict(dtype_counts),
        "bf16_fp16_ratio": bf16_fp16_ratio,
        "bf16_fp16_bytes_mb": bf16_fp16_bytes / 1024 / 1024,
        "has_embed_tokens": has_embed_tokens,
        "has_lm_head": has_lm_head,
        "embedding_lm_head_duplicate": has_embed_tokens and has_lm_head,
        "top_20_tensors": tensor_info[:20],
    }
    
    if verbose:
        print("\n" + "="*70)
        print("[검증 결과] 저장된 모델 구조 분석")
        print("="*70)
        
        print(f"\n[1] 기본 통계")
        print(f"  파일 크기: {report['file_size_mb']:.2f} MB")
        print(f"  총 텐서 개수: {report['total_tensors']}")
        print(f"  dtype 분포: {dict(dtype_counts)}")
        
        print(f"\n[2] BF16/FP16 분석")
        print(f"  BF16/FP16 비중: {report['bf16_fp16_ratio']:.1%}")
        print(f"  BF16/FP16 용량: {report['bf16_fp16_bytes_mb']:.2f} MB")
        
        if report['bf16_fp16_ratio'] > 0.1:
            print(f"  ⚠️  경고: BF16/FP16이 10% 이상 → 부분 양자화/복원 모델 가능성")
        else:
            print(f"  ✓ 정상: BF16/FP16 비중 낮음")
        
        print(f"\n[3] Embedding/lm_head 중복 체크")
        print(f"  embed_tokens 존재: {'✓' if has_embed_tokens else '❌'}")
        print(f"  lm_head 존재: {'✓' if has_lm_head else '❌'}")
        
        if report['embedding_lm_head_duplicate']:
            print(f"  ⚠️  경고: 둘 다 존재 → 중복 저장 가능성 (weight tying 미복구)")
        else:
            print(f"  ✓ 정상: 한쪽만 존재하거나 tied 상태")
        
        print(f"\n[4] 용량 기여도 상위 10개 텐서")
        print(f"  {'Rank':<5} {'Name':<40} {'DType':<15} {'MB':<10}")
        print(f"  {'-'*70}")
        for i, info in enumerate(tensor_info[:10], 1):
            print(f"  {i:<5} {info['name']:<40} {info['dtype']:<15} {info['mb']:>8.2f}")
    
    return report


# =============================================================================
# [5] Safetensors 정정 (출장/로딩 호환성)
# =============================================================================

def normalize_safetensors_keys(output_dir, verbose=True):
    """
    safetensors 파일의 가중치 키를 정규화합니다.
    - "model.model." 중복 제거
    - 호환성 보장
    
    사용 위치:
    - main_GPTQ.py: save_pretrained 직후
    
    Args:
        output_dir: 모델 저장 디렉토리
        verbose: 로깅 여부
    """
    
    for safetensors_file in Path(output_dir).glob("*.safetensors"):
        try:
            weights = load_file(str(safetensors_file), device="cpu")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  {safetensors_file.name} 로드 실패: {e}")
            continue
        
        fixed_weights = {}
        changed = 0
        
        for k, v in weights.items():
            # "model.model.*" 형태면 "model." 하나만 남기기
            new_key = k
            if k.startswith("model.model."):
                new_key = k.replace("model.model.", "model.", 1)
                changed += 1
            
            fixed_weights[new_key] = v
        
        if changed > 0:
            save_file(fixed_weights, str(safetensors_file))
            if verbose:
                print(f"✓ {safetensors_file.name} 정규화 완료 ({changed}개 키 수정)")
        else:
            if verbose:
                print(f"✓ {safetensors_file.name} 정규화 불필요")


# =============================================================================
# [6] 편의함수: 전체 파이프라인 한 번에
# =============================================================================

def prepare_model_for_submission(model, tokenizer=None, output_dir=None, 
                                 remove_lm_head=False, verbose=True):
    """
    모델을 제출용으로 완전히 준비합니다 (weight tying, generation config, 검증).
    
    사용 위치:
    - main_kd_gemini.py 또는 main_GPTQ.py의 save_pretrained 직후
    
    Args:
        model: 모델 객체
        tokenizer: 토크나이저 (optional)
        output_dir: 저장 디렉토리
        remove_lm_head: lm_head 제거 여부
        verbose: 로깅 여부
    """
    
    print(f"\n{'='*70}")
    print("[제출용 모델 준비]")
    print(f"{'='*70}")
    
    # [1] Weight tying 복구
    print("\n[Step 1] Weight tying 복구...")
    ensure_weight_tying(model, verbose=verbose)
    
    # [2] lm_head 제거 (선택사항)
    if remove_lm_head:
        print("\n[Step 2] lm_head 제거...")
        remove_lm_head_from_state_dict(model, verbose=verbose)
    else:
        print("\n[Step 2] lm_head 제거 스킵")
    
    # [3] 저장
    if output_dir is not None and not os.path.exists(output_dir):
        print(f"\n[Step 3] 모델 저장 중: {output_dir}")
        model.save_pretrained(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        print(f"✓ 저장 완료")
    
    # [4] generation_config 생성
    print("\n[Step 4] generation_config 생성...")
    create_generation_config(model, tokenizer, output_dir, verbose=verbose)
    
    # [5] safetensors 키 정규화
    if output_dir is not None:
        print("\n[Step 5] safetensors 정규화...")
        normalize_safetensors_keys(output_dir, verbose=verbose)
    
    # [6] 검증
    if output_dir is not None:
        print("\n[Step 6] 최종 검증...")
        validate_model_structure(output_dir, verbose=verbose)
    
    print(f"\n{'='*70}")
    print("✓ 제출용 모델 준비 완료!")
    print(f"{'='*70}\n")
