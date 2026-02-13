#!/usr/bin/env python3
"""
KD 모델 학습 문제 진단
"""
import torch
from transformers import AutoTokenizer


def analyze_tokens():
    """토큰 560, EOS 토큰 분석"""
    model_path = "./KD_student_model_v4"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 토큰 정보
    print(f"{'='*60}")
    print("🔍 토크나이저 정보")
    print(f"{'='*60}")
    print(f"✓ Vocab Size: {tokenizer.vocab_size}")
    print(f"✓ EOS Token ID: {tokenizer.eos_token_id}")
    print(f"✓ EOS Token: '{tokenizer.eos_token}'")
    print(f"✓ Pad Token ID: {tokenizer.pad_token_id}")
    print(f"✓ Pad Token: '{tokenizer.pad_token}'")
    print(f"✓ BOS Token ID: {tokenizer.bos_token_id}")
    print(f"✓ BOS Token: '{tokenizer.bos_token}'")
    
    print(f"\n{'='*60}")
    print("🔎 토큰 560 디코드")
    print(f"{'='*60}")
    token_560 = tokenizer.decode([560])
    print(f"Token 560: '{token_560}' (repr: {repr(token_560)})")
    print(f"Length: {len(token_560)}")
    print(f"Bytes: {token_560.encode('utf-8')}")
    
    # 몇 가지 다른 토큰도 확인
    print(f"\n{'='*60}")
    print("📊 주변 토큰들")
    print(f"{'='*60}")
    for tid in [358, 359, 360, 361, 362, 363, 550, 560, 570]:
        token = tokenizer.decode([tid])
        print(f"  Token {tid:3d}: '{token}' (repr: {repr(token)})")


def check_dataset_eos():
    """데이터셋에서 EOS 토큰이 제대로 포함되는지 확인"""
    from datasets import load_dataset
    
    print(f"\n{'='*60}")
    print("📂 데이터셋 샘플 확인")
    print(f"{'='*60}")
    
    try:
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train[:3]")  # 3개만 로드
        
        for i, ex in enumerate(ds):
            print(f"\n[샘플 {i}]")
            conv = ex.get("conversations", [])
            print(f"❌ 문제: dataset에 'conversations' 필드 확인")
            print(f"❌ 사용 가능한 필드: {list(ex.keys())}")
            break
            
    except Exception as e:
        print(f"❌ 데이터셋 로드 오류: {e}")


def test_eos_learning():
    """EOS 토큰 학습 상황 시뮬레이션"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained("./base_model", trust_remote_code=True)
    
    print(f"\n{'='*60}")
    print("📚 EOS 토큰 학습 검증")
    print(f"{'='*60}")
    
    # 간단한 테스트 텍스트
    test_texts = [
        "Hi, how are you?",  # EOS 없음
        f"Hi, how are you?{tokenizer.eos_token}",  # EOS 있음
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        print(f"\nText: {repr(text)}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {repr(tokenizer.decode(tokens))}")
        print(f"Has EOS: {tokenizer.eos_token_id in tokens}")


if __name__ == "__main__":
    analyze_tokens()
    check_dataset_eos()
    test_eos_learning()
