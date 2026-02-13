#!/usr/bin/env python3
"""
KD 모델 디버그: 원본과 비교 테스트
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_model_direct(model_path: str, prompt: str = "당신은 누구입니까?"):
    """로컬에서 직접 모델 테스트 (vLLM 없이)"""
    print(f"\n{'='*60}")
    print(f"📊 모델: {model_path}")
    print(f"{'='*60}")
    
    try:
        # 모델 & 토크나이저 로드
        print("📥 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        
        # 모델 정보
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 파라미터: {n_params/1e9:.2f}B")
        print(f"✓ EOS Token ID: {tokenizer.eos_token_id}")
        print(f"✓ Vocab Size: {tokenizer.vocab_size}")
        
        # 토크나이즈
        print(f"\n📝 프롬프트: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        print(f"✓ Input tokens: {inputs['input_ids'].shape}")
        
        # 생성
        print(f"\n⏳ 생성 중...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=False,  # 결정적 생성
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # 디코드
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n✅ 응답 (원본 토큰):")
        print("━" * 60)
        print(response)
        print("━" * 60)
        
        print(f"\n✅ 응답 (정리됨):")
        print("━" * 60)
        print(response_clean)
        print("━" * 60)
        
        # 토큰 분석
        print(f"\n🔍 토큰 분석:")
        print(f"  - 입력 토큰: {outputs[0][:inputs['input_ids'].shape[1]].tolist()}")
        print(f"  - 생성된 토큰 수: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
        print(f"  - 마지막 10개 토큰: {outputs[0][-10:].tolist()}")
        
        # EOS 토큰 포함 여부 확인
        eos_id = tokenizer.eos_token_id
        if eos_id in outputs[0]:
            eos_idx = (outputs[0] == eos_id).nonzero(as_tuple=True)[0]
            print(f"  - EOS 토큰 위치: {eos_idx.tolist()}")
        else:
            print(f"  - EOS 토큰 없음 ⚠️")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_models():
    """원본 vs KD 모델 비교"""
    BASE = "./base_model"
    KD_V3 = "./KD_student_model_v3"
    KD_V4 = "./KD_student_model_v4"
    
    prompt = "당신은 누구입니까?"
    
    # 원본 테스트
    test_model_direct(BASE, prompt)
    
    # KD v3 테스트
    test_model_direct(KD_V3, prompt)
    
    # KD v4 테스트 (현재 문제 모델)
    test_model_direct(KD_V4, prompt)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./KD_student_model_v4",
                        help="테스트할 모델 경로")
    parser.add_argument("--prompt", type=str, default="당신은 누구입니까?",
                        help="테스트 프롬프트")
    parser.add_argument("--compare", action="store_true",
                        help="원본 vs KD 모델 비교")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        test_model_direct(args.model, args.prompt)
