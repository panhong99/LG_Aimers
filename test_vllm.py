#!/usr/bin/env python3
"""
vLLM 모델 응답 테스트 스크립트
"""
import requests
import json
import time
from typing import Optional

def test_vllm_chat(
    model_path: str = "./KD_student_model_v4",
    prompt: str = "What is the capital of France?",
    max_tokens: int = 100,
    temperature: float = 0.7,
    server_url: str = "http://localhost:8000"
) -> Optional[dict]:
    """
    vLLM 서버에 채팅 요청 전송
    
    Args:
        model_path: 모델 경로
        prompt: 질문 문장
        max_tokens: 최대 생성 토큰 수
        temperature: 생성 다양성 (0~1, 낮을수록 결정적)
        server_url: vLLM 서버 URL
    
    Returns:
        API 응답 (JSON)
    """
    url = f"{server_url}/v1/chat/completions"
    
    payload = {
        "model": model_path,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        print(f"📤 요청 송신: {prompt[:50]}...")
        start_time = time.time()
        
        response = requests.post(url, json=payload, timeout=30)
        elapsed = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        # 응답 출력
        if "choices" in result and len(result["choices"]) > 0:
            assistant_message = result["choices"][0]["message"]["content"]
            print(f"\n✅ 응답 (소요시간: {elapsed:.2f}초):")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(assistant_message)
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"\n📊 토큰 정보:")
            print(f"  - Prompt tokens: {result.get('usage', {}).get('prompt_tokens', 'N/A')}")
            print(f"  - Completion tokens: {result.get('usage', {}).get('completion_tokens', 'N/A')}")
            print(f"  - Total tokens: {result.get('usage', {}).get('total_tokens', 'N/A')}")
        
        return result
    
    except requests.exceptions.ConnectionError:
        print("❌ 에러: vLLM 서버에 연결할 수 없습니다.")
        print(f"   확인: vllm serve ./KD_student_model_v4 --host 0.0.0.0 --port 8000")
        return None
    
    except requests.exceptions.Timeout:
        print("❌ 에러: 요청 타임아웃 (서버 응답 없음)")
        return None
    
    except Exception as e:
        print(f"❌ 에러: {e}")
        return None


def test_multiple_prompts(
    model_path: str = "./KD_student_model_v4",
    prompts: Optional[list] = None,
    server_url: str = "http://localhost:8000"
) -> None:
    """
    여러 프롬프트 테스트
    
    Args:
        model_path: 모델 경로
        prompts: 테스트 프롬프트 리스트
        server_url: vLLM 서버 URL
    """
    if prompts is None:
        prompts = [
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "Write a short poem about spring.",
        ]
    
    print(f"\n🚀 vLLM 모델 테스트 시작")
    print(f"   모델: {model_path}")
    print(f"   서버: {server_url}")
    print(f"   프롬프트 수: {len(prompts)}")
    print("=" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[테스트 {i}/{len(prompts)}]")
        test_vllm_chat(
            model_path=model_path,
            prompt=prompt,
            server_url=server_url
        )
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM 모델 응답 테스트")
    parser.add_argument("--model", type=str, default="./KD_student_model_v4",
                        help="모델 경로")
    parser.add_argument("--prompt", type=str, default=None,
                        help="테스트 프롬프트 (지정하지 않으면 기본값 사용)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="최대 생성 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="생성 다양성 (0~1)")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000",
                        help="vLLM 서버 URL")
    
    args = parser.parse_args()
    
    if args.prompt:
        # 단일 프롬프트 테스트
        test_vllm_chat(
            model_path=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            server_url=args.server_url
        )
    else:
        # 기본 프롬프트들로 테스트
        test_multiple_prompts(
            model_path=args.model,
            server_url=args.server_url
        )
