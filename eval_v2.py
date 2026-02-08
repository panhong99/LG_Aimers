import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def measure_speed(model_path, model_name="Model", num_test=5):
    print(f"\n{'='*40}")
    print(f"🚀 측정 시작: {model_name}")
    print(f"   경로: {model_path}")
    print(f"{'='*40}")

    # 1. 모델 & 토크나이저 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        ).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # VRAM 측정 (로드 직후)
    vram_used = torch.cuda.memory_allocated() / 1024**3
    print(f"💾 모델 VRAM 사용량: {vram_used:.2f} GB")

    # 2. 테스트 프롬프트 (긴 생성을 유도)
    prompt = "인공지능의 미래와 인간의 역할에 대해 자세히 설명해줘."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # 생성 설정 (일관된 비교를 위해 고정)
    gen_config = {
        "max_new_tokens": 512,  # 512 토큰 생성 시간 측정
        "do_sample": False,     # 랜덤성 제거 (속도 측정용)
        "pad_token_id": tokenizer.eos_token_id
    }

    # 3. 웜업 (Warmup) - 처음 한 번은 컴파일 등으로 느릴 수 있음
    print("🔥 웜업 중... (첫 실행)")
    with torch.no_grad():
        model.generate(**inputs, **gen_config)
    torch.cuda.synchronize()

    # 4. 실제 속도 측정
    print(f"⏱️  {num_test}회 반복 측정 시작...")
    total_tokens = 0
    total_time = 0

    for i in range(num_test):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        
        with torch.no_grad():
            output = model.generate(**inputs, **gen_config)
        
        end_event.record()
        torch.cuda.synchronize()

        # 시간 계산 (밀리초 -> 초)
        elapsed_sec = start_event.elapsed_time(end_event) / 1000
        
        # 생성된 토큰 수 계산 (입력 토큰 제외)
        gen_tokens = len(output[0]) - len(inputs["input_ids"][0])
        
        total_tokens += gen_tokens
        total_time += elapsed_sec
        
        print(f"   [{i+1}/{num_test}] {gen_tokens} tokens in {elapsed_sec:.2f}s "
              f"({gen_tokens/elapsed_sec:.2f} tok/s)")

    # 5. 결과 요약
    avg_tps = total_tokens / total_time
    print(f"\n✅ [결과] {model_name}")
    print(f"   - 평균 속도 : {avg_tps:.2f} tokens/sec")
    print(f"   - 메모리    : {vram_used:.2f} GB")
    
    # 메모리 정리
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    # 비교할 모델 경로 설정
    BASE_MODEL = "./base_model"       # 원본 (Teacher) 경로
    STUDENT_MODEL = "./awq_base"  # 학습된 (Student) 경로

    # 측정 실행
    measure_speed(BASE_MODEL, "Teacher (Base 30L)")
    measure_speed(STUDENT_MODEL, "Student (AWQ W4A16)")