import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# 1. 모델 파라미터 출력
# ============================================================================
def print_model_params(model_path: str) -> None:
    print("\n" + "=" * 70)
    print("모델 정보")
    print("=" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"파라미터: {total_params / 1e9:.2f}B")
    
    del model


# ============================================================================
# 2. eval-harness 평가
# ============================================================================
def run_eval(args) -> None:
    print("\n" + "=" * 70)
    print("lm-eval 성능 평가 시작")
    print("=" * 70)
    
    if not Path(args.lm_eval_dir).exists():
        raise RuntimeError(f"lm-eval-harness 디렉토리 없음: {args.lm_eval_dir}")
    
    tasks = ",".join(args.lm_eval_tasks)
    timestamp = datetime.now().isoformat().replace(":", "-")
    output_file = Path(args.lm_eval_dir) / f"eval_{timestamp}.json"
    
    cmd = f"""CUDA_VISIBLE_DEVICES=1 python -m lm_eval \
        --model hf \
        --model_args pretrained={args.model_path},trust_remote_code=True \
        --tasks {tasks} \
        --num_fewshot 0 \
        --batch_size {args.lm_eval_batch_size} \
        --device {args.lm_eval_device} \
        --output_path {output_file}"""
    
    result = subprocess.run(cmd, shell=True, cwd=str(os.getcwd()))
    
    if result.returncode != 0:
        raise RuntimeError("eval 실패")
    
    # 결과 파일 찾기
    result_files = sorted(Path(os.getcwd()).glob(f"eval_{timestamp.split('T')[0]}*.json"))
    if result_files:
        with result_files[-1].open("r", encoding="utf-8") as f:
            results = json.load(f)
            task_results = results.get("results", {})
            print(f"\n[결과]")
            for task, metrics in task_results.items():
                for key in metrics.keys():
                    if "acc" in key and "stderr" not in key:
                        print(f"  {task}: {metrics[key]:.4f}")
                        break


# ============================================================================
# 3. vLLM 속도 측정
# ============================================================================
def measure_vllm_speed(args) -> None:
    print("\n" + "=" * 70)
    print("vLLM 속도 측정")
    print("=" * 70)
    print(f"[안내] vLLM 서버가 실행 중인지 확인하세요.")
    print(f"       vllm serve {args.model_path}")
    input("[대기] 준비되면 Enter를 누르세요... ")
    
    from urllib import request
    from urllib.error import URLError
    
    total_time = 0.0
    total_tokens = 0
    
    payload_template = {
        "model": args.model_path,
        "messages": [{"role": "user", "content": args.vllm_prompt}],
        "temperature": 0.0,
        "max_tokens": int(args.vllm_max_tokens),
    }
    
    for i in range(int(args.vllm_num_requests)):
        try:
            start = time.perf_counter()
            data = json.dumps(payload_template).encode("utf-8")
            req = request.Request(
                f"{args.vllm_base_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - start
            
            completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
            total_time += elapsed
            total_tokens += completion_tokens
            
            print(f"  [{i+1}/{int(args.vllm_num_requests)}] {elapsed:.3f}s, {completion_tokens} tokens")
        except URLError as e:
            print(f"[ERROR] vLLM 연결 실패: {e}")
            return
    
    if total_tokens > 0:
        tpt = total_time / total_tokens
        print(f"\n[결과] TPT: {tpt:.6f}s/token ({1/tpt:.2f} tokens/s)")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="모델 평가 스크립트")
    parser.add_argument("--model_path", type=str, default="./KD_student_model", help="평가할 모델 경로")
    parser.add_argument("--lm_eval_tasks", type=str, default="hellaswag", help="평가할 작업 (쉼표로 구분)")
    parser.add_argument("--lm_eval_batch_size", type=str, default="auto", help="배치 크기")
    parser.add_argument("--lm_eval_device", type=str, default="cuda:0", help="평가 디바이스")
    parser.add_argument("--vllm_base_url", type=str, default="http://127.0.0.1:8000", help="vLLM 서버 주소")
    parser.add_argument("--vllm_prompt", type=str, default="where is the capital of france?", help="vLLM 프롬프트")
    parser.add_argument("--vllm_num_requests", type=int, default=5, help="vLLM 요청 횟수")
    parser.add_argument("--vllm_max_tokens", type=int, default=256, help="최대 생성 토큰")
    parser.add_argument("--lm_eval_dir", type=str, default="./lm-evaluation-harness", help="lm-eval 디렉토리")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM 텐서 병렬화 크기")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="GPU 메모리 사용률")
    parser.add_argument("--apply_chat_template", type=bool, default=True, help="채팅 템플릿 적용")
    
    args = parser.parse_args()
    
    # lm_eval_tasks를 리스트로 변환
    args.lm_eval_tasks = [t.strip() for t in args.lm_eval_tasks.split(",")]
    
    # print_model_params(args.model_path)
    # run_eval(args)
    measure_vllm_speed(args)
    print("\n" + "=" * 70)
    print("완료")
    print("=" * 70)
