import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json as json_lib

# ============================================================================
# 1. 모델 파라미터 출력
# ============================================================================
def print_model_params(model_path: str) -> None:
    print("\n" + "=" * 70)
    print("모델 정보")
    print("=" * 70)
    
    # GPTQ 모델 감지
    config_path = Path(model_path) / "config.json"
    is_gptq = False
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json_lib.load(f)
        is_gptq = 'quantization_config' in config and config['quantization_config'].get('quant_method') == 'gptq'
    
    # 모델 로드 (GPTQ 여부에 따라)
    if is_gptq:
        print("[INFO] GPTQ 모델 감지 - AutoGPTQForCausalLM으로 로드")
        try:
            from auto_gptq import AutoGPTQForCausalLM
            model = AutoGPTQForCausalLM.from_quantized(
                model_path,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[WARNING] GPTQ 로드 실패: {e}")
            print("[FALLBACK] AutoModelForCausalLM으로 시도")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
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
    total_prompt_tokens = 0

    # 로컬에서 프롬프트 토큰 길이 추정 (서버 usage가 없을 때 대비)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    
    messages = [{"role": "user", "content": args.vllm_prompt}]
    payload_template = {
        "model": args.model_path,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": int(args.vllm_max_tokens),
    }

    # 프롬프트 토큰 수 추정
    try:
        if args.apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
            prompt_tokens_est = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            prompt_tokens_est = len(prompt_tokens_est)
        else:
            prompt_tokens_est = len(tokenizer.encode(args.vllm_prompt))
        print(f"[프롬프트 토큰 추정] {prompt_tokens_est} tokens")
    except Exception as e:
        print(f"[경고] 프롬프트 토큰 추정 실패: {e}")
        prompt_tokens_est = None
    
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
            
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            total_time += elapsed
            total_tokens += completion_tokens
            total_prompt_tokens += prompt_tokens
            
            # 첫 번째 요청일 때만 질문과 답변 출력
            if i == 0:
                print(f"\n[질문 (Prompt)]")
                print(f"  {args.vllm_prompt}")
                
                choices = result.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "")
                    print(f"\n[답변 (Response)]")
                    print(f"  {response_text}")
                print()
            
            if prompt_tokens:
                print(f"  [{i+1}/{int(args.vllm_num_requests)}] {elapsed:.3f}s, prompt {prompt_tokens}, completion {completion_tokens} tokens")
            else:
                print(f"  [{i+1}/{int(args.vllm_num_requests)}] {elapsed:.3f}s, {completion_tokens} tokens")
        except URLError as e:
            print(f"[ERROR] vLLM 연결 실패: {e}")
            return
    
    if total_tokens > 0:
        tpt = total_time / total_tokens
        tps = 1 / tpt
        print(f"\n[결과] TPT: {tpt:.6f}s/token ({tps:.2f} tokens/s)")
        if total_prompt_tokens > 0:
            avg_prompt = total_prompt_tokens / int(args.vllm_num_requests)
            print(f"[결과] 평균 프롬프트 토큰: {avg_prompt:.1f}")
        if prompt_tokens_est:
            est_total = prompt_tokens_est + int(args.vllm_max_tokens)
            est_time = est_total / tps
            print(f"[예상] (프롬프트+생성) {est_total} tokens -> {est_time/60:.2f} min")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="모델 평가 스크립트")
    parser.add_argument("--model_path", type=str, default="./trainer_output_v6", help="평가할 모델 경로")
    parser.add_argument("--lm_eval_tasks", type=str, default="hellaswag", help="평가할 작업 (쉼표로 구분)")
    parser.add_argument("--lm_eval_batch_size", type=str, default="auto", help="배치 크기")
    parser.add_argument("--lm_eval_device", type=str, default="cuda:0", help="평가 디바이스")
    parser.add_argument("--vllm_base_url", type=str, default="http://127.0.0.1:8000", help="vLLM 서버 주소")
    parser.add_argument("--vllm_prompt", type=str, default="where is the capital of france?", help="vLLM 프롬프트")
    parser.add_argument("--vllm_num_requests", type=int, default=5, help="vLLM 요청 횟수")
    parser.add_argument("--vllm_max_tokens", type=int, default=4096, help="최대 생성 토큰")
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
