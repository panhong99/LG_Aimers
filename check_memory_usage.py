# repro_leaderboard_like.py
import os
import gc
import time
import json
import psutil
import torch

def rss_gb() -> float:
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024**3)

def vram_gb(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024**3)

def log(stage: str):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"[{stage}] RSS={rss_gb():.3f} GB | VRAM(alloc)={vram_gb():.3f} GB")

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def main():
    # ====== CONFIG (보스 환경에 맞게 수정) ======
    MODEL_DIR = os.environ.get("MODEL_DIR", "./models/model_KD_v6_GPTQ_v2")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 리더보드 고정 옵션(스샷 기준)
    MAX_GEN_TOKS = int(os.environ.get("MAX_GEN_TOKS", "16384"))
    GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.85"))
    TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
    APPLY_CHAT_TEMPLATE = os.environ.get("APPLY_CHAT_TEMPLATE", "true").lower() == "true"

    # 리더보드가 prompt 길이를 공개하지 않으므로, 길이를 바꿔가며 재현 가능하게 함
    # SHORT / MEDIUM / LONG 프롬프트를 단계별로 테스트
    prompts = {
        "short": "Explain what mutual information is in simple terms.",
        "medium": "You are an assistant. Summarize the pros and cons of quantization for LLMs, "
                  "and suggest safe defaults for serving on a single GPU.",
        "long": ("You are a helpful assistant. " * 512) + "\nPlease answer briefly.\n",
    }

    print("=== Leaderboard-like repro ===")
    print(f"MODEL_DIR={MODEL_DIR}")
    print(f"MAX_GEN_TOKS={MAX_GEN_TOKS}, GPU_MEM_UTIL={GPU_MEM_UTIL}, TP_SIZE={TP_SIZE}, APPLY_CHAT_TEMPLATE={APPLY_CHAT_TEMPLATE}")
    log("start")

    # ========= (1) Transformers 로딩 단계 =========
    cleanup()
    log("before transformers import")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cleanup()
    log("before tokenizer load")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    log("after tokenizer load")

    cleanup()
    log("before model load (HF)")
    # 리더보드 문서에는 torch_dtype/low_cpu_mem_usage 지정이 없음 -> 동일 조건 재현을 위해 일부러 생략
    # 필요하면 아래 주석 해제해서 dtype 강제로 비교 가능
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    log("after model load (HF)")

    # 모델을 GPU로 올리는 단계(리더보드가 내부적으로 어떻게 하는지 불명이라 비교용)
    cleanup()
    log("before model.to(cuda)")
    if torch.cuda.is_available():
        model = model.to(DEVICE)
    log("after model.to(cuda)")

    # ========= (2) vLLM 엔진 초기화 단계 =========
    # 리더보드는 vLLM 0.14.1 사용. 여기서는 LLM 엔진을 직접 띄워 KV cache/버퍼 예약을 재현.
    cleanup()
    log("before vLLM init")

    try:
        from vllm import LLM, SamplingParams
        # vLLM이 max_model_len을 따로 잡을 수 있음.
        # 리더보드는 max_gen_toks 고정이므로, 재현용으로 max_model_len을 크게 주면(또는 None) 더 빡셈.
        # 여기서는 안전하게 "prompt 길이 + max_gen" 근사로 잡되, 테스트마다 바꿀 수 있게 함.
        llm = LLM(
            model=MODEL_DIR,
            tensor_parallel_size=TP_SIZE,
            gpu_memory_utilization=GPU_MEM_UTIL,
            trust_remote_code=True,
            # dtype은 리더보드가 명시 안함 -> 기본 경로 재현
            # dtype="bfloat16",  # 비교용: 주석 해제 가능
        )
        log("after vLLM init")
    except Exception as e:
        log("vLLM init FAILED")
        print("vLLM init exception:", repr(e))
        raise

    # ========= (3) 실제 생성 단계 =========
    # 리더보드 옵션: max_gen_toks=16384, apply_chat_template=true
    # apply_chat_template은 tokenizer에 chat_template이 있는 경우 대회 스크립트에서 적용될 수 있으니,
    # 여기서도 가능한 한 비슷하게 적용.
    def maybe_apply_chat_template(text: str) -> str:
        if not APPLY_CHAT_TEMPLATE:
            return text
        # chat_template이 있으면 apply. 없으면 원문 유지.
        if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
            messages = [{"role": "user", "content": text}]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text

    for name, raw in prompts.items():
        cleanup()
        log(f"before generate [{name}]")
        prompt = maybe_apply_chat_template(raw)

        sampling = SamplingParams(
            max_tokens=MAX_GEN_TOKS,
            temperature=0.0,
        )
        try:
            t0 = time.time()
            out = llm.generate([prompt], sampling_params=sampling)
            dt = time.time() - t0
            log(f"after generate [{name}] (t={dt:.2f}s)")
            # 출력 길이만 간단히 확인
            gen_text = out[0].outputs[0].text if out and out[0].outputs else ""
            print(f"--- [{name}] generated chars: {len(gen_text)} ---")
        except Exception as e:
            log(f"generate FAILED [{name}]")
            print(f"generate exception [{name}]:", repr(e))
            # 여기서 죽으면 거의 확실히 KV cache / 길이 / 버퍼 관련
            raise

    print("=== DONE ===")

if __name__ == "__main__":
    main()