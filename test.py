#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DACON Aimers 8기 모델 경량화 해커톤 - 로컬 점수 추정기 (비공식)

- 공개 규정에서 확인 가능한 정의:
  * PerfNorm = Perf_model / Perf_base  (기본 모델 대비 성능 비율)  ※ baseline: EXAONE-4.0-1.2B
  * SpeedNorm = (t_base - t_model) / t_base  (토큰당 추론시간 감소 비율, token-time 기준)
    - '제출 탭 시간'은 전체 실행시간이며, 평가 산식은 '토큰당 추론 시간'이라는 점을 명시
    - 따라서 측정은 모델 로드/전처리 제외, '실제 생성 구간'만 시간 측정

주의:
- 테스트 벤치셋은 비공개라서 Perf_model(진짜 성능)은 완전히 동일하게 재현 불가.
- 여기서는 "프록시 성능"으로 Perplexity 기반 점수를 사용(낮을수록 좋음 → perf=exp(-loss) 로 변환).
- SpeedNorm은 vLLM으로 '생성 구간만' 측정(가능하면 규정과 가장 유사).
- 최종 Score 결합식(가중치)은 규정 텍스트에서 확인 불가 → 옵션으로 둠.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Data
# -------------------------
def build_prompts_from_manta(tokenizer, n: int, start: int = 0) -> List[str]:
    """
    MANTA-1M conversations -> apply_chat_template(add_generation_prompt=True)
    """
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[{start}:{start+n}]")
    prompts = []
    for ex in ds:
        text = tokenizer.apply_chat_template(
            ex["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(text)
    return prompts


# -------------------------
# Perf proxy: perplexity-based
# -------------------------
@torch.no_grad()
def measure_proxy_perf_ppl(model, tokenizer, prompts: List[str], max_len: int, device: torch.device) -> float:
    """
    프록시 성능: prompts를 그대로 teacher-forcing해서 평균 CE loss 측정.
    perf = exp(-loss) 로 변환 (값이 클수록 좋음)
    """
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_tokens = 0

    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        # next-token LM loss
        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        loss = float(out.loss)
        # 토큰 수 가중 평균
        toks = int(attn.sum().item())
        total_loss += loss * toks
        total_tokens += toks

    mean_loss = total_loss / max(1, total_tokens)
    perf = math.exp(-mean_loss)
    return perf


# -------------------------
# Speed: vLLM token-time (preferred)
# -------------------------
def try_measure_token_time_vllm(model_path: str, tokenizer_path: str, prompts: List[str], max_new_tokens: int) -> float:
    """
    vLLM로 "생성 구간만" 시간을 측정해서 token_time(초/토큰) 반환.
    실패하면 예외 던짐.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_num_seqs=16,
    )

    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    # 워밍업(짧게)
    _ = llm.generate(prompts[:1], params)

    # 생성 구간 시간 측정
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    t1 = time.perf_counter()

    # 생성 토큰 수 집계
    gen_tokens = 0
    for o in outputs:
        # vLLM outputs: o.outputs[0].token_ids length == generated tokens
        if o.outputs:
            gen_tokens += len(o.outputs[0].token_ids)

    if gen_tokens == 0:
        raise RuntimeError("Generated tokens == 0; cannot compute token time.")

    return (t1 - t0) / gen_tokens


# -------------------------
# Speed: transformers fallback
# -------------------------
@torch.no_grad()
def measure_token_time_hf(model, tokenizer, prompts: List[str], max_new_tokens: int, device: torch.device) -> float:
    """
    HF generate로 대체 측정 (정확히 vLLM과 동일하진 않지만 로컬 확인용).
    """
    model.eval()
    model.to(device)

    # 워밍업
    _ = model.generate(**tokenizer(prompts[0], return_tensors="pt").to(device), max_new_tokens=8)

    total_gen_tokens = 0
    start = time.perf_counter()
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        # 생성 토큰 수(대략): output_len - input_len
        total_gen_tokens += max(0, out.shape[1] - enc["input_ids"].shape[1])
    end = time.perf_counter()

    if total_gen_tokens == 0:
        raise RuntimeError("Generated tokens == 0; cannot compute token time.")
    return (end - start) / total_gen_tokens


# -------------------------
# Main scoring
# -------------------------
@dataclass
class ScoreResult:
    perf_base: float
    perf_model: float
    perf_norm: float
    token_time_base: float
    token_time_model: float
    speed_norm: float
    score: float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="제출 모델 경로(./model 같은 로컬 폴더)")
    ap.add_argument("--base", default="LGAI-EXAONE/EXAONE-4.0-1.2B", help="기본 모델(기본값: HF ID)")
    ap.add_argument("--n_prompts", type=int, default=128, help="측정 프롬프트 개수")
    ap.add_argument("--start", type=int, default=0, help="MANTA에서 시작 인덱스")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--w_perf", type=float, default=0.5, help="최종 score 가중치(추정용)")
    ap.add_argument("--w_speed", type=float, default=0.5, help="최종 score 가중치(추정용)")
    ap.add_argument("--no_vllm", action="store_true", help="vLLM 측정 강제 비활성화(HF fallback)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer는 base로 로드(채팅 템플릿 일치)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    prompts = build_prompts_from_manta(tok, n=args.n_prompts, start=args.start)

    # --- Perf proxy ---
    base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    perf_base = measure_proxy_perf_ppl(base_model, tok, prompts, args.max_len, device)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    perf_model = measure_proxy_perf_ppl(model, tok, prompts, args.max_len, device)

    perf_norm = perf_model / max(1e-12, perf_base)

    # --- Speed token-time ---
    token_time_base = None
    token_time_model = None

    if (not args.no_vllm) and device.type == "cuda":
        try:
            token_time_base = try_measure_token_time_vllm(args.base, args.base, prompts, args.max_new_tokens)
            token_time_model = try_measure_token_time_vllm(args.model, args.model, prompts, args.max_new_tokens)
        except Exception as e:
            print(f"[WARN] vLLM 측정 실패 → HF fallback 사용: {e}")

    if token_time_base is None or token_time_model is None:
        token_time_base = measure_token_time_hf(base_model, tok, prompts, args.max_new_tokens, device)
        token_time_model = measure_token_time_hf(model, tok, prompts, args.max_new_tokens, device)

    # SpeedNorm: 감소 비율
    speed_norm = (token_time_base - token_time_model) / max(1e-12, token_time_base)

    # (추정) 최종 Score
    score = args.w_perf * perf_norm + args.w_speed * speed_norm

    res = ScoreResult(
        perf_base=perf_base,
        perf_model=perf_model,
        perf_norm=perf_norm,
        token_time_base=token_time_base,
        token_time_model=token_time_model,
        speed_norm=speed_norm,
        score=score,
    )

    print("\n" + "=" * 70)
    print("ESTIMATED SCORE (비공식 로컬 추정)")
    print("=" * 70)
    print(f"Perf_base  : {res.perf_base:.6f}   (proxy perf = exp(-loss))")
    print(f"Perf_model : {res.perf_model:.6f}")
    print(f"PerfNorm   : {res.perf_norm:.6f}   (= Perf_model / Perf_base)")
    print("-" * 70)
    print(f"TokenTime_base  : {res.token_time_base:.6e} sec/token")
    print(f"TokenTime_model : {res.token_time_model:.6e} sec/token")
    print(f"SpeedNorm       : {res.speed_norm:.6f}   (= (t_base - t_model)/t_base)")
    print("-" * 70)
    print(f"Score (w_perf={args.w_perf}, w_speed={args.w_speed}) : {res.score:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
