import os
import json
import time
from datetime import datetime
import torch
import shutil
import subprocess
from urllib import request
from urllib.error import URLError, HTTPError
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# ============================================================================
# 경로 설정 (절대경로로 통일)
# ============================================================================
WORKSPACE_DIR = Path(__file__).parent.resolve()
MODEL_BASE_PATH = WORKSPACE_DIR / "base_model"
MODEL_INT4_PATH = WORKSPACE_DIR / "model"
LM_EVAL_DIR = WORKSPACE_DIR / "lm-evaluation-harness"

# ============================================================================
# 데이터셋 설정
# ============================================================================
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# ============================================================================
# 양자화 설정
# ============================================================================
QUANTIZATION_SCHEME = "W4A16"
QUANTIZATION_TARGETS = ["Linear"]
QUANTIZATION_IGNORE = ["embed_tokens", "lm_head"]

# ============================================================================
# lm-eval-harness 설정
# ============================================================================
LM_EVAL_TASKS = ["hellaswag", "arc_challenge", "mmlu"]
LM_EVAL_NUM_FEWSHOT = 0
LM_EVAL_BATCH_SIZE = "auto"
LM_EVAL_DEVICE = "cuda:0"

# ============================================================================
# vLLM 설정 (OpenAI-compatible API)
# ============================================================================
VLLM_BASE_URL = "http://127.0.0.1:8000"
VLLM_MODEL_NAME_BASE = "./base_model"
VLLM_MODEL_NAME_INT4 = "./model"
VLLM_MAX_TOKENS = 128
VLLM_TEMPERATURE = 0.0
VLLM_NUM_REQUESTS = 5
VLLM_PROMPT = "안녕, 한 줄로 자기소개해줘."
VLLM_ENDPOINT = "auto"  # "auto", "chat", "completions", "responses"

# ============================================================================
# 파이프라인 제어
# ============================================================================
RUN_QUANTIZATION = False
RUN_EVAL_BASE = False
RUN_EVAL_INT4 = False
RUN_SPEED_BASE = True
RUN_SPEED_INT4 = True
RUN_ZIP = False

# ============================================================================
# 유틸리티 함수
# ============================================================================
def _ensure_path_exists(path: Path, is_dir: bool = True) -> Path:
    """경로 존재 확인 및 생성"""
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp() -> str:
    return datetime.now().isoformat().replace(":", "-")


def _load_latest_json(prefix: str) -> dict | None:
    files = sorted(WORKSPACE_DIR.glob(f"{prefix}_*.json"))
    if not files:
        return None
    with files[-1].open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_speed_json(prefix: str, model_name: str, endpoint: str, tpt: float, total_time: float, total_tokens: int) -> Path:
    payload = {
        "model": model_name,
        "endpoint": endpoint,
        "tpt": tpt,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "timestamp": _timestamp(),
    }
    path = WORKSPACE_DIR / f"{prefix}_{payload['timestamp']}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _run_lm_eval(model_path: Path, output_name: str) -> dict:
    """lm-eval-harness 실행
    
    Args:
        model_path: 모델 경로
        output_name: 결과 저장 이름 (예: "eval_base")
    """
    if not LM_EVAL_DIR.exists():
        raise RuntimeError(f"lm-eval-harness 디렉토리가 없습니다: {LM_EVAL_DIR}")
    
    tasks = ",".join(LM_EVAL_TASKS)
    
    # output_path가 디렉토리면 lm-eval이 모델명 하위 폴더를 만들어 저장함
    # 따라서 "파일 경로"를 넘겨서 결과가 WORKSPACE_DIR에 직접 떨어지게 함
    output_file = (WORKSPACE_DIR / f"{output_name}.json").resolve()
    
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", tasks,
        "--num_fewshot", str(LM_EVAL_NUM_FEWSHOT),
        "--batch_size", LM_EVAL_BATCH_SIZE,
        "--device", LM_EVAL_DEVICE,
        "--output_path", str(output_file),
    ]

    print(f"\n[INFO] lm-eval 실행: {tasks}")
    print(f"[INFO] 모델: {model_path}")
    print(f"[INFO] 결과 저장 경로: {output_file.parent}")
    
    # WORKSPACE_DIR에서 실행
    subprocess.run(cmd, cwd=str(WORKSPACE_DIR), check=True)

    # 결과 파일 찾기
    print(f"\n[INFO] 결과 파일 검색 중...")
    
    # 패턴 1: eval_base_*.json
    result_files = sorted(WORKSPACE_DIR.glob(f"{output_name}_*.json"))
    print(f"  패턴 '{output_name}_*.json' 검색: {len(result_files)}개 발견")
    
    # 패턴 2: results_*.json (모든 결과 파일)
    all_results = sorted(WORKSPACE_DIR.glob("results_*.json"))
    print(f"  패턴 'results_*.json' 검색: {len(all_results)}개 발견")
    
    # 패턴 3: WORKSPACE_DIR의 모든 json 파일 확인
    all_json = sorted(WORKSPACE_DIR.glob("*.json"))
    print(f"  WORKSPACE_DIR의 모든 JSON: {len(all_json)}개")
    if all_json:
        for jf in all_json[-5:]:  # 최근 5개만 보여줌
            print(f"    - {jf.name}")
    
    # 최적 파일 선택
    result_files = sorted(WORKSPACE_DIR.glob(f"{output_name}_*.json")) + sorted(WORKSPACE_DIR.glob("results_*.json"))
    
    if not result_files:
        # 모든 json 파일 목록 출력
        print(f"\n[ERROR] 결과 파일을 찾을 수 없습니다!")
        print(f"  찾는 패턴: {output_name}_*.json 또는 results_*.json")
        print(f"  검색 위치: {WORKSPACE_DIR}")
        print(f"  WORKSPACE_DIR 내용:")
        for item in sorted(WORKSPACE_DIR.glob("*")):
            if item.is_file() and item.suffix == '.json':
                print(f"    - {item.name}")
        raise FileNotFoundError(f"lm-eval 결과 파일을 찾을 수 없습니다: {output_name}_*.json or results_*.json in {WORKSPACE_DIR}")
    
    result_path = result_files[-1]  # 가장 최신 파일 사용
    print(f"\n[INFO] 선택된 결과 파일: {result_path.name}")
    
    with result_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    
    print(f"[INFO] 결과 로드 완료: {result_path}")
    return results


def _extract_perf_score(results: dict) -> float:
    """lm-eval 결과에서 평균 성능 추출"""
    task_results = results.get("results", {})
    scores = []
    
    for task in LM_EVAL_TASKS:
        metrics = task_results.get(task, {})
        
        # 다양한 키 형식 처리 (acc_norm, acc, acc,none, acc_norm,none 등)
        score = None
        for key in metrics.keys():
            # 'acc'를 포함하면서 'stderr'는 제외하는 키 찾기
            if 'acc' in key and 'stderr' not in key:
                score = metrics[key]
                break
        
        if score is None:
            print(f"[WARNING] {task}에서 accuracy 메트릭을 찾지 못함")
            print(f"         사용 가능한 키: {list(metrics.keys())}")
            continue
        
        scores.append(score)
        print(f"  {task}: {score:.4f}")
    
    if not scores:
        raise ValueError(f"평가 점수를 추출할 수 없습니다")
    
    avg_score = sum(scores) / len(scores)
    return avg_score


def _vllm_request(base_url: str, endpoint: str, payload: dict) -> dict:
    """vLLM OpenAI-compatible API 호출 (chat/completions/responses)"""
    if endpoint not in ("chat", "completions", "responses"):
        raise ValueError(f"지원하지 않는 endpoint: {endpoint}")
    if endpoint == "chat":
        path = "/v1/chat/completions"
    elif endpoint == "completions":
        path = "/v1/completions"
    else:
        path = "/v1/responses"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError) as exc:
        raise RuntimeError(f"vLLM 호출 실패 ({base_url}, endpoint={endpoint}): {exc}") from exc

def _vllm_request_auto(base_url: str, payload_chat: dict, payload_completion: dict, payload_responses: dict) -> dict:
    """chat/completions/responses 엔드포인트를 자동 탐색하여 호출"""
    try:
        return _vllm_request(base_url, "chat", payload_chat)
    except RuntimeError as exc:
        if "HTTP Error 404" not in str(exc):
            raise
    try:
        return _vllm_request(base_url, "completions", payload_completion)
    except RuntimeError as exc:
        if "HTTP Error 404" not in str(exc):
            raise
    return _vllm_request(base_url, "responses", payload_responses)

def _measure_vllm_tpt(base_url: str, model_name: str) -> tuple[float, float, int]:
    """vLLM을 통한 처리량(throughput) 측정: Time Per Token"""
    total_time = 0.0
    total_completion_tokens = 0

    if VLLM_ENDPOINT == "chat":
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": VLLM_PROMPT}],
            "temperature": VLLM_TEMPERATURE,
            "max_tokens": VLLM_MAX_TOKENS,
        }
    elif VLLM_ENDPOINT == "completions":
        payload = {
            "model": model_name,
            "prompt": VLLM_PROMPT,
            "temperature": VLLM_TEMPERATURE,
            "max_tokens": VLLM_MAX_TOKENS,
        }
    elif VLLM_ENDPOINT == "responses":
        payload = {
            "model": model_name,
            "input": [{"role": "user", "content": VLLM_PROMPT}],
            "temperature": VLLM_TEMPERATURE,
            "max_output_tokens": VLLM_MAX_TOKENS,
        }
    else:
        payload = None

    print(f"\n[INFO] vLLM 속도 측정 시작")
    print(f"  모델: {model_name}")
    print(f"  요청 수: {VLLM_NUM_REQUESTS}")
    
    for i in range(VLLM_NUM_REQUESTS):
        start = time.perf_counter()
        if VLLM_ENDPOINT == "auto":
            payload_chat = {
                "model": model_name,
                "messages": [{"role": "user", "content": VLLM_PROMPT}],
                "temperature": VLLM_TEMPERATURE,
                "max_tokens": VLLM_MAX_TOKENS,
            }
            payload_completion = {
                "model": model_name,
                "prompt": VLLM_PROMPT,
                "temperature": VLLM_TEMPERATURE,
                "max_tokens": VLLM_MAX_TOKENS,
            }
            payload_responses = {
                "model": model_name,
                "input": [{"role": "user", "content": VLLM_PROMPT}],
                "temperature": VLLM_TEMPERATURE,
                "max_output_tokens": VLLM_MAX_TOKENS,
            }
            resp = _vllm_request_auto(base_url, payload_chat, payload_completion, payload_responses)
        else:
            resp = _vllm_request(base_url, VLLM_ENDPOINT, payload)
        elapsed = time.perf_counter() - start

        usage = resp.get("usage", {})
        completion_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        
        if completion_tokens == 0:
            raise RuntimeError(f"vLLM 응답에 completion_tokens가 없습니다: {resp}")

        total_time += elapsed
        total_completion_tokens += completion_tokens
        print(f"  [{i+1}/{VLLM_NUM_REQUESTS}] {elapsed:.3f}s, {completion_tokens} tokens")

    tpt = total_time / total_completion_tokens
    print(f"[INFO] TPT (Time Per Token) = {tpt:.6f}s/token")
    return tpt, total_time, total_completion_tokens


# ============================================================================
# 메인 파이프라인
# ============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("LLM 양자화 평가 파이프라인")
    print("=" * 70)
    
    # 경로 검증
    print(f"\n[검증] 경로 확인:")
    print(f"  워크스페이스: {WORKSPACE_DIR}")
    print(f"  Base 모델: {MODEL_BASE_PATH} ({'✓' if MODEL_BASE_PATH.exists() else '✗'})")
    print(f"  INT4 모델: {MODEL_INT4_PATH} ({'✓' if MODEL_INT4_PATH.exists() else '✗'})")
    print(f"  lm-eval: {LM_EVAL_DIR} ({'✓' if LM_EVAL_DIR.exists() else '✗'})")
    
    # 단계 1: 양자화
    if RUN_QUANTIZATION:
        print("\n" + "=" * 70)
        print("단계 1: GPTQ 양자화")
        print("=" * 70)
        
        if not MODEL_BASE_PATH.exists():
            raise RuntimeError(f"Base 모델이 없습니다: {MODEL_BASE_PATH}")
        
        print("[INFO] 모델 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_BASE_PATH),
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_BASE_PATH),
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("[INFO] 완료")

        print("[INFO] 캘리브레이션 데이터 로드 중...")
        ds = load_dataset(
            DATASET_ID,
            split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
        )

        def preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["conversations"],
                    add_generation_prompt=True,
                    tokenize=False)
            }

        ds = ds.map(preprocess)
        print("[INFO] 완료")

        print(f"[INFO] GPTQ 양자화 시작 (scheme={QUANTIZATION_SCHEME})...")
        recipe = [
            GPTQModifier(
                scheme=QUANTIZATION_SCHEME,
                targets=QUANTIZATION_TARGETS,
                ignore=QUANTIZATION_IGNORE,
            )
        ]

        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        )
        print("[INFO] 완료")

        _ensure_path_exists(MODEL_INT4_PATH)
        model.save_pretrained(MODEL_INT4_PATH, save_compressed=True)
        tokenizer.save_pretrained(MODEL_INT4_PATH)
        print(f"[INFO] 모델 저장됨: {MODEL_INT4_PATH}")

    # 단계 2: Base 모델 평가
    base_perf = None
    if RUN_EVAL_BASE:
        print("\n" + "=" * 70)
        print("단계 2: Base 모델 성능 평가 (lm-eval)")
        print("=" * 70)
        base_results = _run_lm_eval(MODEL_BASE_PATH, "eval_base")
        base_perf = _extract_perf_score(base_results)
        print(f"[결과] Base Perf = {base_perf:.6f}")

    # 단계 3: INT4 모델 평가
    int4_perf = None
    if RUN_EVAL_INT4:
        print("\n" + "=" * 70)
        print("단계 3: INT4 모델 성능 평가 (lm-eval)")
        print("=" * 70)
        int4_results = _run_lm_eval(MODEL_INT4_PATH, "eval_int4")
        int4_perf = _extract_perf_score(int4_results)
        print(f"[결과] INT4 Perf = {int4_perf:.6f}")

    # 단계 4: Base 모델 속도 측정
    base_tpt = None
    if RUN_SPEED_BASE:
        print("\n" + "=" * 70)
        print("단계 4: Base 모델 속도 측정 (vLLM)")
        print("=" * 70)
        print("[안내] vLLM 서버가 Base 모델로 실행 중인지 확인하세요.")
        print("[안내] 실행: vllm serve ./base_model --gpu-memory-utilization 0.9")
        input("[대기] 준비되면 Enter를 누르세요... ")
        base_tpt, base_time, base_tokens = _measure_vllm_tpt(VLLM_BASE_URL, VLLM_MODEL_NAME_BASE)
        speed_path = _write_speed_json("speed_base", VLLM_MODEL_NAME_BASE, VLLM_ENDPOINT, base_tpt, base_time, base_tokens)
        print(f"[INFO] 속도 결과 저장: {speed_path.name}")
        print(f"[결과] Base TPT = {base_tpt:.6f}s/token")

    # 단계 5: INT4 모델 속도 측정
    int4_tpt = None
    if RUN_SPEED_INT4:
        print("\n" + "=" * 70)
        print("단계 5: INT4 모델 속도 측정 (vLLM)")
        print("=" * 70)
        print("[안내] vLLM 서버를 INT4 모델로 바꿔주세요.")
        print("[안내] 실행: vllm serve ./model --gpu-memory-utilization 0.9")
        input("[대기] 준비되면 Enter를 누르세요... ")
        int4_tpt, int4_time, int4_tokens = _measure_vllm_tpt(VLLM_BASE_URL, VLLM_MODEL_NAME_INT4)
        speed_path = _write_speed_json("speed_int4", VLLM_MODEL_NAME_INT4, VLLM_ENDPOINT, int4_tpt, int4_time, int4_tokens)
        print(f"[INFO] 속도 결과 저장: {speed_path.name}")
        print(f"[결과] INT4 TPT = {int4_tpt:.6f}s/token")

    # 단계 6: 결과 계산 및 출력
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)

    perf_norm = None
    speed_norm = None
    score_hat = None

    if base_perf is None:
        base_saved = _load_latest_json("eval_base")
        if base_saved:
            base_perf = _extract_perf_score(base_saved)
            print(f"\n[로컬] Base Perf (json) = {base_perf:.6f}")

    if int4_perf is None:
        int4_saved = _load_latest_json("eval_int4")
        if int4_saved:
            int4_perf = _extract_perf_score(int4_saved)
            print(f"[로컬] INT4 Perf (json) = {int4_perf:.6f}")

    if base_tpt is None:
        base_speed = _load_latest_json("speed_base")
        if base_speed:
            base_tpt = base_speed.get("tpt")
            print(f"\n[로컬] Base TPT (json) = {base_tpt:.6f}s/token")

    if int4_tpt is None:
        int4_speed = _load_latest_json("speed_int4")
        if int4_speed:
            int4_tpt = int4_speed.get("tpt")
            print(f"[로컬] INT4 TPT (json) = {int4_tpt:.6f}s/token")

    if base_perf is not None and int4_perf is not None:
        perf_norm = int4_perf / base_perf
        print(f"\n[성능] PerfNorm = INT4_Perf / Base_Perf")
        print(f"       = {int4_perf:.6f} / {base_perf:.6f}")
        print(f"       = {perf_norm:.6f}")
        print(f"       {'✓ 양자화로 성능 유지' if perf_norm >= 0.95 else '✗ 성능 저하'}")

    if base_tpt is not None and int4_tpt is not None:
        speed_norm = 1.0 - (int4_tpt / base_tpt)
        speedup_ratio = base_tpt / int4_tpt
        print(f"\n[속도] SpeedNorm = 1 - (INT4_TPT / Base_TPT)")
        print(f"       = 1 - ({int4_tpt:.6f} / {base_tpt:.6f})")
        print(f"       = {speed_norm:.6f}")
        print(f"       처리 속도: {speedup_ratio:.2f}배 {'빠름' if speedup_ratio > 1 else '느림'}")

    if perf_norm is not None and speed_norm is not None:
        score_hat = 0.5 * perf_norm + 0.5 * speed_norm
        print(f"\n[최종] Score_hat = 0.5 * PerfNorm + 0.5 * SpeedNorm")
        print(f"       = 0.5 * {perf_norm:.6f} + 0.5 * {speed_norm:.6f}")
        print(f"       = {score_hat:.6f}")

    # 단계 7: ZIP 생성
    if RUN_ZIP:
        print("\n" + "=" * 70)
        print("단계 6: 제출 파일 생성")
        print("=" * 70)
        zip_name = str(WORKSPACE_DIR / "baseline_submit")
        print(f"[INFO] 생성 중: {zip_name}.zip")
        shutil.make_archive(
            base_name=zip_name,
            format="zip",
            root_dir=str(WORKSPACE_DIR),
            base_dir="model",
        )
        print(f"[INFO] 완료: {zip_name}.zip")

    print("\n" + "=" * 70)
    print("파이프라인 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
