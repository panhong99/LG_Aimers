"""
Knowledge Distillation for Causal LM — Single GPU (v4)

설계:
  - Bottom-Heavy + Top 레이어 선택 전략 (Feature 연속성 보존)
  - Teacher 4-bit NF4 양자화 (OOM 방지)
  - SDPA Attention (Teacher & Student 모두)
  - 강화된 EOS 가중치 학습 (기본 10.0) → 생성 시 EOS 정상 출력
  - 단순화: 단일 GPU, 일반 PyTorch만 사용

3가지 KD 방법:
  kd     : CE + KL-Divergence (soft targets)
  cosine : CE + Cosine Embedding Loss (hidden states)
  mse    : CE + MSE Loss (hidden states, optional projector)

Usage:
  python train_kd.py --epochs 5 --eos_weight 10.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm


# ───────────────────────────── Enum ─────────────────────────────

class Method(str, Enum):
    KD     = "kd"
    COSINE = "cosine"
    MSE    = "mse"


# ─────────────── Student Builder (Bottom-Heavy + Top) ───────────

def build_student(teacher, n_layers: int = 15):
    """
    Bottom-Heavy + Top 전략으로 Student 생성.

    Teacher 30L → Student 15L 예시:
      Layer 0~13 (하위 14개, 연속) + Layer 29 (최상위 1개) = 총 15개
      → Feature 연속성 보존 + 최종 추론 레이어 유지

    Args:
        teacher: Teacher 모델
        n_layers: Student 레이어 수
    """
    cfg = teacher.config.to_dict()
    orig = cfg["num_hidden_layers"]

    # ── Bottom-Heavy + Top 인덱스 계산 ──
    n_bottom = n_layers - 1
    top_idx  = orig - 1
    indices  = list(range(n_bottom)) + [top_idx]

    assert len(indices) == n_layers, f"인덱스 수 불일치: {len(indices)} != {n_layers}"

    cfg["num_hidden_layers"] = n_layers
    if isinstance(cfg.get("layer_types"), list):
        cfg["layer_types"] = [cfg["layer_types"][i] for i in indices]

    student = teacher.__class__(teacher.config.__class__.from_dict(cfg))

    # ── 가중치 복사 ──
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    copied = 0

    # 비-레이어 파라미터 (embedding, lm_head, final_norm 등)
    layer_pat = re.compile(r"(\.layers\.)(\d+)(\..*)")
    embed_ignore = ["embed_tokens", "lm_head"]
    
    for k in t_sd:
        if not layer_pat.search(k):
            # embedding/lm_head는 스킵
            if any(ig in k for ig in embed_ignore):
                continue
            if k in s_sd and t_sd[k].shape == s_sd[k].shape:
                s_sd[k] = t_sd[k].clone()
                copied += 1

    # 레이어 파라미터: teacher[old_idx] → student[new_idx]
    for new_idx, old_idx in enumerate(indices):
        for k, v in t_sd.items():
            m = layer_pat.search(k)
            if m and int(m.group(2)) == old_idx:
                new_k = k[: m.start()] + m.group(1) + str(new_idx) + m.group(3)
                if new_k in s_sd and v.shape == s_sd[new_k].shape:
                    s_sd[new_k] = v.clone()
                    copied += 1

    student.load_state_dict(s_sd, strict=False)
    print(f"[Student] {orig}L → {n_layers}L  strategy=Bottom-Heavy+Top")
    print(f"  indices={indices}  copied={copied} params")
    return student


# ──────────────────── Projector (MSE method) ────────────────────

class Projector(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        return self.fc(x)


# ───────────────────────── Loss Functions ───────────────────────

def loss_kd(s_logits, t_logits, labels, T: float, alpha: float,
           eos_id: int | None = None, eos_weight: float = 1.0):
    """CE + KL-Divergence soft target loss (EOS 가중치 지원)."""
    s = s_logits[:, :-1].contiguous()
    t = t_logits[:, :-1].contiguous()
    lab = labels[:, 1:].contiguous()

    # ── EOS-weighted CE ──
    if eos_id is not None and eos_weight > 1.0:
        ce_per_tok = F.cross_entropy(
            s.view(-1, s.size(-1)), lab.view(-1),
            ignore_index=-100, reduction="none",
        )
        eos_mask = (lab.view(-1) == eos_id).float()
        weights = torch.ones_like(ce_per_tok) + eos_mask * (eos_weight - 1.0)
        valid = (lab.view(-1) != -100).float()
        ce = (ce_per_tok * weights * valid).sum() / (valid.sum() + 1e-8)
    else:
        ce = F.cross_entropy(s.view(-1, s.size(-1)), lab.view(-1), ignore_index=-100)

    kl = F.kl_div(
        F.log_softmax(s / T, dim=-1),
        F.softmax(t / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

    return alpha * ce + (1 - alpha) * kl, ce.item(), kl.item()


def loss_cosine(s_logits, t_logits, s_hid, t_hid, labels, alpha: float, beta: float):
    """CE + Cosine similarity on last hidden states."""
    s = s_logits[:, :-1].contiguous()
    lab = labels[:, 1:].contiguous()

    ce = F.cross_entropy(s.view(-1, s.size(-1)), lab.view(-1), ignore_index=-100)
    cos = nn.CosineEmbeddingLoss()(
        s_hid.mean(1),
        t_hid.mean(1),
        torch.ones(s_hid.size(0), device=s_hid.device),
    )

    return alpha * ce + beta * cos, ce.item(), cos.item()


def loss_mse(s_logits, t_logits, s_hid, t_hid, labels,
             alpha: float, beta: float, proj=None):
    """CE + MSE on hidden states (with optional projector)."""
    s = s_logits[:, :-1].contiguous()
    lab = labels[:, 1:].contiguous()

    ce = F.cross_entropy(s.view(-1, s.size(-1)), lab.view(-1), ignore_index=-100)
    if proj is not None:
        s_hid = proj(s_hid)
    mse = F.mse_loss(s_hid, t_hid)

    return alpha * ce + beta * mse, ce.item(), mse.item()


# ──────────────────────── Dataset Prep ──────────────────────────

def make_dataset(tokenizer, dataset_id: str, split: str, n_samples: int, max_len: int):
    """
    LGAI-EXAONE/MANTA-1M 데이터셋 로드 & 토크나이즈.
    
    ⚠️ 중요: EOS 토큰 처리
    ────────────────────────────────────
    
    [잘못된 방식] - 모델이 언제 끝나는지 모름
      text = f"{prompt}\n{response}"  # EOS 없음
      → 모델이 계속 생성 (max_new_tokens까지)
    
    [올바른 방식] - 모델이 EOS 학습
      text = f"{prompt}\n{response}{tokenizer.eos_token}"
      → 모델이 EOS 생성하는 법 배움
    
    max_seq_len 설정과 무관하게:
      - max_seq_len=512로 설정해도, 데이터의 마지막 토큰이 잘릴 수 있음
      - 따라서 tokenize 전에 반드시 EOS 토큰을 텍스트에 추가해야 함
      - tokenize 후 truncate되면 EOS가 손실될 수 있으므로 미리 붙임
    """
    ds = load_dataset(dataset_id, split=f"{split}[:{n_samples}]")

    eos_token = tokenizer.eos_token or ""

    def _chat_to_text(ex):
        text = tokenizer.apply_chat_template(
            ex["conversations"],
            add_generation_prompt=False,   # 학습: 전체 대화 포함
            tokenize=False,
        )
        # ✓ EOS 토큰을 명시적으로 추가 (tokenize 전)
        # 이렇게 해야 truncate되어도 일부 EOS 신호가 남음
        if eos_token and not text.rstrip().endswith(eos_token):
            text = text.rstrip() + eos_token
        return {"text": text}

    ds = ds.map(_chat_to_text, remove_columns=ds.column_names)
    ds = ds.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_len),
        batched=True,
        remove_columns=["text"],
    )
    return ds


# ───────────────────────────── Main ─────────────────────────────

def main():
    # ── CLI Args ──
    parser = argparse.ArgumentParser(description="KD for CausalLM (DeepSpeed v3)")
    parser.add_argument("--teacher_model",    default="./base_model")
    parser.add_argument("--out_dir",          default="./KD_student_model_v4")
    parser.add_argument("--student_layers",   type=int, default=15)
    parser.add_argument("--dataset_id",       default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split",    default="train")
    parser.add_argument("--num_samples",      type=int, default=2048)
    parser.add_argument("--max_seq_len",      type=int, default=1024,
                        help="충분히 길어야 EOS 토큰 학습 가능 (권장: 512+)")
    parser.add_argument("--epochs",           type=int, default=5)
    parser.add_argument("--batch_size",       type=int, default=2)
    parser.add_argument("--method",           type=Method, default=Method.KD,
                        choices=list(Method))
    parser.add_argument("--temperature",      type=float, default=1.5)
    parser.add_argument("--alpha_ce",         type=float, default=0.8)
    parser.add_argument("--beta_aux",         type=float, default=0.2)
    parser.add_argument("--eos_weight",       type=float, default=10.0,
                        help="EOS 토큰 CE 가중치 (기본 10.0 = 10배 강조)")
    parser.add_argument("--log_every",        type=int, default=10)
    parser.add_argument("--save_freq",       type=int, default=100,
                        help="Checkpoint 저장 주기 (steps)")
    parser.add_argument("--checkpoint_dir",  type=str, default="./checkpoints",
                        help="Checkpoint 저장 디렉토리")
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--weight_decay",    type=float, default=0.0)
    args = parser.parse_args()

    # ── 단일 GPU 설정 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[WARNING] GPU를 사용할 수 없습니다. CPU로 실행합니다.")
    rank = 0

    # ── Tokenizer ──
    tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Phase 1: Teacher bf16 로드 → Student 가중치 복사 ──
    if rank == 0:
        print(f"[INFO] Teacher 로드 (bf16, 가중치 복사용): {args.teacher_model}")

    teacher_bf16 = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if rank == 0:
        print(f"[INFO] Student 생성 ({args.student_layers} layers, Bottom-Heavy+Top)")

    student = build_student(teacher_bf16, args.student_layers)
    student = student.to(dtype=torch.bfloat16, device=device)

    if hasattr(student.config, "_attn_implementation"):
        student.config._attn_implementation = "sdpa"

    # ── Phase 2: Teacher bf16 유지 (정확한 KD target) ──
    # bf16 teacher는 그대로 유지 (4-bit는 정보 손실 → KD 성능 저하)
    # VRAM: ~6GB (batch size 2에서 총 ~17GB, 충분함)
    teacher = teacher_bf16.to(device)
    if rank == 0:
        print(f"[INFO] Teacher 유지 (bf16 정밀도, KD target 정확도 최대): {args.teacher_model}")

    for p in teacher.parameters():
        p.requires_grad = False

    # ── Projector (MSE method) ──
    proj = None
    extra_params = []
    if args.method == Method.MSE:
        sd, td = student.config.hidden_size, teacher.config.hidden_size
        if sd != td:
            proj = Projector(sd, td).to(dtype=torch.bfloat16, device=device)
            extra_params = list(proj.parameters())
            if rank == 0:
                print(f"[INFO] Projector: {sd} → {td}")

    # ── Dataset & DataLoader ──
    if rank == 0:
        print("[INFO] 데이터셋 준비 중...")
    ds = make_dataset(tok, args.dataset_id, args.dataset_split,
                      args.num_samples, args.max_seq_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    loader   = DataLoader(ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collator)

    if rank == 0:
        print(f"[INFO] 데이터셋: {len(ds)} 샘플, Steps/epoch: {len(loader)}")

    # ── 학습 정보 출력 ──
    if rank == 0:
        t_p = sum(p.numel() for p in teacher.parameters()) / 1e9
        s_p = sum(p.numel() for p in student.parameters()) / 1e9
        print(f"\n{'='*60}")
        print(f"  Method          : {args.method.value}")
        print(f"  Layer Strategy  : Bottom-Heavy + Top")
        print(f"  Teacher         : {t_p:.2f}B params (4-bit NF4)")
        print(f"  Student         : {s_p:.2f}B params ({s_p/t_p*100:.0f}%)")
        print(f"  Temperature     : {args.temperature}")
        print(f"  α(CE)={args.alpha_ce}  β(aux)={args.beta_aux}")
        print(f"  Epochs={args.epochs}  Batch size={args.batch_size}  Device={device}")
        print(f"  max_seq_len     : {args.max_seq_len}")
        print(f"  num_samples     : {args.num_samples}")
        print(f"  Steps/epoch     : {len(loader)}")
        print(f"  Optimizer       : AdamW (lr={args.lr}, wd={args.weight_decay})")
        print(f"  EOS weight      : {args.eos_weight} (가중치 학습)")
        print(f"{'='*60}\n")

    params = list(student.parameters()) + extra_params
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # ── Checkpoint 디렉토리 준비 ──
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    best_ckpt_path = ckpt_dir / "best_model"

    # ── Training Loop ──
    need_hidden = args.method != Method.KD
    global_step = 0

    for epoch in range(args.epochs):
        student.train()
        running = {"loss": 0.0, "ce": 0.0, "aux": 0.0}

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in pbar:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch.get("labels", ids.clone()).to(device)

            # Teacher forward (no grad, 4-bit inference)
            with torch.no_grad():
                t_out = teacher(
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=need_hidden,
                )

            # Student forward
            s_out = student(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=need_hidden,
            )

            # Loss 계산
            if args.method == Method.KD:
                loss, ce, aux = loss_kd(
                    s_out.logits, t_out.logits, labs,
                    args.temperature, args.alpha_ce,
                    eos_id=tok.eos_token_id, eos_weight=args.eos_weight,
                )
            elif args.method == Method.COSINE:
                loss, ce, aux = loss_cosine(
                    s_out.logits, t_out.logits,
                    s_out.hidden_states[-1], t_out.hidden_states[-1],
                    labs, args.alpha_ce, args.beta_aux,
                )
            else:  # MSE
                loss, ce, aux = loss_mse(
                    s_out.logits, t_out.logits,
                    s_out.hidden_states[-1], t_out.hidden_states[-1],
                    labs, args.alpha_ce, args.beta_aux, proj,
                )

            # Backward + step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Logging
            running["loss"] += loss.item()
            running["ce"]   += ce
            running["aux"]  += aux
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # ── Periodic Checkpoint ──
            if global_step % args.save_freq == 0:
                ckpt_path = ckpt_dir / f"step_{global_step}"
                student.save_pretrained(ckpt_path, safe_serialization=True)
                if proj is not None:
                    torch.save(proj.state_dict(), ckpt_path / "projector.pt")
                print(f"  [CKPT] Step {global_step}: {ckpt_path}")

            if global_step % args.log_every == 0:
                n = args.log_every
                pbar.set_postfix(
                    loss=f"{running['loss']/n:.4f}",
                    ce=f"{running['ce']/n:.4f}",
                    aux=f"{running['aux']/n:.4f}",
                )
                running = {"loss": 0.0, "ce": 0.0, "aux": 0.0}

        # ── Epoch 완료: Best loss 확인 ──
        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"  ✓ Epoch {epoch+1} 완료 (avg_loss={avg_epoch_loss:.4f}, global_step={global_step})")
        
        # ── Best model 저장 ──
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            student.save_pretrained(best_ckpt_path, safe_serialization=True)
            tok.save_pretrained(best_ckpt_path)
            if proj is not None:
                torch.save(proj.state_dict(), best_ckpt_path / "projector.pt")
            print(f"  ★ Best model 업데이트: loss={best_loss:.4f} → {best_ckpt_path}")

    # ── 모델 저장 (단순화: DDP 미사용) ──
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    student.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    if proj is not None:
        torch.save(proj.state_dict(), out / "projector.pt")

    # 저장 검증: lm_head shape 확인
    from safetensors.torch import safe_open
    sf_path = out / "model.safetensors"
    if sf_path.exists():
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "lm_head" in key or "embed" in key:
                    t = f.get_tensor(key)
                    status = "✓" if t.numel() > 0 else "✗ BROKEN"
                    print(f"  {status} {key}: {t.shape}")

    print(f"\n[INFO] 저장 완료: {out}")


if __name__ == "__main__":
    main()
