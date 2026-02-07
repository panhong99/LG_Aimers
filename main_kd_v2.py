"""
Knowledge Distillation for Causal LM — DeepSpeed Native

3가지 방법:
  kd     : CE + KL-Divergence (soft targets)
  cosine : CE + Cosine Embedding Loss (hidden states)
  mse    : CE + MSE Loss (hidden states, optional projector)

Usage:
  deepspeed --num_gpus=2 main_kd_v2.py --method kd --student_layers 15
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import deepspeed
from tqdm import tqdm


# ───────────────────────────── Enum ─────────────────────────────

class Method(str, Enum):
    KD     = "kd"
    COSINE = "cosine"
    MSE    = "mse"


# ───────────────────────── Student Builder ──────────────────────

def build_student(teacher, n_layers: int):
    """Teacher에서 균등 간격으로 레이어를 선택해 Student 생성."""
    cfg = teacher.config.to_dict()
    orig = cfg["num_hidden_layers"]
    indices = np.linspace(0, orig - 1, n_layers, dtype=int).tolist()

    cfg["num_hidden_layers"] = n_layers
    if isinstance(cfg.get("layer_types"), list):
        cfg["layer_types"] = [cfg["layer_types"][i] for i in indices]

    student = teacher.__class__(teacher.config.__class__.from_dict(cfg))

    # ── 가중치 복사 ──
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    copied = 0

    # 비-레이어 파라미터 먼저 복사 (embedding, lm_head 등)
    layer_pat = re.compile(r"(\.layers\.)(\d+)(\..*)")
    non_layer_keys = [k for k in t_sd if not layer_pat.search(k)]
    for k in non_layer_keys:
        if k in s_sd and t_sd[k].shape == s_sd[k].shape:
            s_sd[k] = t_sd[k].clone()
            copied += 1

    # 레이어 파라미터 복사
    for new_idx, old_idx in enumerate(indices):
        for k, v in t_sd.items():
            m = layer_pat.search(k)
            if m and int(m.group(2)) == old_idx:
                new_k = k[: m.start()] + m.group(1) + str(new_idx) + m.group(3)
                if new_k in s_sd and v.shape == s_sd[new_k].shape:
                    s_sd[new_k] = v.clone()
                    copied += 1

    student.load_state_dict(s_sd, strict=False)
    print(f"[Student] {orig}L → {n_layers}L  indices={indices}  copied={copied}")
    return student


# ──────────────────── Projector (MSE method) ────────────────────

class Projector(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        return self.fc(x)


# ───────────────────────── Loss Functions ───────────────────────

def loss_kd(s_logits, t_logits, labels, T: float, alpha: float):
    """CE + KL-Divergence soft target loss."""
    s = s_logits[:, :-1].contiguous()
    t = t_logits[:, :-1].contiguous()
    lab = labels[:, 1:].contiguous()

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
    """LGAI-EXAONE/MANTA-1M 데이터셋 로드 & 토크나이즈."""
    ds = load_dataset(dataset_id, split=f"{split}[:{n_samples}]")

    def _chat_to_text(ex):
        return {
            "text": tokenizer.apply_chat_template(
                ex["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
        }

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
    parser = argparse.ArgumentParser(description="KD for CausalLM (DeepSpeed)")
    parser.add_argument("--teacher_model",    default="./base_model")
    parser.add_argument("--out_dir",          default="./KD_student_model_v2")
    parser.add_argument("--student_layers",   type=int, default=15)
    parser.add_argument("--dataset_id",       default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split",    default="train")
    parser.add_argument("--num_samples",      type=int, default=4096)
    parser.add_argument("--max_seq_len",      type=int, default=512)
    parser.add_argument("--epochs",           type=int, default=3)
    parser.add_argument("--batch_size",       type=int, default=1)
    parser.add_argument("--method",           type=Method, default=Method.KD, choices=list(Method))
    parser.add_argument("--temperature",      type=float, default=1.5)
    parser.add_argument("--alpha_ce",         type=float, default=0.8)
    parser.add_argument("--beta_aux",         type=float, default=0.2)
    parser.add_argument("--log_every",        type=int, default=10)
    parser.add_argument("--ds_config",        type=str, default="./ds_kd_zero3.json",
                        help="DeepSpeed JSON config 경로")
    # DeepSpeed가 자동으로 --local_rank 주입
    parser.add_argument("--local_rank",       type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # ── Distributed 환경 ──
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # ── Tokenizer ──
    tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Teacher (frozen, 각 GPU에 복사) ──
    if rank == 0:
        print(f"[INFO] Teacher 로드: {args.teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── Student ──
    if rank == 0:
        print(f"[INFO] Student 생성 ({args.student_layers} layers)")
    student = build_student(teacher, args.student_layers).to(device)

    # ── Projector (MSE method) ──
    proj = None
    extra_params = []
    if args.method == Method.MSE:
        sd, td = student.config.hidden_size, teacher.config.hidden_size
        if sd != td:
            proj = Projector(sd, td).to(device)
            extra_params = list(proj.parameters())
            if rank == 0:
                print(f"[INFO] Projector: {sd} → {td}")

    # ── Dataset & DataLoader ──
    if rank == 0:
        print("[INFO] 데이터셋 준비 중...")
    ds = make_dataset(tok, args.dataset_id, args.dataset_split,
                      args.num_samples, args.max_seq_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    sampler  = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader   = DataLoader(ds, batch_size=args.batch_size,
                          sampler=sampler, collate_fn=collator)

    if rank == 0:
        print(f"[INFO] 데이터셋: {len(ds)} 샘플, Steps/epoch: {len(loader)}")

    # ── DeepSpeed 초기화 ──
    ds_config_path = args.ds_config
    with open(ds_config_path) as f:
        ds_config = json.load(f)

    params = list(student.parameters()) + extra_params

    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=student,
        model_parameters=params,
        config=ds_config,
    )

    # ── 학습 정보 출력 ──
    if rank == 0:
        t_p = sum(p.numel() for p in teacher.parameters()) / 1e9
        s_p = sum(p.numel() for p in student.parameters()) / 1e9
        print(f"\n{'='*55}")
        print(f"  Method       : {args.method.value}")
        print(f"  Teacher      : {t_p:.2f}B params")
        print(f"  Student      : {s_p:.2f}B params  ({s_p/t_p*100:.0f}%)")
        print(f"  Temperature  : {args.temperature}")
        print(f"  α(CE)={args.alpha_ce}  β(aux)={args.beta_aux}")
        print(f"  Epochs={args.epochs}  Batch/GPU={args.batch_size}  GPUs={world_size}")
        print(f"  Steps/epoch  : {len(loader)}")
        print(f"{'='*55}\n")

    # ── Training Loop ──
    need_hidden = args.method != Method.KD
    global_step = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        engine.train()
        running = {"loss": 0.0, "ce": 0.0, "aux": 0.0}

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=(rank != 0))

        for batch in pbar:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch.get("labels", ids.clone()).to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                t_out = teacher(
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=need_hidden,
                )

            # Student forward (through DeepSpeed engine)
            s_out = engine(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=need_hidden,
            )

            # Loss 계산
            if args.method == Method.KD:
                loss, ce, aux = loss_kd(
                    s_out.logits, t_out.logits, labs,
                    args.temperature, args.alpha_ce,
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

            # DeepSpeed backward + step (optimizer.step, zero_grad 자동)
            engine.backward(loss)
            engine.step()

            # Logging
            running["loss"] += loss.item()
            running["ce"]   += ce
            running["aux"]  += aux
            global_step += 1

            if global_step % args.log_every == 0 and rank == 0:
                n = args.log_every
                pbar.set_postfix(
                    loss=f"{running['loss']/n:.4f}",
                    ce=f"{running['ce']/n:.4f}",
                    aux=f"{running['aux']/n:.4f}",
                )
                running = {"loss": 0.0, "ce": 0.0, "aux": 0.0}

        if rank == 0:
            print(f"  ✓ Epoch {epoch+1} 완료 (global_step={global_step})")

    # ── 모델 저장 (rank 0) ──
    if rank == 0:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_to_save = engine.module
        model_to_save.save_pretrained(out)
        tok.save_pretrained(out)
        if proj is not None:
            torch.save(proj.state_dict(), out / "projector.pt")
        print(f"\n[INFO] 저장 완료: {out}")


if __name__ == "__main__":
    main()
