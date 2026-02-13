#!/usr/bin/env python3
"""
Knowledge Distillation for Causal LM — 개선된 버전

🔧 주요 개선사항:
  1. EOS 토큰을 반드시 데이터에 포함
  2. 학습 후 모델 검증 (EOS 테스트)
  3. generation_config.json 저장
  4. 더 강한 EOS 가중치 (→ 100.0)

실행:
  python train_kd_fixed.py --epochs 3 --eos_weight 100.0
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
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm


class Method(str, Enum):
    KD     = "kd"
    COSINE = "cosine"
    MSE    = "mse"


# ────────────── Student Builder ─────────────

def build_student(teacher, n_layers: int = 15):
    """
    Bottom-Heavy + Top 전략으로 Student 생성.
    """
    cfg = teacher.config.to_dict()
    orig = cfg["num_hidden_layers"]

    # ── Bottom-Heavy + Top 인덱스 계산 ──
    n_bottom = n_layers - 1
    top_idx  = orig - 1
    indices  = list(range(n_bottom)) + [top_idx]

    cfg["num_hidden_layers"] = n_layers
    if isinstance(cfg.get("layer_types"), list):
        cfg["layer_types"] = [cfg["layer_types"][i] for i in indices]

    student = teacher.__class__(teacher.config.__class__.from_dict(cfg))

    # ── 가중치 복사 ──
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    copied = 0

    # 비-레이어 파라미터
    layer_pat = re.compile(r"(\.layers\.)(\d+)(\..*)")
    embed_ignore = ["embed_tokens", "lm_head"]
    
    for k in t_sd:
        if not layer_pat.search(k):
            if any(ig in k for ig in embed_ignore):
                continue
            if k in s_sd and t_sd[k].shape == s_sd[k].shape:
                s_sd[k] = t_sd[k].clone()
                copied += 1

    # 레이어 파라미터
    for new_idx, old_idx in enumerate(indices):
        for k, v in t_sd.items():
            m = layer_pat.search(k)
            if m and int(m.group(2)) == old_idx:
                new_k = k[: m.start()] + m.group(1) + str(new_idx) + m.group(3)
                if new_k in s_sd and v.shape == s_sd[new_k].shape:
                    s_sd[new_k] = v.clone()
                    copied += 1

    student.load_state_dict(s_sd, strict=False)
    print(f"[Student] {orig}L → {n_layers}L (copied={copied} params)")
    return student


# ────────────── Loss Functions ─────────────

def loss_kd(s_logits, t_logits, labels, T: float, alpha: float,
           eos_id: int | None = None, eos_weight: float = 1.0):
    """CE + KL-Divergence (EOS 강조)"""
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


# ────────────── Custom Dataset ─────────────

class EOSAwareDataset(Dataset):
    """EOS 토큰을 확실히 포함하는 데이터셋"""
    
    def __init__(self, texts, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        eos_token = tokenizer.eos_token or ""
        
        for text in texts:
            # EOS 토큰을 반드시 추가
            if not text.rstrip().endswith(eos_token):
                text = text.rstrip() + eos_token
            
            tokens = tokenizer.encode(
                text,
                add_special_tokens=False,  # 이미 EOS 있음
                truncation=True,
                max_length=max_len,
            )
            
            # EOS가 토큰에 포함되어 있는지 확인
            has_eos = tokenizer.eos_token_id in tokens
            
            self.samples.append({
                'tokens': tokens,
                'has_eos': has_eos,
                'text': text[:100] if len(text) > 100 else text,
            })
        
        # 통계 출력
        with_eos = sum(1 for s in self.samples if s['has_eos'])
        print(f"📚 Dataset 통계:")
        print(f"   - 총 샘플: {len(self.samples)}")
        print(f"   - EOS 포함: {with_eos}/{len(self.samples)}")
        if with_eos < len(self.samples) * 0.8:
            print(f"   ⚠️  경고: {100*(1-with_eos/len(self.samples)):.1f}%의 샘플에 EOS 없음!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample['tokens']
        
        # Padding
        if len(tokens) < self.max_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor([1] * len(sample['tokens']) + 
                                           [0] * (self.max_len - len(sample['tokens'])),
                                           dtype=torch.long),
        }


# ────────────── Dataset Prep ─────────────

def make_dataset_eos_safe(tokenizer, dataset_id: str, n_samples: int, max_len: int):
    """
    EOS를 반드시 포함하는 데이터셋 구성
    """
    print(f"[INFO] 데이터셋 로드: {dataset_id} ({n_samples} samples)")
    
    try:
        # MANTA-1M 로드
        ds = load_dataset(dataset_id, split=f"train[:{n_samples}]")
        
        texts = []
        for ex in ds:
            try:
                # Chat template 적용
                text = tokenizer.apply_chat_template(
                    ex["conversations"],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                texts.append(text)
            except Exception as e:
                print(f"  ⚠️  샘플 스킵: {e}")
        
        print(f"[INFO] 유효한 샘플: {len(texts)}/{n_samples}")
        
        # Custom Dataset (EOS 강제 포함)
        dataset = EOSAwareDataset(texts, tokenizer, max_len)
        return dataset
        
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        print("[대안] 더미 데이터셋 생성 중...")
        
        # Fallback: 간단한 테스트 데이터
        texts = [
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a powerful tool for data analysis.",
        ] * (n_samples // 3)
        
        dataset = EOSAwareDataset(texts, tokenizer, max_len)
        return dataset


# ────────────── Validation ─────────────

def validate_eos_learning(model, tokenizer, device):
    """학습 후 EOS 토큰 학습 검증"""
    print(f"\n{'='*60}")
    print("✅ EOS 토큰 학습 검증 중...")
    print(f"{'='*60}")
    
    model.eval()
    
    prompts = [
        "What is AI?",
        "안녕하세요.",
        "Tell me a story.",
    ]
    
    eos_token_id = tokenizer.eos_token_id
    early_stops = 0
    runs_to_max = 0
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # EOS 토큰 있는지 확인
        if eos_token_id in outputs[0]:
            early_stops += 1
            eos_idx = (outputs[0] == eos_token_id).nonzero(as_tuple=True)[0]
            status = "✓ EOS 학습됨"
        else:
            runs_to_max += 1
            status = "✗ EOS 미학습 (max_new_tokens 도달)"
        
        print(f"  {status}: '{prompt[:30]}...'")
    
    print(f"\n결과: {early_stops}/{len(prompts)} 정상 (EOS 생성)")
    
    if early_stops >= len(prompts) * 0.8:
        print("✅ EOS 학습이 양호합니다!")
        return True
    else:
        print("❌ EOS 학습이 부족합니다. 더 많은 epoch이 필요할 수 있습니다.")
        return False


# ────────────── Main ─────────────

def main():
    parser = argparse.ArgumentParser(description="KD 학습 (개선 버전)")
    parser.add_argument("--teacher_model",    default="./base_model")
    parser.add_argument("--out_dir",          default="./KD_student_model_v4_fixed")
    parser.add_argument("--student_layers",   type=int, default=15)
    parser.add_argument("--dataset_id",       default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--num_samples",      type=int, default=2048)
    parser.add_argument("--max_seq_len",      type=int, default=1024)
    parser.add_argument("--epochs",           type=int, default=3)
    parser.add_argument("--batch_size",       type=int, default=4)
    parser.add_argument("--method",           type=Method, default=Method.KD)
    parser.add_argument("--temperature",      type=float, default=1.5)
    parser.add_argument("--alpha_ce",         type=float, default=0.8)
    parser.add_argument("--eos_weight",       type=float, default=100.0,
                        help="⭐ EOS 토큰 가중치 (기본 100.0 = 100배 강조)")
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--log_every",       type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Tokenizer ──
    tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Teacher & Student ──
    print("[INFO] Teacher 로드...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map='cpu',
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    print(f"[INFO] Student 생성 ({args.student_layers} layers)...")
    student = build_student(teacher, args.student_layers)
    student = student.to(dtype=torch.bfloat16, device=device)

    teacher = teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # ── Dataset 로드 (EOS 포함 보장) ──
    dataset = make_dataset_eos_safe(tok, args.dataset_id, args.num_samples, args.max_seq_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    loader   = DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collator)

    # ── 학습 정보 ──
    print(f"\n{'='*60}")
    print(f"  Method          : {args.method.value}")
    print(f"  Student Layers  : {args.student_layers}")
    print(f"  Temperature     : {args.temperature}")
    print(f"  EOS Weight      : ⭐ {args.eos_weight}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch Size      : {args.batch_size}")
    print(f"  LR              : {args.lr}")
    print(f"  Steps/epoch     : {len(loader)}")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    # ── Training Loop ──
    best_loss = float('inf')
    best_ckpt = Path(args.out_dir) / "best"

    for epoch in range(args.epochs):
        student.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch.get("labels", ids.clone()).to(device)

            # Teacher forward
            with torch.no_grad():
                t_out = teacher(
                    input_ids=ids,
                    attention_mask=mask,
                )

            # Student forward
            s_out = student(
                input_ids=ids,
                attention_mask=mask,
            )

            # Loss (EOS 강조)
            loss, ce, kl = loss_kd(
                s_out.logits, t_out.logits, labs,
                args.temperature, args.alpha_ce,
                eos_id=tok.eos_token_id, eos_weight=args.eos_weight,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"✓ Epoch {epoch+1} avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(best_ckpt, safe_serialization=True)
            tok.save_pretrained(best_ckpt)
            print(f"★ Best model saved: {best_ckpt}")

    # ── 최종 모델 저장 ──
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    student.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    # ✅ generation_config.json 저장
    gen_config = {
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": False,
    }
    with open(out / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    print(f"\n✅ 모델 저장 완료: {out}")

    # ── 학습 검증 ──
    print("\n[검증] EOS 토큰 학습 확인 중...")
    is_good = validate_eos_learning(student, tok, device)

    if is_good:
        print("\n✅ 학습이 성공적으로 완료되었습니다!")
        print(f"📦 모델 경로: {out}")
    else:
        print("\n⚠️  EOS 학습이 부족합니다. 더 많은 epoch을 시도하세요:")
        print(f"    python train_kd_fixed.py --epochs 5 --eos_weight 200.0")


if __name__ == "__main__":
    main()
