#!/usr/bin/env python3
"""
Layer-reduced KD training pipeline (with optional LoRA refinement)

핵심 목표:
1) 레이어 축소 + KD로 지식 유지
2) 단일 데이터셋 편향 완화 (멀티 데이터셋 혼합)
3) EOS/종료 토큰 학습 안정화
4) KD 직후 품질 저하 시 LoRA 보강 학습 옵션 제공

예시:
python main_kd.py \
  --teacher_model ./base_model \
  --out_dir ./KD_student_strategy \
  --student_layers 24 \
  --dataset_specs "LGAI-EXAONE/MANTA-1M:0.6,HuggingFaceH4/ultrachat_200k:0.4" \
  --num_samples 4096 \
  --epochs 2 \
  --run_lora_refine \
  --lora_epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


@dataclass
class DatasetSpec:
    dataset_id: str
    weight: float
    split: str = "train"


def parse_dataset_specs(raw: str) -> list[DatasetSpec]:
    """
    format:
      dataset_id:weight,dataset_id:weight
    optional split:
      dataset_id@split:weight
    """
    specs: list[DatasetSpec] = []
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("--dataset_specs가 비어있습니다.")
    for item in items:
        left, weight_str = item.rsplit(":", 1)
        weight = float(weight_str)
        if "@" in left:
            dataset_id, split = left.split("@", 1)
        else:
            dataset_id, split = left, "train"
        specs.append(DatasetSpec(dataset_id=dataset_id, split=split, weight=weight))
    total_w = sum(s.weight for s in specs)
    if total_w <= 0:
        raise ValueError("dataset weight 합이 0 이하입니다.")
    for s in specs:
        s.weight = s.weight / total_w
    return specs


def _extract_messages_like(example: dict[str, Any]) -> list[dict[str, str]] | None:
    for key in ("conversations", "messages"):
        if key in example and isinstance(example[key], list) and len(example[key]) > 0:
            msgs: list[dict[str, str]] = []
            for x in example[key]:
                if isinstance(x, dict):
                    role = x.get("role") or x.get("from") or "user"
                    content = x.get("content") or x.get("value") or ""
                    if isinstance(content, str) and content.strip():
                        msgs.append({"role": str(role), "content": content})
            if msgs:
                return msgs
    return None


def _fallback_text(example: dict[str, Any]) -> str | None:
    text_keys = ["text", "prompt", "instruction", "input", "output", "response", "answer"]
    parts: list[str] = []
    for k in text_keys:
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    if parts:
        return "\n".join(parts)
    return None


def render_training_text(example: dict[str, Any], tok) -> str | None:
    msgs = _extract_messages_like(example)
    if msgs is not None and hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                tokenize=False,
            )
        except Exception:
            pass
    return _fallback_text(example)


def load_mixed_texts(tok, specs: list[DatasetSpec], total_samples: int, seed: int) -> list[str]:
    random.seed(seed)
    texts: list[str] = []
    eos_token = tok.eos_token or ""

    for spec in specs:
        n_take = max(1, int(total_samples * spec.weight))
        print(f"[DATA] {spec.dataset_id}@{spec.split} -> target {n_take} samples")
        try:
            ds = load_dataset(spec.dataset_id, split=f"{spec.split}[:{n_take}]")
        except Exception as e:
            print(f"[WARN] dataset load 실패: {spec.dataset_id} ({e})")
            continue

        cur: list[str] = []
        for ex in ds:
            txt = render_training_text(ex, tok)
            if not txt:
                continue
            if eos_token and not txt.rstrip().endswith(eos_token):
                txt = txt.rstrip() + eos_token
            cur.append(txt)
        print(f"[DATA] valid={len(cur)}")
        texts.extend(cur)

    random.shuffle(texts)
    texts = texts[:total_samples]
    if not texts:
        raise RuntimeError("유효 텍스트가 0개입니다. dataset specs를 확인하세요.")
    print(f"[DATA] total valid texts={len(texts)}")
    return texts


def build_hf_dataset(tok, texts: list[str], max_len: int):
    from datasets import Dataset

    ds = Dataset.from_dict({"text": texts})

    def _tok(batch):
        out = tok(
            batch["text"],
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )
        return {"input_ids": out["input_ids"]}

    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    ds = ds.filter(lambda x: len(x["input_ids"]) > 1)
    return ds


def choose_layer_indices(
    orig_layers: int,
    target_layers: int,
    strategy: str,
    keep_first: int,
    keep_last: int,
) -> list[int]:
    if target_layers >= orig_layers:
        return list(range(orig_layers))

    if strategy == "uniform":
        return sorted(set(round(i * (orig_layers - 1) / (target_layers - 1)) for i in range(target_layers)))

    if strategy == "bottom_top":
        k_bottom = target_layers // 2
        bottom = list(range(k_bottom))
        top = list(range(orig_layers - (target_layers - k_bottom), orig_layers))
        return sorted(set((bottom + top)))[:target_layers]

    # sandwich (default): 초반/후반 유지 + 중간 균등 샘플
    keep_first = min(keep_first, target_layers)
    keep_last = min(keep_last, target_layers - keep_first)
    fixed = list(range(keep_first)) + list(range(orig_layers - keep_last, orig_layers))
    fixed = sorted(set(fixed))
    remain = target_layers - len(fixed)
    if remain <= 0:
        return fixed[:target_layers]

    middle_candidates = [i for i in range(orig_layers) if i not in fixed]
    picked: list[int] = []
    for j in range(remain):
        idx = round(j * (len(middle_candidates) - 1) / max(remain - 1, 1))
        picked.append(middle_candidates[idx])
    return sorted(set(fixed + picked))[:target_layers]


def build_student_from_teacher(
    teacher,
    target_layers: int,
    layer_strategy: str,
    keep_first: int,
    keep_last: int,
):
    cfg = teacher.config.to_dict()
    orig_layers = int(cfg["num_hidden_layers"])
    indices = choose_layer_indices(orig_layers, target_layers, layer_strategy, keep_first, keep_last)
    cfg["num_hidden_layers"] = len(indices)
    if isinstance(cfg.get("layer_types"), list):
        cfg["layer_types"] = [cfg["layer_types"][i] for i in indices]

    student = teacher.__class__(teacher.config.__class__.from_dict(cfg))
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()

    layer_pat = re.compile(r"(\.layers\.)(\d+)(\..*)")
    copied = 0

    # non-layer weights copy
    for k, v in t_sd.items():
        if not layer_pat.search(k) and k in s_sd and s_sd[k].shape == v.shape:
            s_sd[k] = v.clone()
            copied += 1

    # selected layer remap
    for new_idx, old_idx in enumerate(indices):
        for k, v in t_sd.items():
            m = layer_pat.search(k)
            if m and int(m.group(2)) == old_idx:
                new_k = k[: m.start()] + m.group(1) + str(new_idx) + m.group(3)
                if new_k in s_sd and s_sd[new_k].shape == v.shape:
                    s_sd[new_k] = v.clone()
                    copied += 1

    student.load_state_dict(s_sd, strict=False)
    print(f"[Student] {orig_layers}L -> {len(indices)}L, strategy={layer_strategy}, indices={indices}")
    print(f"[Student] copied tensors={copied}")
    return student, indices


def kd_loss_eos_aware(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha_ce: float,
    eos_id: int | None,
    eos_weight: float,
) -> tuple[torch.Tensor, float, float]:
    s = s_logits[:, :-1, :].contiguous()
    t = t_logits[:, :-1, :].contiguous()
    y = labels[:, 1:].contiguous()
    bsz, seqlen, vocab = s.shape

    ce_tok = F.cross_entropy(
        s.view(-1, vocab),
        y.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(bsz, seqlen)
    valid = (y != -100).float()

    if eos_id is not None and eos_weight > 1.0:
        eos_mask = (y == eos_id).float()
        weights = 1.0 + eos_mask * (eos_weight - 1.0)
        ce = (ce_tok * weights * valid).sum() / (valid.sum() + 1e-8)
    else:
        ce = (ce_tok * valid).sum() / (valid.sum() + 1e-8)

    kl_all = F.kl_div(
        F.log_softmax(s / temperature, dim=-1),
        F.softmax(t / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (temperature * temperature)

    if eos_id is not None:
        non_eos = ((y != eos_id) & (y != -100)).float()
        kl = (kl_all * non_eos).sum() / (non_eos.sum() + 1e-8)
    else:
        kl = (kl_all * valid).sum() / (valid.sum() + 1e-8)

    loss = alpha_ce * ce + (1.0 - alpha_ce) * kl
    return loss, float(ce.detach().cpu()), float(kl.detach().cpu())


def quick_generation_check(model, tok, device, max_new_tokens: int = 128):
    prompts = [
        "안녕하세요. 당신은 누구입니까?",
        "인공지능의 미래에 대해 설명해주세요.",
        "파이썬과 자바의 차이를 간단히 알려줘.",
    ]
    model.eval()
    stop_count = 0
    for p in prompts:
        x = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **x,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        gen = out.shape[1] - x["input_ids"].shape[1]
        has_eos = tok.eos_token_id in out[0][x["input_ids"].shape[1]:]
        if has_eos:
            stop_count += 1
        print(f"[CHECK] gen={gen:4d}, eos={has_eos}, prompt='{p[:20]}'")
    print(f"[CHECK] eos-stop-rate={stop_count}/{len(prompts)}")


def run_optional_lora_refine(args, student, tok, train_loader, device):
    if not args.run_lora_refine or args.lora_epochs <= 0:
        return student
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        print(f"[WARN] peft import 실패로 LoRA 단계 스킵: {e}")
        return student

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(student, lora_cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lora_lr, weight_decay=0.01)

    print(f"[LoRA] start refine epochs={args.lora_epochs}, modules={target_modules}")
    for ep in range(args.lora_epochs):
        running = 0.0
        pbar = tqdm(train_loader, desc=f"LoRA {ep+1}/{args.lora_epochs}")
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu())
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        print(f"[LoRA] epoch={ep+1} avg_loss={running/max(len(train_loader),1):.4f}")

    # merge and return base model
    merged = model.merge_and_unload()
    print("[LoRA] merged into base student model")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Strategic layer-reduced KD training")
    parser.add_argument("--teacher_model", type=str, default="./base_model")
    parser.add_argument("--out_dir", type=str, default="./KD_student_v1")
    parser.add_argument("--student_layers", type=int, default=20)
    parser.add_argument("--layer_strategy", type=str, default="sandwich", choices=["sandwich", "uniform", "bottom_top"])
    parser.add_argument("--keep_first", type=int, default=6)
    parser.add_argument("--keep_last", type=int, default=4)

    parser.add_argument(
        "--dataset_specs",
        type=str,
        default="LGAI-EXAONE/MANTA-1M:0.7,HuggingFaceH4/ultrachat_200k:0.3",
        help="dataset@split:weight,dataset@split:weight",
    )
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--alpha_ce", type=float, default=0.8)
    parser.add_argument("--eos_weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run_lora_refine", action="store_true")
    parser.add_argument("--lora_epochs", type=int, default=1)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    added_pad = False
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
        added_pad = True
    print(f"[INFO] eos={tok.eos_token_id} {repr(tok.eos_token)}, pad={tok.pad_token_id} {repr(tok.pad_token)}")

    print("[INFO] loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    if added_pad:
        teacher.resize_token_embeddings(len(tok))

    student, picked = build_student_from_teacher(
        teacher=teacher,
        target_layers=args.student_layers,
        layer_strategy=args.layer_strategy,
        keep_first=args.keep_first,
        keep_last=args.keep_last,
    )
    if added_pad:
        student.resize_token_embeddings(len(tok))

    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = student.to(device).train()

    specs = parse_dataset_specs(args.dataset_specs)
    texts = load_mixed_texts(tok, specs, args.num_samples, args.seed)
    train_ds = build_hf_dataset(tok, texts, args.max_seq_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    print("[INFO] KD training start")
    print(f"[INFO] layers={len(picked)} strategy={args.layer_strategy} samples={len(train_ds)} steps/epoch={len(train_loader)}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    best_loss = math.inf
    best_dir = Path(args.out_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        student.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"KD {ep+1}/{args.epochs}")
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                t_logits = teacher(input_ids=ids, attention_mask=mask).logits
            s_logits = student(input_ids=ids, attention_mask=mask).logits
            loss, ce, kl = kd_loss_eos_aware(
                s_logits=s_logits,
                t_logits=t_logits,
                labels=labels,
                temperature=args.temperature,
                alpha_ce=args.alpha_ce,
                eos_id=tok.eos_token_id,
                eos_weight=args.eos_weight,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            lv = float(loss.detach().cpu())
            running += lv
            pbar.set_postfix(loss=f"{lv:.4f}", ce=f"{ce:.4f}", kl=f"{kl:.4f}")

        avg = running / max(len(train_loader), 1)
        print(f"[KD] epoch={ep+1} avg_loss={avg:.4f}")
        quick_generation_check(student, tok, device, max_new_tokens=128)

        if avg < best_loss:
            best_loss = avg
            student.save_pretrained(best_dir, safe_serialization=True)
            tok.save_pretrained(best_dir)
            print(f"[KD] best saved -> {best_dir}")

    student = run_optional_lora_refine(args, student, tok, train_loader, device)
    quick_generation_check(student, tok, device, max_new_tokens=128)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    gen_cfg = {
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "do_sample": False,
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    with open(out / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, ensure_ascii=False, indent=2)

    meta = {
        "teacher_model": args.teacher_model,
        "student_layers": len(picked),
        "picked_layer_indices": picked,
        "layer_strategy": args.layer_strategy,
        "dataset_specs": [spec.__dict__ for spec in specs],
        "num_samples": len(train_ds),
        "kd": {
            "epochs": args.epochs,
            "temperature": args.temperature,
            "alpha_ce": args.alpha_ce,
            "eos_weight": args.eos_weight,
            "lr": args.lr,
        },
        "lora_refine": {
            "enabled": args.run_lora_refine,
            "epochs": args.lora_epochs,
            "lr": args.lora_lr,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        },
    }
    with open(out / "training_recipe.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved: {out}")
    print("[NEXT] KD 모델 품질 확인 후 AWQ/GPTQ를 적용하세요.")


if __name__ == "__main__":
    main()
