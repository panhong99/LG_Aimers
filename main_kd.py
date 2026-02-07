"""Teacher-Student Knowledge Distillation for Causal LM"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def build_student_from_teacher(teacher_model, num_hidden_layers: int, 
                               hidden_size: int = None, num_attention_heads: int = None, 
                               intermediate_size: int = None):
    """Create a smaller student model by slicing layers and reducing dimensions."""
    # Clone config and shrink layers & dimensions
    cfg_dict = teacher_model.config.to_dict()
    cfg_dict["num_hidden_layers"] = num_hidden_layers
    
    # Shrink embedding and FFN dimensions
    if hidden_size is not None:
        cfg_dict["hidden_size"] = hidden_size
    if num_attention_heads is not None:
        cfg_dict["num_attention_heads"] = num_attention_heads
    if intermediate_size is not None:
        cfg_dict["intermediate_size"] = intermediate_size
    
    if "layer_types" in cfg_dict and isinstance(cfg_dict["layer_types"], list):
        cfg_dict["layer_types"] = cfg_dict["layer_types"][:num_hidden_layers]

    config_cls = teacher_model.config.__class__
    student_config = config_cls.from_dict(cfg_dict)

    # Instantiate student model from same class
    student_model = teacher_model.__class__(student_config)

    # Copy matching weights from teacher to student (only matching shapes)
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    copied, skipped = 0, 0
    for k, v in student_state.items():
        tv = teacher_state.get(k)
        if tv is not None and tv.shape == v.shape:
            student_state[k] = tv.detach().clone()
            copied += 1
        else:
            skipped += 1
    student_model.load_state_dict(student_state, strict=False)

    print(f"[INFO] 학생 모델 설정: hidden_size={student_config.hidden_size}, "
          f"num_attention_heads={student_config.num_attention_heads}, "
          f"num_hidden_layers={student_config.num_hidden_layers}")
    print(f"[INFO] 학생 모델 가중치 복사: copied={copied}, skipped={skipped}")
    return student_model

class KDTrainer(Trainer):
    def __init__(self, teacher_model, temperature: float, alpha_ce: float, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha_ce = alpha_ce

        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits

        # Teacher 모델을 student와 같은 디바이스로 옮김
        device = student_logits.device
        self.teacher_model.to(device)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Shift for causal LM
        shift_student = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous() if labels is not None else None

        # CE loss
        if shift_labels is None:
            loss_ce = torch.tensor(0.0, device=student_logits.device)
        else:
            loss_ce = F.cross_entropy(
                shift_student.view(-1, shift_student.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # KD loss (mask padding)
        T = self.temperature
        if shift_labels is None:
            loss_kd = torch.tensor(0.0, device=student_logits.device)
        else:
            mask = shift_labels != -100
            if mask.any():
                s = F.log_softmax(shift_student / T, dim=-1)
                t = F.softmax(shift_teacher / T, dim=-1)
                s = s[mask]
                t = t[mask]
                loss_kd = F.kl_div(s, t, reduction="batchmean") * (T * T)
            else:
                loss_kd = torch.tensor(0.0, device=student_logits.device)

        loss = self.alpha_ce * loss_ce + (1.0 - self.alpha_ce) * loss_kd
        return (loss, outputs) if return_outputs else loss


def prepare_dataset(tokenizer, dataset_id: str, dataset_split: str, num_samples: int, max_seq_len: int):
    print("[INFO] 데이터셋 로드 중...")
    ds = load_dataset(dataset_id, split=f"{dataset_split}[:{num_samples}]")

    def preprocess(example):
        text = tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
        return {"text": text}

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_len,
            return_tensors=None,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    print(f"[INFO] 데이터셋 준비 완료: {len(ds)} 샘플")
    return ds


def main(args):
    print("[INFO] Teacher 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )

    print("[INFO] Student 모델 생성 중...")
    student_model = build_student_from_teacher(
        teacher_model, 
        num_hidden_layers=args.student_layers,
        hidden_size=args.student_hidden_size,
        num_attention_heads=args.student_num_attention_heads,
        intermediate_size=args.student_intermediate_size,
    )

    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        student_model.config.use_cache = False

    # Explicit device placement (teacher: cuda:0, student: cuda:1)
    teacher_model.to(args.teacher_device)
    student_model.to(args.student_device)

    dataset = prepare_dataset(
        tokenizer,
        args.dataset_id,
        args.dataset_split,
        args.num_train_samples,
        args.max_sequence_length,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        do_train=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        fp16=False,
        bf16=True,
        report_to="none",
        deepspeed=args.deepspeed,
    )

    trainer = KDTrainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        teacher_model=teacher_model,
        temperature=args.temperature,
        alpha_ce=args.alpha_ce,
    )

    print("[INFO] KD 학습 시작...")
    trainer.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[INFO] 학생 모델 저장 완료: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher-Student Knowledge Distillation")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed 분산 학습용 (자동 설정됨)")
    parser.add_argument("--teacher_model", type=str, default="./base_model")
    parser.add_argument("--out_dir", type=str, default="./KD_student_model")
    parser.add_argument("--student_layers", type=int, default=15)
    parser.add_argument("--student_hidden_size", type=int, default=None, help="hidden_size를 줄이지 않으면 None")
    parser.add_argument("--student_num_attention_heads", type=int, default=None, help="num_attention_heads를 줄이지 않으면 None")
    parser.add_argument("--student_intermediate_size", type=int, default=None, help="intermediate_size를 줄이지 않으면 None")

    parser.add_argument("--dataset_id", type=str, default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--num_train_samples", type=int, default=5000)
    parser.add_argument("--max_sequence_length", type=int, default=1024)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--logging_dir", type=str, default="./logs_kd")

    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha_ce", type=float, default=0.5)
    parser.add_argument("--no_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed 설정 (분산 학습용, 기본값 None)")
    parser.add_argument("--teacher_device", type=str, default="cuda:0")
    parser.add_argument("--student_device", type=str, default="cuda:1")

    args = parser.parse_args()
    args.gradient_checkpointing = not args.no_gradient_checkpointing
    main(args)
