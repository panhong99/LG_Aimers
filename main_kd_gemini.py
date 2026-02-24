import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    BitsAndBytesConfig
)
from datasets import load_dataset
import wandb

# 제출용 모델 준비 유틸리티
from model_utils import ensure_weight_tying, create_generation_config, normalize_safetensors_keys


# =============================================================================
# 1. 설정 및 하이퍼파라미터
# =============================================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="./base_model")
    target_layers: int = field(default=18)  # ★ 수정: 24 (현재) → 18 (중간) → 15 (강함)
    teacher_model_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    extra_dataset_name: str = field(default="LGAI-EXAONE/MANTA-1M")
    max_seq_length: int = field(default=256)  # 512 -> 256 (할루시네이션 방지)
    dataset_size: int = field(default=20000)

@dataclass
class CustomTrainingArguments(TrainingArguments):
    alpha_kd: float = field(default=0.7)      # ★ 0.5 → 0.7 (프루닝 40%: 강한 KD)
    beta_hidden: float = field(default=0.08)   # ★ 0.05 → 0.08 (hidden feature 더 강하게)
    temperature: float = field(default=1.0)
    
    per_device_train_batch_size: int = field(default=8)  # ★ 4 → 8 (메모리 여유)
    gradient_accumulation_steps: int = field(default=4)  # ★ 8 → 4 (유효 BS 유지: 8*4=32)
    gradient_checkpointing: bool = field(default=True)
    optim: str = field(default="paged_adamw_8bit")
    bf16: bool = field(default=True)
    num_train_epochs: int = field(default=30)  # ★ 20 → 30 (프루닝 40% 보정, 1.5배)
    learning_rate: float = field(default=1.5e-5)  # 유지 (또는 2e-5로 약간 올림)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.1)
    save_strategy: str = field(default="epoch")
    logging_steps: int = field(default=10)
    max_grad_norm: float = field(default=1.0)
    report_to: str = field(default="wandb")
    logging_strategy: str = field(default="steps")
    run_name: str = field(default="KD-Training")
    wandb_project: str = field(default="LG-Aimers-KD", metadata={"help": "WandB 프로젝트 이름"})
    wandb_entity: str = field(default="", metadata={"help": "WandB entity (팀/사용자명, 비우면 default)"})

# =============================================================================
# 2. 스마트 프루닝 함수 (구조 유지)
# =============================================================================
def smart_prune_model(model, target_layers):
    total_layers = model.config.num_hidden_layers
    if target_layers >= total_layers: return model

    # 고정 프루닝: 중간 레이어 제거
    if total_layers == 30 and target_layers == 24:
        # 중간 12~17번 레이어 제거 (6개 삭제)
        selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    elif total_layers == 30 and target_layers == 18:
        # ★ 강 프루닝: 중간 8~21번 레이어 제거 (12개 삭제 = 40%)
        selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31][:18]
        # 더 균형잡힌 샌드위치: 앞 4, 뒤 4 + 중간 샘플링
        keep_start, keep_end = 4, 4
        middle_indices = list(range(keep_start, total_layers - keep_end, (total_layers - keep_start - keep_end) // (target_layers - keep_start - keep_end)))
        selected_indices = sorted(list(range(keep_start)) + middle_indices + list(range(total_layers - keep_end, total_layers)))[:target_layers]
    elif total_layers == 30 and target_layers == 15:
        # ★★ 극강 프루닝: 절반 제거 (50%)
        # 앞 3, 뒤 3 + 중간 9
        keep_start, keep_end = 3, 3
        middle_indices = list(range(keep_start, total_layers - keep_end, (total_layers - keep_start - keep_end) // (target_layers - keep_start - keep_end)))
        selected_indices = sorted(list(range(keep_start)) + middle_indices + list(range(total_layers - keep_end, total_layers)))[:target_layers]
    else:
        # 기본 샌드위치 전략: 앞 2, 뒤 4 보존
        keep_start, keep_end = 2, 4
        middle_total = total_layers - keep_start - keep_end
        middle_target = target_layers - keep_start - keep_end
        
        step = middle_total / middle_target
        middle_indices = [keep_start + int(i * step) for i in range(middle_target)]
        selected_indices = sorted(list(set(range(keep_start)) | set(middle_indices) | set(range(total_layers - keep_end, total_layers))))[:target_layers]
    
    print(f"✂️ Pruning: {total_layers} → {target_layers} layers (감소율: {(1-target_layers/total_layers)*100:.1f}%)")
    print(f"   선택된 레이어: {selected_indices}")

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = nn.ModuleList([model.model.layers[i] for i in selected_indices])
    model.config.num_hidden_layers = target_layers
    return model

# =============================================================================
# 3. 최적화된 Custom KD Trainer
# =============================================================================
class KDTrainer(Trainer):
    def __init__(self, teacher_model=None, alpha_kd=1.0, beta_hidden=1.0, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.alpha_kd = alpha_kd
        self.beta_hidden = beta_hidden
        self.temperature = temperature
        self.teacher.eval() # Teacher는 항상 eval 모드

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Student 추론 (Hidden States 포함)
        outputs_student = model(**inputs, output_hidden_states=True)
        
        # 2. Teacher 추론 (No Grad, 메모리 정리 필수)
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs, output_hidden_states=True)

        # 3. Loss 계산
        loss_task = outputs_student.loss # 기본 LM Loss

        # Logit KD Loss
        s_logits = outputs_student.logits / self.temperature
        t_logits = (outputs_teacher.logits / self.temperature).to(s_logits.dtype)
        
        loss_kd = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        # Hidden State MSE Loss (마지막 레이어 비교)
        s_hidden = outputs_student.hidden_states[-1]
        t_hidden = outputs_teacher.hidden_states[-1].to(s_hidden.dtype)
        loss_hidden = F.mse_loss(s_hidden, t_hidden)

        # 최종 가중치 합
        total_loss = loss_task + (self.alpha_kd * loss_kd) + (self.beta_hidden * loss_hidden)

        # ★ 각 손실값을 WandB에 로깅
        self.log({
            "loss_task": loss_task.item(),
            "loss_kd": loss_kd.item(),
            "loss_hidden": loss_hidden.item(),
            "total_loss": total_loss.item()
        })

        # 메모리 해제: Teacher 결과물은 즉시 삭제하여 OOM 방지
        del outputs_teacher
        if not return_outputs:
            torch.cuda.empty_cache() # 매우 적은 메모리 부족 해결용

        return (total_loss, outputs_student) if return_outputs else total_loss

# =============================================================================
# 4. 실행
# =============================================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ★ WandB 초기화 (entity, project 설정)
    wandb.init(
        project=training_args.wandb_project,
        entity=training_args.wandb_entity,
        name=training_args.run_name,
        config={
            "alpha_kd": training_args.alpha_kd,
            "beta_hidden": training_args.beta_hidden,
            "temperature": training_args.temperature,
            "target_layers": model_args.target_layers,
            "max_seq_length": data_args.max_seq_length,
            "dataset_size": data_args.dataset_size,
            "learning_rate": training_args.learning_rate,
            "num_epochs": training_args.num_train_epochs,
        }
    )

    # [1] Student 로드
    student_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto" # Student는 GPU에 자동 할당
    )

    # ★ 이미 프루닝된 모델이면 스킵, 아니면 프루닝
    if student_model.config.num_hidden_layers != model_args.target_layers:
        student_model = smart_prune_model(student_model, model_args.target_layers)
        print(f"✂️ 프루닝 실행: {student_model.config.num_hidden_layers} → {model_args.target_layers}")
    else:
        print(f"✅ 이미 {model_args.target_layers}개 레이어 구조 - 프루닝 스킵")

    student_model.gradient_checkpointing_enable()

    # [2] Teacher 로드 (4-bit Quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_args.teacher_model_path or model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto", # GPU 공간이 있으면 GPU, 없으면 CPU로 자동 배치
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # [3] 데이터셋 준비
    ds = load_dataset(data_args.extra_dataset_name, split=f"train[:{data_args.dataset_size}]")
    def preprocess(ex):
        input_ids_list = []
        labels_list = []
        for conversation in ex['conversations']:
            user_msg = f"### User: {conversation[0]['content']}\n### Assistant: "
            # 300자 제한 유지
            assistant_content = conversation[1]['content'][:300] if len(conversation[1]['content']) > 300 else conversation[1]['content']
            assistant_msg = f"{assistant_content}{tokenizer.eos_token}\n"
            
            full_text = user_msg + assistant_msg
            tokenized = tokenizer(full_text, truncation=True, max_length=256, padding="max_length")
            
            # 유저 질문 길이를 계산해서 해당 구간은 -100 (Loss 계산 제외) 처리
            user_ids = tokenizer.encode(user_msg, add_special_tokens=False)
            user_len = len(user_ids)
            
            # 기본적으로 input_ids를 복사
            labels = list(tokenized['input_ids'])
            # 유저 질문 구간 masking
            for i in range(min(user_len, 256)):
                labels[i] = -100
            # 패딩 구간 masking
            labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]
            
            input_ids_list.append(tokenized['input_ids'])
            labels_list.append(labels)
            
        return {"input_ids": input_ids_list, "labels": labels_list}    
    tokenized_ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    # [4] 학습 실행
    trainer = KDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        alpha_kd=training_args.alpha_kd,
        beta_hidden=training_args.beta_hidden,
        temperature=training_args.temperature,
    )

    trainer.train()
    
    # [5] ★ 저장 전 Weight Tying 복구 및 generation_config 생성
    print("\n" + "="*70)
    print("[제출용 모델 준비]")
    print("="*70)
    
    # Step 1: Weight tying 복구 (embedding ↔ lm_head)
    print("\n[Step 1] Weight tying 복구...")
    ensure_weight_tying(student_model, config=student_model.config, verbose=True)
    
    # Step 2: 모델 저장
    print(f"\n[Step 2] 모델 저장 중: {training_args.output_dir}")
    student_model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"✓ 저장 완료")
    
    # Step 3: generation_config.json 생성/저장
    print(f"\n[Step 3] generation_config 생성...")
    create_generation_config(
        student_model, 
        tokenizer=tokenizer,
        output_dir=training_args.output_dir,
        verbose=True
    )
    
    # Step 4: safetensors 키 정규화
    print(f"\n[Step 4] safetensors 키 정규화...")
    normalize_safetensors_keys(training_args.output_dir, verbose=True)
    
    print("\n" + "="*70)
    print("✓ 제출용 모델 준비 완료!")
    print("="*70)
    
    # ★ WandB 종료
    wandb.finish()

if __name__ == "__main__":
    main()