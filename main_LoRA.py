"""LoRA 파인튜닝 + AWQ 양자화"""

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ============================================================================
# 설정
# ============================================================================
WORKSPACE_DIR = Path(__file__).parent.resolve()
MODEL_BASE_PATH = WORKSPACE_DIR / "base_model"
MODEL_LORA_PATH = WORKSPACE_DIR / "lora_model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_TRAIN_SAMPLES = 1000  # 적게 훈련
MAX_SEQUENCE_LENGTH = 512

# LoRA 설정
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # 모델에 맞게 수정 필요

# 훈련 설정
LEARNING_RATE = 5e-4
NUM_EPOCHS = 1
BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 4
MAX_STEPS = -1  # -1이면 NUM_EPOCHS 사용

# ============================================================================
# 1. 데이터셋 준비
# ============================================================================
def prepare_dataset():
    print("\n[1/4] 데이터셋 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_BASE_PATH), trust_remote_code=True)
    
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_TRAIN_SAMPLES}]",
    )
    
    def preprocess(example):
        text = tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
        return {"text": text}
    
    ds = ds.map(preprocess, remove_columns=ds.column_names)
    
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors=None,
        )
    
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    print(f"완료: {len(ds)} 샘플")
    return tokenizer, ds


# ============================================================================
# 2. LoRA 설정 + 모델 준비
# ============================================================================
def setup_lora_model(tokenizer):
    print("\n[2/4] LoRA 모델 준비 중...")
    
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_BASE_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


# ============================================================================
# 3. LoRA 훈련
# ============================================================================
def train_lora(model, tokenizer, dataset):
    print("\n[3/4] LoRA 훈련 시작...")
    
    training_args = TrainingArguments(
        output_dir=str(MODEL_LORA_PATH),
        overwrite_output_dir=True,
        do_train=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        max_steps=MAX_STEPS,
        save_strategy="epoch",
        logging_steps=10,
        logging_dir="./logs",
        fp16=False,
        bf16=True,
        report_to="none",
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    print("훈련 완료")
    
    # LoRA 가중치만 저장
    print("\n[4/4] LoRA 가중치 저장 중...")
    MODEL_LORA_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_LORA_PATH)
    tokenizer.save_pretrained(MODEL_LORA_PATH)
    print(f"저장됨: {MODEL_LORA_PATH}")


# ============================================================================
# 4. LoRA 병합 (선택사항)
# ============================================================================
def merge_lora():
    """LoRA를 원본 모델과 병합"""
    print("\n[병합] LoRA 병합 중...")
    
    from peft import AutoPeftModelForCausalLM
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(MODEL_LORA_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    merged_model = model.merge_and_unload()
    
    merged_path = WORKSPACE_DIR / "lora_merged_model"
    merged_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_path, save_compressed=True)
    
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_LORA_PATH))
    tokenizer.save_pretrained(merged_path)
    
    print(f"병합 완료: {merged_path}")
    return merged_path


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("LoRA 파인튜닝")
    print("=" * 70)
    
    if not MODEL_BASE_PATH.exists():
        raise RuntimeError(f"Base 모델이 없습니다: {MODEL_BASE_PATH}")
    
    tokenizer, dataset = prepare_dataset()
    model = setup_lora_model(tokenizer)
    train_lora(model, tokenizer, dataset)
    
    # 병합 여부 선택
    merge_choice = input("\n[선택] LoRA를 원본 모델과 병합할까요? (y/n): ").strip().lower()
    if merge_choice == "y":
        merged_path = merge_lora()
        print(f"\n다음 단계: python main_AWQ.py --model_path {merged_path}")
    else:
        print(f"\n다음 단계: python main_AWQ.py --model_path {MODEL_LORA_PATH}")
    
    print("\n" + "=" * 70)
    print("완료")
    print("=" * 70)
