# ============================================================
# CELL 2-B: AWQ + GPTQ 조합 (단독 대신 이걸 실행)
# 반드시 CELL 1 다시 실행 후 (모델 새로 로드) 실행할 것
# ============================================================
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier

OUTPUT_DIR_AWQ_GPTQ = "/content/drive/MyDrive/EXAONE_Quantized/model_AWQ_GPTQ"

recipe_awq_gptq = [
    AWQModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["lm_head"],
    ),
    GPTQModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["lm_head"],
        dampening_frac=0.01,
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe_awq_gptq,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    processor=tokenizer,
    output_dir=OUTPUT_DIR_AWQ_GPTQ,
)
print("AWQ+GPTQ 완료:", OUTPUT_DIR_AWQ_GPTQ)
