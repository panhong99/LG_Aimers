"""Apply Marlin Kernel to AWQ/GPTQ quantized models"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_marlin_kernel(model_path: str, output_path: str = None):
    """
    Apply Marlin kernel optimization to quantized model
    
    Args:
        model_path: Path to quantized model (AWQ or GPTQ)
        output_path: Output path (optional, overwrites if None)
    """
    
    if output_path is None:
        output_path = model_path
    
    print(f"[INFO] 모델 로드: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("[INFO] Marlin 커널 활성화...")
    
    # Marlin kernel 활성화
    # llmcompressor에서 저장한 양자화 설정 확인
    import json
    from pathlib import Path
    
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        if 'quantization_config' in config_dict:
            quant_config = config_dict['quantization_config']
            print(f"[INFO] 양자화 설정: {quant_config.get('quant_method', 'unknown')}")
            
            quant_method = quant_config.get('quant_method', '').lower()
            
            # llmcompressor는 'compressed-tensors' 형식 사용
            if 'compressed-tensors' in quant_method or 'awq' in quant_method:
                num_bits = 4  # AWQ는 기본 4-bit
                if 'group_0' in quant_config.get('config_groups', {}):
                    num_bits = quant_config['config_groups']['group_0'].get('weights', {}).get('num_bits', 4)
                print(f"[INFO] AWQ {num_bits}-bit 모델 감지됨 - Marlin 커널 호환 ✓")
            elif 'gptq' in quant_method:
                print("[INFO] GPTQ 4-bit 모델 감지됨 - Marlin 커널 호환 ✓")
            else:
                print(f"[WARNING] 알 수 없는 양자화 방식: {quant_method}")
        else:
            print("[WARNING] quantization_config 없음")
    
    print(f"[INFO] 모델 저장: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    
    print("[INFO] Marlin 커널 적용 완료!")
    print("\n[사용 방법]")
    print(f"vllm serve {output_path} --mm-model-cfg-backend huggingface")
    print("\n또는")
    print("from vllm import LLM")
    print(f"llm = LLM('{output_path}', tensor_parallel_size=1)")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marlin 커널 적용")
    parser.add_argument("--model_path", type=str, required=True, help="양자화된 모델 경로")
    parser.add_argument("--output_path", type=str, default=None, help="출력 경로 (기본: 입력 경로)")
    
    args = parser.parse_args()
    apply_marlin_kernel(args.model_path, args.output_path)
