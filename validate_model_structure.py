"""
제출 전 모델 검증 스크립트

사용법:
  python3 validate_model_structure.py ./models/model_KD_v6_GPTQ_v2
  
목적:
- dtype 분포 분석 (bf16/fp16 비중)
- 상위 텐서 크기 출력
- embedding/lm_head 중복 여부 체크
- weight tying 상태 확인
"""

import os
import sys
from model_utils import validate_model_structure

def main():
    if len(sys.argv) < 2:
        print("사용법: python3 validate_model_structure.py <model_dir>")
        print("예시: python3 validate_model_structure.py ./models/model_KD_v6_GPTQ_v2")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    
    if not os.path.exists(model_dir):
        print(f"❌ 오류: {model_dir} 폴더를 찾을 수 없습니다")
        sys.exit(1)
    
    report = validate_model_structure(model_dir, verbose=True)
    
    if report is None:
        sys.exit(1)
    
    # 문제 여부에 따른 종료 코드
    if report['embedding_lm_head_duplicate']:
        print("\n⚠️  CRITICAL: embedding/lm_head 중복 검출!")
        print("  → model_utils.remove_lm_head_from_state_dict() 또는")
        print("  → ensure_weight_tying() 재실행 필요")
        sys.exit(1)
    
    if report['bf16_fp16_ratio'] > 0.15:
        print("\n⚠️  경고: BF16/FP16이 15% 이상")
        print("  → 부분 양자화 상태 가능성")
        sys.exit(1)
    
    print("\n✓ 모델 검증 통과!")
    sys.exit(0)

if __name__ == "__main__":
    main()
