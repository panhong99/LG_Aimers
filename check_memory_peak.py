import torch
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import threading
import time

class MemoryMonitor:
    """메모리 사용량을 실시간으로 모니터링"""
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_memory = 0
        self.memory_history = []
        self.is_monitoring = False
        self.start_memory = 0
        
    def get_memory_info(self):
        """현재 메모리 사용량 반환 (GB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    
    def monitor(self):
        """백그라운드에서 메모리 모니터링"""
        self.start_memory = self.get_memory_info()
        self.peak_memory = self.start_memory
        
        while self.is_monitoring:
            current_memory = self.get_memory_info()
            self.memory_history.append(current_memory)
            
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            
            time.sleep(self.interval)
    
    def start(self):
        """모니터링 시작"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """모니터링 중지"""
        self.is_monitoring = False
        self.monitor_thread.join()
    
    def get_peak_memory(self):
        """피크 메모리 반환"""
        return self.peak_memory
    
    def get_memory_increase(self):
        """시작부터의 메모리 증가량"""
        return self.peak_memory - self.start_memory

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_model_memory_with_peak(model_path, model_name):
    """모델 로드 중 피크 메모리 사용량 측정"""
    print_separator(f"{model_name} 피크 메모리 측정")
    
    # 초기화
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)
    
    # 메모리 모니터 시작
    monitor = MemoryMonitor(interval=0.05)  # 50ms 간격으로 모니터링
    
    initial_memory = monitor.get_memory_info()
    print(f"모델 로드 전 메모리: {initial_memory:.3f} GB")
    
    try:
        monitor.start()
        
        # 모델 로드
        print(f"모델 로딩 중... ({model_path})")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cpu"
        )
        
        load_time = time.time() - start_time
        monitor.stop()
        
        # 결과 출력
        peak_memory = monitor.get_peak_memory()
        memory_increase = monitor.get_memory_increase()
        final_memory = monitor.get_memory_info()
        
        print(f"✅ 모델 로딩 완료 (소요 시간: {load_time:.2f}초)")
        print(f"\n📊 메모리 사용량:")
        print(f"  - 로드 전:      {initial_memory:.3f} GB")
        print(f"  - 피크 메모리:  {peak_memory:.3f} GB")
        print(f"  - 로드 후:      {final_memory:.3f} GB")
        print(f"  - 피크 증가:    {memory_increase:+.3f} GB")
        
        # 모델 정보
        total_params = sum(p.numel() for p in model.parameters())
        model_size = total_params * 2 / (1024 ** 3)  # float16
        
        print(f"\n📈 모델 정보:")
        print(f"  - 파라미터 수:  {total_params:,}")
        print(f"  - 모델 메모리:  {model_size:.3f} GB (float16 기준)")
        
        del model
        gc.collect()
        
        return peak_memory
        
    except Exception as e:
        monitor.stop()
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  모델 로딩 시 RSS 피크 메모리 측정")
    print("="*60)
    
    results = {}
    
    # Base Model 확인
    base_peak = check_model_memory_with_peak(
        "./base_model",
        "Base Model (base_model/)"
    )
    results["Base Model"] = base_peak
    
    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    
    # Model_KD_v6_GPTQ_v2 확인
    gptq_peak = check_model_memory_with_peak(
        "./models/model_KD_v6_GPTQ_v2",
        "GPTQ Model (model_KD_v6_GPTQ_v2)"
    )
    results["GPTQ Model"] = gptq_peak
    
    # 최종 요약
    print_separator("최종 요약: RSS 피크 메모리")
    for model_name, peak in results.items():
        if peak is not None:
            print(f"{model_name:30s} | 피크: {peak:.3f} GB")
    
    if base_peak is not None and gptq_peak is not None:
        reduction = ((base_peak - gptq_peak) / base_peak) * 100
        print(f"\n메모리 절감 (GPTQ vs Base): {reduction:.1f}% ({base_peak - gptq_peak:.3f} GB)")
    
    print("\n" + "="*60 + "\n")
