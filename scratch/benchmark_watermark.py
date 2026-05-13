import time
import torch
import numpy as np
import perth

def benchmark_watermark():
    watermarker = perth.PerthImplicitWatermarker()
    sr = 24000
    # 40ms of audio (chunk_size=2)
    audio = np.random.randn(int(0.04 * sr)).astype(np.float32)
    
    print(f"Benchmarking watermark on {len(audio)} samples ({len(audio)/sr*1000:.1f}ms)...")
    
    # Warmup
    _ = watermarker.apply_watermark(audio, sample_rate=sr)
    
    start = time.time()
    for _ in range(100):
        _ = watermarker.apply_watermark(audio, sample_rate=sr)
    end = time.time()
    
    avg_ms = (end - start) * 1000 / 100
    print(f"Average watermark time: {avg_ms:.2f}ms per chunk")

if __name__ == "__main__":
    benchmark_watermark()
