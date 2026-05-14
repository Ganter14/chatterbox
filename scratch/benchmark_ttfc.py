import torch
import time
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def benchmark_ttfc():
    device = "cuda"
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=True)
    text = "This is a test of the first chunk time."
    
    # Warmup
    for _ in range(3):
        for _ in model.generate_streaming(text, language_id="en"):
            pass
            
    # Measure TTFC
    n_iters = 5
    for _ in range(n_iters):
        start = time.perf_counter()
        for chunk, sr, timing in model.generate_streaming(text, language_id="en"):
            ttfc = time.perf_counter() - start
            print(f"TTFC: {ttfc*1000:.2f}ms, Timing: {timing}")
            break
            
if __name__ == "__main__":
    benchmark_ttfc()
