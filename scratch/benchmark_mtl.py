import torch
import time
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def benchmark_mtl():
    device = "cuda"
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=True)
    text = "Это длинное предложение для замера производительности мультиязычной модели в режиме реального времени."
    
    # Warmup
    for _ in range(3):
        model.generate(text, language_id="ru")
    
    # Measure
    n_iters = 5
    start = time.perf_counter()
    total_samples = 0
    for _ in range(n_iters):
        wav = model.generate(text, language_id="ru")
        total_samples += wav.shape[-1]
    end = time.perf_counter()
    
    duration = end - start
    avg_time = duration / n_iters
    rtf = avg_time / (total_samples / n_iters / 24000)
    
    print(f"MTL-RU Average Time: {avg_time:.4f}s")
    print(f"MTL-RU RTF: {rtf:.4f}")

if __name__ == "__main__":
    benchmark_mtl()
