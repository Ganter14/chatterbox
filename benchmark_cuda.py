import gc
import time

import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def benchmark_model(model_class, text, name, use_cuda_graph=False, **kwargs):
    print(f"\nBenchmarking {name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    start_load = time.time()
    model = model_class.from_pretrained(device=device, use_cuda_graph=use_cuda_graph)
    end_load = time.time()
    print(f"Model loaded in {end_load - start_load:.2f}s")

    # Warmup
    print("Warmup...")
    model.generate(text, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print("Running benchmark...")
    iterations = 3
    total_time = 0
    total_ttfc = 0
    
    for i in range(iterations):
        # 1. Full generation
        start = time.time()
        wav = model.generate(text, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        dur = end - start
        total_time += dur
        
        # 2. Streaming TTFC
        start_st = time.time()
        for audio_chunk, sr, timing in model.generate_streaming(text, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ttfc = (time.time() - start_st) * 1000
            total_ttfc += ttfc
            break # We only care about the first chunk for TTFC
            
        print(f"Iteration {i+1}: full={dur:.4f}s, ttfc={ttfc:.2f}ms")

    avg_time = total_time / iterations
    avg_ttfc = total_ttfc / iterations
    print(f"Average generation time for {name}: {avg_time:.4f}s")
    print(f"Average TTFC for {name}: {avg_ttfc:.2f}ms")
    del model
    _free_cuda()
    return avg_time, avg_ttfc

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Benchmarking on CPU (not recommended for CUDA Graphs comparison).")
    
    results = {}
    
    # 1. Turbo TTS (English)
    results["Turbo-EN-Base"] = benchmark_model(
        ChatterboxTurboTTS, 
        "The quick brown fox jumps over the lazy dog.", 
        "Chatterbox-Turbo (English) - Baseline"
    )
    results["Turbo-EN-Optimized"] = benchmark_model(
        ChatterboxTurboTTS, 
        "The quick brown fox jumps over the lazy dog.", 
        "Chatterbox-Turbo (English) - Optimized",
        use_cuda_graph=True
    )

    # 2. Multilingual TTS (Russian)
    results["MTL-RU-Base"] = benchmark_model(
        ChatterboxMultilingualTTS, 
        "Быстрая коричневая лиса прыгает через ленивую собаку.", 
        "Chatterbox-Multilingual (Russian) - Baseline",
        language_id="ru"
    )
    results["MTL-RU-Optimized"] = benchmark_model(
        ChatterboxMultilingualTTS, 
        "Быстрая коричневая лиса прыгает через ленивую собаку.", 
        "Chatterbox-Multilingual (Russian) - Optimized",
        language_id="ru",
        use_cuda_graph=True
    )

    print("\n" + "="*80)
    print("COMPARISON RESULTS (feat/streaming branch)")
    print(f"{'Model':<25} | {'Base (s)':<10} | {'Opt (s)':<10} | {'Speedup':<10} | {'TTFC (ms)':<10}")
    print("-" * 80)
    
    def print_row(base_key, opt_key, label):
        base_time, base_ttfc = results[base_key]
        opt_time, opt_ttfc = results[opt_key]
        speedup = base_time / opt_time
        print(f"{label:<25} | {base_time:<10.4f} | {opt_time:<10.4f} | {speedup:<10.2f}x | {opt_ttfc:<10.2f}")

    print_row("Turbo-EN-Base", "Turbo-EN-Optimized", "Turbo-EN")
    print_row("MTL-RU-Base", "MTL-RU-Optimized", "MTL-RU")
    print("="*80)
