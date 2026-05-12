import time
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

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
    torch.cuda.synchronize()

    # Benchmark
    print("Running benchmark...")
    iterations = 3
    total_time = 0
    total_tokens = 0
    
    for i in range(iterations):
        start = time.time()
        # We need to capture the number of generated tokens. 
        # For now, let's just measure the whole generation time.
        # In a real benchmark we'd hook into the loop.
        wav = model.generate(text, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        
        dur = end - start
        total_time += dur
        print(f"Iteration {i+1}: {dur:.4f}s")

    avg_time = total_time / iterations
    print(f"Average generation time for {name}: {avg_time:.4f}s")
    return avg_time

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

    print("\n" + "="*40)
    print("COMPARISON RESULTS (feat/cuda-graphs branch)")
    print(f"{'Model':<25} | {'Base':<10} | {'Opt':<10} | {'Speedup':<10}")
    print("-" * 60)
    
    def print_row(base_key, opt_key, label):
        base = results[base_key]
        opt = results[opt_key]
        speedup = base / opt
        print(f"{label:<25} | {base:<10.4f} | {opt:<10.4f} | {speedup:<10.2f}x")

    print_row("Turbo-EN-Base", "Turbo-EN-Optimized", "Turbo-EN")
    print_row("MTL-RU-Base", "MTL-RU-Optimized", "MTL-RU")
    print("="*40)
