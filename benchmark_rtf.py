import time
import torch
import numpy as np
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Test phrases for RTF measurement
TEST_PHRASES = [
    ("Short phrase", "Short", "Писечки сисечки попочки.", "ru"),
    ("Medium phrase", "Medium", "В чащах юга жил бы цитрус? Да, но фальшивый экземпляр!", "ru"),
    ("Long phrase", "Long", "Привет! Как твои дела? Надеюсь, у тебя всё отлично и ты готов к новым свершениям в области искусственного интеллекта и синтеза речи.", "ru")
]

def measure_rtf(model, text, lang=None, chunk_size=2, skip_watermark=True):
    print(f"\n[Benchmarking] Text: \"{text[:50]}...\"")
    print(f"Settings: chunk_size={chunk_size}, skip_watermark={skip_watermark}")
    
    start_total = time.time()
    chunks_info = []
    total_samples = 0
    
    # Streaming benchmark
    gen_start = time.time()
    first_chunk_time = None
    
    kwargs = {'chunk_size': chunk_size, 'skip_watermark': skip_watermark, 'n_cfm_timesteps': 2}
    if lang:
        kwargs['language_id'] = lang
    
    # We can pass additional kwargs to t3.inference_stream? No.
    # But we can't easily pass n_cfm_timesteps to streamer.stream via current API.
    # I'll modify mtl_tts.py to allow it.
        
    for audio_chunk, sr, timing in model.generate_streaming(text, **kwargs):
        if first_chunk_time is None:
            first_chunk_time = time.time() - gen_start
        
        total_samples += len(audio_chunk)
        chunks_info.append(timing)
    
    gen_end = time.time()
    
    total_gen_time = gen_end - gen_start
    audio_duration = total_samples / sr
    rtf = total_gen_time / audio_duration
    
    avg_decode_ms = np.mean([c['decode_ms'] for c in chunks_info])
    
    print(f"Results:")
    print(f"  - Audio duration: {audio_duration:.2f}s")
    print(f"  - Generation time: {total_gen_time:.2f}s")
    print(f"  - RTF: {rtf:.4f} ({'PASSED RTF < 1.0' if rtf < 1.0 else 'FAILED'})")
    print(f"  - TTFC (First Chunk): {first_chunk_time * 1000:.2f}ms")
    print(f"  - Avg Chunk Decode: {avg_decode_ms:.2f}ms")
    
    return {
        'rtf': rtf,
        'ttfc': first_chunk_time * 1000,
        'avg_decode': avg_decode_ms,
        'duration': audio_duration
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Benchmarking on CPU. Results will likely be RTF > 1.0")

    print("Loading Multilingual TTS...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=True)
    
    # Warmup
    print("Warmup...")
    model.generate("Warmup text", language_id="en", skip_watermark=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    results = []
    for label, size, text, lang in TEST_PHRASES:
        res = measure_rtf(model, text, lang=lang, chunk_size=12, skip_watermark=True)
        results.append((label, res))

    print("\n" + "="*60)
    print(f"{'Phrase Type':<15} | {'Duration (s)':<12} | {'RTF':<10} | {'TTFC (ms)':<10}")
    print("-" * 60)
    for label, res in results:
        print(f"{label:<15} | {res['duration']:<12.2f} | {res['rtf']:<10.4f} | {res['ttfc']:<10.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
