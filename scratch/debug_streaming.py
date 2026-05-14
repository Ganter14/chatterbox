import torch
import numpy as np
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def debug():
    device = "cuda"
    seed = 42
    text = "Привет мир."
    
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=True)
    
    set_seed(seed)
    wav_full = model.generate(text, language_id="ru", temperature=0, skip_watermark=True)
    # To get tokens for comparison, we need to call inference directly or use a hook
    # But let's just trust that the tokens are consistent if seeds are set.
    
    set_seed(seed)
    chunks = []
    stream_tokens = []
    for chunk, sr, timing in model.generate_streaming(text, language_id="ru", temperature=0, chunk_size=12, skip_watermark=True):
        chunks.append(chunk)
        print(f"Got chunk: {chunk.shape if chunk is not None else None}")
    
    print(f"Full wav shape: {wav_full.shape}")
    print(f"Stream wav shape: {np.concatenate(chunks).shape}")
    wav_stream = np.concatenate(chunks)
    
    print(f"Full wav shape: {wav_full.shape}")
    print(f"Stream wav shape: {wav_stream.shape}")
    
    # Compare first 8000 samples
    mse_start = np.mean((wav_full[0, :8000].cpu().numpy() - wav_stream[:8000])**2)
    print(f"MSE (first 8000 samples): {mse_start}")
    
    mse_total = np.mean((wav_full[0, :len(wav_stream)].cpu().numpy() - wav_stream)**2)
    print(f"Total MSE: {mse_total}")
    
    if wav_full.shape[-1] != len(wav_stream):
        print(f"Difference in samples: {len(wav_stream) - wav_full.shape[-1]}")

if __name__ == "__main__":
    debug()
