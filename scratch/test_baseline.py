import torch
import numpy as np
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def test_baseline_streaming():
    device = "cuda"
    # Create model WITHOUT my optimizations
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=False)
    
    # We need to make sure we are not using the threaded streamer
    # I'll manually call the non-threaded generate_streaming logic if I can,
    # or just use the current one and see.
    # Wait, the current generate_streaming IS threaded.
    
    text = "Проверка консистентности стриминга."
    seed = 42
    
    def get_wav(streaming):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if streaming:
            chunks = []
            for chunk, sr, timing in model.generate_streaming(text, language_id="ru", temperature=0.001):
                chunks.append(chunk)
            return np.concatenate(chunks)
        else:
            wav = model.generate(text, language_id="ru", temperature=0.001)
            return wav.detach().cpu().numpy().squeeze()

    wav_full = get_wav(False)
    wav_stream = get_wav(True)
    
    mse = np.mean((wav_full[:min(len(wav_full), len(wav_stream))] - wav_stream[:min(len(wav_full), len(wav_stream))])**2)
    print(f"Baseline Streaming MSE: {mse:.2e}")

if __name__ == "__main__":
    test_baseline_streaming()
