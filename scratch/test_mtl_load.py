import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import gc

def test():
    device = "cuda"
    print("Loading base...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=False)
    print("Generating base...")
    wav = model.generate("Привет", language_id="ru")
    print(f"Base shape: {wav.shape}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Loading opt...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=True)
    print("Generating opt...")
    wav = model.generate("Привет", language_id="ru")
    print(f"Opt shape: {wav.shape}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Done!")

if __name__ == "__main__":
    test()
