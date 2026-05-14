import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import time

def test_streaming():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=True)
    
    text = "Привет, как дела?"
    print(f"Testing streaming with text: {text}")
    
    try:
        count = 0
        for audio_chunk, sr, timing in model.generate_streaming(text, language_id="ru", chunk_size=12):
            print(f"Received chunk {count}, size: {len(audio_chunk)}, timing: {timing}")
            count += 1
        print(f"Streaming finished. Total chunks: {count}")
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_streaming()
