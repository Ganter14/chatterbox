import os
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import whisper
import sys

def verify():
    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torchaudio version: {ta.__version__}")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        device = "cuda"
    else:
        print("CUDA not available, using CPU")
        device = "cpu"

    # 1. Generate test audio
    print("Generating test audio...")
    model = ChatterboxTTS.from_pretrained(device=device)
    test_text = "Hello, this is a test of the chatterbox text to speech system."
    wav = model.generate(test_text)
    
    output_file = "verification_test.wav"
    import soundfile as sf
    # Ensure wav is in the right format for soundfile (numpy array, [samples, channels] or [samples])
    # Chatterbox usually returns [1, samples]
    wav_numpy = wav.squeeze().cpu().numpy()
    sf.write(output_file, wav_numpy, model.sr)
    print(f"Audio saved to {output_file}")

    # 2. Recognize text using Whisper
    print("Recognizing text using Whisper...")
    whisper_model = whisper.load_model("base", device=device)
    result = whisper_model.transcribe(output_file)
    recognized_text = result["text"].strip().lower()
    
    print(f"Expected text: {test_text.lower()}")
    print(f"Recognized text: {recognized_text}")

    # 3. Verify
    # Clean up punctuation for comparison
    import re
    expected_clean = re.sub(r'[^\w\s]', '', test_text.lower())
    recognized_clean = re.sub(r'[^\w\s]', '', recognized_text)

    if expected_clean in recognized_clean or recognized_clean in expected_clean:
        print("SUCCESS: Recognized text matches expected text (ignoring punctuation).")
        return True
    else:
        print("FAILURE: Recognized text does not match expected text.")
        return False

if __name__ == "__main__":
    try:
        if verify():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
