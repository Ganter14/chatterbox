import argparse
import time
import queue
import numpy as np
import torch
try:
    import sounddevice as sd
except ImportError:
    sd = None
from chatterbox.tts_turbo import ChatterboxTurboTTS

def main():
    parser = argparse.ArgumentParser(description="Streaming TTS Playback Example")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog. This is a test of the streaming output system in Chatterbox.", help="Text to synthesize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--use_cuda_graph", action="store_true", help="Use CUDA Graphs for acceleration")
    parser.add_argument("--chunk_size", type=int, default=12, help="Number of tokens per chunk")
    args = parser.parse_args()

    if sd is None:
        print("Error: 'sounddevice' not found. Please install it with 'pip install sounddevice'.")
        print("Running in dry-run mode (synthesis only, no playback).")

    print(f"Loading model on {args.device} (CUDA Graphs: {args.use_cuda_graph})...")
    model = ChatterboxTurboTTS.from_pretrained(device=args.device, use_cuda_graph=args.use_cuda_graph)

    # Audio playback queue (bounded to 100 blocks for backpressure)
    audio_q = queue.Queue(maxsize=100)

    def callback(outdata, frames, time_info, status):
        if status:
            print(status)
        try:
            data = audio_q.get_nowait()
            if len(data) < frames:
                outdata[:len(data), 0] = data
                outdata[len(data):, 0] = 0
            else:
                outdata[:, 0] = data[:frames]
        except queue.Empty:
            outdata.fill(0)

    # Use a larger blocksize for smoother playback
    blocksize = 1024
    
    print("\nStarting streaming synthesis...")
    print(f"Text: {args.text}\n")

    if sd:
        stream = sd.OutputStream(samplerate=model.sr, channels=1, callback=callback, blocksize=blocksize)
        stream.start()
    
    start_gen = time.time()
    first_chunk = True
    
    try:
        for audio_chunk, sr, timing in model.generate_streaming(args.text, chunk_size=args.chunk_size):
            if first_chunk:
                ttfc = (time.time() - start_gen) * 1000
                print(f"TTFC (Time To First Chunk): {ttfc:.2f}ms")
                first_chunk = False
            
            print(f"Chunk {timing['chunk_index']}: steps={timing['chunk_steps']}, decode={timing['decode_ms']:.2f}ms")
            
            if sd:
                # Put audio into queue in blocks
                for i in range(0, len(audio_chunk), blocksize):
                    audio_q.put(audio_chunk[i:i+blocksize])
            else:
                # Just simulate processing time if no sounddevice
                time.sleep(0.01)

        if sd:
            # Wait for queue to empty
            while not audio_q.empty():
                time.sleep(0.1)
            time.sleep(0.5)
            stream.stop()
            stream.close()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print("\nDone.")

if __name__ == "__main__":
    main()
