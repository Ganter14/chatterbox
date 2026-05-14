import queue
import threading
import torch
import time

class ThreadedS3GenStreamer:
    """
    Wraps S3GenStreamer to run vocoder inference in a background thread.
    This allows T3 token generation and S3Gen audio synthesis to overlap.
    """
    def __init__(self, streamer, ref_dict):
        self.streamer = streamer
        self.ref_dict = ref_dict
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self._sentinel = object()
        
        # Capture current CUDA stream to ensure worker thread uses it or syncs correctly
        self.main_stream = torch.cuda.current_stream() if torch.cuda.is_available() else None
        
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        self.token_count = 0
        self.chunk_count = 0
        self._exception = None

    def _worker(self):
        try:
            while True:
                item = self.input_queue.get()
                if item is self._sentinel:
                    break
                
                tokens, finalize, start_time = item
                
                # S3GenStreamer handles finalize by flushing its internal buffers
                audio_chunk = self.streamer.stream(tokens, self.ref_dict, finalize=finalize)
                
                if audio_chunk is not None:
                    decode_ms = (time.time() - start_time) * 1000
                    self.output_queue.put((audio_chunk, tokens.shape[1], decode_ms, finalize))
                
                self.input_queue.task_done()
        except Exception as e:
            self._exception = e
        finally:
            # Signal end of output
            self.output_queue.put(self._sentinel)

    def push(self, tokens, finalize=False):
        if self._exception:
            raise self._exception
        
        # Record start time for this chunk's decode measurement
        start_time = time.time()
        self.input_queue.put((tokens, finalize, start_time))
        self.token_count += tokens.shape[1]

    def close(self):
        self.input_queue.put(self._sentinel)

    def get_chunk(self, timeout=None):
        if self._exception:
            raise self._exception
            
        try:
            item = self.output_queue.get(timeout=timeout)
            return item
        except queue.Empty:
            return None

    def is_sentinel(self, item):
        return item is self._sentinel
