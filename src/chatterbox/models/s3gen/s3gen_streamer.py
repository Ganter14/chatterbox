import torch
import numpy as np
from .s3gen import S3GEN_SR
from ..s3tokenizer import S3_TOKEN_RATE

class S3GenStreamer:
    """
    Stateful streamer for S3Gen (Flow + Vocoder).
    Uses a sliding window/accumulated decode approach to ensure continuity.
    """
    def __init__(self, s3gen):
        self.s3gen = s3gen
        self.all_tokens = []
        self.prev_audio_len = 0
        self.pre_lookahead = getattr(s3gen.flow, 'pre_lookahead_len', 3)

    @torch.inference_mode()
    def stream(self, new_tokens, ref_dict, finalize=False):
        """
        Process new tokens and yield new audio samples.
        
        Args:
            new_tokens (torch.Tensor): New tokens [1, T] or [T]
            ref_dict (dict): Conditionals for S3Gen
            finalize (bool): If True, processes all remaining tokens
            
        Returns:
            np.ndarray: New audio samples
        """
        if not isinstance(new_tokens, torch.Tensor):
            raise TypeError(f"Expected new_tokens to be torch.Tensor, got {type(new_tokens)}")

        if new_tokens.ndim == 1:
            new_tokens = new_tokens.unsqueeze(0)
        
        self.all_tokens.append(new_tokens)
        all_flat = torch.cat(self.all_tokens, dim=1)
        
        if not finalize and all_flat.shape[1] <= self.pre_lookahead:
            return None

        # Call flow inference with current finalize status
        # Note: we use flow_inference directly to control the finalize flag
        output_mels = self.s3gen.flow_inference(
            all_flat,
            ref_dict=ref_dict,
            finalize=finalize
        )
        
        # Vocoder inference
        output_wavs, _ = self.s3gen.hift_inference(output_mels)
        
        # Convert to numpy and trim
        wav = output_wavs.squeeze(0).detach().cpu().numpy()
        
        # If not the first chunk, we might want to crossfade or just trim.
        # Given the sliding window on tokens, trimming is usually enough for ISTFTNet.
        new_audio = wav[self.prev_audio_len:]
        self.prev_audio_len = len(wav)
        
        return new_audio
