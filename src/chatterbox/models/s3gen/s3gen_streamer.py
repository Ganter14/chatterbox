import torch
import numpy as np
from .s3gen import S3GEN_SR
from ..s3tokenizer import S3_TOKEN_RATE
from .utils.mask import make_pad_mask

class S3GenStreamer:
    """
    Stateful streamer for S3Gen (Flow + Vocoder).
    Uses accumulated tokens with prompt caching to ensure efficiency.
    Implemented with sliding window for O(N) complexity.
    """
    def __init__(self, s3gen, n_cfm_timesteps=None, window_size=120):
        self.s3gen = s3gen
        self.all_tokens = []
        self.prev_audio_len = 0
        self.pre_lookahead = getattr(s3gen.flow, 'pre_lookahead_len', 3)
        self.n_cfm_timesteps = n_cfm_timesteps
        self.window_size = window_size # tokens
        self.samples_per_token = S3GEN_SR // S3_TOKEN_RATE
        
        # Optimization: Pre-encoded prompt
        self.prompt_enc = None
        self.prompt_mask = None
        
        # Vocoder state for continuity
        self.last_s = None

    @torch.inference_mode()
    def stream(self, new_tokens, ref_dict, finalize=False):
        """
        Process new tokens and yield new audio samples.
        """
        if not isinstance(new_tokens, torch.Tensor):
            raise TypeError(f"Expected new_tokens to be torch.Tensor, got {type(new_tokens)}")

        if new_tokens.ndim == 1:
            new_tokens = new_tokens.unsqueeze(0)
        
        # Pre-encode prompt if not already done
        if self.prompt_enc is None:
            prompt_token = ref_dict['prompt_token']
            prompt_token_len = ref_dict['prompt_token_len']
            
            # Embed and encode prompt
            # We must use the same logic as in flow.py
            mask = (~make_pad_mask(prompt_token_len, prompt_token.size(1))).unsqueeze(-1).to(prompt_token.device)
            token_emb = self.s3gen.flow.input_embedding(prompt_token.long()) * mask.to(self.s3gen.flow.input_embedding.weight.dtype)
            self.prompt_enc, self.prompt_mask = self.s3gen.flow.encoder(token_emb, prompt_token_len)
        
        # Accumulate speech tokens for the decoder context (history)
        self.all_tokens.append(new_tokens)
        all_speech_tokens = torch.cat(self.all_tokens, dim=1)
        
        # Sliding window optimization
        if not finalize and all_speech_tokens.shape[1] > self.window_size + self.pre_lookahead:
            overflow = all_speech_tokens.shape[1] - (self.window_size + self.pre_lookahead)
            all_speech_tokens = all_speech_tokens[:, overflow:]
            self.all_tokens = [all_speech_tokens]
            
            if self.last_s is not None:
                overflow_samples = overflow * self.samples_per_token
                if self.last_s.shape[2] > overflow_samples:
                    self.last_s = self.last_s[:, :, overflow_samples:]
                else:
                    self.last_s = None
            self.prev_audio_len = max(0, self.prev_audio_len - overflow * self.samples_per_token)

        if not finalize and all_speech_tokens.shape[1] <= self.pre_lookahead:
            return None

        # Call flow inference WITH prompt_enc
        # We pass ONLY the accumulated speech tokens (not the prompt)
        output_mels = self.s3gen.flow_inference(
            all_speech_tokens,
            ref_dict=ref_dict,
            finalize=finalize,
            n_cfm_timesteps=self.n_cfm_timesteps,
            prompt_enc=self.prompt_enc,
            prompt_mask=self.prompt_mask
        )
        
        # Ensure correct dtype (important for fp16 mode)
        output_mels = output_mels.to(dtype=self.s3gen.dtype)
        
        # Vocoder inference with caching for continuity
        output_wavs, self.last_s = self.s3gen.hift_inference(output_mels, cache_source=self.last_s)

        # output_wavs is (1, total_samples) on GPU; we only need the tail past
        # what previous chunks already yielded. Sliding the window leaves us
        # with prev_audio_len already aligned to the surviving prefix (see the
        # overflow branch above), so the slice is well-defined.
        # Slice on GPU so the host copy is just the new audio, not the full
        # ~57 KB window; this also avoids forcing cudaSynchronize on samples
        # the consumer is going to discard immediately.
        total_samples = output_wavs.shape[-1]
        new_audio_gpu = output_wavs[0, self.prev_audio_len:total_samples].contiguous()
        new_audio = new_audio_gpu.detach().cpu().numpy()
        self.prev_audio_len = total_samples

        return new_audio



