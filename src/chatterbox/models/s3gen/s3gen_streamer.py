import torch
import numpy as np
from .s3gen import S3GEN_SR
from ..s3tokenizer import S3_TOKEN_RATE

class S3GenStreamer:
    """
    Stateful streamer for S3Gen (Flow + Vocoder).
    Uses accumulated tokens with prompt caching to ensure efficiency.
    """
    def __init__(self, s3gen, n_cfm_timesteps=None):
        self.s3gen = s3gen
        self.all_tokens = []
        self.prev_audio_len = 0
        self.pre_lookahead = getattr(s3gen.flow, 'pre_lookahead_len', 3)
        self.n_cfm_timesteps = n_cfm_timesteps
        
        # Optimization: Pre-encoded prompt
        self.prompt_enc = None
        self.prompt_mask = None
        
        # Vocoder state for continuity
        self.last_s = None

    @torch.inference_mode()
    def _get_prompt_enc(self, ref_dict):
        """Pre-encode prompt tokens to avoid redundant computation."""
        if self.prompt_enc is not None:
            return self.prompt_enc, self.prompt_mask
        
        prompt_token = ref_dict['prompt_token']
        prompt_token_len = ref_dict['prompt_token_len']
        
        # We need to access the flow's encoder and input_embedding
        flow = self.s3gen.flow
        B = prompt_token.size(0)
        
        # batching logic same as in flow.inference
        from .flow import _repeat_batch_dim
        prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)
        prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)
        
        from .utils.mask import make_pad_mask
        mask = (~make_pad_mask(prompt_token_len)).unsqueeze(-1).to(flow.spk_embed_affine_layer.weight.device)
        token_emb = flow.input_embedding(prompt_token.long()) * mask
        
        self.prompt_enc, self.prompt_mask = flow.encoder(token_emb, prompt_token_len)
        return self.prompt_enc, self.prompt_mask

    @torch.inference_mode()
    def stream(self, new_tokens, ref_dict, finalize=False):
        """
        Process new tokens and yield new audio samples.
        """
        if not isinstance(new_tokens, torch.Tensor):
            raise TypeError(f"Expected new_tokens to be torch.Tensor, got {type(new_tokens)}")

        if new_tokens.ndim == 1:
            new_tokens = new_tokens.unsqueeze(0)
        
        self.all_tokens.append(new_tokens)
        all_flat = torch.cat(self.all_tokens, dim=1)
        
        if not finalize and all_flat.shape[1] <= self.pre_lookahead:
            return None

        # Pre-encode prompt if not already done
        p_enc, p_mask = self._get_prompt_enc(ref_dict)
        
        # Call flow inference with pre-encoded prompt
        output_mels = self.s3gen.flow_inference(
            all_flat,
            ref_dict=ref_dict,
            finalize=finalize,
            prompt_enc=p_enc,
            prompt_mask=p_mask,
            n_cfm_timesteps=self.n_cfm_timesteps
        )
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        # Vocoder inference with caching for continuity
        output_wavs, self.last_s = self.s3gen.hift_inference(output_mels, cache_source=self.last_s)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        # Convert to numpy and trim what we already yielded
        wav = output_wavs.squeeze(0).detach().cpu().numpy()
        
        new_audio = wav[self.prev_audio_len:]
        self.prev_audio_len = len(wav)
        
        return new_audio



