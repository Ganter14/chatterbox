import torch
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask
import time

class T3Graph:
    """
    Captures T3's single-token decode step as a CUDA graph.
    Uses transformers StaticCache for KV cache management.
    """
    def __init__(self, t3_model, batch_size=1, max_seq_len=2048, device='cuda', dtype=None):
        self.t3 = t3_model
        self.device = device
        self.dtype = dtype or next(t3_model.parameters()).dtype
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        
        # References to model components
        self.tfmr = t3_model.tfmr
        self.speech_head = t3_model.speech_head
        self.speech_emb = t3_model.speech_emb
        
        # StaticCache initialization
        self.static_cache = StaticCache(
            config=self.tfmr.config, 
            max_batch_size=batch_size, 
            max_cache_len=max_seq_len, 
            device=device, 
            dtype=self.dtype
        )
        
        # Static I/O buffers
        self.input_buf = torch.zeros(batch_size, 1, self.tfmr.config.hidden_size, dtype=self.dtype, device=device)
        self.logits_buf = torch.zeros(batch_size, 1, self.t3.hp.speech_tokens_dict_size, dtype=self.dtype, device=device)
        
        # Position buffer
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # Attention Mask Table (pre-computed for all positions)
        self.attn_mask_table = []
        self.captured = False
        self.graph = None

    def _init_cache_layers(self):
        """Force lazy initialization of StaticCache layers."""
        config = self.tfmr.config
        num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        dummy_k = torch.zeros(self.batch_size, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k, dummy_k)

    def _build_attention_masks(self):
        """Pre-compute causal masks for all possible positions."""
        self.attn_mask_table = []
        mask_val = torch.finfo(self.dtype).min
        for i in range(self.max_seq_len):
            mask = torch.full((self.batch_size, 1, 1, self.max_seq_len), mask_val, device=self.device, dtype=self.dtype)
            mask[:, :, :, :i+1] = 0
            self.attn_mask_table.append(mask)
        
        # Current active mask (static buffer for graph)
        self.active_mask = self.attn_mask_table[0].clone()

    def _decode_step(self):
        """Single-token decode on static buffers."""
        # Note: We use tfmr.forward directly to avoid T3's complex conditioning in every step
        # T3 conditioning is only needed for prefill.
        out = self.tfmr(
            inputs_embeds=self.input_buf,
            past_key_values=self.static_cache,
            cache_position=self.cache_position,
            position_ids=self.position_ids,
            attention_mask=self.active_mask,
            use_cache=True,
            return_dict=True
        )
        hidden_states = out.last_hidden_state
        logits = self.speech_head(hidden_states)
        self.logits_buf.copy_(logits)

    @torch.inference_mode()
    def capture(self, warmup_runs=3):
        """Warmup and capture the CUDA graph."""
        print(f"Capturing T3 CUDA graph (max_seq_len={self.max_seq_len})...")
        self._init_cache_layers()
        self._build_attention_masks()

        # Warmup
        self.cache_position[0] = 100 # arbitrary position
        self.active_mask.copy_(self.attn_mask_table[100])
        for _ in range(warmup_runs):
            self._decode_step()
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # One more warmup in the stream
            self._decode_step()
            torch.cuda.synchronize()
            
            with torch.cuda.graph(self.graph):
                self._decode_step()
        
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.captured = True
        print("T3 CUDA graph captured successfully!")

    def prefill_kv(self, past_key_values):
        """Copy DynamicCache from prefill into StaticCache."""
        self.static_cache.reset()
        seq_len = 0
        
        # Check if it's GPT2 (Tuple of tuples) or Llama (DynamicCache object)
        if isinstance(past_key_values, tuple):
            # GPT-2 style: tuple(tuple(k, v), ...)
            num_layers = len(past_key_values)
            for li in range(num_layers):
                k, v = past_key_values[li] # [B, heads, seq, head_dim]
                seq_len = k.shape[2]
                self.static_cache.layers[li].keys[:, :, :seq_len, :].copy_(k)
                self.static_cache.layers[li].values[:, :, :seq_len, :].copy_(v)
        else:
            # Llama style: DynamicCache (modern transformers style)
            num_layers = len(past_key_values.layers)
            for li in range(num_layers):
                k, v = past_key_values.layers[li].keys, past_key_values.layers[li].values
                seq_len = k.shape[2]
                # Direct copy into the static tensors
                self.static_cache.layers[li].keys[:, :, :seq_len, :].copy_(k)
                self.static_cache.layers[li].values[:, :, :seq_len, :].copy_(v)
            
            self.static_cache.seen_tokens = seq_len
        
        return seq_len

    @torch.inference_mode()
    def run(self, next_token_embed, position):
        """Run one decode step using the captured graph."""
        if not self.captured:
            raise RuntimeError("Graph not captured. Call capture() first.")
        
        if position >= self.max_seq_len:
            raise ValueError(f"Position {position} exceeds max_seq_len {self.max_seq_len}")

        self.input_buf.copy_(next_token_embed)
        self.cache_position[0] = position
        self.position_ids.fill_(position)
        self.active_mask.copy_(self.attn_mask_table[position])
        
        self.graph.replay()
        return self.logits_buf.clone()
