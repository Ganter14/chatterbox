import torch
from transformers import GPT2Model, GPT2Config, StaticCache

config = GPT2Config(n_layer=2, n_head=4, n_embd=128)
model = GPT2Model(config).cuda().half()
cache = StaticCache(config, max_cache_len=100, device="cuda", dtype=torch.half)

input_ids = torch.randint(0, 100, (1, 1)).cuda()
try:
    out = model(input_ids, past_key_values=cache, use_cache=True)
    print("GPT2Model supports StaticCache directly!")
except Exception as e:
    print(f"GPT2Model failed with StaticCache: {e}")

from transformers import LlamaModel, LlamaConfig
config = LlamaConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=128, intermediate_size=256)
model = LlamaModel(config).cuda().half()
cache = StaticCache(config, max_cache_len=100, device="cuda", dtype=torch.half)
try:
    out = model(input_ids, past_key_values=cache, use_cache=True)
    print("LlamaModel supports StaticCache directly!")
except Exception as e:
    print(f"LlamaModel failed with StaticCache: {e}")
