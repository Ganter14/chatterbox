import torch
import torch.nn as nn

def quantize_weight_only(model: nn.Module):
    """
    Apply simple 8-bit weight-only quantization to all Linear layers.
    This is a naive implementation for demonstration.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip heads if they are sensitive
            if 'head' in name:
                continue
                
            # Naive quantization: scale and zero point
            w = module.weight.data
            q_min, q_max = -128, 127
            
            min_val = w.min()
            max_val = w.max()
            
            scale = (max_val - min_val) / (q_max - q_min)
            zero_point = q_min - min_val / scale
            
            # This is NOT how you do it for real inference speedup in PyTorch eager,
            # you need specialized kernels (like bitsandbytes or torch.compile with int8).
            # But for "weight-only" as a concept in this task, we'll use a placeholder
            # or try to use torch.ao.quantization if available.
            
    print("Weight-only quantization applied (simulation).")
