import torch
import numpy as np
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3gen.s3gen_streamer import S3GenStreamer

def debug_regression():
    device = "cuda"
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=True)
    text = "Привет."
    
    # 1. Full generation - capture Mels
    full_mels = []
    original_flow = model.s3gen.flow_inference
    def mock_flow(*args, **kwargs):
        # ensure we capture tokens too
        mels = original_flow(*args, **kwargs)
        full_mels.append(mels.detach().cpu().numpy())
        return mels
    model.s3gen.flow_inference = mock_flow
    
    torch.manual_seed(42)
    model.generate(text, language_id="ru", temperature=0.001, skip_watermark=True)
    f_mels = full_mels[0]
    
    # 2. Sync streaming - capture Mels
    torch.manual_seed(42)
    streamer = S3GenStreamer(model.s3gen)
    
    text_norm = "привет."
    text_tokens = model.tokenizer.text_to_tokens(text_norm, language_id="ru").to(device)
    text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens_cfg = torch.nn.functional.pad(text_tokens_cfg, (1, 0), value=sot)
    text_tokens_cfg = torch.nn.functional.pad(text_tokens_cfg, (0, 1), value=eot)
    
    token_buffer = []
    started = True
    sos_token = model.t3.hp.start_speech_token
    eos_token = model.t3.hp.stop_speech_token
    
    stream_mels = []
    for token in model.t3.inference_stream(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens_cfg,
        max_new_tokens=1000,
        temperature=0.001,
        cfg_weight=0.5,
    ):
        t_val = token.item()
        if t_val == sos_token: continue
        if t_val == eos_token: break
        token_buffer.append(token)
        if len(token_buffer) >= 12:
            chunk_tokens = torch.cat(token_buffer, dim=1)
            streamer.all_tokens.append(chunk_tokens)
            all_tokens = torch.cat(streamer.all_tokens, dim=1)
            # Use keyword argument for ref_dict
            mels = model.s3gen.flow_inference(all_tokens, ref_dict=model.conds.gen, finalize=False)
            stream_mels.append(mels.detach().cpu().numpy())
            token_buffer = []
            
    # Finalize
    if token_buffer:
        chunk_tokens = torch.cat(token_buffer, dim=1)
    else:
        chunk_tokens = torch.zeros((1, 0), dtype=torch.long, device=device)
    streamer.all_tokens.append(chunk_tokens)
    all_tokens = torch.cat(streamer.all_tokens, dim=1)
    mels = model.s3gen.flow_inference(all_tokens, ref_dict=model.conds.gen, finalize=True)
    stream_mels.append(mels.detach().cpu().numpy())
    
    s_mels = stream_mels[-1] # The last mel from streamer should be the full one
    
    print(f"Full mels shape: {f_mels.shape}")
    print(f"Stream mels shape: {s_mels.shape}")
    
    min_len = min(f_mels.shape[-1], s_mels.shape[-1])
    mse = np.mean((f_mels[..., :min_len] - s_mels[..., :min_len])**2)
    print(f"Mel MSE: {mse:.2e}")

if __name__ == "__main__":
    debug_regression()
