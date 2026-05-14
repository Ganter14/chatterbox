import torch
import torch.nn as nn
import logging

class S3GenGraph:
    """
    CUDA Graph capture for S3Gen components (ConditionalDecoder).
    """
    def __init__(self, estimator, max_frames=256, device="cuda"):
        self.estimator = estimator
        self.max_frames = max_frames
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.graph = None
        
        # Static buffers
        # B=2 for CFG
        self.static_x = torch.zeros((2, 80, max_frames), device=device, dtype=estimator.dtype)
        self.static_mu = torch.zeros((2, 80, max_frames), device=device, dtype=estimator.dtype)
        self.static_mask = torch.zeros((2, 1, max_frames), device=device, dtype=torch.bool)
        self.static_t = torch.zeros((2,), device=device, dtype=estimator.dtype)
        self.static_spks = torch.zeros((2, 80), device=device, dtype=estimator.dtype)
        self.static_cond = torch.zeros((2, 80, max_frames), device=device, dtype=estimator.dtype)
        self.static_r = torch.zeros((2,), device=device, dtype=estimator.dtype)
        
        # Output buffer
        self.static_dxdt = None

    def capture(self):
        print(f"Capturing S3Gen Estimator CUDA graph (max_frames={self.max_frames})...", flush=True)
        
        with torch.inference_mode():
            # Warmup
            for _ in range(3):
                self.estimator(
                    x=self.static_x,
                    mask=self.static_mask,
                    mu=self.static_mu,
                    t=self.static_t,
                    spks=self.static_spks,
                    cond=self.static_cond,
                    r=self.static_r
                )
            
            torch.cuda.synchronize()
            
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self.stream):
                with torch.cuda.graph(self.graph):
                    self.static_dxdt = self.estimator(
                        x=self.static_x,
                        mask=self.static_mask,
                        mu=self.static_mu,
                        t=self.static_t,
                        spks=self.static_spks,
                        cond=self.static_cond,
                        r=self.static_r
                    )
        
        torch.cuda.synchronize()
        print("S3Gen Estimator CUDA graph captured successfully!", flush=True)

    def run(self, x, mask, mu, t, spks, cond, r):
        # Full run (fallback or single call)
        T = x.size(2)
        if T > self.max_frames:
            return self.estimator(x, mask, mu, t, spks, cond, r)
        
        self.static_x[:, :, :T].copy_(x)
        self.static_mu[:, :, :T].copy_(mu)
        self.static_mask[:, :, :T].copy_(mask)
        self.static_cond[:, :, :T].copy_(cond)
        self.static_t.copy_(t)
        self.static_spks.copy_(spks)
        self.static_r.copy_(r)
        
        self.graph.replay()
        return self.static_dxdt[:, :, :T]

    def prepare(self, mu, mask, spks, cond):
        """Prepare constants for the ODE loop."""
        T = mu.size(2)
        if T <= self.max_frames:
            self.static_mu[:, :, :T].copy_(mu)
            self.static_mask[:, :, :T].copy_(mask)
            self.static_cond[:, :, :T].copy_(cond)
            self.static_spks.copy_(spks)

    def step(self, x, t, r):
        """Single ODE step."""
        T = x.size(2)
        self.static_x[:, :, :T].copy_(x)
        self.static_t.copy_(t)
        self.static_r.copy_(r)
        self.graph.replay()
        return self.static_dxdt[:, :, :T]
