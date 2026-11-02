# -*- coding: utf-8 -*-
"""
Маленький трансформер у стилі llama2.c (RMSNorm, RoPE, SwiGLU, tied embeddings).
Архітектура навмисно проста, щоб її легко було портувати на C для ESP32-S3.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    hidden_dim: int = 768      # FFN (SwiGLU)
    max_seq_len: int = 256
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)


PRESETS = {
    # ~2.4M параметрів: швидкий експеримент на CPU
    "tiny":  dict(dim=192, n_layers=5, n_heads=6, hidden_dim=512),
    # ~7.2M параметрів: основна модель
    "small": dict(dim=288, n_layers=6, n_heads=6, hidden_dim=768),
    # ~11M параметрів: максимум для 16 МБ flash, краще тренувати в Colab
    "base":  dict(dim=320, n_layers=8, n_heads=8, hidden_dim=864),
}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(head_dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    f = torch.outer(t, freqs)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd), пари (0,1), (2,3), ... як у llama2.c
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.dim)
        self.attention = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.feed_forward = FeedForward(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attention_norm(x), cos, sin)
        return x + self.feed_forward(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight  # tied
        cos, sin = rope_tables(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, loss_mask=None):
        T = idx.shape[1]
        x = self.drop(self.tok_embeddings(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = self.output(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               reduction="none")
        if loss_mask is not None:
            m = loss_mask.view(-1).float()
            loss = (loss * m).sum() / m.sum().clamp(min=1.0)
        else:
            loss = loss.mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, stop_ids=()):
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if nxt.item() in stop_ids:
                break
        return idx


def budget(cfg: ModelConfig, kv_bytes=2):
    """Бюджет пам'яті на ESP32-S3: ваги int8 (+ float-масштаби груп по 64)."""
    m = Transformer(cfg)
    n = m.num_params()
    norm_params = cfg.dim * (2 * cfg.n_layers + 1)
    q_params = n - norm_params
    weights_int8 = q_params + (q_params // 64) * 4 + norm_params * 4
    kv = cfg.n_layers * cfg.max_seq_len * cfg.dim * 2 * kv_bytes
    acts = 4 * (cfg.dim * 6 + cfg.hidden_dim * 2 + cfg.n_heads * cfg.max_seq_len
                + cfg.vocab_size)
    return n, weights_int8, kv, acts


if __name__ == "__main__":
    for name, p in PRESETS.items():
        cfg = ModelConfig(**p)
        n, w, kv, a = budget(cfg)
        print(f"{name:6s} params {n / 1e6:5.2f}M | flash int8 {w / 1e6:5.2f} MB | "
              f"KV-cache fp16 {kv / 1e3:6.0f} KB | activations {a / 1e3:4.0f} KB")
