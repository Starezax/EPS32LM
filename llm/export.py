# -*- coding: utf-8 -*-
"""
Експорт моделі для C-інференсу (ПК і ESP32-S3).

Вихід:
  llm/export/model.bin      ваги: int8 + float32-масштаби на групу з GS значень
  llm/export/tokenizer.bin  таблиця токенів для декодування

Формат model.bin (little-endian):
  header: 12 x int32
    magic 0x314D4C44 ("DLM1"), version=1, dim, hidden_dim, n_layers, n_heads,
    vocab_size, max_seq_len, group_size, 0, 0, 0
  float32  rms_att[n_layers][dim]
  float32  rms_ffn[n_layers][dim]
  float32  rms_final[dim]
  QTensor  tok_embeddings [vocab][dim]       (вона ж вихідний шар)
  для кожного шару l:
    QTensor wq[dim][dim], wk[dim][dim], wv[dim][dim], wo[dim][dim]
    QTensor w1[hidden][dim], w2[dim][hidden], w3[hidden][dim]
  QTensor = int8 q[n] , потім float32 s[n / group_size]   (n = rows*cols, row-major)

Формат tokenizer.bin:
  int32 magic 0x314B4F54 ("TOK1"), int32 vocab_size, int32 max_piece_len
  для кожного id: uint8 len, bytes (UTF-8, "▁" замінено на пробіл)

Запуск:  python llm/export.py --ckpt llm/out/finetune_base.pt
"""
import argparse
import os
import struct
import sys

import numpy as np
import sentencepiece as spm
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ModelConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "export")
GS = 32
MAGIC_MODEL = 0x314D4C44
MAGIC_TOK = 0x314B4F54


def quantize(w, gs=GS):
    """w: (rows, cols) float -> (int8 [n], float32 scales [n/gs])."""
    w = w.detach().float().cpu().numpy().reshape(-1, gs)
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q.reshape(-1), scale.astype(np.float32)


def dequantize(q, s, shape, gs=GS):
    return torch.from_numpy((q.reshape(-1, gs).astype(np.float32) * s[:, None]).reshape(shape))


def layer_tensors(sd, l):
    p = f"layers.{l}."
    return [sd[p + "attention.wq.weight"], sd[p + "attention.wk.weight"],
            sd[p + "attention.wv.weight"], sd[p + "attention.wo.weight"],
            sd[p + "feed_forward.w1.weight"], sd[p + "feed_forward.w2.weight"],
            sd[p + "feed_forward.w3.weight"]]


def export_model(ck, path):
    cfg = ModelConfig(**ck["config"])
    sd = ck["model"]
    for d in (cfg.dim, cfg.hidden_dim):
        assert d % GS == 0, f"{d} не ділиться на group_size {GS}"
    with open(path, "wb") as f:
        f.write(struct.pack("<12i", MAGIC_MODEL, 1, cfg.dim, cfg.hidden_dim, cfg.n_layers,
                            cfg.n_heads, cfg.vocab_size, cfg.max_seq_len, GS, 0, 0, 0))
        for name in ("attention_norm", "ffn_norm"):
            for l in range(cfg.n_layers):
                f.write(sd[f"layers.{l}.{name}.weight"].float().numpy().tobytes())
        f.write(sd["norm.weight"].float().numpy().tobytes())
        tensors = [sd["tok_embeddings.weight"]]
        for l in range(cfg.n_layers):
            tensors += layer_tensors(sd, l)
        for t in tensors:
            q, s = quantize(t)
            f.write(q.tobytes())
            f.write(s.tobytes())
    return cfg


def quantized_state_dict(ck):
    """Ті самі ваги, але пропущені через int8 (для порівняння з C)."""
    sd = {k: v.clone() for k, v in ck["model"].items()}
    for k, v in sd.items():
        if v.dim() == 2:
            q, s = quantize(v)
            sd[k] = dequantize(q, s, v.shape)
    sd["output.weight"] = sd["tok_embeddings.weight"]
    return sd


def export_tokenizer(sp, path):
    pieces = []
    for i in range(sp.get_piece_size()):
        p = sp.id_to_piece(i).replace("▁", " ").encode("utf-8")
        assert len(p) < 256
        pieces.append(p)
    with open(path, "wb") as f:
        f.write(struct.pack("<3i", MAGIC_TOK, len(pieces), max(len(p) for p in pieces)))
        for p in pieces:
            f.write(struct.pack("<B", len(p)))
            f.write(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    mpath = os.path.join(OUT_DIR, "model.bin")
    tpath = os.path.join(OUT_DIR, "tokenizer.bin")
    cfg = export_model(ck, mpath)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(HERE, "data", "tok.model"))
    export_tokenizer(sp, tpath)

    msize = os.path.getsize(mpath)
    kv = 2 * cfg.n_layers * cfg.max_seq_len * cfg.dim * 4
    print(f"Чекпойнт: {args.ckpt} (етап {ck.get('stage')}, ітерація {ck.get('iter')})")
    print(f"model.bin:     {msize / 1e6:.2f} MB  -> {mpath}")
    print(f"tokenizer.bin: {os.path.getsize(tpath) / 1e3:.1f} KB -> {tpath}")
    print(f"RAM для інференсу (KV-кеш float, seq {cfg.max_seq_len}): {kv / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
