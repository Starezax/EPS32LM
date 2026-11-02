# -*- coding: utf-8 -*-
"""
Генерація фраз з натренованої моделі.

Після pretrain (вільний текст):
  python llm/sample.py --ckpt llm/out/pretrain_base.pt --prompt "Вона усміхнулась і"
Після finetune (за категорією):
  python llm/sample.py --ckpt llm/out/finetune_base.pt --cat C --n 10
Категорії: C L S M W P V D O T
"""
import argparse
import os
import sys

import sentencepiece as spm
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ModelConfig, Transformer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cat", default=None)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=os.path.join(HERE, "data", "tok.model"))
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**{**ck["config"], "dropout": 0.0})
    model = Transformer(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    prefix = [sp.bos_id()]
    if args.cat:
        prefix.append(sp.piece_to_id(f"<{args.cat}>"))
    if args.prompt:
        prefix += sp.encode(args.prompt)
    for _ in range(args.n):
        idx = torch.tensor([prefix])
        out = model.generate(idx, args.max_tokens, args.temperature, args.top_k,
                             stop_ids=(sp.eos_id(), sp.bos_id()))
        ids = [t for t in out[0, len(prefix):].tolist() if t not in (sp.eos_id(), sp.bos_id())]
        print("-", (args.prompt + " " if args.prompt else "") + sp.decode(ids))


if __name__ == "__main__":
    main()
