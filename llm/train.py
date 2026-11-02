# -*- coding: utf-8 -*-
"""
Тренування моделі (CPU або GPU).

Етап 1, pretrain (загальна українська):
  python llm/train.py --stage pretrain --preset base --max_iters 20000

Етап 2, finetune (наші фрази, з чекпойнта pretrain):
  python llm/train.py --stage finetune --preset base --init llm/out/pretrain_base.pt --max_iters 1500

Тренування можна перервати Ctrl+C і продовжити з --resume.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PRESETS, ModelConfig, Transformer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")


class Loader:
    def __init__(self, stage, split, seq_len, batch, device):
        self.ids = np.memmap(os.path.join(DATA, f"{stage}_{split}.bin"), dtype=np.uint16, mode="r")
        mpath = os.path.join(DATA, f"{stage}_{split}.mask")
        self.mask = np.memmap(mpath, dtype=np.uint8, mode="r") if os.path.exists(mpath) else None
        self.seq_len, self.batch, self.device = seq_len, batch, device

    def get(self):
        ix = np.random.randint(0, len(self.ids) - self.seq_len - 1, self.batch)
        x = np.stack([self.ids[i:i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.ids[i + 1:i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        m = None
        if self.mask is not None:
            m = torch.from_numpy(np.stack([self.mask[i + 1:i + 1 + self.seq_len] for i in ix]).astype(np.int64))
            m = m.to(self.device)
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device), m


@torch.no_grad()
def evaluate(model, loader, iters):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y, m = loader.get()
        _, loss = model(x, y, m)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def lr_at(it, args):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    p = min(1.0, (it - args.warmup) / max(1, args.max_iters - args.warmup))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--preset", default="base", choices=list(PRESETS))
    ap.add_argument("--init", default=None, help="чекпойнт для старту (для finetune)")
    ap.add_argument("--resume", action="store_true", help="продовжити з останнього чекпойнта")
    ap.add_argument("--max_iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--min_lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=40)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ft = args.stage == "finetune"
    args.lr = args.lr or (3e-4 if ft else 1e-3)
    args.min_lr = args.min_lr or args.lr / 10
    args.warmup = args.warmup if args.warmup is not None else (50 if ft else 500)
    dropout = args.dropout if args.dropout is not None else (0.1 if ft else 0.0)

    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    name = f"{args.stage}_{args.preset}"
    ckpt_last = os.path.join(OUT, f"{name}_last.pt")
    ckpt_best = os.path.join(OUT, f"{name}.pt")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=os.path.join(DATA, "tok.model"))
    cfg = ModelConfig(vocab_size=sp.get_piece_size(), max_seq_len=args.seq_len,
                      dropout=dropout, **PRESETS[args.preset])
    model = Transformer(cfg).to(args.device)

    start_it, best_val = 0, float("inf")
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": args.weight_decay},
         {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    if args.resume and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=args.device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_it, best_val = ck["iter"], ck["best_val"]
        print(f"Продовжую з {ckpt_last}, ітерація {start_it}")
    elif args.init:
        ck = torch.load(args.init, map_location=args.device)
        model.load_state_dict(ck["model"])
        print(f"Старт з ваг {args.init}")

    print(f"Модель {args.preset}: {model.num_params() / 1e6:.2f}M параметрів, "
          f"пристрій {args.device}, потоків {args.threads}")
    train = Loader(args.stage, "train", args.seq_len, args.batch, args.device)
    val = Loader(args.stage, "val", args.seq_len, args.batch, args.device)
    print(f"Токенів: train {len(train.ids) / 1e6:.2f}M, val {len(val.ids) / 1e6:.2f}M")
    tok_per_it = args.batch * args.seq_len

    def save(path, it):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it,
                    "best_val": best_val, "config": cfg.to_dict(), "stage": args.stage}, path)

    model.train()
    t0 = time.time()
    it = start_it
    try:
        while it < args.max_iters:
            if it % args.eval_every == 0 and it > start_it or it == args.max_iters - 1:
                vl = evaluate(model, val, args.eval_iters)
                tag = ""
                if vl < best_val:
                    best_val = vl
                    save(ckpt_best, it)
                    tag = " (найкраща, збережено)"
                save(ckpt_last, it)
                print(f"--- iter {it}: val loss {vl:.3f}{tag}")

            for g in opt.param_groups:
                g["lr"] = lr_at(it, args)
            x, y, m = train.get()
            _, loss = model(x, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            it += 1

            if it % 20 == 0:
                dt = time.time() - t0
                t0 = time.time()
                tps = 20 * tok_per_it / dt
                left = (args.max_iters - it) * tok_per_it / tps / 3600
                print(f"iter {it}/{args.max_iters} loss {loss.item():.3f} "
                      f"lr {lr_at(it, args):.1e} | {tps:.0f} ток/с | лишилось ~{left:.1f} год")
    except KeyboardInterrupt:
        print("\nЗупинено. Зберігаю чекпойнт для --resume.")
    save(ckpt_last, it)
    print(f"Готово. Найкращий val loss {best_val:.3f}, ваги: {ckpt_best}")


if __name__ == "__main__":
    main()
