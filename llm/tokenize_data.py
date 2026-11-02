# -*- coding: utf-8 -*-
"""
Перетворення текстів у токени (numpy uint16) для тренування.

pretrain:  llm/data/pretrain.txt -> pretrain_train.bin, pretrain_val.bin
           кожен шматок: <bos> текст <eos>
finetune:  dataset/phrases.tsv (+ llm/data/telegram.tsv) -> finetune_{train,val}.bin + .mask
           кожна фраза: <bos> <КАТЕГОРІЯ> текст <eos>; loss рахується лише на тексті і <eos>

Запуск:
  python llm/tokenize_data.py pretrain
  python llm/tokenize_data.py finetune
"""
import argparse
import os
import random
import sys

import numpy as np
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from charset import ALLOWED, normalize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ROOT = os.path.dirname(HERE)


def load_sp():
    return spm.SentencePieceProcessor(model_file=os.path.join(DATA, "tok.model"))


def do_pretrain(sp, val_frac):
    rng = random.Random(0)
    train, val = [], []
    chunk = []
    n_chunks = 0

    def flush():
        nonlocal n_chunks
        if not chunk:
            return
        ids = [sp.bos_id()]
        for p in sp.encode(chunk):
            ids += p
        ids.append(sp.eos_id())
        (val if rng.random() < val_frac else train).append(np.array(ids, dtype=np.uint16))
        n_chunks += 1
        if n_chunks % 5000 == 0:
            print(f"  шматків {n_chunks}")

    with open(os.path.join(DATA, "pretrain.txt"), encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                chunk.append(ln)
            else:
                flush()
                chunk = []
    flush()

    for name, parts in (("train", train), ("val", val)):
        arr = np.concatenate(parts)
        arr.tofile(os.path.join(DATA, f"pretrain_{name}.bin"))
        print(f"pretrain_{name}.bin: {len(arr) / 1e6:.1f}M токенів")


def domain_rows():
    rows = []
    for path in (os.path.join(ROOT, "dataset", "phrases.tsv"),
                 os.path.join(DATA, "telegram.tsv")):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if "\t" not in ln:
                    continue
                cat, text = ln.rstrip("\n").split("\t", 1)
                text = normalize(text)
                if text and all(ch in ALLOWED for ch in text):
                    rows.append((cat, text))
        print(f"  {os.path.relpath(path, ROOT)}: {len(rows)} рядків разом")
    return rows


def do_finetune(sp, val_frac):
    rng = random.Random(0)
    rows = domain_rows()
    rng.shuffle(rows)
    n_val = max(50, int(len(rows) * val_frac))
    splits = {"val": rows[:n_val], "train": rows[n_val:]}
    for name, rs in splits.items():
        ids, mask = [], []
        for cat, text in rs:
            tag = sp.piece_to_id(f"<{cat}>")
            seq = [sp.bos_id(), tag] + sp.encode(text) + [sp.eos_id()]
            ids += seq
            # mask[i] = 1, якщо токен i є ЦІЛЛЮ, яку треба вчити (текст і eos)
            mask += [0, 0] + [1] * (len(seq) - 2)
        np.array(ids, dtype=np.uint16).tofile(os.path.join(DATA, f"finetune_{name}.bin"))
        np.array(mask, dtype=np.uint8).tofile(os.path.join(DATA, f"finetune_{name}.mask"))
        print(f"finetune_{name}: {len(rs)} фраз, {len(ids)} токенів")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pretrain", "finetune"])
    ap.add_argument("--val_frac", type=float, default=0.005)
    args = ap.parse_args()
    sp = load_sp()
    if args.stage == "pretrain":
        do_pretrain(sp, args.val_frac)
    else:
        do_finetune(sp, max(args.val_frac, 0.03))


if __name__ == "__main__":
    main()
