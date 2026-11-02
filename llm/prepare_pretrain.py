# -*- coding: utf-8 -*-
"""
Підготовка корпусу для попереднього тренування (pretrain).

Вхід:  llm/data/raw/fiction.txt.bz2  (UberText 2.0, fiction, cleansed)
Вихід: llm/data/pretrain.txt         (абзаци, розділені порожнім рядком)

Що робить:
  - склеює рядки, розбиті по ~80 символів, назад в абзаци;
  - репліки діалогу ("— ...") лишає окремими рядками;
  - нормалізує лапки, апострофи, тире, три крапки;
  - викидає абзаци з російськими/латинськими літерами чи рідкісними символами
    (щоб шрифт на пристрої мав лише потрібні гліфи);
  - бере випадкову частку документів до ліміту символів.

Запуск:  python llm/prepare_pretrain.py --max_chars 300000000
"""
import argparse
import bz2
import os
import random
import re

from charset import ALLOWED, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw", "fiction.txt.bz2")
OUT = os.path.join(HERE, "data", "pretrain.txt")

BAD_RE = re.compile(r"[ыэъёЫЭЪЁa-zA-Z]")
FOOTNOTE_RE = re.compile(r"\s*\[\d+\]")


def iter_docs(path):
    """Документ = блок рядків між порожніми рядками."""
    buf = []
    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                buf.append(line)
            elif buf:
                yield buf
                buf = []
    if buf:
        yield buf


def doc_to_paragraphs(lines):
    """Склеює рядки. Нова репліка "— " починає новий рядок."""
    paras, cur = [], ""
    for ln in lines:
        ln = ln.strip()
        if ln.startswith(("—", "–", "-")) and cur:
            paras.append(cur)
            cur = ln
        elif cur.endswith("-") and not cur.endswith(" -"):
            cur = cur + ln       # перенос слова з дефісом лишаємо як є
        else:
            cur = (cur + " " + ln) if cur else ln
    if cur:
        paras.append(cur)
    return paras


def clean_para(p):
    p = FOOTNOTE_RE.sub("", p)
    p = normalize(p)
    if BAD_RE.search(p):
        return None
    if any(ch not in ALLOWED for ch in p):
        return None
    if len(p) < 3:
        return None
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_chars", type=int, default=300_000_000)
    ap.add_argument("--keep_prob", type=float, default=0.35,
                    help="частка документів, яку беремо (для різноманіття)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total, n_docs, n_paras, dropped = 0, 0, 0, 0
    with open(OUT, "w", encoding="utf-8", newline="\n") as out:
        for lines in iter_docs(RAW):
            if rng.random() > args.keep_prob:
                continue
            # великий документ ріжемо на шматки ~ по 40 абзаців
            paras = []
            for p in doc_to_paragraphs(lines):
                c = clean_para(p)
                if c is None:
                    dropped += 1
                else:
                    paras.append(c)
            if len(paras) < 2:
                continue
            for i in range(0, len(paras), 40):
                chunk = "\n".join(paras[i:i + 40])
                out.write(chunk + "\n\n")
                total += len(chunk)
                n_paras += len(paras[i:i + 40])
            n_docs += 1
            if n_docs % 2000 == 0:
                print(f"  docs {n_docs}, chars {total / 1e6:.1f}M")
            if total >= args.max_chars:
                break

    print(f"Записано: {OUT}")
    print(f"Документів: {n_docs}, абзаців: {n_paras}, відкинуто абзаців: {dropped}")
    print(f"Символів: {total / 1e6:.1f}M (~{total / 3.6 / 1e6:.0f}M токенів)")


if __name__ == "__main__":
    main()
