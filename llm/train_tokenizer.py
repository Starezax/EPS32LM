# -*- coding: utf-8 -*-
"""
Тренування BPE-токенізатора (SentencePiece).

Вхід:  llm/data/pretrain.txt + dataset/phrases.tsv (+ llm/data/telegram.tsv, якщо є)
Вихід: llm/data/tok.model, llm/data/tok.vocab

Запуск:  python llm/train_tokenizer.py --vocab_size 4096
"""
import argparse
import os
import random

import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ROOT = os.path.dirname(HERE)

# Теги категорій: окремі токени.
# C компл., L кохання, S підтримка, M сумую, W побажання, P жарт,
# V Відень, D перше побачення, O Олександрія, T стиль з телеграму
CATEGORY_TAGS = ["<C>", "<L>", "<S>", "<M>", "<W>", "<P>", "<V>", "<D>", "<O>", "<T>"]


def domain_lines():
    out = []
    for name in (os.path.join(ROOT, "dataset", "phrases.tsv"),
                 os.path.join(DATA, "telegram.tsv")):
        if os.path.exists(name):
            with open(name, encoding="utf-8") as f:
                out += [ln.rstrip("\n").split("\t", 1)[1] for ln in f if "\t" in ln]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--sample_chars", type=int, default=40_000_000)
    ap.add_argument("--domain_repeat", type=int, default=20,
                    help="скільки разів повторити доменні фрази, щоб їх слова стали токенами")
    args = ap.parse_args()

    rng = random.Random(0)
    lines, total = [], 0
    with open(os.path.join(DATA, "pretrain.txt"), encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and rng.random() < 0.5:
                lines.append(ln)
                total += len(ln)
                if total >= args.sample_chars:
                    break
    dom = domain_lines()
    lines += dom * args.domain_repeat
    rng.shuffle(lines)

    train_txt = os.path.join(DATA, "tok_train.txt")
    with open(train_txt, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"Текст для токенізатора: {total / 1e6:.1f}M символів корпусу + {len(dom)} доменних фраз x{args.domain_repeat}")

    spm.SentencePieceTrainer.train(
        input=train_txt,
        model_prefix=os.path.join(DATA, "tok"),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        normalization_rule_name="identity",
        split_digits=True,
        byte_fallback=False,
        max_sentence_length=8192,
        input_sentence_size=3_000_000,
        shuffle_input_sentence=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
        user_defined_symbols=CATEGORY_TAGS,
        num_threads=os.cpu_count(),
    )
    os.remove(train_txt)

    sp = spm.SentencePieceProcessor(model_file=os.path.join(DATA, "tok.model"))
    for s in ["Дарцю, у тебе дуже теплі очі. Просто щоб ти знала.",
              "Пам'ятаєш Відень? Я досі згадую, як ми гуляли там удвох."]:
        p = sp.encode(s, out_type=str)
        print(f"{len(s)} симв -> {len(p)} токенів: {p}")


if __name__ == "__main__":
    main()
