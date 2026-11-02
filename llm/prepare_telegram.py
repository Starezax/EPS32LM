# -*- coding: utf-8 -*-
"""
Розбір експорту чату з Telegram Desktop (формат JSON).

Як експортувати: Telegram Desktop -> відкрити чат -> ⋮ -> Export chat history
  -> зняти всі галочки з медіа -> Format: Machine-readable JSON -> Export.
Вийде папка з файлом result.json.

1) Подивитись, хто є в чаті і скільки повідомлень:
   python llm/prepare_telegram.py --json "C:/шлях/до/result.json"
2) Зібрати ТВОЇ повідомлення як навчальні дані (категорія T):
   python llm/prepare_telegram.py --json "C:/шлях/до/result.json" --me "Твоє ім'я в Telegram"
   -> llm/data/telegram.tsv
   -> llm/data/telegram_memories.txt (усі повідомлення про Відень/Олександрію/побачення,
      з обох сторін, щоб ТИ переглянув і виписав реальні деталі в dataset/memories.txt)

Дані нікуди не відправляються, все лишається на твоєму комп'ютері.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from charset import ALLOWED, normalize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_TSV = os.path.join(HERE, "data", "telegram.tsv")
OUT_MEM = os.path.join(HERE, "data", "telegram_memories.txt")

MERGE_SECONDS = 120     # склеювати повідомлення одного автора, якщо пауза менша
MIN_LEN, MAX_LEN = 12, 160

MEMORY_RE = re.compile(r"відн|відень|олександрі|побачен", re.IGNORECASE)
PRIVATE_RE = re.compile(
    r"https?://|www\.|@\w|\+?\d[\d\s\-()]{7,}\d|\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b|пароль|код\b",
    re.IGNORECASE)
EMOJI_HEART_RE = re.compile(r"[❤♥💕💖💗💓💞💘💝😍🥰😘]\ufe0f?")


def msg_text(m):
    t = m.get("text", "")
    if isinstance(t, list):
        t = "".join(x if isinstance(x, str) else x.get("text", "") for x in t)
    return t


def clean(t):
    t = EMOJI_HEART_RE.sub(" ♥ ", t)
    t = normalize(t.replace("\n", " "))
    t = "".join(ch for ch in t if ch in ALLOWED)   # прибирає інші емодзі
    t = re.sub(r"(♥\s*){2,}", "♥ ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def ok(t):
    if not (MIN_LEN <= len(t) <= MAX_LEN):
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters >= 0.6 * len(t.replace(" ", ""))


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    msgs = []
    for m in d.get("messages", []):
        if m.get("type") != "message":
            continue
        t = msg_text(m)
        if not t.strip():
            continue
        msgs.append((m.get("from") or "?", datetime.fromisoformat(m["date"]), t))
    return msgs


def merge(msgs):
    out = []
    for who, dt, t in msgs:
        if out and out[-1][0] == who and (dt - out[-1][1]).total_seconds() < MERGE_SECONDS:
            out[-1] = (who, dt, out[-1][2] + " " + t)
        else:
            out.append((who, dt, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--me", default=None, help="твоє ім'я в полі 'from'")
    args = ap.parse_args()

    msgs = load(args.json)
    counts = Counter(w for w, _, _ in msgs)
    print(f"Повідомлень: {len(msgs)}")
    for who, n in counts.most_common():
        print(f"  {who!r}: {n}")
    if not args.me:
        print("\nДодай --me \"ім'я\" (як вище), щоб зібрати дані.")
        return
    if args.me not in counts:
        sys.exit(f"Немає автора {args.me!r}")

    merged = merge(msgs)
    rows, skipped_private = [], 0
    for who, _, t in merged:
        if who != args.me:
            continue
        if PRIVATE_RE.search(t):
            skipped_private += 1
            continue
        c = clean(t)
        if ok(c):
            rows.append(c)
    rows = list(dict.fromkeys(rows))
    with open(OUT_TSV, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(f"T\t{r}\n")

    with open(OUT_MEM, "w", encoding="utf-8", newline="\n") as f:
        for who, dt, t in merged:
            if MEMORY_RE.search(t):
                f.write(f"[{dt:%Y-%m-%d}] {who}: {t.strip()}\n")

    print(f"\nТвоїх фраз після фільтрів: {len(rows)} -> {OUT_TSV}")
    print(f"Відкинуто через посилання/номери/паролі: {skipped_private}")
    print(f"Згадки про спогади -> {OUT_MEM}")
    print("ПЕРЕГЛЯНЬ telegram.tsv і видали все, що не хочеш бачити на екрані!")


if __name__ == "__main__":
    main()
