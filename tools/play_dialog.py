# -*- coding: utf-8 -*-
"""
Перевірка і програвання сценарію діалогу в терміналі.

Поки моделі немає, вузли "gen" беруть випадкову фразу з датасету
потрібної категорії. Пізніше тут буде виклик моделі.

Керування: a = ліва кнопка, d = права кнопка, q = вихід.
Запуск:  python tools/play_dialog.py
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIALOG_PATH = os.path.join(ROOT, "dialog", "dialog.json")
PHRASES_PATH = os.path.join(ROOT, "dataset", "phrases.tsv")

CATEGORIES = set("CLSMWPVDO")
MAX_LABEL = 10     # шрифт 12x24: 10 символів * 12 px = 120 px (пів екрана)
MAX_SAY = 110      # 20 символів/рядок * 8 рядків, із запасом на перенос
IDLE_PHRASES = 3


def load_phrases():
    by_cat = {c: [] for c in CATEGORIES}
    with open(PHRASES_PATH, encoding="utf-8") as f:
        for line in f:
            cat, text = line.rstrip("\n").split("\t", 1)
            by_cat[cat].append(text)
    return by_cat


def validate(d):
    errors = []
    nodes = d["nodes"]
    longest_addr = max(d["addr"], key=len)
    if d["start"] not in nodes:
        errors.append(f"start '{d['start']}' не існує")
    for nid, n in nodes.items():
        if ("say" in n) == ("gen" in n):
            errors.append(f"{nid}: має бути рівно одне з 'say' або 'gen'")
        if "gen" in n and n["gen"] not in CATEGORIES:
            errors.append(f"{nid}: невідома категорія '{n['gen']}'")
        for s in n.get("say", []):
            worst = s.replace("{addr}", longest_addr).replace("{Addr}", longest_addr)
            if len(worst) > MAX_SAY:
                errors.append(f"{nid}: текст задовгий ({len(worst)}): {s}")
        for side in ("left", "right"):
            b = n.get(side)
            if not b:
                errors.append(f"{nid}: немає кнопки '{side}'")
                continue
            if len(b["label"]) > MAX_LABEL:
                errors.append(f"{nid}.{side}: підпис задовгий: {b['label']}")
            if b["next"] != "IDLE" and b["next"] not in nodes:
                errors.append(f"{nid}.{side}: next '{b['next']}' не існує")

    # досяжність
    seen, stack = set(), [d["start"]]
    while stack:
        nid = stack.pop()
        if nid in seen or nid == "IDLE":
            continue
        seen.add(nid)
        for side in ("left", "right"):
            stack.append(nodes[nid][side]["next"])
    for nid in nodes:
        if nid not in seen:
            errors.append(f"{nid}: недосяжний зі start")
    return errors


def fill(text, addrs):
    a = random.choice(addrs)
    return text.replace("{Addr}", a[:1].upper() + a[1:]).replace("{addr}", a)


def wrap(text, width=20):
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines


def show(text, left, right):
    # імітація екрана 240x240, шрифт 12x24: 20 символів у рядку
    print()
    print("+" + "-" * 22 + "+")
    for line in wrap(text):
        print(f"| {line:<20} |")
    print("+" + "-" * 22 + "+")
    print(f"| {left:<10}{right:>10} |")
    print("+" + "-" * 22 + "+")
    print("  [a] ліва      [d] права")


def main():
    with open(DIALOG_PATH, encoding="utf-8") as f:
        d = json.load(f)
    errors = validate(d)
    if errors:
        print("ПОМИЛКИ В СЦЕНАРІЇ:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"Сценарій OK: {len(d['nodes'])} вузлів.")

    phrases = load_phrases()
    nodes = d["nodes"]
    nid = d["start"]
    while True:
        if nid == "IDLE":
            print("\n~~~ режим очікування (на пристрої: фраза кожні 10 с) ~~~")
            for _ in range(IDLE_PHRASES):
                cat = random.choice(sorted(CATEGORIES))
                print(f"  ♥ {random.choice(phrases[cat])}")
            input("\n(Enter = натиснути кнопку, почати діалог знову)")
            nid = d["start"]
            continue
        n = nodes[nid]
        if "gen" in n:
            text = random.choice(phrases[n["gen"]])
        else:
            text = fill(random.choice(n["say"]), d["addr"])
        show(text, n["left"]["label"], n["right"]["label"])
        while True:
            k = input("> ").strip().lower()
            if k in ("a", "d", "q"):
                break
        if k == "q":
            break
        nid = n["left" if k == "a" else "right"]["next"]


if __name__ == "__main__":
    main()
