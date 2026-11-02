# -*- coding: utf-8 -*-
"""
Єдиний набір символів для всього проєкту (датасети, токенізатор, шрифт на пристрої).
Все, що не входить в ALLOWED, відкидається на етапі підготовки даних.
"""
import re

UA_UPPER = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
UA_LOWER = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
DIGITS = "0123456789"
PUNCT = " .,!?:;-—'\"«»()…♥%№+="

ALLOWED = set(UA_UPPER + UA_LOWER + DIGITS + PUNCT + "\n")

_REPL = [
    (re.compile(r"[’ʼ`´‘]"), "'"),
    (re.compile(r"[„“”]"), '"'),
    (re.compile(r"[‒–―]"), "—"),
    (re.compile(r"\.\.\."), "…"),
    (re.compile(r"[   \t]"), " "),
    (re.compile(r" {2,}"), " "),
]


def normalize(s):
    for rx, r in _REPL:
        s = rx.sub(r, s)
    # "—бо" усередині слова (Агей–бо) -> дефіс
    s = re.sub(r"(?<=\w)—(?=\w)", "-", s)
    return s.strip()
