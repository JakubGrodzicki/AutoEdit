"""Normalizacja tekstu polskiego do porównań znakowych i forced alignmentu."""
from __future__ import annotations

import re
import unicodedata

from num2words import num2words

PL_LETTERS = set("ąćęłńóśźż")
_MARKUP = re.compile(r"<[^>]*>|\{[^}]*\}|\[[^\]]*\]|\([^)]*\)")
_STARS = re.compile(r"\*[^*]*\*")
_APOS = re.compile(r"['’`´]")
_DIGITS = re.compile(r"\d+")
_REPEAT = re.compile(r"(.)\1{2,}")
_LAUGH = re.compile(r"^(?:h[aeoiy]){2,}h?$|^(?:a?h+a+)+h?$")


def strip_markup(text: str) -> str:
    return _MARKUP.sub(" ", text)


def _number(m: re.Match) -> str:
    n = int(m.group())
    if n > 999_999_999:
        return " "
    return " " + num2words(n, lang="pl") + " "


def normalize(text: str) -> str:
    """Małe litery, bez znaczników i interpunkcji, liczby słownie, q->k."""
    t = strip_markup(_STARS.sub(" ", text or ""))
    t = _APOS.sub("", t)
    t = _DIGITS.sub(_number, t).casefold()
    out = []
    for ch in t:
        if ch in PL_LETTERS or ("a" <= ch <= "z" and ch != "q"):
            out.append(ch)
        elif ch == "q":
            out.append("k")
        else:
            base = "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))
            if len(base) == 1 and "a" <= base <= "z":
                out.append("k" if base == "q" else base)
            else:
                out.append(" ")
    return " ".join("".join(out).split())


_ALIGNER_MAP = (("th", "t"), ("ph", "f"), ("x", "ks"), ("v", "w"))


def for_aligner(norm: str) -> str:
    """Pisownia obcych nazw -> przybliżona polska fonetyka, wyłącznie dla alignera akustycznego.
    Tylko zamiany, które nie dotykają rodzimych polskich słów (nie ruszamy np. y, w, ch)."""
    for a, b in _ALIGNER_MAP:
        norm = norm.replace(a, b)
    return norm


def classify(raw: str) -> str:
    """speech | vocal | nonverbal na podstawie surowej komórki."""
    s = (raw or "").strip()
    spoken = normalize(s)
    if not spoken:
        return "nonverbal"
    words = spoken.split()
    if all(_REPEAT.search(w) or _LAUGH.match(w) for w in words):
        return "vocal"
    return "speech"
