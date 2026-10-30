"""Dopasowanie kwestii z Excela do strumienia słów z transkrypcji.

1. find_candidates: dla każdej kwestii wszystkie podobne fragmenty strumienia
   (także powtórki), przyciągnięte do granic słów.
2. best_path: globalne programowanie dynamiczne po osi czasu - każdy fragment
   audio należy do najwyżej jednej kwestii, kolejność z Excela daje premię,
   skoki wstecz i pominięcia są dozwolone (bez premii).
3. finalize: statusy auto / przegląd / brak.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from .models import Candidate, Line, LineResult, Word


@dataclass
class MatchConfig:
    min_score: float = 60.0  # poniżej: fragment nie jest nawet kandydatem
    review_score: float = 70.0
    auto_score: float = 90.0
    max_repeats: int = 8  # ile razy ta sama kwestia może paść w jednym bloku
    base: float = 68.0  # gain kandydata = score - base
    bonus_next: float = 14.0  # kwestia nr N+1 po N
    bonus_same: float = 6.0  # powtórka tej samej kwestii
    bonus_skip: float = 7.0  # N+2..N+4
    overlap_tol: float = 0.15  # s, dopuszczalna nakładka sąsiednich fragmentów
    competitor_margin: float = 5.0
    short_words: int = 3  # tyle słów i mniej: auto tylko z potwierdzeniem sąsiadów
    max_gap: float = 1.2  # forced alignment: maks. luka log-prob na token
    max_tail_gap: float = 1.6


def passes_audio(c: Candidate, cfg: MatchConfig) -> bool:
    if c.audio_gap is None:
        return True
    return c.audio_gap <= cfg.max_gap and (c.audio_tail_gap or 0.0) <= cfg.max_tail_gap


class Stream:
    """Słowa z niepustą normą sklejone w jeden tekst z odwzorowaniem znak -> słowo."""

    def __init__(self, words: list[Word]):
        self.words = [w for w in words if w.norm]
        self.offs: list[int] = []
        parts, pos = [], 0
        for w in self.words:
            self.offs.append(pos)
            parts.append(w.norm)
            pos += len(w.norm) + 1
        self.text = " ".join(parts)

    def span(self, w0: int, w1: int) -> str:
        return " ".join(w.norm for w in self.words[w0:w1 + 1])

    def word_at(self, char: int) -> int:
        return max(0, min(len(self.words) - 1, bisect_right(self.offs, char) - 1))


def _polish(q: str, st: Stream, i0: int, i1: int) -> tuple[int, int, float]:
    """Wspinaczka po granicach słów, żeby maksymalizować podobieństwo do q."""
    best = fuzz.ratio(q, st.span(i0, i1))
    for _ in range(6):
        moves = [(i0 + 1, i1), (i0, i1 - 1), (i0 - 1, i1), (i0, i1 + 1)]
        scored = [(fuzz.ratio(q, st.span(a, b)), a, b) for a, b in moves
                  if 0 <= a <= b < len(st.words)]
        if not scored:
            break
        s, a, b = max(scored)
        if s <= best:
            break
        best, i0, i1 = s, a, b
    return i0, i1, best


def coverage(q: str, text: str) -> tuple[float, float]:
    """Jaka część początku i końca kwestii (po 20% znaków) została faktycznie powiedziana."""
    if not q:
        return 1.0, 1.0
    hit = np.zeros(len(q), dtype=bool)
    for op in Levenshtein.opcodes(q, text):
        if op.tag == "equal":
            hit[op.src_start:op.src_end] = True
    k = max(1, round(0.2 * len(q)))
    return float(hit[:k].mean()), float(hit[-k:].mean())


def _block_size(n_words: int) -> int:
    return 60 if n_words <= 25 else 150 if n_words <= 70 else 400


def find_candidates(lines: list[Line], words: list[Word], cfg: MatchConfig) -> tuple[Stream, list[Candidate]]:
    st = Stream(words)
    cands: list[Candidate] = []
    if not st.words:
        return st, cands
    todo = [ln for ln in lines if ln.kind != "nonverbal" and ln.norm]
    by_size: dict[int, list[Line]] = {}
    for ln in todo:
        by_size.setdefault(_block_size(len(ln.norm.split())), []).append(ln)

    for size, group in by_size.items():
        stride = size // 2
        starts = list(range(0, max(1, len(st.words) - stride), stride))
        blocks = []  # (char0, text)
        for s in starts:
            e = min(len(st.words) - 1, s + size - 1)
            c0 = st.offs[s]
            blocks.append((c0, st.text[c0:st.offs[e] + len(st.words[e].norm)]))
        scores = process.cdist([ln.norm for ln in group], [b[1] for b in blocks],
                               scorer=fuzz.partial_ratio, score_cutoff=cfg.min_score,
                               dtype=np.uint8, workers=-1)
        for li, ln in enumerate(group):
            found: list[Candidate] = []
            for bi in np.nonzero(scores[li] >= cfg.min_score)[0]:
                c0, text = blocks[bi]
                found += _extract(ln, st, c0, text, cfg)
            cands += _dedupe(found)
    cands.sort(key=lambda c: c.start)
    return st, cands


def _extract(ln: Line, st: Stream, c0: int, text: str, cfg: MatchConfig) -> list[Candidate]:
    out = []
    for _ in range(cfg.max_repeats):
        al = fuzz.partial_ratio_alignment(ln.norm, text, score_cutoff=cfg.min_score)
        if al is None:
            break
        ds, de = al.dest_start, max(al.dest_end, al.dest_start + 1)
        i0, i1 = st.word_at(c0 + ds), st.word_at(c0 + de - 1)
        i0, i1, score = _polish(ln.norm, st, i0, i1)
        text = text[:ds] + "#" * (de - ds) + text[de:]  # maska, offsety bez zmian
        if score < cfg.min_score:
            continue
        sc, ec = coverage(ln.norm, st.span(i0, i1))
        out.append(Candidate(ln.idx, i0, i1, st.words[i0].start, st.words[i1].end, float(score), sc, ec))
    return out


def _dedupe(found: list[Candidate]) -> list[Candidate]:
    keep: list[Candidate] = []
    for c in sorted(found, key=lambda c: -c.score):
        if all(_iou(c, k) <= 0.5 for k in keep):
            keep.append(c)
    return keep


def _iou(a: Candidate, b: Candidate) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    if inter <= 0:
        return 0.0
    return inter / (max(a.end, b.end) - min(a.start, b.start))


def best_path(cands: list[Candidate], cfg: MatchConfig) -> None:
    """Zaznacza on_path na kandydatach należących do optymalnej ścieżki przez nagranie."""
    n = len(cands)
    for c in cands:
        c.on_path = False
    if not n:
        return
    order = sorted(range(n), key=lambda i: cands[i].start)
    by_end = sorted(range(n), key=lambda i: cands[i].end)
    dp = [0.0] * n
    prev = [-1] * n
    done = [False] * n
    g_best = (0.0, -1)
    l_best: dict[int, tuple[float, int]] = {}
    p = 0
    for c in order:
        cc = cands[c]
        while p < n and done[by_end[p]] and cands[by_end[p]].end <= cc.start + cfg.overlap_tol:
            j = by_end[p]
            p += 1
            if dp[j] > g_best[0]:
                g_best = (dp[j], j)
            lj = cands[j].line_idx
            if lj not in l_best or dp[j] > l_best[lj][0]:
                l_best[lj] = (dp[j], j)
        opts = [g_best]
        l = cc.line_idx
        for d, bonus in [(0, cfg.bonus_same), (1, cfg.bonus_next)] + [(d, cfg.bonus_skip) for d in (2, 3, 4)]:
            if (l - d) in l_best:
                opts.append((l_best[l - d][0] + bonus, l_best[l - d][1]))
        val, pj = max(opts)
        if val <= 0:
            val, pj = 0.0, -1
        dp[c] = (cc.score - cfg.base) + val
        prev[c] = pj
        done[c] = True
    cur = max(range(n), key=lambda i: dp[i])
    while cur != -1:
        cands[cur].on_path = True
        cur = prev[cur]


def finalize(lines: list[Line], cands: list[Candidate], cfg: MatchConfig) -> list[LineResult]:
    by_line: dict[int, list[Candidate]] = {}
    for c in cands:
        by_line.setdefault(c.line_idx, []).append(c)
    chain = sorted((c for c in cands if c.on_path), key=lambda c: c.start)
    pos = {id(c): k for k, c in enumerate(chain)}
    text_count = Counter(ln.norm for ln in lines if ln.norm)

    def neighbor_confirmed(c: Candidate) -> bool:
        k = pos[id(c)]
        for step, want in ((-1, c.line_idx - 1), (1, c.line_idx + 1)):
            j = k + step
            while 0 <= j < len(chain) and chain[j].line_idx == c.line_idx:
                j += step
            if 0 <= j < len(chain) and chain[j].line_idx == want and chain[j].score >= cfg.review_score:
                return True
        return False

    results = []
    for ln in lines:
        r = LineResult(ln, candidates=sorted(by_line.get(ln.idx, []), key=lambda c: c.start))
        results.append(r)
        if ln.kind == "nonverbal":
            r.reasons.append("odgłos niewerbalny, ręcznie lub z luki")
            continue
        good = [c for c in r.candidates if c.score >= cfg.review_score]
        if not good:
            continue
        on = [c for c in good if c.on_path]
        pool = on or good
        complete = [c for c in pool if c.complete and passes_audio(c, cfg)]
        r.chosen = complete[-1] if complete else max(pool, key=lambda c: c.score)
        r.status = "przegląd"
        c = r.chosen
        if not on:
            r.reasons.append("fragment zajęty przez inną kwestię")
            continue
        takes = [o for o in r.candidates if o.score >= cfg.auto_score and o.complete and passes_audio(o, cfg)]
        if len(takes) > 1:  # kilka pełnych podejść: wybór należy do człowieka, ostatnie jest tylko domyślne
            r.reasons.append(f"{len(takes)} pełnych podejść do odsłuchu, domyślnie ostatnie")
        elif len(on) > 1:
            r.reasons.append(f"{len(on)} podejść, wybrano ostatnie kompletne")
        confirmed = neighbor_confirmed(c)
        n_words = len(ln.norm.split())
        why = []
        if ln.kind != "speech":
            why.append("odgłos głosowy")
        if not c.complete:
            why.append("kwestia niedokończona")
        if c.score < cfg.auto_score:
            why.append(f"podobieństwo {c.score:.0f}%")
        if (text_count[ln.norm] > 1 or n_words <= cfg.short_words) and not confirmed:
            why.append("krótka/powtarzalna, brak potwierdzenia sąsiadami")
        rival = [o for o in cands if o.line_idx != ln.idx and _iou(o, c) > 0.5
                 and o.score >= c.score - cfg.competitor_margin]
        if rival and not confirmed:
            why.append(f"konkurencyjna kwestia #{rival[0].line_idx}")
        if not passes_audio(c, cfg):
            why.append("audio nie potwierdza tekstu")
        if len(takes) > 1:
            why.append("wiele podejść")
        r.reasons += why
        if not why:
            r.status = "auto"
    return results
