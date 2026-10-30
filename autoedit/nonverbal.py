"""Propozycje dla kwestii niewerbalnych (śmiech, warknięcia) z luk między dopasowanymi kwestiami."""
from __future__ import annotations

import numpy as np

from .models import Candidate, LineResult
from .transcribe import SR


def voiced_segments(audio16: np.ndarray, t0: float, t1: float, hop: float = 0.02,
                    merge_gap: float = 0.3, min_len: float = 0.2, max_len: float = 8.0) -> list[tuple[float, float]]:
    a, b = int(t0 * SR), int(t1 * SR)
    h = int(hop * SR)
    n = (b - a) // h
    if n < 3:
        return []
    rms = np.sqrt((audio16[a:a + n * h].reshape(n, h) ** 2).mean(axis=1) + 1e-12)
    thr = max(np.percentile(rms, 10) * 4.0, 5e-4)
    loud = rms > thr
    segs, i = [], 0
    while i < n:
        if loud[i]:
            j = i
            while j < n and loud[j]:
                j += 1
            segs.append([t0 + i * hop, t0 + j * hop])
            i = j
        else:
            i += 1
    merged: list[list[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(s, e) for s, e in merged if min_len <= e - s <= max_len]


def propose(results: list[LineResult], audio16: np.ndarray, max_per_line: int = 5,
            open_window: float = 8.0) -> None:
    """Dla każdej kwestii niewerbalnej dopisuje kandydatów (origin='gap'); nigdy nie nadaje statusu auto."""
    total = len(audio16) / SR
    for k, r in enumerate(results):
        if r.line.kind != "nonverbal":
            continue
        prev = next((x.chosen for x in reversed(results[:k]) if x.chosen is not None), None)
        nxt = next((x.chosen for x in results[k + 1:] if x.chosen is not None), None)
        if prev is None and nxt is None:
            continue
        t0 = (prev.refined[1] if prev and prev.refined else prev.end) if prev else max(0.0, nxt.start - open_window)
        t1 = (nxt.refined[0] if nxt and nxt.refined else nxt.start) if nxt else min(total, prev.end + open_window)
        if t1 - t0 < 0.2:
            continue
        for s, e in voiced_segments(audio16, t0, t1)[:max_per_line]:
            r.candidates.append(Candidate(r.line.idx, -1, -1, s, e, 0.0, origin="gap", refined=(s, e)))
        if r.candidates:
            r.status = "przegląd"
            r.reasons.append("propozycja z luki między sąsiednimi kwestiami, do odsłuchu")
