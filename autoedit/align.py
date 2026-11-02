"""Forced alignment (wav2vec2 CTC) znanego tekstu w oknie wokół kandydata.

Daje (1) dokładniejsze granice niż czasy słów z Whispera i (2) niezależną od
transkrypcji ocenę, czy audio mówi ten tekst: luka log-prob między ścieżką
wymuszoną a swobodnym dekodowaniem CTC, osobno dla końcówki kwestii.
"""
from __future__ import annotations

import json
from bisect import bisect_right

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2ForCTC

from .models import Candidate, Line
from .normalize import for_aligner
from .transcribe import SR

ALIGN_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
GAP_PENALTY = 0.1  # koszt ramki ciszy między głoskami (1 s = ok. 5 nat): prawdziwa pauza zostaje, rzut o kilka sekund nie


def ctc_viterbi(lp: np.ndarray, targets: list[int], blank: int, gap_penalty: float = GAP_PENALTY) -> list[tuple[int, int]]:
    """Wymuszone dopasowanie CTC: dla każdego tokenu (pierwsza ramka, ramka za ostatnią); [] gdy tekst się nie mieści.

    Zwykły forced alignment każe traktować całe okno jako tę kwestię i ciszę. Gdy w oknie jest cudza mowa, pierwsza
    lub ostatnia głoska opłaca się na niej, nawet kilka sekund dalej. Tu ramki przed pierwszym i po ostatnim tokenie
    są darmowe, cokolwiek w nich słychać, a każda ramka ciszy między tokenami coś kosztuje."""
    lp = lp.astype(np.float64)
    n_frames, n_tok = lp.shape[0], len(targets)
    tok = np.array(targets)
    emit, silence, free = lp[:, tok], lp[:, blank], lp.max(axis=1)
    rows = np.arange(n_tok)
    same = np.concatenate(([False], tok[1:] == tok[:-1]))  # ten sam token pod rząd wymaga ciszy między nimi
    before = free[0]  # ścieżka: pierwszy token jeszcze się nie zaczął
    in_tok = np.full(n_tok, -np.inf)  # najlepsza ścieżka kończąca się w tokenie j
    in_gap = np.full(n_tok, -np.inf)  # po tokenie j: cisza; po ostatnim: koniec kwestii
    in_tok[0] = emit[0, 0]
    tok_from = np.zeros((n_frames, n_tok), dtype=np.int8)  # 0: dalej ten token, 1: z ciszy, 2: wprost z poprzedniego tokenu
    gap_opened = np.zeros((n_frames, n_tok), dtype=bool)  # cisza po tokenie j zaczęła się w tej ramce
    tok_from[0, 0] = 1
    for t in range(1, n_frames):
        from_gap = np.concatenate(([before], in_gap[:-1]))
        from_prev = np.concatenate(([-np.inf], in_tok[:-1]))
        from_prev[same] = -np.inf
        options = np.stack([in_tok, from_gap, from_prev])
        tok_from[t] = options.argmax(axis=0)
        gap_opened[t] = in_tok > in_gap
        gap_score = np.full(n_tok, silence[t] - gap_penalty)
        gap_score[-1] = free[t]
        in_tok, in_gap = options[tok_from[t], rows] + emit[t], np.maximum(in_gap, in_tok) + gap_score
        before += free[t]
    if not np.isfinite(max(in_tok[-1], in_gap[-1])):
        return []
    spans: list[tuple[int, int]] = [(0, 0)] * n_tok
    t, j = n_frames - 1, n_tok - 1
    gap = bool(in_gap[-1] > in_tok[-1])
    end = -1  # ramka za ostatnią bieżącego tokenu
    while t >= 0 and j >= 0:
        if gap:
            gap = not gap_opened[t, j]
            t -= 1
            continue
        end = end if end >= 0 else t + 1
        spans[j] = (t, end)
        step = tok_from[t, j]
        if step != 0:
            gap = step == 1
            j -= 1
            end = -1
        t -= 1
    return spans


class Aligner:
    def __init__(self, device: str = "cuda"):
        self.device = device
        with open(hf_hub_download(ALIGN_MODEL, "vocab.json"), encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        self.blank = self.vocab["<pad>"]
        self.model = Wav2Vec2ForCTC.from_pretrained(ALIGN_MODEL).to(device).eval()

    def _log_probs(self, audio: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(audio.astype(np.float32))
        x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)
        with torch.inference_mode():
            logits = self.model(x[None].to(self.device)).logits
        return torch.log_softmax(logits.float(), dim=-1)[0].cpu()

    def _targets(self, norm: str) -> list[int]:
        ids = []
        for ch in for_aligner(norm).replace(" ", "|"):
            if ch in self.vocab:
                ids.append(self.vocab[ch])
        return ids

    def align(self, audio16: np.ndarray, norm: str, t0: float, t1: float, lo: float = 0.0, hi: float | None = None):
        """Zwraca (start, end, gap, tail_gap, span_gap, pause) w sekundach sesji albo None. Czasy Whispera bywają
        przesunięte o ~0,5 s, więc gdy dopasowanie dotyka krawędzi okna, ponawiamy z szerszym. Okno nigdy nie
        wychodzi poza [lo, hi]."""
        hi = len(audio16) / SR if hi is None else min(hi, len(audio16) / SR)
        res = None
        for margin in (0.8, 1.8):
            res, touches_edge = self._align(audio16, norm, t0, t1, margin, lo, hi)
            if res is None or not touches_edge:
                break
        return res

    def _align(self, audio16: np.ndarray, norm: str, t0: float, t1: float, margin: float, lo: float, hi: float):
        w0 = max(lo, t0 - margin)
        w1 = min(hi, t1 + margin)
        seg = audio16[int(w0 * SR):int(w1 * SR)]
        targets = self._targets(norm)
        if len(seg) < SR // 10 or not targets:
            return None, False
        lp = self._log_probs(seg).numpy()  # (T, V)
        T = lp.shape[0]
        if len(targets) > T:
            return None, False
        spans = ctc_viterbi(lp, targets, self.blank)
        if not spans:
            return None, False
        stride = (len(seg) / SR) / T
        first, last = spans[0][0], spans[-1][1]
        free = lp.max(axis=1)
        forced = lp[:, self.blank].copy()
        gaps = []  # luka na token: suma (free - forced) po ramkach tokenu
        for (a, b), token in zip(spans, targets):
            forced[a:b] = lp[a:b, token]
            gaps.append(float((free[a:b] - forced[a:b]).sum()))
        gap = float(np.mean(gaps))
        k = max(3, len(gaps) // 4)
        tail_gap = float(np.mean(gaps[-k:]))
        span_gap = float((free[first:last] - forced[first:last]).sum()) / len(targets)  # także cisza: tam słychać cudzą mowę
        pause = max((b[0] - a[1] for a, b in zip(spans, spans[1:])), default=0) * stride
        start = w0 + first * stride
        end = w0 + last * stride
        edge = 2  # ramki
        touches = (first <= edge and w0 > lo) or (last >= T - edge and w1 < hi)
        return (start, end, gap, tail_gap, span_gap, pause), touches


def energy_extend(audio16: np.ndarray, start: float, end: float, max_left: float = 0.15,
                  max_right: float = 0.25, hop: float = 0.01) -> tuple[float, float]:
    """CTC trafia w środek głoski, więc cofamy początek i wydłużamy koniec dopóki sygnał jest głośny."""
    h = int(hop * SR)
    a = max(0, int((start - 1.0) * SR))
    b = min(len(audio16), int((end + 1.0) * SR))
    frames = audio16[a:b]
    n = len(frames) // h
    if n < 4:
        return start, end
    rms = np.sqrt((frames[:n * h].reshape(n, h) ** 2).mean(axis=1) + 1e-12)
    floor = np.percentile(rms, 10)
    thr = max(floor * 3.0, 1e-4)
    i0 = int((start - a / SR) / hop)
    i1 = int((end - a / SR) / hop)
    left = 0
    while i0 > 0 and rms[i0 - 1] > thr and left < max_left / hop:
        i0 -= 1
        left += 1
    right = 0
    while i1 < n - 1 and rms[i1 + 1] > thr and right < max_right / hop:
        i1 += 1
        right += 1
    return a / SR + i0 * hop, a / SR + (i1 + 1) * hop


def refine(aligner: Aligner, audio16: np.ndarray, line: Line, cand: Candidate, lo: float = 0.0,
           hi: float | None = None) -> None:
    res = aligner.align(audio16, line.norm, cand.start, cand.end, lo, hi)
    if res is None:
        return
    s, e, cand.audio_gap, cand.audio_tail_gap, cand.audio_span_gap, cand.audio_pause = res
    cand.refined = energy_extend(audio16, s, e)


def refine_all(aligner: Aligner, audio16: np.ndarray, lines: list[Line], cands: list[Candidate], min_score: float) -> None:
    """Najpierw ścieżka, potem pozostali kandydaci widoczni w przeglądzie. Okno nie sięga w głąb sąsiedniej kwestii
    ze ścieżki, bo tam głoski miałyby się na czym zaczepić (0,1 s luzu: Whisper może podać koniec za późno)."""
    by_idx = {ln.idx: ln for ln in lines}
    path = sorted((c for c in cands if c.on_path), key=lambda c: c.start)
    prev_end = 0.0
    for c in path:
        refine(aligner, audio16, by_idx[c.line_idx], c, lo=max(0.0, prev_end - 0.1))
        prev_end = c.end
    done = sorted(c.refined for c in path if c.refined)
    starts, ends = [s for s, _ in done], [e for _, e in done]
    for c in cands:
        if c.on_path or c.score < min_score:
            continue
        mid = (c.start + c.end) / 2
        k = bisect_right(ends, mid)
        lo = ends[k - 1] if k else 0.0
        k = bisect_right(starts, mid)
        refine(aligner, audio16, by_idx[c.line_idx], c, lo=lo, hi=starts[k] if k < len(starts) else None)
