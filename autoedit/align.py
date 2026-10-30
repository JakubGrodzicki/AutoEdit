"""Forced alignment (wav2vec2 CTC) znanego tekstu w oknie wokół kandydata.

Daje (1) dokładniejsze granice niż czasy słów z Whispera i (2) niezależną od
transkrypcji ocenę, czy audio mówi ten tekst: luka log-prob między ścieżką
wymuszoną a swobodnym dekodowaniem CTC, osobno dla końcówki kwestii.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torchaudio.functional as AF
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2ForCTC

from .models import Candidate, Line
from .normalize import for_aligner
from .transcribe import SR

ALIGN_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"


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

    def align(self, audio16: np.ndarray, norm: str, t0: float, t1: float):
        """Zwraca (start, end, gap, tail_gap) w sekundach sesji albo None. Czasy Whispera bywają
        przesunięte o ~0,5 s, więc gdy dopasowanie dotyka krawędzi okna, ponawiamy z szerszym."""
        res = None
        for margin in (0.8, 1.8):
            res, touches_edge = self._align(audio16, norm, t0, t1, margin)
            if res is None or not touches_edge:
                break
        return res

    def _align(self, audio16: np.ndarray, norm: str, t0: float, t1: float, margin: float):
        w0 = max(0.0, t0 - margin)
        w1 = min(len(audio16) / SR, t1 + margin)
        seg = audio16[int(w0 * SR):int(w1 * SR)]
        targets = self._targets(norm)
        if len(seg) < SR // 10 or not targets:
            return None, False
        lp = self._log_probs(seg)  # (T, V)
        T = lp.shape[0]
        if len(targets) > T:
            return None, False
        tgt = torch.tensor([targets], dtype=torch.int32)
        aligned, _ = AF.forced_align(lp[None], tgt, blank=self.blank)
        labels = aligned[0]
        per_frame = lp.gather(1, labels[:, None].long())[:, 0]
        free = lp.max(dim=1).values
        spans = AF.merge_tokens(labels, per_frame.exp(), blank=self.blank)
        if not spans:
            return None, False
        stride = (len(seg) / SR) / T
        first, last = spans[0], spans[-1]
        gaps = []  # luka na token: suma (free - forced) po ramkach tokenu
        for s in spans:
            gaps.append(float((free[s.start:s.end] - per_frame[s.start:s.end]).sum()))
        gap = float(np.mean(gaps))
        k = max(3, len(gaps) // 4)
        tail_gap = float(np.mean(gaps[-k:]))
        start = w0 + first.start * stride
        end = w0 + last.end * stride
        edge = 2  # ramki
        touches = (first.start <= edge and w0 > 0.0) or (last.end >= T - edge and w1 < len(audio16) / SR)
        return (start, end, gap, tail_gap), touches


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


def refine(aligner: Aligner, audio16: np.ndarray, line: Line, cand: Candidate) -> None:
    res = aligner.align(audio16, line.norm, cand.start, cand.end)
    if res is None:
        return
    s, e, gap, tail = res
    cand.audio_gap, cand.audio_tail_gap = gap, tail
    cand.refined = energy_extend(audio16, s, e)
