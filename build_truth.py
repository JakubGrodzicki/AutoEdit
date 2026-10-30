"""Dane wzorcowe: gdzie w sesji leży każda gotowa, ręcznie pocięta kwestia.

Użycie:  python build_truth.py sesja.wav edity_nazwane/ -o output/truth.json
Szuka każdego pliku w sesji korelacją cech log-mel (odporna na obróbkę i głośność).
Pliki bez pewnego trafienia są raportowane i wyłączane z pomiaru. Program silnika z tych plików nie korzysta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

from autoedit.transcribe import SR, load_audio16, prepare_audio

HOP = 160  # 10 ms
N_MELS = 40
_mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=400, hop_length=HOP, n_mels=N_MELS)


def logmel(x: np.ndarray) -> np.ndarray:
    """(pasma, ramki) log-mel; odporne na głośność i lekką obróbkę, w przeciwieństwie do obwiedni energii."""
    with torch.inference_mode():
        m = _mel(torch.from_numpy(x.astype(np.float32)))
    return torch.log(m + 1e-6).numpy()


class SessionIndex:
    """Cechy sesji + gotowe FFT, żeby każdy edit kosztował tylko jedną korelację na pasmo."""

    def __init__(self, feats: np.ndarray):
        self.n = feats.shape[1]
        self.size = 1 << int(np.ceil(np.log2(self.n + 2000)))
        self.f = np.fft.rfft(feats, self.size, axis=1)
        c1 = np.concatenate([np.zeros((feats.shape[0], 1)), np.cumsum(feats, axis=1)], axis=1)
        c2 = np.concatenate([np.zeros((feats.shape[0], 1)), np.cumsum(feats ** 2, axis=1)], axis=1)
        self.c1, self.c2 = c1, c2

    def score(self, tmpl: np.ndarray) -> np.ndarray:
        """Średnia po pasmach znormalizowanej korelacji wzajemnej dla każdego przesunięcia."""
        m = tmpl.shape[1]
        t = tmpl - tmpl.mean(axis=1, keepdims=True)
        num = np.fft.irfft(self.f * np.conj(np.fft.rfft(t, self.size, axis=1)), self.size, axis=1)[:, :self.n - m + 1]
        s1 = self.c1[:, m:] - self.c1[:, :-m]
        s2 = self.c2[:, m:] - self.c2[:, :-m]
        var = np.maximum(s2 - s1 ** 2 / m, 1e-2 * m)  # płaskie fragmenty (cyfrowa cisza) nie mogą dawać fałszywie wysokiej korelacji
        den = np.sqrt(var) * np.sqrt((t ** 2).sum(axis=1, keepdims=True)) + 1e-9
        score = (num / den).mean(axis=0)
        # poziom energii okna musi być zbliżony do edita (edytor mógł podgłośnić, ale szum tła != mowa)
        level = s1.sum(axis=0) / (m * s1.shape[0])
        score[np.abs(level - tmpl.mean()) > 2.5] = -1.0
        return score


def load_edit16(path: Path) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = torch.from_numpy(a.mean(axis=1))
    return AF.resample(a, sr, SR).numpy() if sr != SR else a.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("edits", type=Path, help="folder z plikami nazwanymi nazwą skryptu")
    ap.add_argument("-o", "--out", type=Path, default=Path("output/truth.json"))
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--min-ncc", type=float, default=0.5)
    a = ap.parse_args()

    sess = load_audio16(prepare_audio(a.session, a.cache))
    index = SessionIndex(logmel(sess))
    truth, bad = {}, []
    for f in sorted(a.edits.glob("*.wav")):
        e = load_edit16(f)
        c = index.score(logmel(e)[:, :-1])
        k = int(np.argmax(c))
        second = np.max(np.concatenate([c[:max(0, k - 100)], c[k + 100:]])) if len(c) > 200 else 0.0
        truth[f.stem] = {"start": round(k * HOP / SR, 3), "end": round(k * HOP / SR + len(e) / SR, 3),
                         "ncc": round(float(c[k]), 3), "next_best": round(float(second), 3)}
        if c[k] < a.min_ncc or c[k] - second < 0.1:
            bad.append(f.stem)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = len(truth) - len(bad)
    print(f"Znaleziono pewnie: {ok}/{len(truth)} (korelacja >= {a.min_ncc} i wyraźnie lepsza od drugiego miejsca)")
    for name in bad:
        print("  niepewne:", name, truth[name])
    starts = [v["start"] for v in truth.values()]  # kolejność plików = kolejność skryptów = chronologia
    mono = sum(b >= a_ for a_, b in zip(starts, starts[1:])) / max(1, len(starts) - 1)
    print(f"Kontrola: {mono:.0%} kolejnych par editów leży w sesji w rosnącej kolejności (oczekiwane ~100%)")


if __name__ == "__main__":
    sys.exit(main())
