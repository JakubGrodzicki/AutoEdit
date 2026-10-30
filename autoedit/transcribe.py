"""Transkrypcja z timestampami słów + cache JSON (ASR nie biegnie drugi raz)."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .models import Line, Word
from .normalize import normalize

SR = 16000


def file_key(path: Path) -> str:
    """Szybki odcisk: rozmiar + początek i koniec pliku (bez czytania 4 GB)."""
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    chunk = 1 << 16
    with open(path, "rb") as f:  # 16 próbek rozłożonych po pliku (początek/koniec sesji bywa cichy)
        for k in range(16):
            f.seek(max(0, (size - chunk) * k // 15))
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def prepare_audio(path: str | Path, cache_dir: str | Path) -> Path:
    """Kopia 16 kHz mono PCM16 do analizy (tylko do analizy, nie do eksportu)."""
    path, cache_dir = Path(path), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{file_key(path)}.16k.wav"
    if not out.exists():
        tmp = out.with_suffix(".tmp.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
             "-c:a", "pcm_s16le", str(tmp)], check=True)
        tmp.replace(out)
    return out


def load_audio16(wav16: Path) -> np.ndarray:
    audio, sr = sf.read(wav16, dtype="float32")
    assert sr == SR
    return audio


def build_hint(lines: list[Line], limit: int = 40) -> str:
    """Rzadkie nazwy własne z Excela (słowa z wielkiej litery w środku zdania)."""
    seen: dict[str, None] = {}
    for ln in lines:
        for m in re.finditer(r"(?<![.!?…]\s)(?<!^)\b[A-ZĄĆĘŁŃÓŚŹŻ][\w'’-]{3,}", ln.text):
            seen.setdefault(m.group(), None)
    return " ".join(list(seen)[:limit])


CHUNK_S = 180.0  # transkrypcja częściami; jedno przejście przez 85 min gubiło całe fragmenty mowy


def chunk_bounds(audio: np.ndarray, chunk_s: float = CHUNK_S, search_s: float = 8.0) -> list[tuple[int, int]]:
    """Podział na części cięty w najcichszym miejscu w pobliżu celu, żeby nie przeciąć słowa."""
    hop = SR // 10
    n = len(audio) // hop
    rms = np.sqrt((audio[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    cuts, pos = [0], 0
    while len(audio) - pos > chunk_s * 1.5 * SR:
        lo = int((pos + (chunk_s - search_s) * SR) / hop)
        hi = int((pos + (chunk_s + search_s) * SR) / hop)
        pos = (lo + int(np.argmin(rms[lo:hi]))) * hop
        cuts.append(pos)
    cuts.append(len(audio))
    return list(zip(cuts[:-1], cuts[1:]))


def find_holes(audio: np.ndarray, words: list[Word], block_s: float = 120.0, min_len: float = 0.4,
               max_covered: float = 0.1) -> list[tuple[float, float]]:
    """Fragmenty z głosem, w których Whisper nie zapisał (prawie) niczego."""
    from .nonverbal import voiced_segments

    total = len(audio) / SR
    starts = np.array([w.start for w in words])
    ends = np.array([w.end for w in words])
    holes = []
    t = 0.0
    while t < total:
        for s, e in voiced_segments(audio, t, min(total, t + block_s), merge_gap=0.5, min_len=min_len, max_len=60.0):
            covered = np.clip(np.minimum(ends, e) - np.maximum(starts, s), 0, None).sum()
            if covered < max_covered * (e - s):
                holes.append((s, e))
        t += block_s
    return holes


def fill_holes(asr, audio: np.ndarray, words: list[Word], progress=None) -> list[Word]:
    """Druga, celowana transkrypcja miejsc pominiętych w głównym przebiegu (bez VAD, krótkie okna)."""
    holes = find_holes(audio, words)
    merged: list[list[float]] = []
    for s, e in holes:  # sąsiednie dziury razem, do 30 s, żeby Whisper miał kontekst
        if merged and s - merged[-1][1] < 1.5 and e - merged[-1][0] <= 30.0:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    added: list[Word] = []
    for k, (s, e) in enumerate(merged):
        a, b = int(max(0.0, s - 0.3) * SR), int(min(len(audio) / SR, e + 0.3) * SR)
        segments, _ = asr.transcribe(audio[a:b], language="pl", word_timestamps=True, vad_filter=False,
                                     condition_on_previous_text=False)
        for seg in segments:
            if seg.no_speech_prob > 0.6 or seg.avg_logprob < -1.2:  # szum i halucynacje
                continue
            for w in seg.words or []:
                added.append(Word(text=w.word.strip(), start=a / SR + float(w.start), end=a / SR + float(w.end)))
        if progress:
            progress(e, len(audio) / SR)
    return sorted(words + added, key=lambda w: w.start)


def _cache_path(cache_dir: Path, wav16: Path, model: str, hint: str | None) -> Path:
    tag = hashlib.sha1(f"{model}|{hint or ''}|chunks{int(CHUNK_S)}+fill".encode()).hexdigest()[:8]
    return cache_dir / f"{wav16.name.split('.')[0]}.{tag}.words.json"


def set_norms(words: list[Word]) -> list[Word]:
    for w in words:
        w.norm = normalize(w.text)
    return words


def transcribe(wav16: Path, cache_dir: str | Path, model: str = "large-v3", hint: str | None = None,
               progress: Callable[[float, float], None] | None = None) -> list[Word]:
    cache_dir = Path(cache_dir)
    cp = _cache_path(cache_dir, wav16, model, hint)
    if cp.exists():
        return set_norms([Word(**d) for d in json.loads(cp.read_text(encoding="utf-8"))])

    from faster_whisper import WhisperModel

    asr = WhisperModel(model, device="cuda", compute_type="float16")
    audio = load_audio16(wav16)
    total = len(audio) / SR
    words: list[Word] = []
    for a, b in chunk_bounds(audio):
        off = a / SR
        segments, _ = asr.transcribe(
            audio[a:b], language="pl", word_timestamps=True, vad_filter=True,
            condition_on_previous_text=False, hotwords=hint or None)
        for seg in segments:
            for w in seg.words or []:
                words.append(Word(text=w.word.strip(), start=off + float(w.start), end=off + float(w.end)))
        if progress:
            progress(b / SR, total)
    words = fill_holes(asr, audio, words, progress)
    cp.write_text(json.dumps([{"text": w.text, "start": w.start, "end": w.end} for w in words],
                             ensure_ascii=False), encoding="utf-8")
    return set_norms(words)
