"""Cięcie z ORYGINALNEGO pliku sesji (bez zmiany formatu) i raport XLSX."""
from __future__ import annotations

import re
from bisect import bisect_right
from pathlib import Path

import numpy as np
import openpyxl
import soundfile as sf

from .models import Candidate, LineResult, Word

_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(script: str) -> tuple[str, bool]:
    clean = _BAD.sub("_", script).strip(" .")
    return clean, clean != script


def bounds(c: Candidate) -> tuple[float, float]:
    return c.refined if c.refined else (c.start, c.end)


def neighbor_limits(words: list[Word], raw_start: float, raw_end: float) -> tuple[float, float]:
    """Koniec ostatniego słowa przed kwestią i początek pierwszego po niej (cokolwiek, także gadanie
    reżysera). Granice liczymy od surowego zakresu słów kandydata z Whispera, bo to jego własne słowa
    (po alignmencie granice kwestii przesuwają się nawet o 0,5 s i wtedy własne słowa wyglądałyby na sąsiadów)."""
    ends = [w.end for w in words]  # słowa Whispera są posortowane po czasie
    k = bisect_right(ends, raw_start + 0.02)
    prev_end = max((w.end for w in words[max(0, k - 3):k]), default=0.0)
    nxt = next((w.start for w in words[k:] if w.start >= raw_end - 0.02), float("inf"))
    return prev_end, nxt


def cut(src: Path, dst: Path, start: float, end: float, words: list[Word], pad: float = 0.15,
        fade: float = 0.005, raw: tuple[float, float] | None = None) -> dict:
    info = sf.info(src)
    sr = info.samplerate
    prev_end, nxt = neighbor_limits(words, *(raw or (start, end)))
    left = max(0.0, min(pad, start - prev_end - 0.02))
    right = max(0.0, min(pad, nxt - end - 0.02, info.duration - end))
    a = max(0, int(round((start - left) * sr)))
    b = min(info.frames, int(round((end + right) * sr)))
    data, _ = sf.read(src, start=a, stop=b, dtype="float64", always_2d=True)
    lead = int(round(pad * sr)) - int(round((start - a / sr) * sr))
    trail = int(round(pad * sr)) - int(round((b / sr - end) * sr))
    f = min(int(fade * sr), len(data) // 4)
    if f > 1:  # fade na prawdziwym audio PRZED dołożeniem zer, inaczej na styku zostaje twardy skok
        ramp = np.linspace(0, 1, f)[:, None]
        if lead > 0:
            data[:f] *= ramp
        if trail > 0:
            data[-f:] *= ramp[::-1]
    data = np.pad(data, ((max(0, lead), max(0, trail)), (0, 0)))  # brakujący padding = cisza
    subtype = info.subtype if info.subtype in sf.available_subtypes("WAV") else "PCM_24"
    sf.write(dst, data, sr, subtype=subtype)
    return {"duration": len(data) / sr, "pad_left": left, "pad_right": right}


def export_all(results: list[LineResult], src: Path, out_dir: Path, words: list[Word],
               statuses: tuple[str, ...] = ("auto",), pad: float = 0.15,
               overwrite: bool = False) -> dict[int, str]:
    """Zwraca {idx kwestii: komunikat}. Nie nadpisuje istniejących plików bez overwrite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: dict[int, str] = {}
    used: dict[str, int] = {}
    for r in results:
        if r.status not in statuses or r.chosen is None:
            continue
        name, changed = safe_name(r.line.script)
        if not name:
            notes[r.line.idx] = "pusta nazwa skryptu"
            continue
        if name in used:
            notes[r.line.idx] = f"duplikat nazwy skryptu (wiersz {used[name]}), pominięto"
            continue
        used[name] = r.line.idx
        dst = out_dir / f"{name}.wav"
        if dst.exists() and not overwrite:
            notes[r.line.idx] = "plik istnieje, pominięto"
            continue
        s, e = bounds(r.chosen)
        raw = (r.chosen.start, r.chosen.end) if r.chosen.origin == "text" else None
        cut(src, dst, s, e, words, pad, raw=raw)
        notes[r.line.idx] = "zmieniono znaki niedozwolone w nazwie" if changed else "ok"
    return notes


def write_report(results: list[LineResult], path: Path, notes: dict[int, str]) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Raport"
    ws.append(["Skrypt", "Postać", "Treść", "Status", "Podobieństwo %", "Start", "Koniec",
               "Powody", "Eksport", "Kandydaci (start-koniec, %)"])
    for r in results:
        c = r.chosen
        s, e = bounds(c) if c else (None, None)
        cand = "; ".join(f"{bounds(x)[0]:.1f}-{bounds(x)[1]:.1f} ({x.score:.0f}%)"
                         for x in r.candidates[:8])
        ws.append([r.line.script, r.line.character, r.line.text, r.status,
                   round(c.score, 1) if c else None,
                   round(s, 2) if s is not None else None, round(e, 2) if e is not None else None,
                   "; ".join(r.reasons), notes.get(r.line.idx, ""), cand])
    ws.freeze_panes = "A2"
    wb.save(path)


def export_selected(items: list[tuple[str, float, float, tuple[float, float] | None]], src: Path, out_dir: Path,
                    words: list[Word], pad: float = 0.15, overwrite: bool = False) -> dict[str, str]:
    """Eksport zatwierdzonych przez użytkownika wycinków: (nazwa skryptu, start, koniec, surowy zakres Whispera)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: dict[str, str] = {}
    used: set[str] = set()
    for script, s, e, raw in items:
        name, changed = safe_name(script)
        if not name or name in used:
            notes[script] = "pusta lub zduplikowana nazwa, pominięto"
            continue
        used.add(name)
        dst = out_dir / f"{name}.wav"
        if dst.exists() and not overwrite:
            notes[script] = "plik istnieje, pominięto"
            continue
        cut(src, dst, s, e, words, pad, raw=raw)
        notes[script] = "zmieniono znaki niedozwolone w nazwie" if changed else "ok"
    return notes
