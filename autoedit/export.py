"""Cięcie z ORYGINALNEGO pliku sesji (bez zmiany formatu) i raport XLSX."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl
import soundfile as sf

from .models import Candidate, LineResult

_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
FRAME_S = 0.005
FOREIGN_DROP_DB = 15.0  # inna wypowiedź: głośność wraca do poziomu kwestii minus tyle
FOREIGN_QUIET_S = 0.06  # i to po co najmniej takiej ciszy; krótszy dołek to własna głoska lub wybrzmienie


def safe_name(script: str) -> tuple[str, bool]:
    clean = _BAD.sub("_", script).strip(" .")
    return clean, clean != script


def bounds(c: Candidate) -> tuple[float, float]:
    return c.refined if c.refined else (c.start, c.end)


def _levels(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    return 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)


def _free_margin(outward: np.ndarray, thr: float, hop: int, quiet_frames: int) -> int:
    """Ile próbek marginesu (licząc od granicy cięcia na zewnątrz) można wziąć: własne wybrzmienie jest
    dozwolone, a powrót głośności po ciszy to już inna wypowiedź."""
    quiet = 0
    for k, level in enumerate(_levels(outward, hop)):
        if level <= thr:
            quiet += 1
        elif quiet >= quiet_frames:
            return max(0, k - 4) * hop  # 20 ms odstępu od cudzej mowy
        else:
            quiet = 0
    return len(outward)


def cut(src: Path, dst: Path, start: float, end: float, pad: float = 0.15, fade: float = 0.005) -> dict:
    """Padding pochodzi z prawdziwego dźwięku; zerami uzupełniamy tylko to, co przypadłoby na inną wypowiedź lub poza plik."""
    info = sf.info(src)
    sr = info.samplerate
    a = max(0, int(round((start - pad) * sr)))
    b = min(info.frames, int(round((end + pad) * sr)))
    data, _ = sf.read(src, start=a, stop=b, dtype="float64", always_2d=True)
    i0, i1 = int(round(start * sr)) - a, int(round(end * sr)) - a
    hop = max(1, int(FRAME_S * sr))
    quiet_frames = int(round(FOREIGN_QUIET_S / FRAME_S))
    mono = data.mean(axis=1)
    own = _levels(mono[i0:i1], hop)
    thr = (np.percentile(own, 95) if len(own) else -60.0) - FOREIGN_DROP_DB
    left = _free_margin(mono[:i0][::-1], thr, hop, quiet_frames)
    right = _free_margin(mono[i1:], thr, hop, quiet_frames)
    data = data[i0 - left:i1 + right]
    lead = int(round(pad * sr)) - left
    trail = int(round(pad * sr)) - right
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
    return {"duration": len(data) / sr, "pad_left": left / sr, "pad_right": right / sr}


def export_all(results: list[LineResult], src: Path, out_dir: Path, statuses: tuple[str, ...] = ("auto",),
               pad: float = 0.15, overwrite: bool = False) -> dict[int, str]:
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
        cut(src, dst, s, e, pad)
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


def export_selected(items: list[tuple[str, float, float]], src: Path, out_dir: Path, pad: float = 0.15,
                    overwrite: bool = False) -> dict[str, str]:
    """Eksport zatwierdzonych przez użytkownika wycinków: (nazwa skryptu, start, koniec)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: dict[str, str] = {}
    used: set[str] = set()
    for script, s, e in items:
        name, changed = safe_name(script)
        if not name or name in used:
            notes[script] = "pusta lub zduplikowana nazwa, pominięto"
            continue
        used.add(name)
        dst = out_dir / f"{name}.wav"
        if dst.exists() and not overwrite:
            notes[script] = "plik istnieje, pominięto"
            continue
        cut(src, dst, s, e, pad)
        notes[script] = "zmieniono znaki niedozwolone w nazwie" if changed else "ok"
    return notes
