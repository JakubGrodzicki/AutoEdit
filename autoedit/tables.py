"""Wczytanie kwestii z xlsx/csv."""
from __future__ import annotations

import csv
import re
from dataclasses import replace
from pathlib import Path

import openpyxl

from .models import Line
from .normalize import classify, normalize

# Kolejność ma znaczenie: "Czy ta postać jest głównym bohaterem" zawiera "postać",
# więc kolumna "główny" musi zostać przypisana przed "postać".
_FIELDS = [
    ("is_main", ("główn", "glown", "main", "protagon", "bohater", "mc")),
    ("text", ("treść", "tresc", "kwest", "text", "tekst", "dialog")),
    ("script", ("skrypt", "script", "nazwa pliku", "file", "id")),
    ("character", ("postac", "postać", "character", "rola", "role")),
]
_TRUE = {"tak", "t", "yes", "y", "true", "1", "x", "główny", "glowny", "main"}


def detect_columns(header: list[str]) -> dict[str, int] | None:
    found: dict[str, int] = {}
    for field, keys in _FIELDS:
        for i, cell in enumerate(header):
            if i in found.values():
                continue
            if any(k in cell.casefold() for k in keys):
                found[field] = i
                break
    return found if {"script", "text"} <= found.keys() else None


def _rows(path: Path) -> list[list[str]]:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = [["" if c is None else str(c).strip() for c in r] for r in ws.iter_rows(values_only=True)]
        wb.close()
        return rows
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1250")
    dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    return [[c.strip() for c in r] for r in csv.reader(text.splitlines(), dialect)]


def load_lines(path: str | Path, columns: dict[str, int] | None = None, start_idx: int = 0) -> list[Line]:
    """Kolumny: wykrywane po nagłówku, a bez nagłówka pozycyjnie
    (skrypt | postać | główny | treść). `columns` pozwala je wskazać ręcznie."""
    path = Path(path)
    rows = [r for r in _rows(path) if any(r)]
    if not rows:
        return []
    cols = columns or detect_columns(rows[0])
    if cols is not None and columns is None:
        rows = rows[1:]
    elif cols is None:
        cols = {"script": 0, "character": 1, "is_main": 2, "text": 3}
    lines = []
    for r in rows:
        def cell(f: str) -> str:
            i = cols.get(f)
            return r[i] if i is not None and i < len(r) else ""
        script, text = cell("script"), cell("text")
        if not script or text.casefold() == "line":  # nagłówek sekcji: "TYTUŁ | NPC CODE NAME | MAIN HERO | LINE"
            continue
        lines.append(Line(
            idx=start_idx + len(lines), script=script, character=cell("character"),
            is_main=cell("is_main").casefold() in _TRUE, text=text, source=path.name,
            kind=classify(text), norm=normalize(text),
        ))
    return lines


def load_all(paths: list[str | Path]) -> list[Line]:
    lines: list[Line] = []
    for p in paths:
        lines += load_lines(p, start_idx=len(lines))
    return lines


def _safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip().rstrip(".") or "arkusz"


def split_workbook(src: str | Path, out_dir: str | Path) -> list[Path]:
    """Rozbija skoroszyt na osobne pliki .xlsx, po jednym na arkusz (z zachowaniem formatowania)."""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.load_workbook(src, read_only=True)
    names = wb.sheetnames
    wb.close()
    used: set[str] = set()
    out: list[Path] = []
    for name in names:
        # Świeży odczyt dla każdego arkusza, żeby zachować szerokości kolumn, scalone komórki itp.
        wb = openpyxl.load_workbook(src)
        for other in wb.sheetnames:
            if other != name:
                del wb[other]
        base = _safe_name(name)
        fname, i = base, 2
        while fname.casefold() in used:  # nazwy plików w Windows nie rozróżniają wielkości liter
            fname = f"{base}_{i}"
            i += 1
        used.add(fname.casefold())
        target = out_dir / f"{fname}.xlsx"
        wb.save(target)
        wb.close()
        out.append(target)
    return out


ROLE_LABEL ={"npc": "NPC", "main": "MC"}


def select_role(lines: list[Line], who: str | None) -> list[Line]:
    """Kwestie jednego aktora: "npc" (Czy MC = NIE) albo "main" (Czy MC = TAK); None = wszystkie.
    Numeracja od nowa, bo premia ścieżki za kolejną kwestię liczy się po idx + 1."""
    if who is None:
        return lines
    keep = [ln for ln in lines if ln.is_main == (who == "main")]
    if not keep:
        raise ValueError("Arkusz nie ma kwestii oznaczonych jako MC (kolumna Czy MC = TAK)." if who == "main"
                         else "Wszystkie kwestie arkusza są oznaczone jako MC (kolumna Czy MC = TAK).")
    return [replace(ln, idx=i) for i, ln in enumerate(keep)]


def duplicate_scripts(lines: list[Line]) -> dict[str, list[int]]:
    seen: dict[str, list[int]] = {}
    for ln in lines:
        seen.setdefault(ln.script, []).append(ln.idx)
    return {k: v for k, v in seen.items() if len(v) > 1}
