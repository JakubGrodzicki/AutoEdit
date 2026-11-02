"""Wczytanie kwestii z xlsx/csv."""
from __future__ import annotations

import csv
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


def duplicate_scripts(lines: list[Line]) -> dict[str, list[int]]:
    seen: dict[str, list[int]] = {}
    for ln in lines:
        seen.setdefault(ln.script, []).append(ln.idx)
    return {k: v for k, v in seen.items() if len(v) > 1}
