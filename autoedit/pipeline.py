"""Cały proces silnika: tabele -> transkrypcja -> dopasowanie -> alignment -> eksport -> raport."""
from __future__ import annotations

import ctypes
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import export, nonverbal, tables, transcribe
from .matching import MatchConfig, best_path, finalize, find_candidates
from .models import Candidate, Line, LineResult

WORK_DIR = ".autoedit"  # pliki robocze projektu; w samym folderze wynikowym widać tylko WAV-y
WORK_FILES = ("meta.json", "project.json", "decisions.json", "raport.xlsx")


def work_dir(out_dir: Path) -> Path:
    """Folder plików roboczych projektu. Starsze projekty mają je wprost w folderze wynikowym."""
    hidden = out_dir / WORK_DIR
    return out_dir if (out_dir / "meta.json").exists() and not (hidden / "meta.json").exists() else hidden


def has_project(out_dir: Path) -> bool:
    work = work_dir(out_dir)
    return (work / "meta.json").exists() and (work / "project.json").exists()


def prepare_work_dir(out_dir: Path) -> Path:
    """Tworzy ukryty folder roboczy i przenosi do niego pliki starszego projektu z folderu wynikowego."""
    work = out_dir / WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        ctypes.windll.kernel32.SetFileAttributesW(str(work), 0x02)  # FILE_ATTRIBUTE_HIDDEN, sama kropka nic nie daje
    for name in WORK_FILES:
        if (out_dir / name).exists():
            (out_dir / name).replace(work / name)
    return work


def archive_decisions(work: Path) -> None:
    """Decyzje wskazują kandydatów po numerze i trzymają stare granice, więc nowego przebiegu już nie dotyczą."""
    if (work / "decisions.json").exists():
        (work / "decisions.json").replace(work / "decisions.old.json")


def save_state(results: list[LineResult], path: Path) -> None:
    data = [{"idx": r.line.idx, "script": r.line.script, "status": r.status, "reasons": r.reasons,
             "chosen": asdict(r.chosen) if r.chosen else None,
             "candidates": [asdict(c) for c in r.candidates]} for r in results]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _cand(d: dict) -> Candidate:
    d = dict(d)
    if d.get("refined"):
        d["refined"] = tuple(d["refined"])
    return Candidate(**d)


def load_project(out_dir: Path) -> tuple[dict, list[LineResult]]:
    """Wczytuje wynik wcześniejszego przebiegu (meta.json + project.json). Wiersze arkusza szukamy po nazwie skryptu,
    bo numeracja zależy od tego, które wiersze są pomijane; tych, których już nie ma w arkuszu, nie pokazujemy."""
    work = work_dir(out_dir)
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    by_script: dict[str, list[Line]] = {}
    for ln in tables.select_role(tables.load_all([Path(t) for t in meta["tables"]]), meta.get("who")):
        by_script.setdefault(ln.script, []).append(ln)
    results = []
    for d in json.loads((work / "project.json").read_text(encoding="utf-8")):
        same = by_script.get(d["script"])
        if not same:
            continue
        r = LineResult(same.pop(0), status=d["status"], reasons=d["reasons"],
                       candidates=[_cand(c) for c in d["candidates"]])
        if d["chosen"]:
            ch = _cand(d["chosen"])
            r.chosen = next((c for c in r.candidates if (c.start, c.end, c.origin) == (ch.start, ch.end, ch.origin)), ch)
        results.append(r)
    return meta, results


def run(audio: Path, table_files: list[Path], out_dir: Path, cache_dir: Path, cfg: MatchConfig | None = None,
        use_hint: bool = False, do_align: bool = True, do_export: bool = True, pad: float = 0.15,
        overwrite: bool = False, who: str | None = None, log: Callable[[str], None] = print) -> list[LineResult]:
    """`who`: czyje kwestie są w nagraniu, "npc" albo "main" (Czy MC = TAK); None = wszystkie z arkusza."""
    cfg = cfg or MatchConfig()
    work = prepare_work_dir(out_dir)

    lines = tables.load_all(table_files)
    if who is not None:
        try:
            chosen = tables.select_role(lines, who)
        except ValueError as e:
            log(f"BŁĄD: {e}")
            raise SystemExit(1) from e
        if who == "npc" and not any(l.is_main for l in lines):
            log("Arkusz nie oznacza kwestii MC, biorę wszystkie kwestie.")
        else:
            other = tables.ROLE_LABEL["main" if who == "npc" else "npc"]
            log(f"Nagranie aktora {tables.ROLE_LABEL[who]}: pominięto {len(lines) - len(chosen)} kwestii {other}")
        lines = chosen
    log(f"Kwestie: {len(lines)} ({sum(l.kind != 'speech' for l in lines)} niewerbalnych/głosowych)")
    for name, idxs in tables.duplicate_scripts(lines).items():
        log(f"UWAGA: nazwa skryptu '{name}' powtarza się w wierszach {idxs}")

    wav16 = transcribe.prepare_audio(audio, cache_dir)
    hint = transcribe.build_hint(lines) if use_hint else None
    log("Transkrypcja (cache jeśli już była)...")
    words = transcribe.transcribe(wav16, cache_dir, hint=hint,
                                  progress=lambda t, total: log(f"  {t / 60:.1f}/{total / 60:.1f} min"))
    log(f"Słów: {len(words)}")
    (work / "meta.json").write_text(json.dumps({
        "audio": str(audio.resolve()), "tables": [str(Path(t).resolve()) for t in table_files], "pad": pad, "who": who,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    log("Dopasowanie...")
    stream, cands = find_candidates(lines, words, cfg)
    best_path(cands, cfg)
    audio16 = None
    if do_align:
        from .align import Aligner, refine_all
        audio16 = transcribe.load_audio16(wav16)
        log("Forced alignment kandydatów...")
        refine_all(Aligner(), audio16, lines, cands, cfg.review_score)
    results = finalize(lines, cands, cfg)
    if audio16 is not None:
        nonverbal.propose(results, audio16)

    notes: dict[int, str] = {}
    if do_export:
        notes = export.export_all(results, audio, out_dir, pad=pad, overwrite=overwrite)
    export.write_report(results, work / "raport.xlsx", notes)
    save_state(results, work / "project.json")
    archive_decisions(work)
    counts = {s: sum(r.status == s for r in results) for s in ("auto", "przegląd", "brak")}
    log(f"Gotowe: {counts}")
    return results
