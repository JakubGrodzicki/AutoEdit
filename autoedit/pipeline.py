"""Cały proces silnika: tabele -> transkrypcja -> dopasowanie -> alignment -> eksport -> raport."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import export, nonverbal, tables, transcribe
from .matching import MatchConfig, best_path, finalize, find_candidates
from .models import Candidate, LineResult, Word


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


def load_project(out_dir: Path) -> tuple[dict, list[LineResult], list[Word]]:
    """Wczytuje wynik wcześniejszego przebiegu (meta.json + project.json + cache słów)."""
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    lines = tables.load_all([Path(t) for t in meta["tables"]])
    results = []
    for d in json.loads((out_dir / "project.json").read_text(encoding="utf-8")):
        r = LineResult(lines[d["idx"]], status=d["status"], reasons=d["reasons"],
                       candidates=[_cand(c) for c in d["candidates"]])
        if d["chosen"]:
            ch = _cand(d["chosen"])
            r.chosen = next((c for c in r.candidates if (c.start, c.end, c.origin) == (ch.start, ch.end, ch.origin)), ch)
        results.append(r)
    words = transcribe.set_norms([Word(**w) for w in json.loads(Path(meta["words"]).read_text(encoding="utf-8"))])
    return meta, results, words


def run(audio: Path, table_files: list[Path], out_dir: Path, cache_dir: Path, cfg: MatchConfig | None = None,
        use_hint: bool = False, do_align: bool = True, do_export: bool = True, pad: float = 0.15,
        overwrite: bool = False, log: Callable[[str], None] = print) -> list[LineResult]:
    cfg = cfg or MatchConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = tables.load_all(table_files)
    log(f"Kwestie: {len(lines)} ({sum(l.kind != 'speech' for l in lines)} niewerbalnych/głosowych)")
    for name, idxs in tables.duplicate_scripts(lines).items():
        log(f"UWAGA: nazwa skryptu '{name}' powtarza się w wierszach {idxs}")

    wav16 = transcribe.prepare_audio(audio, cache_dir)
    hint = transcribe.build_hint(lines) if use_hint else None
    log("Transkrypcja (cache jeśli już była)...")
    words = transcribe.transcribe(wav16, cache_dir, hint=hint,
                                  progress=lambda t, total: log(f"  {t / 60:.1f}/{total / 60:.1f} min"))
    log(f"Słów: {len(words)}")
    (out_dir / "meta.json").write_text(json.dumps({
        "audio": str(audio.resolve()), "tables": [str(Path(t).resolve()) for t in table_files],
        "words": str(transcribe._cache_path(Path(cache_dir), wav16, "large-v3", hint).resolve()), "pad": pad,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    log("Dopasowanie...")
    stream, cands = find_candidates(lines, words, cfg)
    best_path(cands, cfg)
    audio16 = None
    if do_align:
        from .align import Aligner, refine
        audio16 = transcribe.load_audio16(wav16)
        aligner = Aligner()
        by_idx = {l.idx: l for l in lines}
        log("Forced alignment kandydatów na ścieżce...")
        for c in cands:
            if c.on_path:
                refine(aligner, audio16, by_idx[c.line_idx], c)
    results = finalize(lines, cands, cfg)
    if audio16 is not None:
        nonverbal.propose(results, audio16)

    notes: dict[int, str] = {}
    if do_export:
        notes = export.export_all(results, audio, out_dir / "kwestie", words, pad=pad, overwrite=overwrite)
    export.write_report(results, out_dir / "raport.xlsx", notes)
    save_state(results, out_dir / "project.json")
    counts = {s: sum(r.status == s for r in results) for s in ("auto", "przegląd", "brak")}
    log(f"Gotowe: {counts}")
    return results
