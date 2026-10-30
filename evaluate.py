"""Pomiar jakości silnika względem gotowych editów.

Użycie:  python evaluate.py sesja.wav edity_nazwane/ \
                            output/projekt/project.json kwestie.xlsx

Edity są obrobione, więc szukanie ich w całej sesji daje przypadkowe trafienia (cisza, szum tła).
Zamiast tego, dla każdej kwestii, porównujemy edit z KAŻDYM takiem znalezionym przez silnik (wąskie okno
wokół kandydata, korelacja cech log-mel) i pytamy: czy silnik wybrał ten take, który wybrał montażysta?
Gdy żaden take nie przypomina edita (korelacja < MIN), kwestia jest 'nieoceniana', a nie 'błędna'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoedit.tables import load_lines
from autoedit.transcribe import SR, load_audio16, prepare_audio
from build_truth import HOP, SessionIndex, load_edit16, logmel

MIN = 0.45  # poniżej tej korelacji nie twierdzimy, że edit pochodzi z danego take'a
FRAME = HOP / SR


def bounds(c: dict) -> tuple[float, float]:
    return tuple(c["refined"]) if c.get("refined") else (c["start"], c["end"])


def best_in_window(score: np.ndarray, lo_s: float, hi_s: float) -> tuple[float, float]:
    """(korelacja, czas startu edita) - maksimum w oknie czasu [lo_s, hi_s] początku edita."""
    a, b = max(0, int(lo_s / FRAME)), min(len(score), int(hi_s / FRAME) + 1)
    if b <= a:
        return -1.0, 0.0
    k = a + int(np.argmax(score[a:b]))
    return float(score[k]), k * FRAME


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("edits", type=Path, help="folder z editami nazwanymi nazwą skryptu")
    ap.add_argument("state", type=Path)
    ap.add_argument("table", type=Path)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    a = ap.parse_args()

    state = {r["script"]: r for r in json.loads(a.state.read_text(encoding="utf-8"))}
    lines = load_lines(a.table)
    npc, mc = [l for l in lines if not l.is_main], [l for l in lines if l.is_main]
    index = SessionIndex(logmel(load_audio16(prepare_audio(a.session, a.cache))))

    res = {}  # script -> dict
    for l in npc:
        f = a.edits / f"{l.script}.wav"
        r = state[l.script]
        cands = [c for c in r["candidates"] if c["origin"] == "text" and c["score"] >= 70]
        if not f.exists() or not cands:
            res[l.script] = {"verdict": "brak kandydatów", "status": r["status"]}
            continue
        e = load_edit16(f)
        sc = index.score(logmel(e)[:, :-1])
        scored = []
        for c in cands:
            s, e_ = bounds(c)
            corr, t = best_in_window(sc, s - 1.5, s + 1.0)
            scored.append((corr, t, c, s, e_))
        corr, t, best, s, e_ = max(scored, key=lambda x: x[0])
        ch = r["chosen"]
        if corr < MIN:
            verdict = "nieoceniana"
        else:
            same = ch is not None and abs(bounds(ch)[0] - s) < 0.5
            verdict = "ten sam take" if same else "inny take"
        res[l.script] = {"verdict": verdict, "status": r["status"], "corr": corr, "n": len(cands),
                         "lead": s - t, "trail": (t + len(e) / SR) - e_, "reasons": r["reasons"]}

    def n(status, verdict):
        return sum(1 for v in res.values() if v["status"] == status and v["verdict"] == verdict)

    print(f"Kwestie NPC: {len(npc)}   (korelacja >= {MIN} traktowana jako dowód, że edit pochodzi z danego take'a)")
    for status in ("auto", "przegląd", "brak"):
        tot = sum(1 for v in res.values() if v["status"] == status)
        print(f"  {status:9s} {tot:3d} | ten sam take: {n(status, 'ten sam take'):3d} | INNY take: "
              f"{n(status, 'inny take'):3d} | nieoceniane: {n(status, 'nieoceniana'):3d} | "
              f"bez kandydatów: {n(status, 'brak kandydatów'):3d}")

    lead = [v["lead"] for v in res.values() if v["verdict"] == "ten sam take" and v["status"] == "auto"]
    trail = [v["trail"] for v in res.values() if v["verdict"] == "ten sam take" and v["status"] == "auto"]
    if lead:
        print("\nMargines editora wokół naszych granic (auto, ten sam take), ms:")
        print(f"  przód: mediana {np.median(lead) * 1000:.0f}, zakres {min(lead) * 1000:.0f}..{max(lead) * 1000:.0f}")
        print(f"  tył:   mediana {np.median(trail) * 1000:.0f}, zakres {min(trail) * 1000:.0f}..{max(trail) * 1000:.0f}")
        print("  (>0: edit ma więcej audio niż nasze cięcie; <0: nasze cięcie wystaje poza edit)")

    print("\nGranice wystające poza edit (>120 ms), ten sam take:")
    for k, v in res.items():
        if v["verdict"] == "ten sam take" and (v["lead"] < -0.12 or v["trail"] < -0.12):
            ch = state[k]["chosen"]
            print(f"  {k:42s} {v['status']:8s} przód {v['lead'] * 1000:6.0f} ms, tył {v['trail'] * 1000:6.0f} ms | "
                  f"Whisper {ch['start']:.2f}-{ch['end']:.2f} -> alignment {bounds(ch)[0]:.2f}-{bounds(ch)[1]:.2f} "
                  f"gap {ch.get('audio_gap')} tail {ch.get('audio_tail_gap')}")

    print("\nauto, ale montażysta wybrał INNY take:")
    for k, v in res.items():
        if v["status"] == "auto" and v["verdict"] == "inny take":
            print(f"  {k:44s} korelacja {v['corr']:.2f}, podejść {v['n']}")

    print("\nWiersze MC (nie powinny występować w nagraniu):")
    for status in ("auto", "przegląd", "brak"):
        print(f"  {status}: {sum(1 for l in mc if state[l.script]['status'] == status)}")


if __name__ == "__main__":
    main()
