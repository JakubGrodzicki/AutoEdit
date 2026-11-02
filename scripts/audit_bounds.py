"""Kontrola gotowego projektu: czy wycinki zawierają dokładnie swoją kwestię i czy nic nie jest ucięte.

Użycie:  python scripts/audit_bounds.py <folder projektu>

Sprawdza wybór silnika (bez decyzji użytkownika) na prawdziwych plikach z eksportu do folderu tymczasowego:
Whisper na samym wycinku porównany z treścią kwestii (cudzy lub brakujący tekst) oraz krawędzie pliku, bo uciętej
głoski transkrypcja nie zdradzi: koniec wciąż w mowie i cyfrowa cisza zamiast dźwięku. Wypisuje tylko podejrzane kwestie.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rapidfuzz import fuzz  # noqa: E402

from autoedit import export, pipeline  # noqa: E402
from autoedit.normalize import normalize  # noqa: E402
from build_truth import load_edit16  # noqa: E402

MIN_RATIO = 85  # poniżej: Whisper słyszy na wycinku co innego niż treść kwestii
IN_SPEECH_DB = 15.0  # koniec pliku głośniejszy niż poziom kwestii minus tyle: wycinek kończy się w mowie
ZEROS_S = 0.05


def edges(path: Path) -> tuple[float, float, float]:
    """(cisza cyfrowa na początku, na końcu, głośność końcówki pliku względem poziomu kwestii w dB)."""
    a, sr = sf.read(path, dtype="float64", always_2d=True)
    a = a.mean(axis=1)
    nz = np.nonzero(np.abs(a) > 0)[0]
    if not len(nz):
        return len(a) / sr, len(a) / sr, -120.0
    real = a[nz[0]:nz[-1] + 1]
    hop = int(0.005 * sr)
    n = len(real) // hop
    db = 20 * np.log10(np.sqrt((real[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)
    return nz[0] / sr, (len(a) - 1 - nz[-1]) / sr, float(db[-6:].mean() - np.percentile(db, 95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    args = ap.parse_args()

    meta, results = pipeline.load_project(args.project)
    src = Path(meta["audio"])

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")

    counts = {s: sum(r.status == s for r in results) for s in ("auto", "przegląd", "brak")}
    missing = [r.line.script for r in results if r.status == "brak" and not r.line.is_main]
    print(f"Statusy: {counts}. Bez wyniku (NPC): {len(missing)} {missing[:8]}")
    n = low = in_speech = zeros_end = zeros_start = jumps = overlaps = 0
    prev: tuple[float, float] | None = None
    with tempfile.TemporaryDirectory() as tmp:
        for r in results:
            if r.chosen is None:
                continue
            n += 1
            s, e = export.bounds(r.chosen)
            flags = []
            if prev is not None:
                jumps += s < prev[0]
                if s < prev[1] - 0.1 and s >= prev[0]:
                    overlaps += 1
                    flags.append(f"nakładka z poprzednią {prev[1] - s:.2f} s")
            prev = (s, e)
            wav = Path(tmp) / "x.wav"
            export.cut(src, wav, s, e, meta.get("pad", 0.15))
            segs, _ = asr.transcribe(load_edit16(wav), language="pl", vad_filter=False, condition_on_previous_text=False)
            heard = normalize(" ".join(seg.text for seg in segs))
            ratio = fuzz.ratio(r.line.norm, heard)
            z0, z1, end_db = edges(wav)
            if ratio < MIN_RATIO:
                low += 1
                flags.append(f"słychać '{heard[:60]}' ({ratio:.0f}%)")
            if end_db > -IN_SPEECH_DB:
                in_speech += 1
                flags.append(f"koniec w mowie ({end_db:+.0f} dB)")
            if z1 >= ZEROS_S:
                zeros_end += 1
                flags.append(f"cyfrowa cisza na końcu {z1:.2f} s")
            if z0 >= ZEROS_S:
                zeros_start += 1
                flags.append(f"cyfrowa cisza na początku {z0:.2f} s")
            if flags:
                print(f"  {r.line.script:40s} {r.status[:4]} {s:8.2f}-{e:8.2f} | {'; '.join(flags)} | {r.line.norm[:50]}")
    print(f"Wycinków: {n}. Poniżej {MIN_RATIO}%: {low}. Koniec w mowie: {in_speech}. Cyfrowa cisza na końcu: {zeros_end}, "
          f"na początku: {zeros_start}. Skoki wstecz: {jumps}. Nakładki: {overlaps}.")


if __name__ == "__main__":
    main()
