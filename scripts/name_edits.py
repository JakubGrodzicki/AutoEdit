"""Nadaje pociętym kwestiom nazwy skryptów z Excela.

Pliki 01.wav, 02.wav... odpowiadają po kolei wierszom NPC (Czy MC = NIE). Oryginały zostają
nietknięte: kopie z nazwami trafiają do osobnego folderu. Przerywa, gdy liczby się nie zgadzają.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoedit.tables import load_lines  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("table", type=Path)
ap.add_argument("edits", type=Path, help="folder z 01.wav, 02.wav...")
ap.add_argument("-o", "--out", type=Path, help="folder wyjściowy (domyślnie '<edits> (nazwane)')")
a = ap.parse_args()

npc = [l for l in load_lines(a.table) if not l.is_main]
files = sorted((f for f in a.edits.glob("*.wav") if re.fullmatch(r"\d+", f.stem)), key=lambda f: int(f.stem))
nums = [int(f.stem) for f in files]
if nums != list(range(nums[0], nums[0] + len(nums))):
    sys.exit(f"Numeracja plików nie jest ciągła: {nums}")
if len(files) != len(npc):
    sys.exit(f"Plików: {len(files)}, wierszy NPC (Czy MC = NIE): {len(npc)}. Nie zgadza się, przerywam.")
if len({l.script for l in npc}) != len(npc):
    sys.exit("Nazwy skryptów w wierszach NPC nie są unikalne.")

out = a.out or a.edits.with_name(a.edits.name + " (nazwane)")
out.mkdir(parents=True, exist_ok=True)
for f, ln in zip(files, npc):
    shutil.copy2(f, out / f"{ln.script}.wav")
print(f"Skopiowano {len(files)} plików do: {out}")
print(f"  {files[0].name} -> {npc[0].script}.wav")
print(f"  {files[-1].name} -> {npc[-1].script}.wav")
