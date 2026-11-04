from __future__ import annotations

import argparse
from pathlib import Path

from . import pipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="autoedit", description="Automatyczne cięcie sesji dubbingowych")
    ap.add_argument("audio", type=Path, help="plik sesji (oryginał)")
    ap.add_argument("tables", type=Path, nargs="+", help="xlsx/csv z kwestiami (można kilka)")
    ap.add_argument("-o", "--out", type=Path, default=Path("output"))
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--hint", action="store_true", help="podpowiedź nazw własnych z Excela dla Whispera")
    ap.add_argument("--no-align", action="store_true", help="pomiń forced alignment (bez weryfikacji audio)")
    ap.add_argument("--no-export", action="store_true", help="tylko raport, bez cięcia")
    ap.add_argument("--pad", type=float, default=0.15, help="padding w sekundach")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--who", choices=("npc", "main"),
                    help="czyje kwestie są w nagraniu: npc (Czy MC = NIE) albo main (Czy MC = TAK); domyślnie wszystkie")
    a = ap.parse_args()
    pipeline.run(a.audio, a.tables, a.out, a.cache, use_hint=a.hint, do_align=not a.no_align,
                 do_export=not a.no_export, pad=a.pad, overwrite=a.overwrite, who=a.who)


if __name__ == "__main__":
    main()
