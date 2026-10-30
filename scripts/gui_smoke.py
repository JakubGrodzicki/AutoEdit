"""Smoke test GUI: lista, okno przeglądu, sortowanie kandydatów, przeciąganie granic, decyzje, zrzuty, eksport.
Użycie: python scripts/gui_smoke.py <folder wyniku> <folder na zrzuty> <folder eksportu próbnego>"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from autoedit.gui.app import MainWindow, ReviewDialog  # noqa: E402

import shutil, tempfile  # noqa: E402

src = Path(sys.argv[1])
out = Path(tempfile.mkdtemp()) / src.name  # kopia: test nie może dotykać decisions.json użytkownika
shutil.copytree(src, out, ignore=shutil.ignore_patterns("decisions.json", "kwestie"))
shots, exp = Path(sys.argv[2]), Path(sys.argv[3])
shots.mkdir(parents=True, exist_ok=True)
app = QApplication([])
w = MainWindow()
w.show()
w.load(out)
w.f_who.setCurrentIndex(1)  # tylko NPC
print("lista:", w.table.rowCount(), "|", w.summary.text().strip())
w.grab().save(str(shots / "lista.png"))

w.f_state.setCurrentIndex(1)  # do przeglądu
row = next(k for k, i in enumerate(w.shown) if len(w.results[i].candidates) >= 3)
w.table.selectRow(row)
dlg = ReviewDialog(w, row)  # nie-modalnie, żeby test mógł dalej działać
dlg.show()
app.processEvents()
r = dlg.r
scores = [r.candidates[i].score for i in dlg.order]
print("kwestia:", r.line.script, "| posortowane malejąco:", scores == sorted(scores, reverse=True), scores)
print("zaznaczony wiersz kandydatów:", [i.row() for i in dlg.cands.selectionModel().selectedRows()])

# przeciąganie czerwonego znacznika startu
s0, e0 = dlg.wave.start, dlg.wave.end
x = dlg.wave._x(s0)
QTest.mousePress(dlg.wave, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(x), 40))
QTest.mouseMove(dlg.wave, QPoint(int(x) + 60, 40))
QTest.mouseRelease(dlg.wave, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(x) + 60, 40))
print(f"start po przeciągnięciu: {s0:.3f} -> {dlg.wave.start:.3f} (pole: {dlg.sp_start.value():.3f}), koniec bez zmian: {dlg.wave.end == e0}")
dlg.grab().save(str(shots / "przeglad.png"))
print("decyzja zapamiętana z nowym startem:", abs(w.decisions[r.line.script]["start"] - dlg.wave.start) < 1e-6)

dlg.decide("accepted")
print("po zatwierdzeniu:", w.state_of(r), "| otwarta kolejna:", dlg.r.line.script != r.line.script)
dlg.close()

import autoedit.export as ex  # noqa: E402

items = []
for rr in w.results:
    if w.state_of(rr) == "accepted" and not rr.line.is_main:
        sel = w.selection_of(rr)
        c = rr.candidates[sel[0]]
        items.append((rr.line.script, sel[1], sel[2], (c.start, c.end)))
print("zatwierdzonych do eksportu:", len(items))
notes = ex.export_selected(items[:5], Path(w.meta["audio"]), exp, w.words, 0.15, True)
print("eksport próbny:", list(notes.values()))
