"""Smoke test GUI: lista, okno przeglądu, sortowanie kandydatów, przeciąganie granic, decyzje, zrzuty, eksport.
Użycie: python scripts/gui_smoke.py <folder wyniku> <folder na zrzuty>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from autoedit import pipeline  # noqa: E402
from autoedit.gui.app import MainWindow, ReviewDialog  # noqa: E402

import shutil, tempfile  # noqa: E402

src = Path(sys.argv[1])
out = Path(tempfile.mkdtemp()) / src.name  # kopia: test nie może dotykać decisions.json ani plików użytkownika
shutil.copytree(src, out, ignore=shutil.ignore_patterns("decisions.json", "kwestie", "*.wav"))
shots = Path(sys.argv[2])
shots.mkdir(parents=True, exist_ok=True)
app = QApplication([])
w = MainWindow()
w.show()
w.load(out)
w.f_who.setCurrentIndex(1)  # tylko NPC
print("lista:", w.table.rowCount(), "|", w.summary.text().strip())
w.grab().save(str(shots / "lista.png"))

w.f_state.setCurrentIndex(1)  # do przeglądu
row = next((k for k, i in enumerate(w.shown) if len(w.results[i].candidates) >= 3), 0)
w.table.selectRow(row)
dlg = ReviewDialog(w, row)  # nie-modalnie, żeby test mógł dalej działać
dlg.show()
app.processEvents()
r = dlg.r
scores = [r.candidates[i].score for i in dlg.order]
print("kwestia:", r.line.script, "| kandydaci (fragmenty cudzych kwestii na końcu):", scores)
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
print("decyzje w plikach roboczych:", (pipeline.work_dir(out) / "decisions.json").exists())


def no_dialog(*_a, **_k):
    raise AssertionError("eksport nie może pytać o folder")


QFileDialog.getExistingDirectory = staticmethod(no_dialog)
QMessageBox.information = staticmethod(lambda _parent, title, msg: print(f"[{title}] {msg.splitlines()[0]}"))
QMessageBox.question = staticmethod(lambda *_a, **_k: QMessageBox.StandardButton.Yes)

accepted = sum(w.state_of(x) == "accepted" and w.selection_of(x) is not None for x in w.results)
w.export_accepted()
names = sorted(p.name for p in out.iterdir())
others = [n for n in names if not n.endswith(".wav") and n != pipeline.WORK_DIR]
print(f"zatwierdzonych: {accepted} | plików WAV: {sum(n.endswith('.wav') for n in names)} | inne pliki w folderze: {others}")
w.export_accepted()  # drugi raz: pliki już są, pytanie o nadpisanie zamiast błędu
shutil.rmtree(out.parent, ignore_errors=True)
