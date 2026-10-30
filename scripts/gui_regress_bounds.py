"""Regresja: granice kandydata w oknie przeglądu muszą odpowiadać aktualnej kwestii po zatwierdzeniu.
Użycie: python scripts/gui_regress_bounds.py <folder wyniku>  (najlepiej kopia, bo test zatwierdza kwestie)."""
import sys
from pathlib import Path
sys.path.insert(0, ".")
from PySide6.QtWidgets import QApplication
from autoedit.gui.app import MainWindow, ReviewDialog, bounds
app = QApplication([])
w = MainWindow(); w.show(); w.load(Path(sys.argv[1])); w.f_who.setCurrentIndex(1); w.f_state.setCurrentIndex(1)
dlg = ReviewDialog(w, 0); dlg.show(); app.processEvents()
bad = 0
for step in range(6):
    r = dlg.r; idx = dlg.cand_index()
    s, e = dlg.sp_start.value(), dlg.sp_end.value()
    cs, ce = bounds(r.candidates[idx])
    ok = abs(s - cs) < 1e-3 and abs(e - ce) < 1e-3
    bad += not ok
    print(f"{r.line.script:34s} kand {idx}: pola {s:.2f}-{e:.2f} | kandydat {cs:.2f}-{ce:.2f} | {'OK' if ok else 'PRZECIEK'}")
    dlg.decide("accepted")
    app.processEvents()
print("przecieków:", bad)
