"""Okno główne: lista kwestii. Wybór kandydata odbywa się w dużym oknie przeglądu (ReviewDialog)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import export, pipeline, tables
from ..models import Candidate, LineResult
from .player import Player
from .waveform import Waveform

ROOT = Path(__file__).resolve().parents[2]
STATE_LABEL = {"accepted": "✓ zatwierdzona", "rejected": "✗ odrzucona", "pending": "? do przeglądu",
               "none": "— nie znaleziono"}
STATE_COLOR = {"accepted": QColor(200, 235, 200), "rejected": QColor(235, 200, 200),
               "pending": QColor(250, 240, 195), "none": QColor(225, 225, 225)}
TEXT_COLOR = QColor(25, 25, 25)  # jasne tła wierszy wymagają ciemnego tekstu także w ciemnym motywie


def bounds(c: Candidate) -> tuple[float, float]:
    return c.refined if c.refined else (c.start, c.end)


class NewProjectDialog(QDialog):
    """Wybór sesji, tabel i folderu wyjściowego, potem przetwarzanie silnikiem w osobnym procesie."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nowy projekt")
        self.resize(720, 460)
        self.audio, self.out = QLineEdit(), QLineEdit()
        self.tables = QLineEdit()
        form = QFormLayout()
        for label, edit, fn in (("Nagranie sesji", self.audio, self.pick_audio),
                                ("Tabele (xlsx/csv)", self.tables, self.pick_tables),
                                ("Folder wynikowy", self.out, self.pick_out)):
            row = QHBoxLayout()
            row.addWidget(edit)
            b = QPushButton("Wybierz…")
            b.clicked.connect(fn)
            row.addWidget(b)
            form.addRow(label, row)
        self.who = QComboBox()  # kwestie drugiego aktora z tego samego arkusza nie biorą udziału w dopasowaniu
        self.who.addItem("NPC (Czy MC = NIE)", "npc")
        self.who.addItem("Głównego bohatera (Czy MC = TAK)", "main")
        self.who.setCurrentIndex(max(0, self.who.findData(QSettings("AutoEdit", "AutoEdit").value("who", "npc"))))
        form.addRow("Nagranie aktora", self.who)
        self.log = QPlainTextEdit(readOnly=True)
        self.run_btn = QPushButton("Uruchom przetwarzanie")
        self.run_btn.clicked.connect(self.start)
        self.close_btn = QPushButton("Zamknij")
        self.close_btn.clicked.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.log)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.run_btn)
        row.addWidget(self.close_btn)
        lay.addLayout(row)
        self.proc: QProcess | None = None
        self.result_dir: Path | None = None

    def pick_audio(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Nagranie sesji", "", "Audio (*.wav *.w64 *.flac *.mp3 *.m4a)")
        if f:
            self.audio.setText(f)

    def pick_tables(self) -> None:
        fs, _ = QFileDialog.getOpenFileNames(self, "Tabele z kwestiami", "", "Tabele (*.xlsx *.xlsm *.csv)")
        if fs:
            self.tables.setText(";".join(fs))

    def pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder wynikowy")
        if d:
            self.out.setText(d)

    def start(self) -> None:
        tables = [t for t in self.tables.text().split(";") if t]
        if not (self.audio.text() and tables and self.out.text()):
            QMessageBox.warning(self, "Brakuje danych", "Wskaż nagranie, co najmniej jedną tabelę i folder wynikowy.")
            return
        if pipeline.has_project(Path(self.out.text())):
            ans = QMessageBox.question(self, "W tym folderze jest już projekt",
                                       "Przetworzenie od nowa zastąpi wyniki, a zapisane decyzje (zatwierdzenia i ręczne "
                                       "granice) przestaną obowiązywać. Ich kopia zostanie w decisions.old.json.\n\n"
                                       "Wyeksportowane wcześniej pliki WAV zostaną w folderze, dopóki eksport ich nie "
                                       "nadpisze.\n\nKontynuować?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(lambda: self.log.appendPlainText(
            bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace").rstrip()))
        self.proc.finished.connect(self.on_finished)
        self.run_btn.setEnabled(False)
        self.log.appendPlainText("Start…")
        who = self.who.currentData()
        QSettings("AutoEdit", "AutoEdit").setValue("who", who)
        self.proc.start(sys.executable, ["-m", "autoedit.cli", self.audio.text(), *tables, "-o", self.out.text(),
                                         "--cache", str(ROOT / "cache"), "--no-export", "--who", who])

    def on_finished(self, code: int, _status) -> None:  # nie "done": to nadpisałoby QDialog.done(int)
        self.run_btn.setEnabled(True)
        if code == 0:
            self.result_dir = Path(self.out.text())
            self.accept()
        else:
            self.log.appendPlainText(f"Proces zakończył się błędem (kod {code}).")

    def reject(self) -> None:
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()
        super().reject()


class ReviewDialog(QDialog):
    """Duże okno do wyboru kandydata jednej kwestii: odsłuch, przeciąganie granic, zatwierdzenie."""

    def __init__(self, win: "MainWindow", row: int) -> None:
        super().__init__(win)
        self.win = win
        self.row = row  # wiersz w tabeli głównej
        self.order: list[int] = []  # wiersz tabeli kandydatów -> indeks w r.candidates
        self._loading = False
        self.setWindowTitle("Wybór kandydata")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.62), int(screen.height() * 0.78))

        self.counter = QLabel("")
        self.title = QLabel("")
        self.title.setStyleSheet("font-size: 22px;")
        self.text = QLabel("")
        self.text.setWordWrap(True)
        self.text.setStyleSheet("font-size: 34px; font-weight: bold;")  # treść kwestii decyduje o wyborze
        self.reasons = QLabel("")
        self.reasons.setWordWrap(True)
        self.state = QLabel("")
        self.state.setStyleSheet("font-size: 15px; font-weight: bold;")

        self.cands = QTableWidget(0, 5)
        self.cands.setHorizontalHeaderLabels(["Podobieństwo", "Start", "Koniec", "Dł. (s)", "Audio / kompletność"])
        self.cands.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.cands.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cands.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cands.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cands.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.cands.verticalHeader().setVisible(False)
        self.cands.itemSelectionChanged.connect(self.on_cand)

        self.wave = Waveform()
        self.wave.changed.connect(self.on_wave)
        hint = QLabel("Czerwone znaczniki na fali można przeciągać myszą.")
        self.sp_start, self.sp_end = QDoubleSpinBox(), QDoubleSpinBox()
        for sp in (self.sp_start, self.sp_end):
            sp.setDecimals(3)
            sp.setSingleStep(0.01)
            sp.setRange(0, 1e5)
            sp.valueChanged.connect(self.on_spin)
        self.btn_play = QPushButton("▶ Odsłuchaj  (Spacja)")
        self.btn_ok = QPushButton("✓ Zatwierdź  (Enter)")
        self.btn_no = QPushButton("✗ Odrzuć  (Del)")
        self.btn_back = QPushButton("← Wróć do listy  (Esc)")
        for b in (self.btn_play, self.btn_ok, self.btn_no, self.btn_back):
            b.setMinimumHeight(44)
        self.btn_play.clicked.connect(self.play)
        self.btn_ok.clicked.connect(lambda: self.decide("accepted"))
        self.btn_no.clicked.connect(lambda: self.decide("rejected"))
        self.btn_back.clicked.connect(self.accept)

        head = QHBoxLayout()
        head.addWidget(self.title)
        head.addStretch()
        head.addWidget(self.state)
        head.addWidget(self.counter)
        bar = QHBoxLayout()
        for w in (QLabel("Start"), self.sp_start, QLabel("Koniec"), self.sp_end, hint):
            bar.addWidget(w)
        bar.addStretch()
        btns = QHBoxLayout()
        for b in (self.btn_back, self.btn_play, self.btn_no, self.btn_ok):
            btns.addWidget(b)
        lay = QVBoxLayout(self)
        lay.addLayout(head)
        lay.addWidget(self.text)
        lay.addWidget(self.reasons)
        lay.addWidget(self.cands)  # wysokość dopasowana do liczby kandydatów w show_row
        lay.addWidget(self.wave, 1)  # fala dostaje całą resztę miejsca
        lay.addLayout(bar)
        lay.addLayout(btns)

        for key, fn in (("Space", self.play), ("Return", lambda: self.decide("accepted")),
                        ("Enter", lambda: self.decide("accepted")), ("Delete", lambda: self.decide("rejected"))):
            QShortcut(QKeySequence(key), self, activated=fn)
        self.show_row(row)

    def reject(self) -> None:  # Esc = wróć do listy
        self.accept()

    def done(self, r: int) -> None:
        self.win.player.stop()
        super().done(r)

    # ---- zawartość ----
    @property
    def r(self) -> LineResult:
        return self.win.results[self.win.shown[self.row]]

    def show_row(self, row: int) -> None:
        self.row = row
        r, w = self.r, self.win
        w.player.stop()
        pending = [k for k, i in enumerate(w.shown) if w.state_of(w.results[i]) == "pending"]
        self.counter.setText(f"   do przeglądu: {len(pending)}")
        self.title.setText(r.line.script)
        self.text.setText(r.line.text.upper())
        self.reasons.setText("; ".join(r.reasons))
        self.state.setText(STATE_LABEL[w.state_of(r)])
        self._loading = True
        # fragmenty cudzych kwestii na końcu, żeby nie przysłaniały prawdziwych podejść
        by_score = sorted(range(len(r.candidates)), key=lambda i: (r.candidates[i].inside_other, -r.candidates[i].score))
        good = [i for i in by_score if r.candidates[i].score >= 70 or r.candidates[i].origin == "gap"]
        self.order = good or by_score[:5]  # słabe dopasowania pokazujemy tylko, gdy nie ma nic lepszego
        self.cands.setRowCount(len(self.order))
        for k, i in enumerate(self.order):
            c = r.candidates[i]
            s, e = bounds(c)
            if c.origin == "gap":
                sim, audio = "-", "propozycja z luki"
            else:
                sim = f"{c.score:.0f}%"
                gap = "" if c.audio_gap is None else f"luka {c.audio_gap:.1f} / {c.audio_tail_gap:.1f}, "
                audio = gap + ("kompletna" if c.complete else "niedokończona")
            for col, v in enumerate([sim, f"{s:.2f}", f"{e:.2f}", f"{e - s:.2f}", audio]):
                self.cands.setItem(k, col, QTableWidgetItem(v))
        rows = min(len(self.order), 6)  # tabela tylko tak wysoka, jak trzeba (do 6 wierszy, potem przewijanie)
        self.cands.setFixedHeight(self.cands.horizontalHeader().height() + rows * 37 + 6)
        sel = w.selection_of(r)
        self.cands.clearSelection()  # inaczej ten sam numer wiersza nie wywoła zmiany zaznaczenia
        self._loading = False
        for b in (self.btn_play, self.btn_ok):
            b.setEnabled(sel is not None)
        if sel is not None:
            self.cands.selectRow(self.order.index(sel[0]))
            self.on_cand()  # pola, fala i decyzja zawsze odpowiadają nowej kwestii
        else:
            self.wave.env = self.wave.env[:0]
            self.wave.update()

    def cand_index(self) -> int | None:
        rows = self.cands.selectionModel().selectedRows()
        return self.order[rows[0].row()] if rows else None

    def on_cand(self) -> None:
        if self._loading:
            return
        idx = self.cand_index()
        if idx is None:
            return
        r, w = self.r, self.win
        d = w.decisions.get(r.line.script, {})
        s, e = bounds(r.candidates[idx])
        if d.get("cand") == idx:
            s, e = d.get("start", s), d.get("end", e)
        self._set_bounds(s, e)
        self.wave.load(w.meta["audio"], s, e)
        w.remember(r, idx, s, e)

    def _set_bounds(self, s: float, e: float) -> None:
        self._loading = True
        self.sp_start.setValue(s)
        self.sp_end.setValue(e)
        self._loading = False

    def on_wave(self, s: float, e: float) -> None:
        self._set_bounds(s, e)
        idx = self.cand_index()
        if idx is not None:
            self.win.remember(self.r, idx, s, e)

    def on_spin(self) -> None:
        if self._loading:
            return
        s, e = self.sp_start.value(), self.sp_end.value()
        self.wave.set_range(s, e)
        idx = self.cand_index()
        if idx is not None:
            self.win.remember(self.r, idx, s, e)

    # ---- akcje ----
    def play(self) -> None:
        w = self.win
        if w.player.playing:
            w.player.stop()
            return
        if self.cand_index() is None:
            return
        if not w.player.play(w.meta["audio"], self.sp_start.value(), self.sp_end.value(), w.meta.get("pad", 0.15)):
            self.reasons.setText("Brak urządzenia audio lub nieobsługiwany format.")

    def decide(self, state: str) -> None:
        w, r = self.win, self.r
        idx = self.cand_index()
        if state == "accepted" and idx is None:
            return
        d = w.decisions.setdefault(r.line.script, {})
        d["state"] = state
        if idx is not None:
            d.update({"cand": idx, "start": self.sp_start.value(), "end": self.sp_end.value()})
        w.save_decisions()
        w.paint_row(self.row, r)
        w.summary_refresh()
        pend = [k for k in range(len(w.shown)) if w.state_of(w.results[w.shown[k]]) == "pending"]
        nxt = next((k for k in pend if k > self.row), pend[0] if pend else None)  # dalej, a potem od początku
        if nxt is None:
            self.accept()
            w.statusBar().showMessage("Nie ma już kwestii do przeglądu.")
        else:
            self.show_row(nxt)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AutoEdit")
        self.resize(1280, 780)
        self.player = Player()
        self.out_dir: Path | None = None
        self.work: Path | None = None  # folder plików roboczych projektu
        self.meta: dict = {}
        self.results: list[LineResult] = []
        self.decisions: dict[str, dict] = {}
        self.shown: list[int] = []  # indeksy results widoczne w tabeli
        self._review: ReviewDialog | None = None

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Skrypt", "Postać", "Stan", "Podejść", "Najlepsze", "Treść"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.activated.connect(self.open_review)  # Enter i dwuklik
        # wiersze mają własne tło (stan), więc zaznaczenie musi je jawnie przesłonić
        self.table.setStyleSheet("QTableWidget::item:selected { background-color: #1f5fbf; color: white; "
                                 "border-top: 2px solid #8ab4ff; border-bottom: 2px solid #8ab4ff; }")

        self.f_state = QComboBox()
        self.f_state.addItems(["Wszystkie", "Do przeglądu", "Zatwierdzone", "Odrzucone", "Nie znaleziono"])
        self.f_who = QComboBox()
        self.f_who.addItems(["Wszystkie", "Tylko NPC (Czy MC = NIE)", "Tylko MC"])
        self.f_state.currentIndexChanged.connect(self.fill_table)
        self.f_who.currentIndexChanged.connect(self.fill_table)
        self.summary = QLabel("")
        self.btn_review = QPushButton("Przeglądaj kolejne do przeglądu →")
        self.btn_review.clicked.connect(self.review_next)
        self.btn_new = QPushButton("＋ Nowy projekt…")
        self.btn_open = QPushButton("📂 Otwórz wyniki")
        self.btn_load = QPushButton("Wczytaj projekt…")
        self.btn_export = QPushButton("⬇ Eksportuj zatwierdzone")
        for b in (self.btn_new, self.btn_open, self.btn_load, self.btn_export, self.btn_review):
            b.setMinimumHeight(38)
        self.btn_export.setMinimumHeight(46)
        self.btn_export.setStyleSheet("QPushButton { background-color: #1f7a3d; color: white; font-weight: bold; "
                                      "padding: 0 22px; } QPushButton:hover { background-color: #26964b; }")
        self.btn_new.clicked.connect(self.new_project)
        self.btn_open.clicked.connect(self.show_results_folder)
        self.btn_load.clicked.connect(self.open_project)
        self.btn_export.clicked.connect(self.export_accepted)
        top = QHBoxLayout()
        top.addWidget(self.btn_new)
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_load)
        top.addSpacing(24)
        top.addWidget(QLabel("Stan:"))
        top.addWidget(self.f_state)
        self.f_who_label = QLabel("Kwestie:")
        top.addWidget(self.f_who_label)
        top.addWidget(self.f_who)
        top.addStretch()
        top.addWidget(self.btn_review)
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(top)
        lay.addWidget(self.table)
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Dwuklik lub Enter na kwestii otwiera wybór kandydata. Zatwierdzone automatycznie "
                                "i nie znalezione też można otworzyć i sprawdzić."))
        bottom.addStretch()
        bottom.addWidget(self.btn_export)
        lay.addLayout(bottom)
        self.setCentralWidget(central)
        self.statusBar().addPermanentWidget(self.summary)
        self.statusBar().showMessage("Nowy projekt lub Otwórz wynik z menu Projekt.")

    # ---------- projekt ----------
    def new_project(self) -> None:
        dlg = NewProjectDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_dir:
            self.load(dlg.result_dir)

    def show_results_folder(self) -> None:
        """Otwiera w Eksploratorze folder wynikowy projektu (ten wskazany przy tworzeniu)."""
        folder = self.out_dir or Path(str(QSettings("AutoEdit", "AutoEdit").value("last_results_dir", ROOT / "output")))
        if not folder.is_dir():
            folder = ROOT
        os.startfile(folder)

    def open_project(self) -> None:
        """Wczytanie wcześniejszego wyniku: wskazuje się folder wynikowy podany przy tworzeniu projektu."""
        settings = QSettings("AutoEdit", "AutoEdit")
        start = settings.value("last_results_dir", str(ROOT / "output"))
        d = QFileDialog.getExistingDirectory(self, "Wczytaj projekt - wskaż folder wynikowy", str(start))
        if not d:
            return
        folder = Path(d).parent if Path(d).name == pipeline.WORK_DIR else Path(d)
        if not pipeline.has_project(folder):
            QMessageBox.warning(self, "To nie jest wynik AutoEdit",
                                f"W folderze:\n{folder}\n\nnie ma projektu AutoEdit.\n\n"
                                "Wskaż folder wynikowy podany przy tworzeniu projektu.")
            return
        self.load(folder)

    def load(self, out_dir: Path) -> None:
        try:
            self.meta, self.results = pipeline.load_project(out_dir)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Nie można wczytać", f"{out_dir}\n\n{e}")
            return
        self.out_dir = out_dir
        self.work = pipeline.work_dir(out_dir)
        QSettings("AutoEdit", "AutoEdit").setValue("last_results_dir", str(out_dir.resolve()))
        dp = self.work / "decisions.json"
        self.decisions = json.loads(dp.read_text(encoding="utf-8")) if dp.exists() else {}
        who = self.meta.get("who")
        # projekt z wybranym aktorem ma kwestie tylko jednej grupy, więc filtr NPC/MC zostaje dla starszych projektów
        if who:
            self.f_who.setCurrentIndex(0)
        self.f_who.setVisible(who is None)
        self.f_who_label.setVisible(who is None)
        role = f" ({tables.ROLE_LABEL[who]})" if who else ""
        self.setWindowTitle(f"AutoEdit - {out_dir.name}{role}")
        self.fill_table()
        self.statusBar().showMessage(f"Wczytano {len(self.results)} kwestii.")

    def save_decisions(self) -> None:
        if self.work:
            (self.work / "decisions.json").write_text(json.dumps(self.decisions, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")

    # ---------- stan kwestii ----------
    def state_of(self, r: LineResult) -> str:
        d = self.decisions.get(r.line.script)
        if d and d.get("state"):
            return d["state"]
        return {"auto": "accepted", "przegląd": "pending"}.get(r.status, "none")

    def default_index(self, r: LineResult) -> int | None:
        if r.chosen is not None and r.chosen in r.candidates:
            return r.candidates.index(r.chosen)
        if r.candidates:
            return max(range(len(r.candidates)), key=lambda i: r.candidates[i].score)
        return None

    def selection_of(self, r: LineResult) -> tuple[int, float, float] | None:
        """(indeks kandydata, start, koniec): decyzja użytkownika albo wybór silnika."""
        d = self.decisions.get(r.line.script, {})
        cs = r.candidates
        if "cand" in d and d["cand"] < len(cs):
            s, e = bounds(cs[d["cand"]])
            return d["cand"], d.get("start", s), d.get("end", e)
        i = self.default_index(r)
        if i is None:
            return None
        s, e = bounds(cs[i])
        return i, s, e

    def remember(self, r: LineResult, idx: int, s: float, e: float) -> None:
        d = self.decisions.setdefault(r.line.script, {})
        cs, ce = bounds(r.candidates[idx])
        if idx == self.default_index(r) and (s, e) == (cs, ce) and not d.get("state"):
            self.decisions.pop(r.line.script, None)  # wybór domyślny, nic do zapamiętania
        else:
            d.update({"cand": idx, "start": s, "end": e})
        self.save_decisions()

    # ---------- lista ----------
    def fill_table(self) -> None:
        if not self.results:
            return
        cur = self.table.currentRow()
        keep = self.results[self.shown[cur]].line.script if 0 <= cur < len(self.shown) else None
        want = {1: "pending", 2: "accepted", 3: "rejected", 4: "none"}.get(self.f_state.currentIndex())
        who = self.f_who.currentIndex()
        self.shown = [i for i, r in enumerate(self.results)
                      if (want is None or self.state_of(r) == want)
                      and (who == 0 or (who == 1) == (not r.line.is_main))]
        self.table.setRowCount(len(self.shown))
        for row, i in enumerate(self.shown):
            self.paint_row(row, self.results[i])
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.summary_refresh()
        if keep:
            for row, i in enumerate(self.shown):
                if self.results[i].line.script == keep:
                    self.table.selectRow(row)
                    break

    def paint_row(self, row: int, r: LineResult) -> None:
        st = self.state_of(r)
        texts = [c for c in r.candidates if c.origin == "text" and c.score >= 70 and not c.inside_other]
        best = f"{max(c.score for c in texts):.0f}%" if texts else ""
        vals = [r.line.script, r.line.character, STATE_LABEL[st], str(len(texts)) if texts else "", best, r.line.text]
        for col, v in enumerate(vals):
            it = QTableWidgetItem(v)
            it.setBackground(STATE_COLOR[st])
            it.setForeground(TEXT_COLOR)
            self.table.setItem(row, col, it)

    def summary_refresh(self) -> None:
        who = self.meta.get("who")
        mine = self.results if who else [r for r in self.results if not r.line.is_main]
        counts = {k: sum(self.state_of(r) == k for r in mine) for k in ("accepted", "pending", "rejected", "none")}
        self.summary.setText(f"{tables.ROLE_LABEL[who or 'npc']}: zatwierdzone {counts['accepted']} · do przeglądu {counts['pending']} · "
                             f"odrzucone {counts['rejected']} · nie znaleziono {counts['none']}   ")

    # ---------- przegląd ----------
    def open_review(self, *_) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.shown) or self._review is not None:
            return
        self._review = ReviewDialog(self, row)
        self._review.exec()
        self._review = None
        self.fill_table()

    def review_next(self) -> None:
        for k, i in enumerate(self.shown):
            if self.state_of(self.results[i]) == "pending":
                self.table.selectRow(k)
                self.open_review()
                return
        self.statusBar().showMessage("Nie ma kwestii do przeglądu w bieżącym widoku.")

    def export_accepted(self) -> None:
        if not self.results:
            return
        items = []
        for r in self.results:
            if self.state_of(r) != "accepted":
                continue
            sel = self.selection_of(r)
            if sel is None:
                continue
            items.append((r.line.script, sel[1], sel[2]))
        if not items:
            QMessageBox.information(self, "Eksport", "Brak zatwierdzonych kwestii.")
            return
        out = self.out_dir
        existing = [n for n, *_ in items if (out / f"{export.safe_name(n)[0]}.wav").exists()]
        overwrite = False
        if existing:
            ans = QMessageBox.question(self, "Pliki już istnieją",
                                       f"{len(existing)} plików o tych nazwach już jest w folderze. Nadpisać?\n"
                                       "Nie = pominąć istniejące.",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            overwrite = ans == QMessageBox.StandardButton.Yes
        notes = export.export_selected(items, Path(self.meta["audio"]), out, self.meta.get("pad", 0.15), overwrite)
        good = lambda v: v.startswith("ok") or v.startswith("zmieniono")  # noqa: E731
        skipped = [f"{k}: {v}" for k, v in notes.items() if not good(v)]
        msg = f"Zapisano {sum(good(v) for v in notes.values())} z {len(items)} plików do:\n{out}"
        if skipped:
            msg += "\n\nPominięto:\n" + "\n".join(skipped[:15])
        QMessageBox.information(self, "Eksport", msg)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
        win.load(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
