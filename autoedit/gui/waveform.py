from __future__ import annotations

import numpy as np
import soundfile as sf
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class Waveform(QWidget):
    """Obwiednia fragmentu sesji z dwoma znacznikami cięcia, które użytkownik przeciąga myszą."""

    changed = Signal(float, float)  # start, koniec (s sesji), emitowany podczas przeciągania

    GRAB_PX = 10
    MIN_LEN = 0.05

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self.env = np.zeros(0)
        self.t0 = self.t1 = 0.0
        self.start = self.end = 0.0
        self._drag: str | None = None

    def load(self, path: str, start: float, end: float, margin: float = 1.5) -> None:
        info = sf.info(path)
        sr = info.samplerate
        self.t0, self.t1 = max(0.0, start - margin), min(info.duration, end + margin)
        x, _ = sf.read(path, start=int(self.t0 * sr), stop=int(self.t1 * sr), dtype="float32", always_2d=True)
        mono = np.abs(x).max(axis=1)
        bins = 1600
        step = max(1, len(mono) // bins)
        n = len(mono) // step
        self.env = mono[:n * step].reshape(n, step).max(axis=1) if n else np.zeros(0)
        self.set_range(start, end)

    def set_range(self, start: float, end: float) -> None:
        self.start, self.end = start, end
        self.update()

    # ---- geometria ----
    def _x(self, t: float) -> float:
        return (t - self.t0) / max(1e-9, self.t1 - self.t0) * self.width()

    def _t(self, x: float) -> float:
        return self.t0 + x / max(1, self.width()) * (self.t1 - self.t0)

    def _hit(self, x: float) -> str | None:
        ds, de = abs(x - self._x(self.start)), abs(x - self._x(self.end))
        if min(ds, de) > self.GRAB_PX:
            return None
        return "start" if ds <= de else "end"

    # ---- mysz ----
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton and len(self.env):
            self._drag = self._hit(e.position().x())

    def mouseMoveEvent(self, e) -> None:
        x = e.position().x()
        if self._drag is None:
            self.setCursor(Qt.CursorShape.SizeHorCursor if self._hit(x) else Qt.CursorShape.ArrowCursor)
            return
        t = min(max(self._t(x), self.t0), self.t1)
        if self._drag == "start":
            self.start = min(t, self.end - self.MIN_LEN)
        else:
            self.end = max(t, self.start + self.MIN_LEN)
        self.update()
        self.changed.emit(self.start, self.end)

    def mouseReleaseEvent(self, _) -> None:
        self._drag = None

    # ---- rysowanie ----
    def paintEvent(self, _) -> None:
        p = QPainter(self)
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())
        if len(self.env) == 0 or self.t1 <= self.t0:
            return
        w, h = self.width(), self.height() - 18
        peak = max(float(self.env.max()), 1e-6)
        p.fillRect(QRectF(self._x(self.start), 0, max(1.0, self._x(self.end) - self._x(self.start)), h),
                   QColor(60, 140, 220, 55))
        p.setPen(QPen(QColor(150, 150, 150)))
        for i, v in enumerate(self.env):
            x = i / len(self.env) * w
            half = v / peak * h / 2
            p.drawLine(int(x), int(h / 2 - half), int(x), int(h / 2 + half))
        p.setPen(QPen(pal.text().color()))  # oś czasu: znaczniki co 0,5 s
        step = 0.5
        t = np.ceil(self.t0 / step) * step
        while t < self.t1:
            x = int(self._x(t))
            p.drawLine(x, h, x, h + 4)
            if abs(t - round(t)) < 1e-6:
                p.drawText(x + 2, h + 15, f"{t:.0f} s")
            t += step
        p.setPen(QPen(QColor(235, 70, 70), 3))
        for t in (self.start, self.end):
            p.drawLine(int(self._x(t)), 0, int(self._x(t)), h)
        p.setPen(QPen(QColor(235, 70, 70)))
        p.drawText(int(self._x(self.start)) + 4, 14, f"{self.start:.2f}")
        p.drawText(max(0, int(self._x(self.end)) - 50), 14, f"{self.end:.2f}")
