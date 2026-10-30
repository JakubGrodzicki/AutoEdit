"""Generuje launcher/autoedit.ico (ikona: fala dźwiękowa z czerwonym znacznikiem cięcia).
Użycie: .venv\\Scripts\\python.exe launcher\\make_icon.py   (potrzebuje PySide6, wynik jest w repozytorium)."""
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen  # noqa: E402

app = QGuiApplication(sys.argv)


def render(size: int) -> bytes:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size
    p.setBrush(QColor(32, 36, 44))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(s * 0.03, s * 0.03, s * 0.94, s * 0.94), s * 0.2, s * 0.2)
    bars = [0.18, 0.34, 0.6, 0.82, 0.5, 0.72, 0.9, 0.55, 0.3, 0.16]
    w = s * 0.062
    x = s * 0.17
    p.setBrush(QColor(90, 160, 235))
    for h in bars:
        p.drawRoundedRect(QRectF(x, s * 0.5 - s * 0.36 * h, w, s * 0.72 * h), w / 2, w / 2)
        x += s * 0.073
    p.setPen(QPen(QColor(235, 70, 70), max(2.0, s * 0.045), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(int(s * 0.5), int(s * 0.1), int(s * 0.5), int(s * 0.9))
    p.end()
    buf = QByteArray()
    b = QBuffer(buf)
    b.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(b, "PNG")
    return bytes(buf)


sizes = [16, 32, 48, 64, 128, 256]
pngs = [render(s) for s in sizes]
out = struct.pack("<HHH", 0, 1, len(sizes))
offset = 6 + 16 * len(sizes)
entries = b""
for s, data in zip(sizes, pngs):
    entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
    offset += len(data)
Path(__file__).with_name("autoedit.ico").write_bytes(out + entries + b"".join(pngs))
print("zapisano autoedit.ico,", offset, "B")
