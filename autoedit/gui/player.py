"""Odsłuch wycinka z oryginalnego pliku sesji."""
from __future__ import annotations

import soundfile as sf
from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices


class Player:
    def __init__(self) -> None:
        self.sink: QAudioSink | None = None
        self.buf: QBuffer | None = None
        self.data = QByteArray()

    @staticmethod
    def available() -> bool:
        return bool(QMediaDevices.audioOutputs())

    def play(self, path: str, start: float, end: float, pad: float = 0.15) -> bool:
        self.stop()
        if not self.available():
            return False
        info = sf.info(path)
        sr = info.samplerate
        a, b = max(0, int((start - pad) * sr)), min(info.frames, int((end + pad) * sr))
        samples, _ = sf.read(path, start=a, stop=b, dtype="int16", always_2d=True)
        fmt = QAudioFormat()
        fmt.setSampleRate(sr)
        fmt.setChannelCount(samples.shape[1])
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        dev = QMediaDevices.defaultAudioOutput()
        if not dev.isFormatSupported(fmt):
            return False
        self.data = QByteArray(samples.tobytes())
        self.buf = QBuffer(self.data)
        self.buf.open(QIODevice.OpenModeFlag.ReadOnly)
        self.sink = QAudioSink(dev, fmt)
        self.sink.start(self.buf)
        return True

    def stop(self) -> None:
        if self.sink is not None:
            self.sink.stop()
            self.sink = None
        if self.buf is not None:
            self.buf.close()
            self.buf = None

    @property
    def playing(self) -> bool:
        return self.sink is not None and self.sink.state().name == "ActiveState"
