"""Czy polski model akustyczny 'odpala' litery x, v, th itd. dla nazw z Excela?
Mierzy lukę audio (gap, tail_gap) dla pisowni z Excela vs mapowanej fonetycznie.
Mowa to synteza pl-PL (czytająca nazwy po polsku lub wg zapisu fonetycznego), więc to test mechaniki,
a nie dowód jakości na prawdziwych aktorach."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoedit.align import Aligner  # noqa: E402
from autoedit.normalize import for_aligner, normalize  # noqa: E402
from autoedit.transcribe import SR  # noqa: E402

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)

# (tekst w Excelu, tekst jak wymawia aktor)
CASES = [
    ("Musimy znaleźć Xal'Thorak", "Musimy znaleźć Ksal Torak"),
    ("Idź do Vaelthir i powiedz im", "Idź do Waeltir i powiedz im"),
    ("Strzeż się Qartha", "Strzeż się Kartha"),
    ("Pokonaliśmy Thranduila", "Pokonaliśmy Tranduila"),
    ("To dzieło Kaelena z Ysgard", "To dzieło Kaelena z Isgard"),
    ("Wracamy do Wyzima", "Wracamy do Wyzima"),  # kontrola: zwykły polski
]


def tts(text, path):
    ps = ("Add-Type -AssemblyName System.Speech;$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          "$s.SelectVoice('Microsoft Paulina Desktop');"
          f"$s.SetOutputToWaveFile('{path}');$s.Speak('{text}');$s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    a, sr = sf.read(path, dtype="float32")
    a = a if a.ndim == 1 else a[:, 0]
    n = int(len(a) * SR / sr)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


al = Aligner()
print(f"{'tekst z Excela':34s} {'raw gap/tail':>16s} {'mapped gap/tail':>18s}")
for excel, spoken in CASES:
    audio = tts(spoken, work / "n.wav")
    pad = np.zeros(SR, dtype=np.float32)
    audio = np.concatenate([pad, audio, pad])
    t0, t1 = 0.5, len(audio) / SR - 0.5
    out = []
    for text in (normalize(excel), for_aligner(normalize(excel))):
        r = al.align(audio, text, t0, t1)
        out.append("None" if r is None else f"{r[2]:5.2f}/{r[3]:5.2f}")
    print(f"{excel:34s} {out[0]:>16s} {out[1]:>18s}")
