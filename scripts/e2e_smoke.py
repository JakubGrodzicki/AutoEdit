"""Przejazd end-to-end na syntetycznej sesji (głos Windows pl-PL).
Sprawdza tylko poprawność połączeń modułów; NIE mierzy jakości na prawdziwym nagraniu."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import openpyxl
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoedit import pipeline  # noqa: E402

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)

LINES = [
    ("VO_001", "Nie wiem czy to dobry pomysł"),
    ("VO_002", "Musimy ruszać zanim zapadnie zmrok"),
    ("VO_003", "Zabierz miecz i chodź ze mną"),
    ("VO_004", "Widziałem go przy starej bramie"),
    ("VO_005", "*śmiech*"),
    ("VO_006", "Wracamy do zamku jeszcze dziś wieczorem"),
]


def tts(text: str, path: Path) -> np.ndarray:
    ps = ("Add-Type -AssemblyName System.Speech;"
          "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          "$s.SelectVoice('Microsoft Paulina Desktop');"
          f"$s.SetOutputToWaveFile('{path}');$s.Speak('{text}');$s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    audio, sr = sf.read(path, dtype="float32")
    assert sr in (16000, 22050), sr
    return audio if audio.ndim == 1 else audio[:, 0], sr


pieces, sr = [], None


def add(text):
    global sr
    a, sr = tts(text, work / "tmp.wav")
    pieces.append(a)
    pieces.append(np.zeros(int(0.8 * sr), dtype=np.float32))


add(LINES[0][1])
add("Nie wiem czy to")                  # nieudane podejście
add(LINES[0][1])                        # powtórka, ostatnia ma wygrać
add("poczekaj chwilę zaczniemy jeszcze raz")  # gadanie reżysera/aktora
add(LINES[2][1])                        # kwestie nie po kolei
add(LINES[1][1])
add(LINES[3][1])
add(LINES[5][1])

audio = np.concatenate(pieces)
audio = audio + (np.random.default_rng(0).standard_normal(len(audio)) * 2e-4).astype(np.float32)  # szum tła studia
session = work / "sesja.wav"
sf.write(session, audio, sr, subtype="PCM_24")

xl = work / "kwestie.xlsx"
wb = openpyxl.Workbook()
ws = wb.active
ws.append(["Nazwa skryptu w grze", "Nazwa postaci", "Czy ta postać jest głównym bohaterem", "Treść kwestii"])
for name, text in LINES:
    ws.append([name, "Bohater", "tak", text])
wb.save(xl)

pipeline.run(session, [xl], work / "out", work / "cache", overwrite=True)
