"""Test środowiska: faster-whisper i wav2vec2 (torch) w jednym procesie, na GPU albo (z ostrzeżeniem) na CPU.

Odwzorowuje program: ten sam wybór urządzeń i ładowanie bibliotek CUDA (autoedit.device), Whisper zwalniany przed
wyrównaniem, wyrównanie przez autoedit.align.Aligner (z jego przejściem na CPU) i krótkie okno, jak w programie.
Dzięki temu karta z małą pamięcią nie przerywa instalacji, skoro sam program by sobie poradził."""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from autoedit import device as dev

SR = 16000
dev.prepare_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402


def whisper_run(device: str) -> None:
    ctype = dev.whisper_compute_type(device)
    t = time.time()
    asr = WhisperModel("large-v3", device=device, compute_type=ctype)
    print(f"whisper loaded ({device}, {ctype}) {time.time() - t:.1f}s")
    t = time.time()
    segs, _ = asr.transcribe(audio, language="pl", word_timestamps=True, vad_filter=True)
    print("segments:", len(list(segs)), f"{time.time() - t:.1f}s")
    del asr
    gc.collect()


audio = (np.random.randn(SR * 30) * 0.01).astype(np.float32)

w_dev = dev.whisper_device()
try:
    whisper_run(w_dev)
except Exception as e:  # tak samo jak w transcribe.transcribe
    if w_dev != "cuda" or not dev.is_cuda_error(e):
        raise
    dev.force_cpu(str(e).splitlines()[0], (dev.WHISPER,))
    w_dev = "cpu"
    whisper_run(w_dev)

import torch  # noqa: E402
from autoedit.align import Aligner  # noqa: E402

t = time.time()
aligner = Aligner()
print(f"wav2vec2 loaded ({aligner.device}) {time.time() - t:.1f}s")
lp = aligner._log_probs(audio[:SR * 5])
print("log-probs:", tuple(lp.shape))
if aligner.device == "cuda":
    print("VRAM MB:", torch.cuda.max_memory_allocated() // 2**20)
else:
    print("VRAM MB: wyrównanie na CPU")
print(f"DEVICE whisper={w_dev} torch={aligner.device}")  # po ewentualnych fallbackach
print("OK")
