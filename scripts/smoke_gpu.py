"""Test środowiska: faster-whisper i wav2vec2 (torch) w jednym procesie, na GPU albo (z ostrzeżeniem) na CPU.

Używa tego samego wyboru urządzenia i ładowania bibliotek CUDA (autoedit.device) co program."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from autoedit import device as dev

DEVICE = dev.pick_device()
print("DEVICE", DEVICE)
dev.prepare_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
import torch  # noqa: E402
from transformers import Wav2Vec2ForCTC  # noqa: E402

ALIGN_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"

vocab = json.load(open(hf_hub_download(ALIGN_ID, "vocab.json"), encoding="utf-8"))
print("vocab size:", len(vocab))

t = time.time()
asr = WhisperModel("large-v3", device=DEVICE, compute_type=dev.whisper_compute_type(DEVICE))
print(f"whisper loaded {time.time() - t:.1f}s")

t = time.time()
aligner = Wav2Vec2ForCTC.from_pretrained(ALIGN_ID).to(DEVICE).eval()
print(f"wav2vec2 loaded {time.time() - t:.1f}s")

audio = (np.random.randn(16000 * 30) * 0.01).astype(np.float32)
t = time.time()
segs, info = asr.transcribe(audio, language="pl", word_timestamps=True, vad_filter=True)
print("segments:", len(list(segs)), f"{time.time() - t:.1f}s")

with torch.inference_mode():
    out = aligner(torch.from_numpy(audio)[None].to(DEVICE)).logits
print("logits:", tuple(out.shape))
if DEVICE == "cuda":
    print("VRAM MB:", torch.cuda.max_memory_allocated() // 2**20)
else:
    print("VRAM MB: brak GPU, test na CPU")
print("OK")
