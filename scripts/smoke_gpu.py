"""Etap 0: faster-whisper i wav2vec2 (torch) w jednym procesie na GPU."""
import json
import time

import numpy as np
import torch
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2ForCTC

ALIGN_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"

vocab = json.load(open(hf_hub_download(ALIGN_ID, "vocab.json"), encoding="utf-8"))
print("vocab size:", len(vocab))
print("tokens:", " ".join(sorted(vocab, key=vocab.get)))

t = time.time()
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"whisper loaded {time.time() - t:.1f}s")

t = time.time()
aligner = Wav2Vec2ForCTC.from_pretrained(ALIGN_ID).to("cuda").eval()
print(f"wav2vec2 loaded {time.time() - t:.1f}s")

audio = (np.random.randn(16000 * 30) * 0.01).astype(np.float32)
t = time.time()
segs, info = asr.transcribe(audio, language="pl", word_timestamps=True, vad_filter=True)
print("segments:", len(list(segs)), f"{time.time() - t:.1f}s")

with torch.inference_mode():
    out = aligner(torch.from_numpy(audio)[None].to("cuda")).logits
print("logits:", tuple(out.shape))
print("VRAM MB:", torch.cuda.max_memory_allocated() // 2**20)
print("OK")
