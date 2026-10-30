"""Etap 0: ostrożne pobranie modeli (bez xet, jedno połączenie naraz)."""
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

for repo, ignore in [
    ("Systran/faster-whisper-large-v3", None),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-polish", ["*.msgpack", "*.h5", "*.ot", "flax*", "tf*"]),
]:
    print("pobieram", repo, flush=True)
    path = snapshot_download(repo, max_workers=1, ignore_patterns=ignore)
    print("gotowe", path, flush=True)
print("OK")
