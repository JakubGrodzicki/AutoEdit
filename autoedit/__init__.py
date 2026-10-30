import os

# Sesje są objęte NDA i narzędzie działa offline. Modele pobiera tylko scripts/download_models.py.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
