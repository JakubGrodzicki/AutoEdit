"""Wybór urządzenia obliczeniowego: GPU (CUDA), a gdy go brak albo brakuje bibliotek CUDA, CPU z ostrzeżeniem."""
from __future__ import annotations

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path

_forced = {"cpu": False}

# Wiadomości o zmianie urządzenia; pipeline może je przekierować do logu GUI.
_listeners: list = []


def on_warning(fn) -> None:
    _listeners[:] = [fn]  # jeden odbiorca: ostatni uruchomiony przebieg


def warn(msg: str) -> None:
    text = f"UWAGA: {msg}"
    print(text, file=sys.stderr, flush=True)
    for fn in _listeners:
        try:
            fn(text)
        except Exception:
            pass


def _dll_dirs() -> list[Path]:
    """Foldery z bibliotekami CUDA dostarczonymi przez pip (torch cu12x oraz pakiety nvidia-*)."""
    dirs: list[Path] = []
    for base in map(Path, sys.path):
        for rel in ("torch/lib", "nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_runtime/bin"):
            d = base / rel
            if d.is_dir() and d not in dirs:
                dirs.append(d)
    return dirs


@lru_cache(maxsize=None)
def prepare_cuda_dlls() -> None:
    """Udostępnia CTranslate2 (faster-whisper) biblioteki CUDA z torch/lib, zanim zacznie ich szukać.

    Bez tego cublas64_12.dll nie jest znajdowana, gdy torch nie został jeszcze zaimportowany
    i na komputerze nie ma zainstalowanego CUDA Toolkit."""
    if os.name != "nt":
        return
    for d in _dll_dirs():
        try:
            os.add_dll_directory(str(d))
        except OSError:
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def _can_load(dll: str) -> bool:
    if os.name != "nt":
        return True  # na Linuksie ładowaniem zajmie się ld; błąd i tak złapie fallback w czasie działania
    try:
        ctypes.WinDLL(dll)
        return True
    except OSError:
        return False


@lru_cache(maxsize=None)
def cuda_problem() -> str | None:
    """None, gdy CUDA jest gotowa dla obu bibliotek (CTranslate2 i torch); inaczej opis problemu."""
    prepare_cuda_dlls()
    try:
        import torch
        if not torch.cuda.is_available():
            return "nie wykryto karty NVIDIA z działającym sterownikiem CUDA"
    except Exception as e:
        return f"PyTorch z CUDA niedostępny ({e})"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() < 1:
            return "CTranslate2 nie widzi karty NVIDIA"
    except Exception as e:
        return f"CTranslate2 nie widzi CUDA ({e})"
    for dll in ("cublas64_12.dll", "cudnn64_9.dll"):
        if not _can_load(dll):
            return f"brak biblioteki {dll} (uruchom ponownie install.bat)"
    return None


@lru_cache(maxsize=None)
def pick_device() -> str:
    """'cuda' albo 'cpu'. Zmienna AUTOEDIT_DEVICE=cpu wymusza CPU."""
    if _forced["cpu"]:
        return "cpu"  # ostrzeżenie padło już w force_cpu
    if os.environ.get("AUTOEDIT_DEVICE", "").lower() == "cpu":
        warn("wymuszono CPU (AUTOEDIT_DEVICE=cpu). Przetwarzanie będzie znacznie wolniejsze.")
        return "cpu"
    problem = cuda_problem()
    if problem:
        warn(f"{problem}. Przetwarzanie na CPU będzie znacznie wolniejsze (nagranie godzinne to nawet kilka godzin).")
        return "cpu"
    return "cuda"


def force_cpu(reason: str) -> None:
    """Przełącza cały proces na CPU po błędzie CUDA w trakcie działania."""
    warn(f"błąd CUDA w trakcie pracy ({reason}). Przełączam na CPU, będzie znacznie wolniej.")
    _forced["cpu"] = True
    pick_device.cache_clear()


def whisper_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


CUDA_ERROR_MARKERS = ("cublas", "cudnn", "cuda", ".dll", "out of memory")


def is_cuda_error(e: BaseException) -> bool:
    return isinstance(e, (RuntimeError, OSError)) and any(m in str(e).lower() for m in CUDA_ERROR_MARKERS)
