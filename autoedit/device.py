"""Wybór urządzenia obliczeniowego: GPU (CUDA), a gdy go brak, karta jest za stara albo brakuje bibliotek CUDA,
CPU z ostrzeżeniem.

Whisper (CTranslate2) i wyrównanie (PyTorch) mają osobny wybór: PyTorch z CUDA 12.8 ma kernele tylko dla kart
od sm_75 (GTX 16xx / RTX 20xx), a CTranslate2 działa też na starszych (np. GTX 10xx), tylko bez float16."""
from __future__ import annotations

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path

WHISPER, TORCH = "whisper", "torch"
MIN_VRAM_TORCH_MB = 3000  # wav2vec2-large: ok. 1,3 GB wag + aktywacje
MIN_VRAM_FLOAT16_MB = 6000  # Whisper large-v3 float16 (ok. 3 GB) obok wav2vec2; poniżej int8

_forced: dict[str, bool] = {WHISPER: False, TORCH: False}
_warned: set[str] = set()

# Wiadomości o zmianie urządzenia; pipeline przekierowuje je do logu (GUI łączy stdout i stderr).
_listeners: list = []


def on_warning(fn) -> None:
    _listeners[:] = [fn]  # jeden odbiorca: ostatni uruchomiony przebieg


def warn(msg: str) -> None:
    text = f"UWAGA: {msg}"
    if not _listeners:  # bez odbiorcy (skrypty, instalator) na stderr; z odbiorcą tylko raz, w logu
        print(text, file=sys.stderr, flush=True)
    for fn in _listeners:
        try:
            fn(text)
        except Exception:
            pass


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warn(msg)


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
    """None, gdy sterownik i biblioteki CUDA są gotowe; inaczej opis problemu (dotyczy obu bibliotek)."""
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
def gpu_info() -> tuple[str, tuple[int, int], int] | None:
    """(nazwa, compute capability, VRAM w MB) pierwszej karty albo None."""
    if cuda_problem():
        return None
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        return p.name, (p.major, p.minor), p.total_memory // 2**20
    except Exception:
        return None


def _torch_min_arch() -> tuple[int, int] | None:
    """Najstarsza architektura, dla której zainstalowany PyTorch ma kernele (np. sm_75 -> (7, 5))."""
    try:
        import torch
        archs = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
        nums = [int(a[3:]) for a in archs if a[3:].isdigit()]
        if not nums:
            return None
        m = min(nums)
        return m // 10, m % 10
    except Exception:
        return None


def _env_cpu() -> bool:
    return os.environ.get("AUTOEDIT_DEVICE", "").lower() == "cpu"


def _cpu_reason() -> str | None:
    """Powód, dla którego oba etapy idą na CPU (wspólny dla Whispera i PyTorch)."""
    if _env_cpu():
        return "wymuszono CPU (AUTOEDIT_DEVICE=cpu)"
    return cuda_problem()


SLOW = "Przetwarzanie na CPU będzie znacznie wolniejsze (nagranie godzinne to nawet kilka godzin)."


@lru_cache(maxsize=None)
def whisper_device() -> str:
    """Urządzenie dla Whispera (CTranslate2): 'cuda' albo 'cpu'."""
    if _forced[WHISPER]:
        return "cpu"  # ostrzeżenie padło już w force_cpu
    reason = _cpu_reason()
    if reason:
        _warn_once("common", f"{reason}. {SLOW}")
        return "cpu"
    return "cuda"


@lru_cache(maxsize=None)
def torch_device() -> str:
    """Urządzenie dla wyrównania (PyTorch, wav2vec2): 'cuda' albo 'cpu'."""
    if _forced[TORCH]:
        return "cpu"
    reason = _cpu_reason()
    if reason:
        _warn_once("common", f"{reason}. {SLOW}")
        return "cpu"
    info = gpu_info()
    if info is None:
        _warn_once("torch", "nie udało się odczytać parametrów karty, wyrównanie na CPU (wolniej).")
        return "cpu"
    name, cc, vram = info
    lo = _torch_min_arch()
    if lo is not None and cc < lo:
        _warn_once("torch", f"karta {name} (sm_{cc[0]}{cc[1]}) jest za stara dla PyTorch (wymaga sm_{lo[0]}{lo[1]} "
                            "lub nowszej). Transkrypcja zostaje na GPU, wyrównanie idzie na CPU (wolniej).")
        return "cpu"
    if vram < MIN_VRAM_TORCH_MB:
        _warn_once("torch", f"karta {name} ma tylko {vram} MB pamięci. Wyrównanie idzie na CPU (wolniej).")
        return "cpu"
    return "cuda"


# Od najszybszego; float32 na końcu, bo obsługuje go każda karta (także bez int8, np. Maxwell).
_COMPUTE_CHAIN = ("float16", "int8_float16", "int8_float32", "int8", "float32")


def whisper_compute_type(device: str) -> str:
    try:
        import ctranslate2
        supported = set(ctranslate2.get_supported_compute_types(device))
    except Exception:
        supported = set()
    if not supported:
        return "int8" if device == "cpu" else "float32"
    chain = list(_COMPUTE_CHAIN)
    if device == "cuda":
        info = gpu_info()
        if info is not None and info[2] < MIN_VRAM_FLOAT16_MB:
            chain.remove("float16")  # mało pamięci: wagi int8 zajmują połowę
    return next((t for t in chain if t in supported), "float32")


def force_cpu(reason: str, which: tuple[str, ...] = (WHISPER, TORCH)) -> None:
    """Przełącza wskazane etapy na CPU po błędzie CUDA w trakcie działania."""
    what = {WHISPER: "transkrypcję", TORCH: "wyrównanie"}
    reason = reason.split(". ")[0][:160]  # komunikaty CUDA bywają na pół ekranu; pierwsze zdanie wystarcza
    warn(f"błąd CUDA w trakcie pracy ({reason}). Przełączam {' i '.join(what[w] for w in which)} na CPU, "
         "będzie znacznie wolniej.")
    for w in which:
        _forced[w] = True
    whisper_device.cache_clear()
    torch_device.cache_clear()


CUDA_ERROR_MARKERS = ("cublas", "cudnn", "cuda", ".dll", "out of memory", "float16", "no kernel image",
                      "compute capability")


def is_cuda_error(e: BaseException) -> bool:
    return isinstance(e, (RuntimeError, OSError, ValueError)) and any(m in str(e).lower() for m in CUDA_ERROR_MARKERS)
