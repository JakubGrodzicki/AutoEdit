"""Wybór urządzeń: stare karty, mało pamięci, błędy CUDA w trakcie pracy. Bez prawdziwego GPU (wszystko mockowane)."""
import numpy as np
import pytest

from autoedit import device as dev


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.delenv("AUTOEDIT_DEVICE", raising=False)
    monkeypatch.setattr(dev, "_forced", {dev.WHISPER: False, dev.TORCH: False})
    monkeypatch.setattr(dev, "_warned", set())
    monkeypatch.setattr(dev, "_listeners", [])
    monkeypatch.setattr(dev, "cuda_problem", lambda: None)
    for f in (dev.whisper_device, dev.torch_device):
        f.cache_clear()
    yield
    for f in (dev.whisper_device, dev.torch_device):
        f.cache_clear()


def gpu(monkeypatch, cc, vram, archs=(75, 80, 86, 90, 100, 120)):
    monkeypatch.setattr(dev, "gpu_info", lambda: ("Test GPU", cc, vram))
    monkeypatch.setattr(dev, "_torch_min_arch", lambda: (min(archs) // 10, min(archs) % 10))


def test_modern_gpu_uses_cuda_for_both(monkeypatch):
    gpu(monkeypatch, (8, 6), 12288)
    assert dev.whisper_device() == "cuda"
    assert dev.torch_device() == "cuda"


def test_pascal_keeps_whisper_on_gpu_but_aligns_on_cpu(monkeypatch, capsys):
    gpu(monkeypatch, (6, 1), 8192)
    assert dev.whisper_device() == "cuda"
    assert dev.torch_device() == "cpu"
    assert "sm_61" in capsys.readouterr().err


def test_low_vram_aligns_on_cpu(monkeypatch):
    gpu(monkeypatch, (7, 5), 2048)
    assert dev.torch_device() == "cpu"


def test_env_forces_cpu(monkeypatch):
    monkeypatch.setenv("AUTOEDIT_DEVICE", "cpu")
    assert dev.whisper_device() == "cpu"
    assert dev.torch_device() == "cpu"


def test_cuda_problem_means_cpu_with_single_warning(monkeypatch, capsys):
    monkeypatch.setattr(dev, "cuda_problem", lambda: "brak biblioteki cublas64_12.dll")
    assert dev.whisper_device() == "cpu"
    assert dev.torch_device() == "cpu"
    assert capsys.readouterr().err.count("UWAGA") == 1


def test_compute_type_chain(monkeypatch):
    import ctranslate2
    gpu(monkeypatch, (8, 6), 12288)
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda d: {"float16", "int8_float16", "float32"})
    assert dev.whisper_compute_type("cuda") == "float16"
    gpu(monkeypatch, (7, 5), 4096)  # mało pamięci: bez float16
    assert dev.whisper_compute_type("cuda") == "int8_float16"
    gpu(monkeypatch, (6, 1), 8192)  # Pascal: CTranslate2 nie zgłasza float16
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda d: {"int8", "int8_float32", "float32"})
    assert dev.whisper_compute_type("cuda") == "int8_float32"
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda d: {"float32"})  # Maxwell
    assert dev.whisper_compute_type("cuda") == "float32"
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda d: {"int8", "int8_float32", "float32"})
    assert dev.whisper_compute_type("cpu") == "int8_float32"


def test_is_cuda_error():
    assert dev.is_cuda_error(RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
    assert dev.is_cuda_error(ValueError("Requested float16 compute type, but the target device or backend "
                                        "do not support efficient float16 computation."))
    assert dev.is_cuda_error(RuntimeError("CUDA error: no kernel image is available for execution on the device"))
    assert dev.is_cuda_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not dev.is_cuda_error(ValueError("invalid literal for int()"))
    assert not dev.is_cuda_error(OSError("Repository not found"))


def test_warning_goes_to_listener_only(capsys):
    got = []
    dev.on_warning(got.append)
    dev.warn("test")
    assert got == ["UWAGA: test"]
    assert capsys.readouterr().err == ""


def test_aligner_falls_back_to_cpu_on_cuda_oom(monkeypatch):
    """Błąd CUDA w trakcie wyrównania: to samo okno liczy się ponownie na CPU, kolejne już na CPU."""
    import torch
    from autoedit.align import Aligner

    class FakeModel:
        def __init__(self):
            self.moved_to = []

        def to(self, d):
            self.moved_to.append(d)
            return self

    calls = []

    def fake_forward(self, x):
        calls.append(self.device)
        if self.device == "cuda":
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return torch.zeros(1, len(x) // 320, 5)

    al = Aligner.__new__(Aligner)
    al.device, al.model = "cuda", FakeModel()
    monkeypatch.setattr(Aligner, "_forward", fake_forward)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    lp = al._log_probs(np.zeros(16000, dtype=np.float32))
    assert lp.shape == (50, 5)
    al._log_probs(np.zeros(16000, dtype=np.float32))
    assert calls == ["cuda", "cpu", "cpu"]
    assert al.model.moved_to == ["cpu"]
    assert dev._forced[dev.TORCH] and not dev._forced[dev.WHISPER]


def test_aligner_reraises_non_cuda_errors(monkeypatch):
    from autoedit.align import Aligner

    def boom(self, x):
        raise ValueError("zły kształt")

    al = Aligner.__new__(Aligner)
    al.device, al.model = "cuda", None
    monkeypatch.setattr(Aligner, "_forward", boom)
    with pytest.raises(ValueError, match="zły kształt"):
        al._log_probs(np.zeros(16000, dtype=np.float32))
