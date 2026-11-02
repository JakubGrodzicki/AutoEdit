import numpy as np
import pytest
import soundfile as sf

from autoedit.export import cut, safe_name

SR = 48000


def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float64)


def make_session(tmp_path, seconds=10.0):
    p = tmp_path / "sesja.wav"
    sf.write(p, tone(seconds), SR, subtype="PCM_24")  # ciągły ton = "ambient"
    return p


def make_speech(tmp_path, bursts, seconds=10.0, room=0.001):
    """Cichy pokój (room) i głośne wypowiedzi (start, koniec)."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(int(seconds * SR)) * room
    for s, e in bursts:
        i = int(s * SR)
        burst = tone(e - s)
        a[i:i + len(burst)] = burst
    p = tmp_path / "sesja.wav"
    sf.write(p, a, SR, subtype="PCM_24")
    return p


def edge_zeros_ms(path):
    a, sr = sf.read(path, dtype="float64")
    nz = np.nonzero(np.abs(a) > 0)[0]
    return nz[0] / sr * 1000, (len(a) - 1 - nz[-1]) / sr * 1000


def test_full_real_padding_when_neighbours_far(tmp_path):
    src = make_speech(tmp_path, [(1.0, 1.4), (5.0, 5.5), (8.0, 8.4)])
    out = tmp_path / "x.wav"
    info = cut(src, out, 5.0, 5.5, pad=0.15)
    z0, z1 = edge_zeros_ms(out)
    assert info["pad_left"] == pytest.approx(0.15) and info["pad_right"] == pytest.approx(0.15)
    assert z0 < 5 and z1 < 5  # prawdziwe audio aż do krawędzi, brak zer
    assert abs(sf.info(out).duration - (0.5 + 0.3)) < 0.002


def test_own_decay_is_not_a_foreign_voice(tmp_path):
    src = make_speech(tmp_path, [(5.0, 5.5), (5.53, 5.56)])  # krótki dołek i powrót: wybrzmienie, nie inna wypowiedź
    out = tmp_path / "x.wav"
    info = cut(src, out, 5.0, 5.5, pad=0.15)
    assert info["pad_right"] == pytest.approx(0.15)


def test_close_neighbour_after_gives_zeros_with_faded_join(tmp_path):
    src = make_speech(tmp_path, [(5.0, 5.5), (5.6, 6.5)])
    out = tmp_path / "x.wav"
    info = cut(src, out, 5.0, 5.5, pad=0.15)
    assert info["pad_right"] < 0.1
    _, z1 = edge_zeros_ms(out)
    assert z1 > 50  # reszta paddingu to cisza
    a, sr = sf.read(out, dtype="float64")
    i = len(a) - 1 - int(round(z1 / 1000 * sr))
    assert abs(a[i]) < 0.02  # fade-out na prawdziwym audio, bez twardego skoku na styku


def test_close_neighbour_before_gives_zeros_with_faded_join(tmp_path):
    src = make_speech(tmp_path, [(3.0, 4.9), (5.0, 5.5)])
    out = tmp_path / "x.wav"
    info = cut(src, out, 5.0, 5.5, pad=0.15)
    assert info["pad_left"] < 0.1
    z0, _ = edge_zeros_ms(out)
    assert z0 > 50
    a, sr = sf.read(out, dtype="float64")
    assert abs(a[int(round(z0 / 1000 * sr))]) < 0.02


def test_continuous_sound_keeps_padding(tmp_path):
    src = make_session(tmp_path)
    out = tmp_path / "x.wav"
    info = cut(src, out, 4.9, 5.6, pad=0.15)
    assert info["pad_left"] == pytest.approx(0.15) and info["pad_right"] == pytest.approx(0.15)


def test_session_edges_are_filled_with_silence(tmp_path):
    src = make_session(tmp_path, seconds=1.0)
    out = tmp_path / "x.wav"
    info = cut(src, out, 0.05, 0.95, pad=0.15)
    assert info["pad_left"] == pytest.approx(0.05) and info["pad_right"] == pytest.approx(0.05)
    assert abs(sf.info(out).duration - (0.9 + 0.3)) < 0.002


def test_format_preserved(tmp_path):
    src = make_session(tmp_path)
    out = tmp_path / "x.wav"
    cut(src, out, 4.0, 5.0, pad=0.15)
    i = sf.info(out)
    assert i.samplerate == SR and i.subtype == "PCM_24" and i.channels == 1


def test_safe_name():
    assert safe_name("VO_001") == ("VO_001", False)
    assert safe_name('VO:001?') == ("VO_001_", True)
