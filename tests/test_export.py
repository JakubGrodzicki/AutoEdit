import numpy as np
import soundfile as sf

from autoedit.export import cut, neighbor_limits, safe_name
from autoedit.models import Word

SR = 48000


def make_session(tmp_path, seconds=10.0):
    t = np.arange(int(seconds * SR)) / SR
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float64)  # ciągły ton = "ambient"
    p = tmp_path / "sesja.wav"
    sf.write(p, audio, SR, subtype="PCM_24")
    return p


def leading_zeros_ms(path):
    a, sr = sf.read(path, dtype="float64")
    nz = np.nonzero(np.abs(a) > 0)[0]
    return nz[0] / sr * 1000, (len(a) - 1 - nz[-1]) / sr * 1000


def test_full_real_padding_when_neighbours_far(tmp_path):
    src = make_session(tmp_path)
    words = [Word("a", 1.0, 1.4), Word("b", 5.0, 5.5), Word("c", 8.0, 8.4)]
    out = tmp_path / "x.wav"
    info = cut(src, out, 4.9, 5.6, words, pad=0.15, raw=(5.0, 5.5))
    z0, z1 = leading_zeros_ms(out)
    assert info["pad_left"] == 0.15 and info["pad_right"] == 0.15
    assert z0 < 5 and z1 < 5  # prawdziwe audio aż do krawędzi, brak zer
    assert abs(sf.info(out).duration - (0.7 + 0.3)) < 0.002


def test_own_words_are_not_neighbours_after_refinement(tmp_path):
    """Whisper wyprzedza o ~0,5 s: własne słowa kwestii kończą się przed zaalignowanym startem."""
    src = make_session(tmp_path)
    words = [Word("x", 1.0, 1.2), Word("nie", 4.2, 4.6), Word("wiem", 4.7, 5.1), Word("y", 8.0, 8.3)]
    out = tmp_path / "x.wav"
    info = cut(src, out, 4.7, 5.5, words, pad=0.15, raw=(4.2, 5.1))
    assert info["pad_left"] == 0.15


def test_close_neighbour_gives_zeros_with_faded_join(tmp_path):
    src = make_session(tmp_path)
    words = [Word("a", 3.0, 4.95), Word("b", 5.0, 5.5)]
    out = tmp_path / "x.wav"
    info = cut(src, out, 5.0, 5.5, words, pad=0.15, raw=(5.0, 5.5))
    assert info["pad_left"] < 0.05
    z0, _ = leading_zeros_ms(out)
    assert z0 > 100  # reszta paddingu to cisza
    a, sr = sf.read(out, dtype="float64")
    i = int(round(z0 / 1000 * sr))
    assert abs(a[i]) < 0.02  # fade-in na prawdziwym audio, bez twardego skoku na styku


def test_format_preserved(tmp_path):
    src = make_session(tmp_path)
    out = tmp_path / "x.wav"
    cut(src, out, 4.0, 5.0, [], pad=0.15)
    i = sf.info(out)
    assert i.samplerate == SR and i.subtype == "PCM_24" and i.channels == 1


def test_neighbor_limits_ignore_own_words():
    words = [Word("a", 0.0, 1.0), Word("b", 2.0, 2.4), Word("c", 2.5, 2.9), Word("d", 5.0, 5.5)]
    assert neighbor_limits(words, 2.0, 2.9) == (1.0, 5.0)


def test_safe_name():
    assert safe_name("VO_001") == ("VO_001", False)
    assert safe_name('VO:001?') == ("VO_001_", True)
