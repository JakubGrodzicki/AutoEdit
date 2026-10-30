from autoedit.normalize import classify, normalize
from autoedit.tables import detect_columns, load_lines


def test_normalize_basic():
    assert normalize("Cześć, świecie!") == "cześć świecie"
    assert normalize("Xal'Thorak vs. Qubit") == "xalthorak vs kubit"
    assert normalize("Mam 3 miecze") == "mam trzy miecze"
    assert normalize("(szeptem) Chodź tu {name}") == "chodź tu"
    assert normalize("Café") == "cafe"


def test_classify():
    assert classify("*śmiech*") == "nonverbal"
    assert classify("[grunt]") == "nonverbal"
    assert classify("Aaaargh!") == "vocal"
    assert classify("Hahaha") == "vocal"
    assert classify("Dobrze, idę.") == "speech"


def test_detect_columns_main_before_character():
    h = ["Nazwa skryptu w grze", "Nazwa postaci", "Czy ta postać jest głównym bohaterem", "Treść kwestii"]
    assert detect_columns(h) == {"is_main": 2, "text": 3, "script": 0, "character": 1}


def test_load_csv(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("Skrypt;Postać;Główny;Treść\nVO_001;Geralt;tak;Idę.\nVO_002;Geralt;tak;*śmiech*\n", encoding="utf-8")
    lines = load_lines(p)
    assert [l.script for l in lines] == ["VO_001", "VO_002"]
    assert lines[0].is_main and lines[0].norm == "idę"
    assert lines[1].kind == "nonverbal"


def test_for_aligner_only_touches_foreign_spellings():
    from autoedit.normalize import for_aligner
    assert for_aligner("xalthorak") == "ksaltorak"
    assert for_aligner("vaelthir") == "waeltir"
    assert for_aligner("wyzima") == "wyzima" and for_aligner("chodź ze mną") == "chodź ze mną"


def test_detect_columns_czy_mc():
    h = ["Nazwa Skryptu", "Postać", "Czy MC", "Treść kwestii"]
    assert detect_columns(h) == {"is_main": 2, "text": 3, "script": 0, "character": 1}


def test_chunk_bounds_cut_in_quiet_spot():
    import numpy as np
    from autoedit.transcribe import SR, chunk_bounds
    a = np.ones(SR * 400, dtype=np.float32) * 0.1
    a[int(SR * 183.5):int(SR * 184.0)] = 0.0  # cisza blisko celu 180 s
    b = chunk_bounds(a, chunk_s=180.0)
    assert b[0][0] == 0 and b[-1][1] == len(a)
    assert all(x[1] == y[0] for x, y in zip(b, b[1:]))
    assert 183.4 <= b[0][1] / SR <= 184.1


def test_find_holes_reports_voiced_stretch_without_words():
    import numpy as np
    from autoedit.models import Word
    from autoedit.transcribe import SR, find_holes
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(SR * 30) * 1e-3).astype(np.float32)
    a[SR * 5:SR * 8] += (0.2 * np.sin(np.arange(SR * 3) / 20)).astype(np.float32)    # mowa ze słowami
    a[SR * 15:SR * 18] += (0.2 * np.sin(np.arange(SR * 3) / 20)).astype(np.float32)  # mowa bez słów
    holes = find_holes(a, [Word("x", 5.0, 8.0)])
    assert len(holes) == 1 and 14.8 <= holes[0][0] <= 15.2 and 17.8 <= holes[0][1] <= 18.2
