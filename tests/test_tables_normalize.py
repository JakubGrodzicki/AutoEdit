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


def test_section_header_rows_are_not_lines(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("TYTUŁ;NPC CODE NAME;MAIN HERO;LINE\nVO_001;Gracz;NIE;Idę.\nTYTUŁ 2;NPC CODE NAME;MAIN HERO;LINE\n"
                 "VO_002;Gracz;TAK;Wracam.\n", encoding="utf-8")
    lines = load_lines(p)
    assert [(l.idx, l.script) for l in lines] == [(0, "VO_001"), (1, "VO_002")]
    assert not lines[0].is_main and lines[1].is_main


def test_select_role_keeps_one_actor_and_renumbers(tmp_path):
    import pytest
    from autoedit.tables import select_role
    p = tmp_path / "a.csv"
    p.write_text("Skrypt;Postać;Czy MC;Treść\nVO_001;Kupiec;NIE;Witaj.\nVO_002;Gracz;TAK;Tak.\n"
                 "VO_003;Kupiec;NIE;Coś kupujesz?\n", encoding="utf-8")
    lines = load_lines(p)
    assert select_role(lines, None) is lines
    # kolejna kwestia NPC po przerwie na MC musi mieć idx + 1, inaczej ścieżka nie da premii za kolejność
    assert [(l.idx, l.script) for l in select_role(lines, "npc")] == [(0, "VO_001"), (1, "VO_003")]
    assert [(l.idx, l.script) for l in select_role(lines, "main")] == [(0, "VO_002")]
    assert [l.idx for l in lines] == [0, 1, 2]  # oryginalne wiersze bez zmian
    with pytest.raises(ValueError):
        select_role([l for l in lines if not l.is_main], "main")


def test_speech_map_keeps_first_word_after_pause_in_its_chunk():
    from autoedit.transcribe import SR, SpeechMap
    m = SpeechMap([{"start": 1 * SR, "end": 3 * SR}, {"start": 8 * SR, "end": 10 * SR}])
    assert m.restore(0.5, 1.0) == (1.5, 2.0)  # przed pierwszym fragmentem wycięto 1 s ciszy
    assert m.restore(1.5, 2.0) == (2.5, 3.0)  # koniec dokładnie na granicy: nadal pierwszy fragment
    assert m.restore(2.5, 2.9) == (8.5, 8.9)  # przed drugim wycięto łącznie 6 s
    # słowo zaczyna się na końcu pierwszego fragmentu, kończy w drugim: należy do drugiego (środek wskazałby pierwszy),
    # a początek nie może wypaść przed drugim fragmentem
    assert m.restore(1.8, 2.1) == (8.0, 8.1)
    assert m.restore(3.9, 4.2) == (9.9, 10.2)  # poza ostatnim fragmentem: ostatnie przesunięcie


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
