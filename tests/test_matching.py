from autoedit.matching import MatchConfig, best_path, find_candidates, finalize
from autoedit.models import Candidate, Line, Word
from autoedit.normalize import normalize


def mk_lines(texts):
    return [Line(i, f"VO_{i:03d}", "X", True, t, "t", norm=normalize(t)) for i, t in enumerate(texts)]


def mk_words(items):
    """items: lista (tekst, start); każde słowo trwa 0.3 s."""
    return [Word(t, s, s + 0.3, norm=normalize(t)) for t, s in items]


def say(sentence, t0):
    return [(w, t0 + 0.35 * i) for i, w in enumerate(sentence.split())]


def run(lines, words, cfg=None):
    cfg = cfg or MatchConfig()
    st, cands = find_candidates(lines, words, cfg)
    best_path(cands, cfg)
    return finalize(lines, cands, cfg), cands


LINES = ["Nie wiem czy to dobry pomysł", "Musimy ruszać zanim zapadnie zmrok",
         "Zabierz miecz i chodź ze mną", "Widziałem go przy starej bramie"]


def test_exact_in_order():
    ws = []
    for i, t in enumerate(LINES):
        ws += mk_words(say(t, 10 * i))
    res, _ = run(mk_lines(LINES), ws)
    assert [r.status for r in res] == ["auto"] * 4
    assert res[2].chosen.start == 20.0


def test_repeat_takes_picks_last_complete():
    ws = mk_words(say("Nie wiem czy to", 0) + say("Nie wiem czy to dobry pomysł", 5)
                  + say("Nie wiem czy to dobry pomysł", 12))
    res, _ = run(mk_lines(LINES[:1]), ws)
    r = res[0]
    assert r.chosen.start == 12.0 and r.chosen.complete  # domyślnie ostatnie
    assert len(r.candidates) >= 3
    assert r.status == "przegląd" and "wiele podejść" in r.reasons  # wybór należy do człowieka


def test_incomplete_only_is_review():
    ws = mk_words(say("Musimy ruszać zanim", 0))
    res, _ = run(mk_lines(LINES[1:2]), ws)
    assert res[0].status != "auto"


def test_out_of_order_and_chatter():
    ws = []
    ws += mk_words(say(LINES[0], 0))
    ws += mk_words(say("poczekaj chwilę jeszcze raz", 5))  # reżyser/aktor, bez kwestii
    ws += mk_words(say(LINES[2], 10))
    ws += mk_words(say(LINES[1], 20))  # pominięta wcześniej, wraca później
    res, _ = run(mk_lines(LINES[:3]), ws)
    assert [r.status for r in res] == ["auto"] * 3
    assert res[1].chosen.start == 20.0


def test_mangled_proper_name_still_matches():
    lines = mk_lines(["Idź do Xalthorak i powiedz im wszystko"])
    ws = mk_words(say("Idź do szaltorak i powiedz im wszystko", 0))
    res, _ = run(lines, ws)
    assert res[0].chosen is not None and res[0].chosen.score > 85


def test_short_duplicate_needs_neighbors():
    texts = ["Tak", "Idę na północ", "Tak", "Wracam po zmroku"]
    ws = []
    ws += mk_words(say("Idę na północ", 0)) + mk_words(say("Tak", 4)) + mk_words(say("Wracam po zmroku", 8))
    res, _ = run(mk_lines(texts), ws)
    # "Tak" jako kwestia #2 jest potwierdzona sąsiadami (N-1 i N+1), kwestia #0 nie
    assert res[2].status == "auto"
    assert res[0].status != "auto"


def test_short_word_fragments_do_not_evict_full_line():
    lines = mk_lines(["Nie", "Nigdy więcej nie próbuj mnie okradać nie teraz"])
    ws = mk_words(say(lines[1].text, 0))
    res, cands = run(lines, ws)
    full = next(c for c in cands if c.line_idx == 1 and c.score == 100.0)
    assert full.on_path
    assert res[1].chosen is full and res[1].status == "auto"
    assert not any(c.on_path for c in cands if c.line_idx == 0)  # dwa razy "nie" nie wypiera całej kwestii


def test_line_contained_in_next_line_gets_its_own_take():
    lines = mk_lines(["Jestem zajęty zostaw mnie w spokoju", "Nie widzisz że jestem zajęty zostaw mnie w spokoju"])
    ws = mk_words(say(lines[0].text, 0)) + mk_words(say(lines[1].text, 10))
    res, _ = run(lines, ws)
    assert res[0].chosen.start == 0.0
    assert res[1].chosen.start == 10.0


def test_identical_lines_follow_sheet_order():
    lines = mk_lines(["Idę na północ", "Zostaw mnie w spokoju", "Wracam po zmroku", "Zostaw mnie w spokoju", "Trzymaj się z daleka"])
    ws = []
    for i, ln in enumerate(lines):
        ws += mk_words(say(ln.text, 10 * i))
    res, _ = run(lines, ws)
    assert res[1].chosen.start == 10.0
    assert res[3].chosen.start == 30.0


def mk_cand(line_idx, start, end, on_path=True, score=100.0, chars=30):
    return Candidate(line_idx, 0, 0, start, end, score, on_path=on_path, chars=chars)


def test_standalone_second_take_off_path_still_needs_a_human():
    lines = mk_lines(["Nie wiem czy to dobry pomysł"])
    res = finalize(lines, [mk_cand(0, 0.0, 2.0), mk_cand(0, 5.0, 7.0, on_path=False)], MatchConfig())
    assert res[0].status == "przegląd" and "wiele podejść" in res[0].reasons


def test_fragment_of_other_line_is_not_a_take():
    lines = mk_lines(["Nie wiem czy to dobry pomysł", "Mówię ci że nie wiem czy to dobry pomysł i tyle"])
    fragment = mk_cand(0, 11.0, 13.0, on_path=False)
    cands = [mk_cand(0, 0.0, 2.0), fragment, mk_cand(1, 10.0, 14.0)]
    res = finalize(lines, cands, MatchConfig())
    assert fragment.inside_other and not cands[0].inside_other
    assert res[0].chosen is cands[0] and "wiele podejść" not in res[0].reasons


def test_default_choice_prefers_standalone_over_fragment_when_off_path():
    lines = mk_lines(["Nie wiem czy to dobry pomysł", "Mówię ci że nie wiem czy to dobry pomysł i tyle"])
    standalone = mk_cand(0, 0.0, 2.0, on_path=False)
    fragment = mk_cand(0, 11.0, 13.0, on_path=False)  # późniejszy, więc wygrałby jako "ostatni"
    res = finalize(lines, [standalone, fragment, mk_cand(1, 10.0, 14.0)], MatchConfig())
    assert res[0].chosen is standalone


def test_foreign_speech_or_long_pause_inside_sends_to_review_without_changing_choice():
    lines = mk_lines(["Nie wiem czy to dobry pomysł", "Musimy ruszać zanim zapadnie zmrok"])
    a, b = mk_cand(0, 0.0, 2.0), mk_cand(1, 5.0, 7.0)
    a.audio_span_gap, b.audio_pause = 9.0, 3.5
    res = finalize(lines, [a, b], MatchConfig())
    assert res[0].chosen is a and res[0].status == "przegląd" and any("inny tekst" in w for w in res[0].reasons)
    assert res[1].chosen is b and res[1].status == "przegląd" and any("przerwa" in w for w in res[1].reasons)


def test_nonverbal_never_matched():
    lines = mk_lines(["*śmiech*", "Idę na północ"])
    lines[0].kind = "nonverbal"
    res, _ = run(lines, mk_words(say("Idę na północ", 0)))
    assert res[0].status == "brak" and res[1].status in ("auto", "przegląd")
