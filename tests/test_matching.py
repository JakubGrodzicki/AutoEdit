from autoedit.matching import MatchConfig, best_path, find_candidates, finalize
from autoedit.models import Line, Word
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


def test_nonverbal_never_matched():
    lines = mk_lines(["*śmiech*", "Idę na północ"])
    lines[0].kind = "nonverbal"
    res, _ = run(lines, mk_words(say("Idę na północ", 0)))
    assert res[0].status == "brak" and res[1].status in ("auto", "przegląd")
