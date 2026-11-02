from itertools import product

import numpy as np

from autoedit.align import GAP_PENALTY, ctc_viterbi, refine_all
from autoedit.models import Candidate, Line
from autoedit.transcribe import SR

BLANK, A, B, C = 0, 1, 2, 3
HEARD, UNHEARD = -0.01, -12.0  # log-prawdopodobieństwo: model słyszy / nie słyszy tokenu
UNSURE = -0.7  # niepewna ramka: i token, i cisza są możliwe


def emissions(frames, heard):
    """Macierz ramki x tokeny: domyślnie cisza, w `heard` {ramka: token lub (token, log-prob, log-prob ciszy)}."""
    lp = np.full((frames, 4), UNHEARD)
    lp[:, BLANK] = HEARD
    for t, what in heard.items():
        tok, value, silence = what if isinstance(what, tuple) else (what, HEARD, UNHEARD)
        lp[t, tok] = value
        lp[t, BLANK] = silence
    return lp


def test_tokens_land_where_the_model_hears_them():
    lp = emissions(40, {5: A, 10: B, 11: B, 15: C})
    assert ctc_viterbi(lp, [A, B, C], BLANK) == [(5, 6), (10, 12), (15, 16)]


def test_decoy_far_away_does_not_pull_first_token():
    lp = emissions(300, {5: A, 200: (A, -4.0, UNSURE), 201: B, 202: C})  # podobny dźwięk 4 s wcześniej
    assert ctc_viterbi(lp, [A, B, C], BLANK)[0] == (200, 201)
    assert ctc_viterbi(lp, [A, B, C], BLANK, gap_penalty=0.0)[0] == (5, 6)  # bez kary tak właśnie się działo


def test_foreign_speech_in_window_does_not_attract_tokens():
    # głośna cudza mowa w oknie: blank jest tam bardzo mało prawdopodobny, ale ramki przed kwestią nic nie kosztują
    foreign = {t: A for t in range(0, 40)}
    lp = emissions(120, {**foreign, 100: (A, -3.0, UNSURE), 101: B, 102: C})
    assert ctc_viterbi(lp, [A, B, C], BLANK)[0] == (100, 101)


def test_decoy_at_window_edge_does_not_pull_first_token():
    lp = emissions(160, {1: A, 2: A, 150: (A, -4.0, UNSURE), 151: B, 152: C})  # wabik w 100 ms luzu okna, 3 s przed kwestią
    assert ctc_viterbi(lp, [A, B, C], BLANK)[0] == (150, 151)


def test_decoy_after_the_line_does_not_pull_last_token():
    lp = emissions(300, {10: A, 11: B, 12: (C, -4.0, UNSURE), 250: C})
    assert ctc_viterbi(lp, [A, B, C], BLANK)[-1] == (12, 13)


def test_real_pause_inside_line_survives():
    lp = emissions(100, {5: A, 55: B, 56: C})  # 1 s przerwy, a pierwsza głoska nie ma innego miejsca
    assert ctc_viterbi(lp, [A, B, C], BLANK)[0] == (5, 6)


def test_repeated_token_needs_silence_between():
    lp = emissions(20, {5: A, 6: A, 7: A})
    (s0, e0), (s1, e1) = ctc_viterbi(lp, [A, A], BLANK)
    assert e0 < s1  # między dwoma identycznymi tokenami musi być cisza


def test_text_longer_than_audio_gives_nothing():
    assert ctc_viterbi(emissions(2, {}), [A, A], BLANK) == []


def path_score(lp, labels, targets, gap_penalty):
    """labels: -1 przed kwestią i po niej, -2 cisza między tokenami, inaczej numer tokenu w `targets`."""
    total = 0.0
    for t, lab in enumerate(labels):
        if lab == -1:
            total += lp[t].max()
        elif lab == -2:
            total += lp[t, BLANK] - gap_penalty
        else:
            total += lp[t, targets[lab]]
    return total


def valid_paths(n_frames, targets):
    """Wszystkie dozwolone przebiegi: tokeny po kolei, każdy co najmniej raz, ten sam znak pod rząd tylko przez ciszę."""
    states = (-1, -2) + tuple(range(len(targets)))
    for labels in product(states, repeat=n_frames):
        toks = [x for x in labels if x >= 0]
        if sorted(set(toks)) != list(range(len(targets))) or toks != sorted(toks):
            continue
        first, last = labels.index(0), n_frames - 1 - labels[::-1].index(len(targets) - 1)
        if any(x == -1 for x in labels[first:last + 1]) or any(x == -2 for x in labels[:first] + labels[last + 1:]):
            continue
        if any(targets[labels[t]] == targets[labels[t + 1]] and labels[t] != labels[t + 1]
               for t in range(n_frames - 1) if labels[t] >= 0 and labels[t + 1] >= 0):
            continue
        if any(labels[t] >= 0 and labels[t + 1] >= 0 and labels[t + 1] - labels[t] > 1 for t in range(n_frames - 1)):
            continue
        yield labels


def test_finds_the_best_path_among_all_possible():
    rng = np.random.default_rng(0)
    for targets in ([A, B], [A, A, B], [B, C, A]):
        for _ in range(4):
            raw = rng.standard_normal((7, 4)) * 3
            lp = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))
            best = max(path_score(lp, p, targets, GAP_PENALTY) for p in valid_paths(7, targets))
            spans = ctc_viterbi(lp, targets, BLANK)
            labels = [-1] * 7
            for t in range(spans[0][0], spans[-1][1]):
                labels[t] = -2
            for j, (a, b) in enumerate(spans):
                labels[a:b] = [j] * (b - a)
            assert abs(path_score(lp, labels, targets, GAP_PENALTY) - best) < 1e-9


class RecordingAligner:
    """Zamiast modelu: zapamiętuje okna i potwierdza czasy Whispera."""

    def __init__(self):
        self.windows = {}

    def align(self, _audio, norm, t0, t1, lo, hi):
        self.windows[norm] = (lo, hi)
        return t0, t1, 0.0, 0.0, 0.0, 0.0


def test_refine_all_keeps_windows_out_of_neighbouring_lines():
    lines = [Line(i, f"VO_{i}", "X", False, t, "t", norm=t) for i, t in enumerate(["jeden", "dwa", "trzy", "cztery"])]
    cands = [Candidate(0, 0, 0, 1.0, 2.0, 100.0, on_path=True), Candidate(1, 1, 1, 5.0, 6.0, 100.0, on_path=True),
             Candidate(2, 2, 2, 3.0, 4.0, 90.0), Candidate(3, 3, 3, 3.0, 4.0, 60.0)]
    rec = RecordingAligner()
    refine_all(rec, np.zeros(SR * 8, dtype=np.float32), lines, cands, min_score=70.0)
    assert rec.windows["jeden"] == (0.0, None)
    assert rec.windows["dwa"] == (1.9, None)  # koniec poprzedniego minus 0,1 s luzu
    lo, hi = rec.windows["trzy"]  # poza ścieżką: między sąsiadami
    assert abs(lo - 2.0) < 0.02 and abs(hi - 5.0) < 0.02
    assert "cztery" not in rec.windows  # poniżej progu przeglądu nie wyrównujemy
    assert all(c.refined for c in cands[:3])
