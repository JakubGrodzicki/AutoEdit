import ctypes
import json
import os

from autoedit import pipeline
from autoedit.models import Candidate


def write_sheet(folder):
    p = folder / "kwestie.csv"
    p.write_text("TYTUŁ;NPC CODE NAME;MAIN HERO;LINE\nVO_001;X;NIE;Idę na północ.\nVO_002;X;NIE;Wracam po zmroku.\n",
                 encoding="utf-8")
    return p


def write_project(work, sheet, records):
    work.mkdir(parents=True, exist_ok=True)
    (work / "meta.json").write_text(json.dumps({"audio": "x.wav", "tables": [str(sheet)], "pad": 0.15}), encoding="utf-8")
    (work / "project.json").write_text(json.dumps(records), encoding="utf-8")


def record(idx, script):
    return {"idx": idx, "script": script, "status": "brak", "reasons": [], "chosen": None, "candidates": []}


def test_new_layout_keeps_only_wavs_visible(tmp_path):
    out = tmp_path / "wynik"
    out.mkdir()
    work = pipeline.prepare_work_dir(out)
    assert work == out / pipeline.WORK_DIR and work.is_dir()
    if os.name == "nt":
        assert ctypes.windll.kernel32.GetFileAttributesW(str(work)) & 0x02  # Eksplorator ma go nie pokazywać
    assert pipeline.work_dir(out) == work and not pipeline.has_project(out)
    write_project(work, write_sheet(tmp_path), [])
    assert pipeline.has_project(out) and pipeline.work_dir(out) == work


def test_old_layout_is_still_found_and_moved_on_reprocessing(tmp_path):
    out = tmp_path / "wynik"
    write_project(out, write_sheet(tmp_path), [record(1, "VO_001")])
    (out / "decisions.json").write_text("{}", encoding="utf-8")
    (out / "raport.xlsx").write_text("x", encoding="utf-8")
    (out / "VO_001.wav").write_text("x", encoding="utf-8")
    assert pipeline.work_dir(out) == out and pipeline.has_project(out)

    work = pipeline.prepare_work_dir(out)
    assert sorted(p.name for p in out.iterdir()) == [pipeline.WORK_DIR, "VO_001.wav"]
    assert pipeline.work_dir(out) == work and pipeline.has_project(out)  # przerwany przebieg nie gubi projektu
    assert (work / "decisions.json").exists()
    pipeline.archive_decisions(work)
    assert (work / "decisions.old.json").exists() and not (work / "decisions.json").exists()


def test_load_matches_results_to_sheet_by_script_not_index(tmp_path):
    """Stary projekt liczył nagłówek sekcji jako wiersz, więc jego numery są przesunięte."""
    out = tmp_path / "wynik"
    write_project(out, write_sheet(tmp_path), [record(0, "TYTUŁ"), record(1, "VO_001"), record(2, "VO_002")])
    meta, results = pipeline.load_project(out)
    assert meta["pad"] == 0.15
    assert [(r.line.idx, r.line.script, r.line.text) for r in results] == [(0, "VO_001", "Idę na północ."),
                                                                          (1, "VO_002", "Wracam po zmroku.")]


def test_old_candidates_load_without_new_fields(tmp_path):
    out = tmp_path / "wynik"
    cand = {"line_idx": 0, "w0": 0, "w1": 1, "start": 1.0, "end": 2.0, "score": 100.0, "start_cov": 1.0, "end_cov": 1.0,
            "audio_gap": 0.0, "audio_tail_gap": 0.0, "refined": [1.1, 2.1], "on_path": True, "origin": "text"}
    rec = dict(record(0, "VO_001"), status="auto", chosen=cand, candidates=[cand])
    write_project(out, write_sheet(tmp_path), [rec])
    _, results = pipeline.load_project(out)
    r = results[0]
    assert isinstance(r.chosen, Candidate) and r.chosen is r.candidates[0]
    assert r.chosen.refined == (1.1, 2.1) and not r.chosen.inside_other
