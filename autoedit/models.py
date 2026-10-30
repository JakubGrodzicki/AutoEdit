from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Line:
    idx: int
    script: str
    character: str
    is_main: bool
    text: str
    source: str
    kind: str = "speech"  # speech | vocal | nonverbal
    norm: str = ""


@dataclass
class Word:
    text: str
    start: float
    end: float
    norm: str = ""


@dataclass
class Candidate:
    line_idx: int
    w0: int  # indeks pierwszego słowa w strumieniu
    w1: int  # indeks ostatniego słowa (włącznie)
    start: float
    end: float
    score: float  # 0-100, podobieństwo znakowe do tekstu z Excela
    start_cov: float = 1.0
    end_cov: float = 1.0
    audio_gap: float | None = None  # forced alignment: luka log-prob na token
    audio_tail_gap: float | None = None
    refined: tuple[float, float] | None = None  # granice po forced alignment
    on_path: bool = False
    origin: str = "text"  # text | gap

    @property
    def complete(self) -> bool:
        return self.start_cov >= 0.8 and self.end_cov >= 0.8


@dataclass
class LineResult:
    line: Line
    status: str = "brak"  # auto | przegląd | brak
    chosen: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
