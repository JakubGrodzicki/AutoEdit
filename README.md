# AutoEdit

Automatyczne cięcie długich sesji nagraniowych dubbingu na pojedyncze pliki kwestii.

Aktor czyta kwestie z arkusza (xlsx/csv), powstaje jedno długie nagranie, a silnik gry przyjmuje wyłącznie osobne pliki
nazwane jak skrypty z arkusza. AutoEdit transkrybuje sesję, dopasowuje transkrypcję do tekstów z arkusza,
znajduje wszystkie podejścia do każdej kwestii i pozwala wybrać właściwe w prostym oknie odsłuchu.

Program działa w całości lokalnie (offline, na GPU). Nagrania i skrypty gier są zwykle poufne, więc nic nie jest wysyłane
do żadnej usługi. Modele pobiera tylko `scripts/download_models.py`.

## Jak to działa

1. **Arkusze:** jeden lub kilka plików xlsx/csv, kolumny rozpoznawane po nagłówkach (skrypt, postać, czy MC, treść).
2. **Transkrypcja:** `faster-whisper` (large-v3, polski) z czasami słów. Nagranie idzie kawałkami cięty w ciszy,
   a miejsca z głosem bez zapisanych słów są transkrybowane drugi raz osobno. Wynik jest w cache.
3. **Dopasowanie:** podobieństwo znakowe (rapidfuzz), więc zniekształcone nazwy własne nadal się dopasowują.
   Globalna ścieżka po osi czasu (programowanie dynamiczne) obsługuje powtórki, kwestie nie po kolei i rozmowy.
4. **Weryfikacja audio:** forced alignment (wav2vec2 CTC, `torchaudio.functional.forced_align`) daje dokładne granice
   oraz niezależną od Whispera ocenę, czy audio mówi ten tekst, osobno dla końcówki kwestii.
5. **Statusy:** `auto` tylko dla jednego, jednoznacznego, kompletnego take'a. Kwestie z kilkoma podejściami, krótkie,
   niepewne i niewerbalne trafiają do przeglądu. Wybór take'a zawsze należy do człowieka.
6. **GUI (PySide6):** lista kwestii, okno wyboru kandydata z odsłuchem, przeciąganymi granicami na fali i zatwierdzaniem.
7. **Eksport:** cięcie z oryginalnego pliku (ta sama częstotliwość, głębia bitowa i kanały), padding 150 ms z prawdziwego
   dźwięku (skracany, gdy blisko jest inna wypowiedź), krótkie fade'y. Nazwa pliku = nazwa skryptu.

## Wymagania

- Windows 10/11, Python 3.12 i `ffmpeg` w `PATH` (skrypt instalacyjny doinstaluje je przez winget, jeśli ich brakuje)
- karta NVIDIA z CUDA 12.x (testowane: RTX 3080 Ti 12 GB, CUDA 12.4)

## Instalacja

Jeden skrypt robi wszystko: instaluje (przez winget) Pythona 3.12 i ffmpeg, jeśli ich brakuje, tworzy `.venv`,
instaluje PyTorch z CUDA 12.8 i pozostałe pakiety, pobiera modele do cache Hugging Face
(`%USERPROFILE%\.cache\huggingface\hub`) i na końcu sprawdza, czy Whisper i wav2vec2 działają razem na GPU.

```bash
install.bat
```

albo bezpośrednio:

```bash
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Skrypt można uruchamiać wielokrotnie: pomija to, co już jest, a pobieranie modeli wznawia. Parametry:
`-CheckOnly` (tylko sprawdza wymagania), `-SkipModels`, `-SkipTest`, `-Recreate` (buduje `.venv` od nowa),
`-DesktopShortcut` (skrót na pulpicie) i `-LauncherOnly` (tylko buduje `AutoEdit.exe`, nic więcej nie instaluje).
Wymaga ok. 12 GB wolnego miejsca i połączenia z internetem. Po instalacji program działa offline.

Na końcu skrypt tworzy w folderze programu plik **`AutoEdit.exe`**. To mały launcher (ok. 24 KB, kompilowany
wbudowanym w Windows kompilatorem C#, źródło w `launcher/AutoEdit.cs`), więc nie trzeba niczego doinstalowywać.
Plik nie jest częścią repozytorium, bo powstaje przy instalacji.

<details>
<summary>Instalacja ręczna</summary>

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/download_models.py
.venv/Scripts/python.exe scripts/smoke_gpu.py
```

</details>

## Użycie

GUI: dwuklik w **`AutoEdit.exe`**. Okno terminala się nie pojawia. Komunikaty programu trafiają do `logs/autoedit.log`,
a gdy program się wysypie lub nie jest jeszcze zainstalowany, launcher pokazuje czytelny komunikat zamiast zniknąć po cichu.

Z terminala (to samo):

```bash
.venv/Scripts/python.exe -m autoedit.gui
```

Z linii poleceń (przetwarzanie bez eksportu, potem przegląd w GUI):

```bash
.venv/Scripts/python.exe -m autoedit.cli sesja.wav kwestie.xlsx -o output/projekt --no-export
.venv/Scripts/python.exe -m autoedit.gui output/projekt
```

Bez `--no-export` silnik od razu tnie kwestie ze statusem `auto`. Raport `raport.xlsx` w folderze wyniku
pokazuje status, powody i listę kandydatów dla każdej kwestii.

## Testy

```bash
.venv/Scripts/python.exe -m pytest tests
```

Testy jednostkowe nie mierzą jakości na prawdziwym nagraniu. Do tego służą `build_truth.py` i `evaluate.py`
(porównanie z ręcznie pociętymi editami), a `scripts/e2e_smoke.py` robi przejazd na syntetycznej sesji (głos Windows pl-PL).

## Status i ograniczenia

- Progi weryfikacji audio (`max_gap`, `max_tail_gap` w `MatchConfig`) są wstępne i wymagają strojenia na prawdziwych sesjach.
- Jakość mierzona dotąd na jednej sesji (jedna postać, polski).
- Kwestie niewerbalne (śmiech, warknięcia) dostają tylko propozycje z luk między sąsiednimi kwestiami, do ręcznego odsłuchu.
- Podział kwestii MC i NPC jest na razie tylko filtrem na liście i nie wpływa na dopasowanie.
- Nie wybrano jeszcze licencji.
