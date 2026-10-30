<#
.SYNOPSIS
    Instalacja AutoEdit: Python 3.12, ffmpeg, środowisko .venv, pakiety, modele i test GPU.

.DESCRIPTION
    Skrypt można uruchamiać wielokrotnie: pomija to, co już jest zainstalowane, a pobieranie modeli wznawia.
    Modele (Whisper large-v3 i polski wav2vec2, razem ok. 4,5 GB) trafiają do standardowego cache Hugging Face
    (%USERPROFILE%\.cache\huggingface\hub), skąd program je wczytuje offline.

.PARAMETER CheckOnly   Tylko sprawdza wymagania i pokazuje plan, niczego nie instaluje.
.PARAMETER SkipModels  Pomija pobieranie modeli.
.PARAMETER SkipTest    Pomija test GPU i testy jednostkowe na końcu.
.PARAMETER Recreate    Usuwa i tworzy od nowa środowisko .venv.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1
#>
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$SkipModels,
    [switch]$SkipTest,
    [switch]$Recreate
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$TorchVersion = '2.11.0'
$TorchIndex   = 'https://download.pytorch.org/whl/cu128'   # CUDA 12.8; działa na sterownikach z CUDA 12.x i nowszych
$NeedFreeGB   = 12                                          # torch ok. 3 GB + modele ok. 4,5 GB + zapas
$VenvDir      = Join-Path $PSScriptRoot '.venv'
$VenvPy       = Join-Path $VenvDir 'Scripts\python.exe'

function Step($t) { Write-Host "`n== $t" -ForegroundColor Cyan }
function Ok($t)   { Write-Host "   [OK] $t" -ForegroundColor Green }
function Warn($t) { Write-Host "   [!]  $t" -ForegroundColor Yellow }
function Fail($t) { Write-Host "`nBŁĄD: $t" -ForegroundColor Red; exit 1 }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}
function Run([string]$Exe, [string[]]$Arguments, [string]$What) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "$What (kod wyjścia $LASTEXITCODE)." }
}
function Find-Python312 {
    try {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            $p = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { return ($p | Select-Object -First 1).Trim() }
        }
    } catch { }
    $guess = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (Test-Path $guess) { return $guess }
    return $null
}
function Have-Winget { [bool](Get-Command winget -ErrorAction SilentlyContinue) }

Write-Host "AutoEdit: instalacja" -ForegroundColor White
Write-Host "Folder: $PSScriptRoot"

# ---------------------------------------------------------------- wymagania
Step 'Sprawdzanie wymagań'

$drive = (Get-Item -LiteralPath $PSScriptRoot).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB, 1)
if ($freeGB -lt $NeedFreeGB) { Warn "Wolne miejsce na dysku $($drive.Name): $freeGB GB, zalecane co najmniej $NeedFreeGB GB." }
else { Ok "Miejsce na dysku: $freeGB GB wolne" }

$gpuOk = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($gpu) { Ok "Karta NVIDIA: $gpu"; $gpuOk = $true }
}
if (-not $gpuOk) { Warn 'Nie wykryto karty NVIDIA (nvidia-smi). Program wymaga GPU z CUDA 12.x i aktualnego sterownika.' }

$py = Find-Python312
if ($py) { Ok "Python 3.12: $py" } else { Warn 'Brak Pythona 3.12 (zostanie zainstalowany przez winget).' }

$ffmpegOk = [bool](Get-Command ffmpeg -ErrorAction SilentlyContinue)
if ($ffmpegOk) { Ok 'ffmpeg w PATH' } else { Warn 'Brak ffmpeg w PATH (zostanie zainstalowany przez winget).' }

if ((-not $py -or -not $ffmpegOk) -and -not (Have-Winget)) {
    Fail 'Brakuje Pythona 3.12 lub ffmpeg, a winget jest niedostępny. Zainstaluj je ręcznie (python.org, ffmpeg.org) i uruchom skrypt ponownie.'
}

if ($CheckOnly) {
    Write-Host "`nTryb -CheckOnly: nic nie zostało zainstalowane." -ForegroundColor Cyan
    exit 0
}

# ---------------------------------------------------------------- Python i ffmpeg
if (-not $py) {
    Step 'Instalacja Pythona 3.12 (tylko dla bieżącego użytkownika)'
    Run 'winget' @('install', '--id', 'Python.Python.3.12', '--scope', 'user', '--silent',
                   '--accept-package-agreements', '--accept-source-agreements') 'Instalacja Pythona przez winget nie powiodła się'
    Refresh-Path
    $py = Find-Python312
    if (-not $py) { Fail 'Python 3.12 został zainstalowany, ale nie można go znaleźć. Uruchom skrypt ponownie w nowym oknie terminala.' }
    Ok "Python 3.12: $py"
}

if (-not $ffmpegOk) {
    Step 'Instalacja ffmpeg'
    Run 'winget' @('install', '--id', 'Gyan.FFmpeg', '-e', '--silent',
                   '--accept-package-agreements', '--accept-source-agreements') 'Instalacja ffmpeg przez winget nie powiodła się'
    Refresh-Path
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Fail 'ffmpeg został zainstalowany, ale nie ma go jeszcze w PATH. Otwórz nowe okno terminala i uruchom skrypt ponownie.'
    }
    Ok 'ffmpeg zainstalowany'
}

# ---------------------------------------------------------------- środowisko wirtualne
Step 'Środowisko wirtualne (.venv)'
if ($Recreate -and (Test-Path $VenvDir)) {
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
    Ok 'Usunięto stare .venv'
}
if (Test-Path $VenvPy) {
    $is312 = & $VenvPy -c "import sys; print(sys.version_info[:2] == (3, 12))"
    if ($is312 -ne 'True') { Fail 'Istniejące .venv nie jest zbudowane na Pythonie 3.12. Uruchom skrypt z parametrem -Recreate.' }
    Ok '.venv już istnieje'
} else {
    Run $py @('-m', 'venv', $VenvDir) 'Tworzenie .venv nie powiodło się'
    Ok 'Utworzono .venv'
}

Step 'Aktualizacja pip'
Run $VenvPy @('-m', 'pip', 'install', '--quiet', '--upgrade', 'pip') 'Aktualizacja pip nie powiodła się'
Ok 'pip aktualny'

Step "PyTorch $TorchVersion z CUDA 12.8 (ok. 3 GB, pierwszy raz może potrwać kilka minut)"
Run $VenvPy @('-m', 'pip', 'install', "torch==$TorchVersion", "torchaudio==$TorchVersion", '--index-url', $TorchIndex) `
    'Instalacja PyTorch nie powiodła się (sprawdź połączenie z internetem i uruchom skrypt ponownie)'
Ok 'PyTorch zainstalowany'

Step 'Pozostałe pakiety (requirements.txt)'
Write-Host '   (pierwsza instalacja pobiera m.in. PySide6 i faster-whisper, może potrwać kilka minut)'
Run $VenvPy @('-m', 'pip', 'install', '--quiet', '-r', (Join-Path $PSScriptRoot 'requirements.txt')) 'Instalacja pakietów nie powiodła się'
Ok 'Pakiety zainstalowane'

# ---------------------------------------------------------------- modele
if ($SkipModels) {
    Warn 'Pominięto pobieranie modeli (-SkipModels).'
} else {
    Step 'Pobieranie modeli: Whisper large-v3 i polski wav2vec2 (ok. 4,5 GB)'
    Write-Host '   Pobieranie jest ostrożne (jedno połączenie naraz) i wznawia się po przerwaniu.'
    $env:HF_HUB_DISABLE_XET = '1'
    $done = $false
    foreach ($try in 1..3) {
        & $VenvPy (Join-Path $PSScriptRoot 'scripts\download_models.py')
        if ($LASTEXITCODE -eq 0) { $done = $true; break }
        Warn "Próba $try z 3 nie powiodła się."
        if ($try -lt 3) { Start-Sleep -Seconds 10 }
    }
    if (-not $done) { Fail 'Nie udało się pobrać modeli. Sprawdź połączenie i uruchom skrypt ponownie (pobieranie zostanie wznowione).' }
    Ok "Modele w cache: $env:USERPROFILE\.cache\huggingface\hub"
}

# ---------------------------------------------------------------- weryfikacja
if ($SkipTest) {
    Warn 'Pominięto testy (-SkipTest).'
} else {
    Step 'Test GPU: Whisper i wav2vec2 w jednym procesie'
    $env:HF_HUB_OFFLINE = '1'
    # Windows PowerShell 5.1 traktuje tekst na stderr (ostrzeżenia torch) jak błąd, więc na czas testu go łagodzimy.
    $ErrorActionPreference = 'Continue'
    $out = & $VenvPy (Join-Path $PSScriptRoot 'scripts\smoke_gpu.py') 2>&1 | ForEach-Object { "$_" } | Out-String
    $smokeExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($smokeExit -ne 0 -or $out -notmatch '(?m)^OK\s*$') {
        Write-Host $out
        Fail 'Test GPU nie przeszedł. Sprawdź sterownik NVIDIA i komunikaty powyżej.'
    }
    Ok (($out -split "`r?`n" | Where-Object { $_ -match 'VRAM' } | Select-Object -First 1))

    Step 'Testy jednostkowe'
    Run $VenvPy @('-m', 'pytest', '-q', 'tests') 'Testy jednostkowe nie przeszły'
}

Write-Host "`nGotowe. Uruchomienie programu:" -ForegroundColor Green
Write-Host '   .venv\Scripts\python.exe -m autoedit.gui' -ForegroundColor White
