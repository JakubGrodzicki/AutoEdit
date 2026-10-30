@echo off
rem Uruchamia install.ps1 bez zmiany polityki wykonywania skryptow w systemie.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
