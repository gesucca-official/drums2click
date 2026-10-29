@echo off
cd /d "%~dp0"
python -m venv .venv || goto :error
.venv\Scripts\python -m pip install --upgrade pip || goto :error
.venv\Scripts\python -m pip install -r requirements.txt || goto :error
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_shortcut.ps1"
echo.
echo Done. Double-click Drums2Click (the shortcut with the metronome icon) to start.
pause
exit /b 0

:error
echo.
echo Setup failed.
pause
exit /b 1
