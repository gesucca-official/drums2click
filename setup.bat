@echo off
cd /d "%~dp0"
python -m venv .venv || goto :error
.venv\Scripts\python -m pip install --upgrade pip || goto :error
.venv\Scripts\python -m pip install -r requirements.txt || goto :error
echo.
echo Done. Double-click Drums2Click.bat to start.
pause
exit /b 0

:error
echo.
echo Setup failed.
pause
exit /b 1
