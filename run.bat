@echo off
rem Double-click to run LookBox from source. First run creates .venv and installs deps.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment...
    py -3.12 -m venv .venv || python -m venv .venv || goto :fail
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements-dev.txt || goto :fail
)
.venv\Scripts\python -m lookbox %*
exit /b %errorlevel%

:fail
echo.
echo Setup failed. Make sure Python 3.12 is installed from python.org (tick "Add to PATH").
pause
exit /b 1
