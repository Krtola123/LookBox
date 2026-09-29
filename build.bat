@echo off
rem Build RRIPP.exe on this PC: dist\RRIPP\RRIPP.exe (+ a self-test report).
rem Needs the same Python setup as run.bat (it creates it if missing).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    py -3.12 -m venv .venv || python -m venv .venv || goto :fail
)
.venv\Scripts\python -m pip install -r requirements-dev.txt pyinstaller>=6.6 || goto :fail
.venv\Scripts\python packaging\make_version_info.py || goto :fail
.venv\Scripts\python -m PyInstaller packaging\lookbox.spec --noconfirm --distpath dist --workpath build || goto :fail
echo.
echo Checking the build...
rem A windowed exe: "start /wait" waits for it and passes its exit code on.
start "" /wait dist\RRIPP\RRIPP.exe --selftest dist\selftest.txt
set RESULT=%errorlevel%
type dist\selftest.txt
if not "%RESULT%"=="0" goto :fail
echo.
echo Done: dist\RRIPP\RRIPP.exe  (copy the whole dist\RRIPP folder anywhere)
pause
exit /b 0

:fail
echo.
echo Build failed. See the messages above.
pause
exit /b 1
