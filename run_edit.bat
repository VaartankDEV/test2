@echo off
setlocal EnableExtensions
REM ================================================================
REM  Anime Edit Generator - double-click this file to make an edit.
REM  1) checks your clips + music   2) renders a quick preview
REM  3) asks you, then renders the full-quality MP4 into "output"
REM ================================================================
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title Anime Edit Generator

REM ---- find Python 3 (the "py" launcher first, then "python") ----
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python 3 was not found.
    echo Install it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer. Then double-click this file again.
    goto :fail
)

REM ---- numpy is the only Python package needed; ask before installing ----
%PY% -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo The Python package "numpy" is needed for beat and motion analysis but is not installed.
    choice /C YN /M "Install it now with: %PY% -m pip install -r requirements.txt"
    if errorlevel 2 goto :fail
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Installing numpy failed. See the messages above.
        goto :fail
    )
)

echo.
echo === Step 1 of 3: checking your clips and music ===
%PY% scripts\analyze_media.py --no-plan
if errorlevel 1 goto :fail

echo.
echo === Step 2 of 3: rendering a short low-resolution preview ===
%PY% scripts\preview_edit.py --open
if errorlevel 1 goto :fail

echo.
choice /C YN /M "The preview should be playing. Render the full-quality edit now"
if errorlevel 2 goto :done

echo.
echo === Step 3 of 3: rendering the final edit (this can take a few minutes) ===
%PY% scripts\create_edit.py
if errorlevel 1 goto :fail
start "" "output"
goto :done

:fail
echo.
echo Something went wrong - please read the message above.
echo A detailed log file is in the "logs" folder.
pause
exit /b 1

:done
echo.
echo Finished. Your videos are in the "output" folder.
pause
exit /b 0
