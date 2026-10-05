@echo off
rem RU Voice Assistant - launcher for Windows (double-click this file)
setlocal
cd /d "%~dp0"
title RU Voice Assistant

rem --- 1. Find Python -------------------------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python not found. Trying to install Python 3.12 via winget...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    echo.
    echo Python installed. Please CLOSE this window and run launch.bat again.
    pause
    exit /b 0
)

rem --- 2. Create virtual environment ---------------------------------------
if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo Failed to create venv. Install Python 3.10-3.12 from python.org
        echo and tick "Add python.exe to PATH" during installation.
        pause
        exit /b 1
    )
)
set "VENVPY=.venv\Scripts\python.exe"

rem --- 3. Install dependencies (only once) ---------------------------------
if exist .venv\.installed goto RUN
echo Installing dependencies, first run takes 10-20 minutes...
"%VENVPY%" -m pip install --upgrade pip --disable-pip-version-check
"%VENVPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check
if errorlevel 1 goto FAIL
"%VENVPY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 goto FAIL
echo ok> .venv\.installed

:RUN
echo Starting GUI... A browser tab will open at http://127.0.0.1:7861
"%VENVPY%" app.py
pause
exit /b 0

:FAIL
echo.
echo Installation failed. Check your internet connection and try again.
pause
exit /b 1