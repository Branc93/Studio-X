@echo off
title "Studio X - Installer & Launcher"
set VENV_DIR=.\venv
set FLAG_FILE=%VENV_DIR%\install_complete.flag

echo ==================================================
echo         Studio X - Local AI Interface
echo ==================================================
echo.

REM --- Step 1: Activate Virtual Environment ---
REM Create venv if it doesn't exist
if not exist "%VENV_DIR%\Scripts\activate" (
    echo [*] Creating new Python virtual environment...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b
    )
)
REM Activate the venv for this session
call "%VENV_DIR%\Scripts\activate"

REM --- Step 2: Check if installation is already complete ---
if exist "%FLAG_FILE%" (
    echo [*] Installation already complete. Skipping setup.
    goto :launch_app
)

REM --- Step 3: Perform One-Time Installation ---
echo.
echo [***] Performing first-time setup. This may take a while...
echo.

REM CRITICAL: Use --extra-index-url to search both the PyTorch server AND the main PyPI index.
echo [*] Installing all required libraries...
pip install --no-cache-dir -r requirements.txt torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo.
    echo [!!!] ERROR: A failure occurred during library installation.
    echo Please check the error messages above.
    pause
    exit /b
)

echo.
echo [*] Installation successful!
REM Create the flag file to skip this step in the future
echo.> "%FLAG_FILE%"
echo.

:launch_app
REM --- Step 4: Launch the Application ---
echo [*] Starting Studio X...
echo    - Access at: http://127.0.0.1:7860
echo    - Press CTRL+C in this window to close the app.
echo.
python image_app.py

pause
