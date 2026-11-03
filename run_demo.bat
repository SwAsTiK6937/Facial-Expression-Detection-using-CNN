@echo off
setlocal EnableDelayedExpansion
rem ==================================================================
rem  One-click launcher for the facial expression demo (Windows).
rem  First run: creates a Python environment and installs TensorFlow
rem  + OpenCV (about 600 MB, 5-10 minutes). Later runs start directly.
rem  Extra arguments are passed through, e.g.:
rem     run_demo.bat --camera 1
rem     run_demo.bat --video demo.mp4
rem ==================================================================
cd /d "%~dp0"
set "VENV=%USERPROFILE%\.venvs\fer"
set "PY=%VENV%\Scripts\python.exe"

if exist "%VENV%\.installed" goto run

echo.
echo === First-time setup ===
set "BASE="
rem TensorFlow on Windows supports Python 3.10 - 3.13 only
for %%V in (3.12 3.13 3.11 3.10) do (
    if not defined BASE (
        py -%%V -c "import sys" >nul 2>&1 && set "BASE=py -%%V"
    )
)
if not defined BASE (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "BASE=python"
)
if not defined BASE (
    echo.
    echo [ERROR] No compatible Python found. TensorFlow needs Python 3.10 - 3.13.
    echo         Install Python 3.12 from https://www.python.org/downloads/windows/
    echo         ^(tick "Add python.exe to PATH" in the installer^), then run this again.
    pause
    exit /b 1
)
echo Using: !BASE!
if not exist "%PY%" (
    !BASE! -m venv "%VENV%" || (echo [ERROR] could not create venv & pause & exit /b 1)
)
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install "tensorflow>=2.16" "opencv-python>=4.8" "numpy>=1.24"
if errorlevel 1 (
    echo.
    echo [ERROR] Package install failed. Check your internet connection and run again.
    pause
    exit /b 1
)
"%PY%" -c "import tensorflow, cv2; print('TensorFlow', tensorflow.__version__, '| OpenCV', cv2.__version__)"
if errorlevel 1 (
    echo [ERROR] Install finished but imports failed. If the error mentions a DLL, install the
    echo         Microsoft Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
    pause
    exit /b 1
)
echo ok> "%VENV%\.installed"
echo === Setup complete ===

:run
echo.
echo Starting demo... (press q in the video window to quit)
"%PY%" webcam_test.py %*
if errorlevel 1 pause
endlocal
