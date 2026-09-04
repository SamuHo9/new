@echo off
setlocal

set "SLICER_EXE=C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
set "SCRIPT_PATH=%~dp0run_spharm_batch.py"

echo ============================================================
echo Starting Slicer SALT SPHARM Analysis (Interactive Mode)
echo ============================================================
echo.

if not exist "%SLICER_EXE%" (
    echo [ERROR] SlicerSALT not found at: %SLICER_EXE%
    pause
    exit /b 1
)

:: Run SlicerSALT with the script. If no folder arguments are passed, it will prompt for input folder.
"%SLICER_EXE%" --no-main-window --no-splash --python-script "%SCRIPT_PATH%" %*

echo.
echo Process finished. Check spharm_debug_log.txt for details.
pause
