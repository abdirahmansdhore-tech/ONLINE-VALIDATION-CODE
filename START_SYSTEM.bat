@echo off
echo ================================================================
echo Digital Twin Validation System - Enhanced Startup (Fixed Version)
echo ================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo Python detected - OK
echo.

REM Check if we're in the correct directory
if not exist "main.py" (
    echo ERROR: main.py not found in current directory
    echo Please run this script from the DTDC_FIXED directory
    pause
    exit /b 1
)

echo Current directory - OK
echo.

REM Check if virtual environment exists
if exist "arena38env" (
    echo Activating virtual environment...
    call arena38env\Scripts\activate.bat
    if errorlevel 1 (
        echo WARNING: Failed to activate virtual environment
        echo Continuing with system Python...
    ) else (
        echo Virtual environment activated - OK
    )
    echo.
)

REM Install/update dependencies
echo Checking dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: Some dependencies may not have installed correctly
    echo System will attempt to start anyway...
)
echo.

REM Check configuration
if not exist "config\system_config.json" (
    echo ERROR: Configuration file not found
    echo Please ensure config\system_config.json exists
    pause
    exit /b 1
)

echo Configuration file found - OK
echo.

REM Start the enhanced startup script
echo Starting system with enhanced error checking...
echo.
python start_system.py

REM If the enhanced script fails, try the original
if errorlevel 1 (
    echo.
    echo Enhanced startup failed, trying original startup...
    python main.py
)

echo.
echo System has stopped.
pause

