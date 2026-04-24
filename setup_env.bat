@echo off
REM Automated Virtual Environment Setup
REM This script creates and configures the Python virtual environment
REM Run this if start_app.bat fails to find the virtual environment

echo Creating Python virtual environment...
cd /d "%~dp0"

REM Remove old virtual environment if it exists
if exist ".venv" (
    echo Removing old virtual environment...
    rmdir /s /q .venv
)

REM Create new virtual environment
python -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Please ensure Python 3.10+ is installed and added to PATH
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -U pip setuptools wheel
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Virtual environment setup completed successfully!
echo You can now run: start_app.bat
echo.
pause
