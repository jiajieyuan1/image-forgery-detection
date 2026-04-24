@echo off
REM Start from the directory where this script is located
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found!
    echo Please run setup_env.bat first or rebuild with:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    echo This may happen if Python versions don't match.
    echo Please rebuild the environment using setup_env.bat
    pause
    exit /b 1
)

REM Start the Streamlit application
streamlit run app.py
pause
