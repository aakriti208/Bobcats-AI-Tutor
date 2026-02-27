@echo off
REM SimpleRAG Installation Script for Windows
REM This script automates the installation of SimpleRAG for Canvas LMS

setlocal EnableDelayedExpansion

REM Initialize variables
set "CANVAS_API_TOKEN="
set "CANVAS_COURSE_IDS="
set "CANVAS_BASE_URL=https://canvas.yourinstitution.edu/"

REM Parse command-line arguments
:parse_args
if "%~1"=="" goto check_args
if /i "%~1"=="--token" (
    set "CANVAS_API_TOKEN=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--courses" (
    set "CANVAS_COURSE_IDS=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--help" (
    goto show_usage
)
echo [ERROR] Unknown option: %~1
echo.
goto show_usage

:show_usage
echo ==========================================
echo SimpleRAG Installation Script
echo ==========================================
echo.
echo Usage:
echo   %~nx0 --token ^<API_TOKEN^> --courses ^<COURSE_IDS^>
echo.
echo Required Arguments:
echo   --token ^<API_TOKEN^>      Your Canvas API token
echo   --courses ^<COURSE_IDS^>   Comma-separated course IDs (e.g., 12345,67890)
echo.
echo Options:
echo   --help                   Show this help message
echo.
echo Example:
echo   %~nx0 --token abc123xyz --courses 12345,67890
echo.
pause
exit /b 1

:check_args
if "%CANVAS_API_TOKEN%"=="" (
    echo [ERROR] Missing required argument: --token
    echo.
    goto show_usage
)
if "%CANVAS_COURSE_IDS%"=="" (
    echo [ERROR] Missing required argument: --courses
    echo.
    goto show_usage
)

echo ==========================================
echo SimpleRAG Installation Script
echo ==========================================
echo.
echo [INFO] Configuration:
echo   Canvas URL: %CANVAS_BASE_URL%
echo   Course IDs: %CANVAS_COURSE_IDS%
set "TOKEN_PREVIEW=%CANVAS_API_TOKEN:~0,10%"
echo   API Token: %TOKEN_PREVIEW%...
echo.

REM Check for Python
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python found: %PYTHON_VERSION%

REM Create virtual environment
echo [INFO] Creating virtual environment...
if exist "venv" (
    echo [INFO] Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)
python -m venv venv
echo [SUCCESS] Virtual environment created

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
echo [SUCCESS] Virtual environment activated

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [SUCCESS] Pip upgraded

REM Install requirements
echo [INFO] Installing Python dependencies (this may take a few minutes)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [SUCCESS] Python dependencies installed

REM Setup environment file
echo [INFO] Setting up environment configuration...
if exist ".env" (
    echo [INFO] Backing up existing .env file to .env.backup
    copy /Y .env .env.backup >nul
)

REM Create .env file with provided configuration
(
echo # Canvas API Configuration
echo CANVAS_API_TOKEN=%CANVAS_API_TOKEN%
echo CANVAS_BASE_URL=%CANVAS_BASE_URL%
echo CANVAS_COURSE_IDS=%CANVAS_COURSE_IDS%
echo.
echo # Optional: Override default settings
echo EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
echo OLLAMA_MODEL=gemma:2b
echo TOP_K_RESULTS=3
echo SIMILARITY_THRESHOLD=0.5
echo.
echo # Ingestion Settings
echo INGEST_BATCH_SIZE=100
echo INGEST_MAX_WORKERS=4
echo INGEST_CONTENT_TYPES=module,page,assignment,announcement,discussion,file
echo PDF_EXTRACTION_TIMEOUT=30
echo MAX_FILE_SIZE_MB=50
echo LOG_LEVEL=INFO
) > .env

echo [SUCCESS] Created .env file with your configuration

REM Check for Ollama
echo [INFO] Checking for Ollama installation...
ollama --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Ollama is installed

    REM Check if model is installed
    echo [INFO] Checking for gemma:2b model...
    ollama list | findstr /C:"gemma:2b" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [INFO] Installing gemma:2b model (this will take a few minutes)...
        ollama pull gemma:2b
        echo [SUCCESS] gemma:2b model installed
    ) else (
        echo [SUCCESS] gemma:2b model is installed
    )
) else (
    echo [ERROR] Ollama is not installed
    echo.
    echo Ollama is required for answer generation.
    echo Installation instructions:
    echo   1. Visit https://ollama.ai
    echo   2. Download and install Ollama for Windows
    echo   3. Run: ollama serve
    echo   4. Run: ollama pull gemma:2b
)

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "data\chroma_db" mkdir data\chroma_db
if not exist "data\metadata" mkdir data\metadata
echo [SUCCESS] Directories created

echo.
echo ==========================================
echo Installation Complete!
echo ==========================================
echo.
echo Your .env file has been configured with:
echo   Canvas URL: %CANVAS_BASE_URL%
echo   Course IDs: %CANVAS_COURSE_IDS%
echo.
echo Next steps:
echo.
echo 1. Activate the virtual environment (in new terminal sessions):
echo    venv\Scripts\activate.bat
echo.
echo 2. Ingest your Canvas content:
echo    python scripts\ingest_data.py --course YOUR_COURSE_ID --full
echo    (Replace YOUR_COURSE_ID with one of: %CANVAS_COURSE_IDS%)
echo.
echo 3. Start the web interface:
echo    python app.py
echo    Then visit: http://localhost:8000
echo.
echo Note: To modify settings, edit the .env file in the project root
echo.
echo For help, see README.md or contact support
echo.
pause
