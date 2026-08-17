@echo off
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [1/2] Sanal ortam ve paketler kuruluyor — internet gerekir, 5-20 dk surebilir.
  where py >nul 2>&1
  if %ERRORLEVEL% equ 0 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if not exist "%VENV_PY%" (
    echo HATA: Python 3.10+ bulunamadi. https://www.python.org/downloads/
    pause
    exit /b 1
  )
  "%VENV_PY%" -m pip install -U pip
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo HATA: pip install basarisiz.
    pause
    exit /b 1
  )
)

echo [2/2] Streamlit baslatiliyor. Durdurmak icin bu pencerede Ctrl+C.
echo Tarayici acilmazsa: http://localhost:8501
"%VENV_PY%" -m streamlit run streamlit_app.py
pause
