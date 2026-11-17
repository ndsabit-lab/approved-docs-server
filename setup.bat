@echo off
REM Core install. For OCR (photos and scanned PDFs) also run:  pip install -r requirements-ocr.txt
REM For your own trained model also run:  pip install -r requirements-ml.txt
echo Creating Python environment and installing packages...
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
echo.
echo Done. Start the server with:
echo   .venv\Scripts\activate
echo   uvicorn app:app --host 0.0.0.0 --port 8000
pause
