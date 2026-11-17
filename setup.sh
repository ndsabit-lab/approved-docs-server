#!/usr/bin/env sh
# Core install. For OCR (photos and scanned PDFs) also run:  pip install -r requirements-ocr.txt
# For your own trained model also run:  pip install -r requirements-ml.txt
set -e
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
echo
echo "Done. Start the server with:"
echo "  . .venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000"
