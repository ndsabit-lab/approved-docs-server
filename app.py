"""
Approved Documents Archive server.

    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000

What it does:
  POST /analyze           upload a PDF or photos -> text, suggested type/date/offices/names (nothing saved yet)
  POST /documents         save the confirmed document into  year / month / type / file.pdf  and index it
  GET  /documents/search  full-text + filter search (date, type, office, person)
  GET  /tree              counts per year / month / type, for browsing folders
  GET  /files/{id}        the archived PDF
  GET  /config            document types, office list, classifier and storage in use
  GET  /export/training.csv  confirmed documents as (text, label) rows for training your own model
"""
import csv
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn

import classifier
import db
import extract
from settings import API_KEY, DOC_TYPES, LABELS, MAX_UPLOAD_BYTES, MONTHS, OFFICES, TMP_DIR, TYPE_FOLDERS
from storage import get_storage

storage = get_storage()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 24 * 3600  # drop abandoned uploads older than a day
    for folder in TMP_DIR.iterdir():
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
    if not API_KEY:
        print("[security] ARCHIVE_API_KEY is not set: anyone who can reach this server can read the archive. "
              "Set it before using real documents.")
    print(f"[archive] classifier={classifier.classifier_name()} storage={storage.name} ocr={extract.ocr_available()}")
    yield


app = FastAPI(title="Approved Documents Archive", lifespan=lifespan)
cors_origins = [origin.strip() for origin in os.environ.get("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"])


def require_key(x_api_key: str = Header(default=""), key: str = Query(default="")):
    if not API_KEY:
        return
    supplied = (x_api_key or key).encode()
    if not secrets.compare_digest(supplied, API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Missing or wrong API key.")


router = APIRouter(dependencies=[Depends(require_key)])


@app.get("/health")
def health():
    return {"status": "ok"}


@router.get("/config")
def get_config():
    return {
        "doc_types": [{"value": t["value"], "label": t["label"]} for t in DOC_TYPES],
        "offices": [o["name"] for o in OFFICES],
        "classifier": classifier.classifier_name(),
        "storage": storage.name,
        "ocr": extract.ocr_available(),
    }


@router.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    blobs = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"'{f.filename}' is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        blobs.append((f.filename or "page.jpg", f.content_type or "", data))

    try:
        pdf_bytes, text, pages, warnings = extract.build_pdf_and_text(blobs)
    except extract.ExtractError as e:
        raise HTTPException(400, str(e))

    upload_id = uuid.uuid4().hex
    folder = TMP_DIR / upload_id
    folder.mkdir(parents=True)
    (folder / "document.pdf").write_bytes(pdf_bytes)

    prediction = classifier.classify(text)
    fields = extract.extract_fields(text, OFFICES)
    analysis = {
        "upload_id": upload_id,
        "classifier": prediction["classifier"],
        "doc_type": prediction["label"],
        "confidence": prediction["confidence"],
        "scores": prediction["scores"],
        "pages": pages,
        "text_chars": len(text),
        "warnings": warnings,
        **fields,
    }
    (folder / "analysis.json").write_text(json.dumps({**analysis, "text": text}), encoding="utf-8")
    return analysis


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:60] or "document"


def _clean_list(values: List[str]) -> List[str]:
    out, seen = [], set()
    for v in values:
        v = re.sub(r"\s+", " ", v).strip()[:80]
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out[:30]


class SaveIn(BaseModel):
    upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    client_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    updated_at: Optional[str] = None
    doc_type: str
    title: str = Field(min_length=1, max_length=300)
    reference_no: Optional[str] = Field(default=None, max_length=120)
    approved_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    offices: List[str] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    approved_by_president: bool


class UpdateIn(BaseModel):
    updated_at: str
    doc_type: str
    title: str = Field(min_length=1, max_length=300)
    reference_no: Optional[str] = Field(default=None, max_length=120)
    approved_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    offices: List[str] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    approved_by_president: bool


def _normalize_timestamp(value: Optional[str]) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, "updated_at must be an ISO 8601 timestamp.")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _metadata_values(body: SaveIn | UpdateIn, approved: date) -> dict:
    return {
        "title": body.title.strip(),
        "reference_no": (body.reference_no or "").strip() or None,
        "doc_type": body.doc_type,
        "approved_date": approved.isoformat(),
        "year": approved.year,
        "month": approved.month,
        "offices": _clean_list(body.offices),
        "people": _clean_list(body.people),
    }


def _validate_metadata(body: SaveIn | UpdateIn) -> date:
    if not body.approved_by_president:
        raise HTTPException(400, "Confirm that the document is approved and signed by the President before saving.")
    if body.doc_type not in LABELS:
        raise HTTPException(400, "Unknown document type.")
    try:
        return date.fromisoformat(body.approved_date)
    except ValueError:
        raise HTTPException(400, "The approval date is not a real date. Use YYYY-MM-DD.")


@router.post("/documents")
def save_document(body: SaveIn):
    approved = _validate_metadata(body)
    updated_at = _normalize_timestamp(body.updated_at)

    folder = TMP_DIR / body.upload_id
    if body.client_id and db.get(body.client_id):
        shutil.rmtree(folder, ignore_errors=True)
        return db.update_metadata(body.client_id, _metadata_values(body, approved), updated_at)

    pdf_path, analysis_path = folder / "document.pdf", folder / "analysis.json"
    if not pdf_path.exists() or not analysis_path.exists():
        raise HTTPException(404, "This upload has expired. Please scan or choose the file again.")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    existing = db.find_by_hash(sha)
    if existing:
        raise HTTPException(409, f"This file is already archived as \"{existing['title']}\" ({existing['stored_path']}).")

    doc_id = body.client_id or uuid.uuid4().hex
    key = (f"{approved.year}/{approved.month:02d}-{MONTHS[approved.month - 1]}/{TYPE_FOLDERS[body.doc_type]}/"
           f"{approved.isoformat()}_{_slug(body.title)}_{doc_id[:6]}.pdf")
    storage.save(pdf_path, key)

    db.insert(
        {
            "id": doc_id, "title": body.title.strip(), "reference_no": (body.reference_no or "").strip() or None,
            "doc_type": body.doc_type, "approved_date": approved.isoformat(),
            "year": approved.year, "month": approved.month,
            "offices": _clean_list(body.offices), "people": _clean_list(body.people),
            "stored_path": key, "storage_backend": storage.name, "sha256": sha,
            "text_excerpt": (analysis.get("text") or "")[:1500],
            "predicted_type": analysis.get("doc_type"), "predicted_confidence": analysis.get("confidence"),
            "classifier": analysis.get("classifier"),
            "created_at": updated_at,
            "updated_at": updated_at,
        },
        analysis.get("text") or "",
    )
    shutil.rmtree(folder, ignore_errors=True)
    return db.get(doc_id)


@router.put("/documents/{doc_id}")
def update_document(doc_id: str, body: UpdateIn):
    approved = _validate_metadata(body)
    if not db.get(doc_id):
        raise HTTPException(404, "Document not found.")
    updated = db.update_metadata(doc_id, _metadata_values(body, approved), _normalize_timestamp(body.updated_at))
    if not updated:
        raise HTTPException(404, "Document not found.")
    return updated


@router.get("/sync/documents")
def sync_documents():
    return db.all_documents()


@router.get("/documents/search")
def search_documents(
    q: str = "", year: Optional[int] = None, month: Optional[int] = Query(default=None, ge=1, le=12),
    doc_type: Optional[str] = None, office: Optional[str] = None, person: Optional[str] = None,
    limit: int = 50,
):
    return db.search(q, year, month, doc_type, office, person, limit)


@router.get("/documents/{doc_id}")
def get_document(doc_id: str):
    doc = db.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    return doc


@router.get("/tree")
def get_tree():
    return db.tree()


@router.get("/files/{doc_id}")
def get_file(doc_id: str):
    doc = db.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    try:
        return storage.serve(doc["stored_path"], f"{doc['approved_date']} {doc['title']}.pdf")
    except FileNotFoundError:
        raise HTTPException(404, "The file is missing from storage.")


@router.get("/export/training.csv")
def export_training():
    """Confirmed documents as text,label rows: the dataset for train_classifier.py."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["text", "label"])
    for r in db.training_rows():
        writer.writerow([r["text"], r["label"]])
    return PlainTextResponse(out.getvalue(), media_type="text/csv")


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
