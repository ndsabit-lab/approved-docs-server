"""SQLite metadata + full-text search (FTS5) for archived documents."""
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from settings import DATA_DIR, DB_PATH


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with connect() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                reference_no TEXT,
                doc_type TEXT NOT NULL,
                approved_date TEXT NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                offices TEXT NOT NULL DEFAULT '[]',
                people TEXT NOT NULL DEFAULT '[]',
                stored_path TEXT NOT NULL,
                storage_backend TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                text_excerpt TEXT,
                predicted_type TEXT,
                predicted_confidence REAL,
                classifier TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT ''
            )"""
        )
        columns = {row["name"] for row in c.execute("PRAGMA table_info(documents)")}
        if "updated_at" not in columns:
            c.execute("ALTER TABLE documents ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")
        c.execute("UPDATE documents SET updated_at = created_at WHERE updated_at = ''")
        c.execute("CREATE INDEX IF NOT EXISTS idx_docs_folder ON documents(year, month, doc_type)")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_sha ON documents(sha256)")
        try:
            c.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5("
                "id UNINDEXED, title, reference_no, offices, people, body)"
            )
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                "This Python's SQLite was built without FTS5, which search needs. "
                "Install a newer Python from python.org."
            ) from e


def _row(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    d["offices"] = json.loads(d.get("offices") or "[]")
    d["people"] = json.loads(d.get("people") or "[]")
    if d.get("text_excerpt") and not d.get("snippet"):
        d["snippet"] = d["text_excerpt"]
    for private in ("sha256", "text_excerpt"):
        d.pop(private, None)
    return d


def find_by_hash(sha256: str) -> Optional[Dict[str, Any]]:
    with connect() as c:
        r = c.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return _row(r) if r else None


def insert(doc: Dict[str, Any], body_text: str) -> None:
    with connect() as c:
        c.execute(
            """INSERT INTO documents (id, title, reference_no, doc_type, approved_date, year, month,
                offices, people, stored_path, storage_backend, sha256, text_excerpt,
                predicted_type, predicted_confidence, classifier, created_at, updated_at)
               VALUES (:id, :title, :reference_no, :doc_type, :approved_date, :year, :month,
                :offices, :people, :stored_path, :storage_backend, :sha256, :text_excerpt,
                :predicted_type, :predicted_confidence, :classifier, :created_at, :updated_at)""",
            {**doc, "offices": json.dumps(doc["offices"]), "people": json.dumps(doc["people"])},
        )
        c.execute(
            "INSERT INTO documents_fts (id, title, reference_no, offices, people, body) VALUES (?,?,?,?,?,?)",
            (doc["id"], doc["title"], doc.get("reference_no") or "", " ; ".join(doc["offices"]),
             " ; ".join(doc["people"]), body_text),
        )


def get(doc_id: str) -> Optional[Dict[str, Any]]:
    with connect() as c:
        r = c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return _row(r) if r else None


def all_documents() -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute("SELECT * FROM documents ORDER BY updated_at DESC").fetchall()
        return [_row(row) for row in rows]


def update_metadata(doc_id: str, values: Dict[str, Any], updated_at: str) -> Optional[Dict[str, Any]]:
    with connect() as c:
        current = c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not current:
            return None
        incoming_time = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        current_time = datetime.fromisoformat(current["updated_at"].replace("Z", "+00:00"))
        if incoming_time <= current_time:
            return _row(current)

        offices_json = json.dumps(values["offices"])
        people_json = json.dumps(values["people"])
        c.execute(
            """UPDATE documents SET title = ?, reference_no = ?, doc_type = ?,
               approved_date = ?, year = ?, month = ?, offices = ?, people = ?, updated_at = ?
               WHERE id = ?""",
            (
                values["title"], values["reference_no"], values["doc_type"],
                values["approved_date"], values["year"], values["month"],
                offices_json, people_json, updated_at, doc_id,
            ),
        )
        c.execute(
            """UPDATE documents_fts SET title = ?, reference_no = ?, offices = ?, people = ?
               WHERE id = ?""",
            (values["title"], values["reference_no"] or "", " ; ".join(values["offices"]),
             " ; ".join(values["people"]), doc_id),
        )
        updated = c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return _row(updated)


def search(q: str = "", year: Optional[int] = None, month: Optional[int] = None,
           doc_type: Optional[str] = None, office: Optional[str] = None,
           person: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    select = "SELECT d.*"
    frm = " FROM documents d"
    where: List[str] = []
    params: List[Any] = []

    tokens = re.findall(r"\w+", q or "", flags=re.UNICODE)
    if tokens:
        match = " ".join(f'"{t}"*' for t in tokens)  # every word must match, prefix allowed
        select += ", snippet(documents_fts, 5, '', '', '...', 14) AS snippet"
        frm += " JOIN documents_fts ON documents_fts.id = d.id"
        where.append("documents_fts MATCH ?")
        params.append(match)
    if year is not None:
        where.append("d.year = ?"); params.append(year)
    if month is not None:
        where.append("d.month = ?"); params.append(month)
    if doc_type:
        where.append("d.doc_type = ?"); params.append(doc_type)
    if office:
        where.append("d.offices LIKE ?"); params.append(f'%"{office}"%')
    if person:
        where.append("d.people LIKE ?"); params.append(f"%{person}%")

    sql = select + frm + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY d.approved_date DESC, d.created_at DESC LIMIT ?"
    params.append(max(1, min(limit, 200)))
    with connect() as c:
        return [_row(r) for r in c.execute(sql, params).fetchall()]


def tree() -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute(
            "SELECT year, month, doc_type, COUNT(*) AS count FROM documents "
            "GROUP BY year, month, doc_type ORDER BY year DESC, month DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def training_rows() -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute(
            "SELECT text_excerpt AS text, doc_type AS label, predicted_type, predicted_confidence "
            "FROM documents WHERE text_excerpt IS NOT NULL AND length(text_excerpt) > 20"
        ).fetchall()
        return [dict(r) for r in rows]
