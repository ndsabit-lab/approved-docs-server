"""
Turn an uploaded PDF or scan into (a) a PDF to archive, (b) its text, and (c) suggested fields.

- Digital PDFs: text is read directly (PyMuPDF).
- Scanned PDFs and photos: text is read with OCR (RapidOCR), if installed.
Everything found here is a *suggestion*; the app lets a person confirm or correct it.
"""
import io
import re
from datetime import date
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

from settings import MONTHS

MAX_OCR_PAGES = 5  # the first pages carry the title, date and type; keeps scans fast


class ExtractError(Exception):
    pass


# ---------------------------------------------------------------- OCR (optional)
_ocr_engine = None
_ocr_checked = False


def _get_ocr():
    global _ocr_engine, _ocr_checked
    if not _ocr_checked:
        _ocr_checked = True
        try:
            from rapidocr_onnxruntime import RapidOCR

            _ocr_engine = RapidOCR()
        except Exception:  # noqa: BLE001
            _ocr_engine = None
    return _ocr_engine


def ocr_available() -> bool:
    return _get_ocr() is not None


def _ocr_image_bytes(data: bytes) -> str:
    engine = _get_ocr()
    if engine is None:
        return ""
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    result, _ = engine(np.array(image)[:, :, ::-1])  # RGB -> BGR
    if not result:
        return ""
    return "\n".join(item[1] for item in result)


# ---------------------------------------------------------------- building the PDF
def _is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def _image_to_pdf(data: bytes, filename: str) -> bytes:
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg")
    ext = {"jpeg": "jpg"}.get(ext, ext)
    try:
        img = fitz.open(stream=data, filetype=ext)
        return img.convert_to_pdf()
    except Exception as e:  # noqa: BLE001
        raise ExtractError(
            f"Could not read '{filename}' as an image. Use JPG or PNG (iPhone HEIC photos are not supported)."
        ) from e


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_pdf_and_text(blobs: List[Tuple[str, str, bytes]]) -> Tuple[bytes, str, int, List[str]]:
    """blobs: (filename, content_type, data). Returns (pdf_bytes, text, page_count, warnings)."""
    warnings: List[str] = []
    if not blobs:
        raise ExtractError("No file received.")

    pdfs = [b for b in blobs if _is_pdf(b[2])]
    if pdfs and len(blobs) > 1:
        raise ExtractError("Send either one PDF or one or more photos, not both.")

    if pdfs:
        data = blobs[0][2]
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:  # noqa: BLE001
            raise ExtractError("This PDF could not be opened.") from e
        if doc.needs_pass:
            raise ExtractError("This PDF is password-protected. Remove the password and try again.")
        pages = doc.page_count
        text = "\n".join(page.get_text() for page in doc)
        if len(text.strip()) < 40:  # no real text layer: it is a scan
            parts = []
            for i, page in enumerate(doc):
                if i >= MAX_OCR_PAGES:
                    warnings.append(f"Only the first {MAX_OCR_PAGES} pages were read with OCR.")
                    break
                parts.append(_ocr_image_bytes(page.get_pixmap(dpi=150).tobytes("png")))
            text = "\n".join(parts)
            if not ocr_available():
                warnings.append(
                    "This looks like a scanned PDF but OCR is not installed on the server, so no text was read. "
                    "Install it with: pip install -r requirements-ocr.txt (or fill in the details by hand)."
                )
        return data, _clean(text), pages, warnings

    merged = fitz.open()
    parts = []
    for i, (name, _ctype, data) in enumerate(blobs):
        merged.insert_pdf(fitz.open("pdf", _image_to_pdf(data, name)))
        if i < MAX_OCR_PAGES:
            parts.append(_ocr_image_bytes(data))
    if len(blobs) > MAX_OCR_PAGES:
        warnings.append(f"Only the first {MAX_OCR_PAGES} pages were read with OCR.")
    text = _clean("\n".join(parts))
    if not ocr_available():
        warnings.append(
            "OCR is not installed on the server, so the photo text was not read. "
            "Install it with: pip install -r requirements-ocr.txt (or fill in the details by hand)."
        )
    elif len(text) < 40:
        warnings.append("Very little text was read from the photo. Try a flatter, brighter, sharper picture.")
    return merged.tobytes(), text, merged.page_count, warnings


# ---------------------------------------------------------------- suggested fields
_MONTH_NUM = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}
_MONTH_NUM.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
                   "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})
_MONTH_RE = (r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
             r"Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DATE_MDY = re.compile(rf"\b{_MONTH_RE}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DATE_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day of\s+)?{_MONTH_RE}\.?,?\s+(\d{{4}})\b", re.I)
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _find_dates(text: str) -> List[Tuple[int, date]]:
    found: List[Tuple[int, date]] = []

    def add(pos: int, y: int, m: int, d: int) -> None:
        try:
            value = date(y, m, d)
        except ValueError:
            return
        if 2000 <= value.year <= date.today().year + 1:
            found.append((pos, value))

    for m in _DATE_MDY.finditer(text):
        add(m.start(), int(m.group(3)), _MONTH_NUM[m.group(1).lower()], int(m.group(2)))
    for m in _DATE_DMY.finditer(text):
        add(m.start(), int(m.group(3)), _MONTH_NUM[m.group(2).lower()], int(m.group(1)))
    for m in _DATE_ISO.finditer(text):
        add(m.start(), int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return sorted(found)


def guess_approved_date(text: str):
    dates = _find_dates(text)
    if not dates:
        return None
    for m in re.finditer(r"\bapproved\b", text, re.I):  # a date right after "Approved" wins
        for pos, value in dates:
            if m.end() <= pos <= m.end() + 300:
                return value
    return dates[0][1]  # otherwise the first date, usually the document date


_HEADER_HINTS = ("republic of the philippines", "state college", "state university", "province of",
                 "city of", "tel.", "email", "www.", "@", "telefax", "office of the", "barangay", "brgy")


def guess_title(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:60]:
        m = re.match(r"(?i)^(subject|re|regarding)\s*[:\-]\s*(.+)$", line)
        if m:
            return m.group(2).strip()[:200]
    for line in lines[:25]:
        low = line.lower()
        if (6 <= len(line) <= 100 and line.upper() == line and re.search(r"[A-Z]{3}", line)
                and not any(h in low for h in _HEADER_HINTS)):
            return line.title()
    for line in lines[:25]:
        if 12 <= len(line) <= 120 and not any(h in line.lower() for h in _HEADER_HINTS) \
                and not re.fullmatch(r"[\d\W]+", line):
            return line
    return ""


_REF_RE = re.compile(
    r"\b((?:Memorandum Circular|Memorandum|Office Order|Special Order|Travel Order|Board Resolution|"
    r"BAC Resolution|Resolution|Purchase Request|Purchase Order|Disbursement Voucher|Executive Order)"
    r"\s*(?:No\.?|Number|#)\s*[\w\-/.]*\d[\w\-/.]*(?:\s*,?\s*(?:s\.|series of)\s*\d{4})?)", re.I)


def guess_reference(text: str):
    m = _REF_RE.search(text)
    return re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".,") if m else None


def find_offices(text: str, offices: List[Dict]) -> List[str]:
    found = []
    for office in offices:
        for name in [office["name"], *office.get("aliases", [])]:
            pattern = r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])"
            # Single words (Budget, Library...) must match exactly as written, so "budget" in
            # running text is not mistaken for the Budget Office. Multi-word names ignore case.
            if re.search(pattern, text, re.I if " " in name else 0):
                found.append(office["name"])
                break
    return found


_TITLED_NAME = re.compile(
    r"\b((?:Dr|Prof|Engr|Atty|Hon|Mr|Mrs|Ms|Sir|Madam|Dir|Rev)\.?\s+"
    r"[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]*){1,3})")
_CREDENTIAL_NAME = re.compile(
    r"\b([A-Z][A-Z.'-]+(?:\s+[A-Z][A-Z.'-]*){1,3}),\s*(?:Ph\.?D|Ed\.?D|DBA|MPA|MBA|CPA|MIT|MAEd|MSc?|LLB|RN)\b")


def find_people(text: str) -> List[str]:
    names: List[str] = []
    for m in _TITLED_NAME.finditer(text):
        names.append(re.sub(r"\s+", " ", m.group(1)).strip())
    for m in _CREDENTIAL_NAME.finditer(text):
        names.append(m.group(1).title())
    unique, seen = [], set()
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique.append(n)
    return unique[:12]


def extract_fields(text: str, offices: List[Dict]) -> Dict:
    approved = guess_approved_date(text)
    return {
        "approved_date": approved.isoformat() if approved else None,
        "title": guess_title(text),
        "reference_no": guess_reference(text),
        "offices": find_offices(text, offices),
        "people": find_people(text),
    }
