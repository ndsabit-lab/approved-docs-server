"""
Document-type classifier.

Two modes, chosen automatically:
  * model  - a fine-tuned Hugging Face text classifier, when MODEL_ID is set
  * rules  - a keyword baseline, so the whole system works before any model is trained

The model's labels must be exactly: procurement, financial, general_internal, general_external
(train_classifier.py produces a model like that).
"""
import math
import os
from typing import Dict

from settings import LABELS

KEYWORDS: Dict[str, list] = {
    "procurement": [
        "purchase request", "purchase order", "request for quotation", "rfq", "bidding", "bids and awards",
        "notice of award", "notice to proceed", "procurement", "supplier", "quotation", "abstract of quotations",
        "small value procurement", "delivery receipt", "inspection and acceptance", "technical specifications",
        "bac resolution", "philgeps",
    ],
    "financial": [
        "disbursement voucher", "obligation request", "payroll", "allotment", "budget", "cash advance",
        "liquidation", "reimbursement", "statement of account", "financial", "fund", "appropriation",
        "tuition", "fees", "certificate of availability of funds", "check no", "remittance", "audit",
    ],
    "general_internal": [
        "memorandum", "office order", "special order", "all faculty", "all employees", "all offices",
        "this college", "college president", "for your guidance", "for strict compliance", "effective immediately",
        "campus", "department heads", "internal",
    ],
    "general_external": [
        "dear sir", "dear madam", "dear ma'am", "very truly yours", "sincerely yours", "invitation",
        "memorandum of agreement", "memorandum of understanding", "partnership", "ched", "deped", "dbm", "coa",
        "local government unit", "your office", "we are writing", "respectfully yours",
    ],
}

_pipe = None
_loaded = False
_name = "rules-baseline"


def _load() -> None:
    global _pipe, _name, _loaded
    if _loaded:
        return
    _loaded = True
    model_id = os.environ.get("MODEL_ID")
    if not model_id:
        return
    try:
        from transformers import pipeline  # optional dependency

        _pipe = pipeline("text-classification", model=model_id, top_k=None, truncation=True, max_length=512)
        _name = f"model:{model_id}"
    except Exception as e:  # noqa: BLE001
        print(f"[classifier] Could not load MODEL_ID={model_id!r}: {e}\n[classifier] Using the rules baseline instead.")


def _softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


def _rules(text: str) -> Dict[str, float]:
    lower = text.lower()
    hits = [sum(lower.count(k) for k in KEYWORDS[label]) for label in LABELS]
    return dict(zip(LABELS, _softmax([min(h, 6) * 1.5 for h in hits])))


def classifier_name() -> str:
    _load()
    return _name


def classify(text: str) -> dict:
    """Returns {"label", "confidence", "scores", "classifier"}."""
    _load()
    scores = None
    name = _name
    if _pipe is not None and text.strip():
        out = _pipe(text[:4000])
        if out and isinstance(out[0], list):
            out = out[0]
        model_scores = {o["label"].lower(): float(o["score"]) for o in out}
        if all(label in model_scores for label in LABELS):
            scores = {label: model_scores[label] for label in LABELS}
        else:
            print(f"[classifier] Model labels {sorted(model_scores)} do not match {LABELS}. Using rules.")
            name = "rules-baseline"
    if scores is None:
        scores = _rules(text)
        name = "rules-baseline"
    label = max(scores, key=scores.get)
    return {"label": label, "confidence": scores[label], "scores": scores, "classifier": name}
