# nlp_analysis.py
from typing import List, Dict
from functools import lru_cache
import os

LABEL_CATEGORIES = ["Geopolitical Risk","Civil Unrest","Climate Signal","Disinformation Watch"]

# Lightweight keyword fallback (never fails)
_FALLBACK = {
    "Geopolitical Risk": ["sanction","missile","nato","election","border","ceasefire","treaty","invasion"],
    "Civil Unrest": ["protest","strike","riot","demonstration","clash","march"],
    "Climate Signal": ["climate","heatwave","wildfire","drought","flood","hurricane","emissions"],
    "Disinformation Watch": ["propaganda","misinformation","disinformation","troll","bot","fake news"],
}

def _fallback_label(t: str) -> str:
    tl = (t or "").lower()
    best, hits = "Geopolitical Risk", 0
    for lab, kws in _FALLBACK.items():
        c = sum(k in tl for k in kws)
        if c > hits:
            best, hits = lab, c
    return best

@lru_cache(maxsize=1)
def _get_zsl():
    """
    Lazy-load a small HF zero-shot model.
    If import/download fails, return None so we use the fallback.
    """
    try:
        from transformers import pipeline
        model = os.getenv("ZSL_MODEL", "valhalla/distilbart-mnli-12-6")  # small & CPU-friendly
        return pipeline("zero-shot-classification", model=model)
    except Exception:
        return None

def classify_signals(titles: List[str]) -> List[Dict]:
    zsl = _get_zsl()
    out: List[Dict] = []
    if zsl is None:
        for t in titles:
            out.append({"label": _fallback_label(t), "score": None})
        return out

    for t in titles:
        try:
            res = zsl(t, candidate_labels=LABEL_CATEGORIES, multi_label=False)
            out.append({"label": res["labels"][0], "score": float(res["scores"][0])})
        except Exception:
            out.append({"label": _fallback_label(t), "score": None})
    return out
