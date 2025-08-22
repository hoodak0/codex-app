# trending.py
import re
from collections import defaultdict
from typing import List, Dict, Iterable, Tuple

# ---- Curated topics with synonyms ----
CURATED_TOPICS = {
    "AI": ["ai", "artificial intelligence"],
    "Gaza": ["gaza"],
    "Israel": ["israel"],
    "Ukraine": ["ukraine"],
    "Russia": ["russia"],
    "Putin": ["putin"],
    "NATO": ["nato"],
    "China": ["china"],
    "Xi": ["xi"],
    "Iran": ["iran"],
    "Taiwan": ["taiwan"],
    "Oil": ["oil", "opec"],
    "Interest Rates": ["interest rate", "interest rates", "rate hike", "rate hikes"],
    "Climate": ["climate", "heatwave", "wildfire", "drought", "hurricane"],
    "Strike": ["strike", "strikes"],
    "Protest": ["protest", "protests"],
}

def _kw_to_pattern(kw: str) -> re.Pattern:
    parts = [re.escape(p) for p in kw.split()]
    joined = r"[ \-]+".join(parts)
    return re.compile(rf"\b{joined}\b", flags=re.IGNORECASE)

TOPIC_PATTERNS = {t: [_kw_to_pattern(k) for k in kws] for t, kws in CURATED_TOPICS.items()}
ACRONYM_WHITELIST = {"AI", "EU", "UK", "US"}

def trending_curated(articles: List[Dict], min_sources: int = 3):
    topic2sources = defaultdict(set)
    topic2examples = defaultdict(list)
    for art in articles:
        title = art.get("title", "") or ""
        src = art.get("source", "unknown")
        for topic, patterns in TOPIC_PATTERNS.items():
            if any(p.search(title) for p in patterns):
                topic2sources[topic].add(src)
                if len(topic2examples[topic]) < 3:
                    topic2examples[topic].append(f"[{src}] {title}")
    items = [
        {"topic": t, "sources": len(s), "sample_headlines": topic2examples[t]}
        for t, s in topic2sources.items() if len(s) >= min_sources
    ]
    return sorted(items, key=lambda x: x["sources"], reverse=True)

# ---- Optional: Auto-entities via spaCy ----
def _lazy_spacy():
    import spacy
    return spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])

ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "PRODUCT"}
ALIASES = {
    "united states": "US", "u.s.": "US", "usa": "US",
    "united kingdom": "UK", "u.k.": "UK", "britain": "UK",
    "european union": "EU", "artificial intelligence": "AI",
}

def _canon(term: str) -> str:
    t = term.strip()
    tl = t.lower()
    return ALIASES.get(tl, t if (len(t) > 2 or t.upper() in ACRONYM_WHITELIST) else t.upper())

def trending_entities(articles: List[Dict], min_sources: int = 3):
    nlp = _lazy_spacy()
    topic2sources = defaultdict(set)
    topic2examples = defaultdict(list)
    for art in articles:
        title = (art.get("title") or "").strip()
        if not title:
            continue
        src = art.get("source", "unknown")
        ents = set()
        doc = nlp(title)
        for e in doc.ents:
            if e.label_ in ENTITY_LABELS:
                ents.add(_canon(e.text))
        for term in ents:
            topic2sources[term].add(src)
            if len(topic2examples[term]) < 3:
                topic2examples[term].append(f"[{src}] {title}")
    items = [
        {"topic": t, "sources": len(s), "sample_headlines": topic2examples[t]}
        for t, s in topic2sources.items() if len(s) >= min_sources
    ]
    return sorted(items, key=lambda x: x["sources"], reverse=True)
