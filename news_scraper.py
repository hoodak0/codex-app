# news_scraper.py
import feedparser
import logging
from typing import List, Dict, Optional, Iterable, Tuple
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

# ===== Default feeds (edit or add more as you like) =====
FEEDS: List[str] = [
    "https://www.aljazeera.com/xml/rss/all.xml",            # Al Jazeera
    # "https://rss.cnn.com/rss/edition_world.rss",            # CNN World
    "http://feeds.bbci.co.uk/news/world/rss.xml",           # BBC World
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",  # NYT World
    "https://www.rt.com/rss/news/",                         # Russian news (RT)
]

# --------------------------------------------------------

def _normalize_entry(entry, source_title: str, source_url: str) -> Dict:
    """Normalize an RSS entry to a consistent dict."""
    return {
        "title": (entry.get("title") or "").strip() or "No Title",
        "link": entry.get("link", "") or "",
        "summary": entry.get("summary", "") or entry.get("description", "") or "",
        "published": entry.get("published", "") or entry.get("updated", "") or "",
        "source": source_title or source_url,
    }

def _domain(u: str) -> str:
    try:
        return urlsplit(u).netloc
    except Exception:
        return ""

def _fetch_one(feed_url: str, cap: Optional[int] = None) -> List[Dict]:
    """Fetch up to `cap` items from one feed (no side effects on import)."""
    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        log.warning("Parse error for %s: %s", feed_url, e)
        return []

    if getattr(feed, "bozo", 0):
        log.warning("Bozo feed for %s: %s", feed_url, getattr(feed, "bozo_exception", ""))

    source_title = (getattr(feed, "feed", {}) or {}).get("title", "")
    entries = feed.entries or []
    if cap:
        entries = entries[:cap]
    return [_normalize_entry(e, source_title, feed_url) for e in entries]

def _round_robin(lists: List[List[Dict]]) -> Iterable[Tuple[int, Dict]]:
    """Yield items in round-robin across multiple lists until all are exhausted."""
    idx = [0] * len(lists)
    i = 0
    total = sum(len(x) for x in lists)
    if total == 0:
        return
    while total > 0:
        if lists[i] and idx[i] < len(lists[i]):
            item = lists[i][idx[i]]
            idx[i] += 1
            total -= 1
            yield i, item
        i = (i + 1) % len(lists)

def fetch_headlines_round_robin(
    total_limit: int,
    feeds: Optional[List[str]] = None,
    per_feed_cap: Optional[int] = 100,
    dedupe: bool = True,
) -> List[Dict]:
    """
    Round-robin across `feeds` until we collect `total_limit` items.

    Returns: List[Dict] with keys: title, link, summary, published, source.
    """
    urls = feeds or FEEDS
    per_source = [_fetch_one(u, per_feed_cap) for u in urls]
    log.info("Fetched counts: %s", {u: len(lst) for u, lst in zip(urls, per_source)})

    out: List[Dict] = []
    seen = set()

    for src_idx, item in _round_robin(per_source):
        key = (item["title"].lower().strip(), _domain(item["link"]))
        if dedupe and key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= total_limit:
            break

    return out

# Backward-compat single-feed helper (if old code still calls it)
def fetch_headlines(rss_url: str = FEEDS[0], limit: int = 10) -> List[Dict]:
    return fetch_headlines_round_robin(total_limit=limit, feeds=[rss_url], per_feed_cap=limit)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    items = fetch_headlines_round_robin(total_limit=12)
    for i, h in enumerate(items, 1):
        print(f"{i:02d}. [{h['source']}] {h['title']}\n    {h['link']}\n")

