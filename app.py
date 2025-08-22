import streamlit as st
import pandas as pd
from collections import Counter

from trending import trending_curated, trending_entities
from news_scraper import fetch_headlines_round_robin, FEEDS
from nlp_analysis import classify_signals
from codex_insights import infer_beneficiaries, simulate_scenarios
from multi_predict import predict_labels
import os
SAFE_MODE = os.getenv("SAFE_MODE", "0") == "1"
DEFAULT_COUNT = int(os.getenv("DEFAULT_ARTICLES", 20 if SAFE_MODE else 30))


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Codex Sentinel", layout="wide")
st.title("🛰️ Codex Sentinel — Threat Intelligence Dashboard")
st.markdown(
    "Real-time analysis of global headlines.\n"
    "Dual-layer classification: NLP (Zero-shot) + Custom ML (Multi-label)"
)

with st.sidebar:
    user_selected_count = st.slider("Number of articles", 5, 200, DEFAULT_COUNT)
    per_feed_cap = st.number_input("Per-feed cap (per source fetch)", min_value=10, max_value=500, value=100, step=10)
    dedupe = st.checkbox("Dedupe similar titles", value=True)
    st.markdown("---")
    min_sources = st.slider("Trending threshold (# distinct sources)", 2, 5, 3)
    show_auto_trends = st.checkbox("Auto-entity trends (spaCy)", value=False)

if "run" not in st.session_state:
    st.session_state.run = False

run = st.button("▶️ Run analysis")
st.session_state.run = st.session_state.run or run
if not st.session_state.run:
    st.info("Adjust settings in the sidebar and click **Run analysis**.")
    st.stop()


# -----------------------------
# Fetch
# -----------------------------
with st.spinner("Fetching…"):
    articles = fetch_headlines_round_robin(
        total_limit=user_selected_count,
        feeds=FEEDS,
        per_feed_cap=per_feed_cap,
        dedupe=dedupe,
    )

st.caption(f"Fetched {len(articles)} article(s).")
if not articles:
    st.warning("No articles returned. Try increasing article count, per-feed cap, or disabling dedupe.")
    st.stop()

# Quick debug: show source mix
src_counts = pd.Series([a.get("source", "unknown") for a in articles]).value_counts()
with st.expander("Source mix", expanded=False):
    st.write(src_counts)


# -----------------------------
# Analyze (NLP + ML)
# -----------------------------
with st.spinner("Analyzing…"):
    titles = [a["title"] for a in articles]

    # Zero-shot classification: list aligned to titles
    nlp_results = classify_signals(titles)

    # ML multi-label predictions: list[list[str]]
    ml_preds = predict_labels(titles)

# Guard against shape mismatches
n = len(articles)
if len(nlp_results) != n:
    st.warning(f"NLP results length ({len(nlp_results)}) != articles length ({n}). Truncating.")
    nlp_results = nlp_results[:n]
if len(ml_preds) != n:
    st.warning(f"ML results length ({len(ml_preds)}) != articles length ({n}). Truncating.")
    ml_preds = ml_preds[:n]


# -----------------------------
# Build table for display
# -----------------------------
rows = []
for i, item in enumerate(articles):
    bens = infer_beneficiaries(item["title"])          # returns list[str]
    scens = simulate_scenarios(item["title"], k=2)     # returns list[str]

    rows.append({
        "📰 Headline": item["title"],
        "🌍 Source": item.get("source", ""),
        "🔗 Link": item.get("link", ""),
        "🤖 NLP Tag": nlp_results[i].get("label", "") if isinstance(nlp_results[i], dict) else str(nlp_results[i]),
        "🧠 ML Predictions": ", ".join(ml_preds[i]) if isinstance(ml_preds[i], (list, tuple)) else str(ml_preds[i]),
        "🎯 Who Benefits?": ", ".join(bens) if bens else "",
        "🧩 Scenario A": scens[0] if scens else "",
        "🧩 Scenario B": scens[1] if len(scens) > 1 else "",
    })

df = pd.DataFrame(rows)

st.subheader("Results")
st.dataframe(df, use_container_width=True)


# -----------------------------
# What's Trending
# -----------------------------
st.subheader(f"🔥 What’s Trending — Curated (≥ {min_sources} sources)")
cur = trending_curated(articles, min_sources=min_sources)
if not cur:
    st.caption("No curated trends yet. Try increasing article count or per-feed cap.")
else:
    for tr in cur[:10]:
        with st.container():
            st.markdown(f"**{tr['topic']}** — {tr['sources']} sources")
            for ex in tr["sample_headlines"]:
                st.markdown(f"- {ex}")

if show_auto_trends:
    st.subheader(f"🧠 What’s Trending — Auto (Entities, ≥ {min_sources} sources)")
    try:
        ent = trending_entities(articles, min_sources=min_sources)
        if not ent:
            st.caption("No auto-entity trends yet. Try increasing article count.")
        else:
            for tr in ent[:10]:
                with st.container():
                    st.markdown(f"**{tr['topic']}** — {tr['sources']} sources")
                    for ex in tr["sample_headlines"]:
                        st.markdown(f"- {ex}")
    except Exception:
        st.caption("Auto-entities require spaCy `en_core_web_sm`.")
        st.code("pip install spacy && python -m spacy download en_core_web_sm")


# -----------------------------
# Top tags (ML multi-label + NLP label)
# -----------------------------
ml_counter = Counter()
for tags in ml_preds:
    for tag in (tags or []):
        ml_counter[tag] += 1
top_ml_tags = ml_counter.most_common(10)

nlp_counter = Counter()
for r in nlp_results:
    if isinstance(r, dict):
        lab = r.get("label")
        if lab:
            nlp_counter[lab] += 1
    else:
        nlp_counter[str(r)] += 1
top_nlp_tags = nlp_counter.most_common(10)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏷️ Top ML Tags")
    if top_ml_tags:
        st.table(pd.DataFrame(top_ml_tags, columns=["Tag", "Count"]))
    else:
        st.caption("No ML tags found.")

with col2:
    st.subheader("🧪 Top NLP Labels")
    if top_nlp_tags:
        st.table(pd.DataFrame(top_nlp_tags, columns=["Label", "Count"]))
    else:
        st.caption("No NLP labels found.")


# -----------------------------
# Save snapshot
# -----------------------------
if st.button("Save snapshot to CSV"):
    df.to_csv("classified_news_snapshot.csv", index=False)
    st.success("Snapshot saved!")
