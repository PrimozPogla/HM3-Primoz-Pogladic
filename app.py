import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Brand Reputation Monitor (2023)", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
PRODUCTS_PATH = DATA_DIR / "products.json"
TESTIMONIALS_PATH = DATA_DIR / "testimonials.json"
REVIEWS_PATH = DATA_DIR / "reviews.json"


# -----------------------------
# Helpers: load data
# -----------------------------
@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_products_df() -> pd.DataFrame:
    raw = load_json(PRODUCTS_PATH)
    if raw is None:
        return pd.DataFrame()
    return pd.DataFrame(raw)


@st.cache_data
def load_testimonials_df() -> pd.DataFrame:
    raw = load_json(TESTIMONIALS_PATH)
    if raw is None:
        return pd.DataFrame()
    return pd.DataFrame(raw)


@st.cache_data
def load_reviews_df() -> pd.DataFrame:
    raw = load_json(REVIEWS_PATH)
    if raw is None:
        return pd.DataFrame()

    df = pd.DataFrame(raw)

    # Expected columns: rid, date, rating, text
    # Make sure date is parsed and month is available
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    df["month"] = df["date"].dt.to_period("M").astype(str)  # e.g. "2023-03"
    # Keep only 2023 for this assignment scope (optional safety)
    df = df[df["month"].str.startswith("2023-", na=False)].copy()

    # Clean types
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Sort newest first (common in business dashboards)
    df = df.sort_values(by="date", ascending=False, na_position="last").reset_index(drop=True)
    return df


# -----------------------------
# Helpers: month select
# -----------------------------
def month_label(ym: str) -> str:
    # ym like "2023-03"
    dt = datetime.strptime(ym + "-01", "%Y-%m-%d")
    return dt.strftime("%B %Y")  # March 2023


def build_month_options_2023():
    months = [f"2023-{m:02d}" for m in range(1, 13)]
    labels = [month_label(m) for m in months]
    # mapping label -> ym
    return labels, months, dict(zip(labels, months))


# -----------------------------
# Sentiment pipeline
# -----------------------------
@st.cache_resource
def get_sentiment_pipeline():
    """
    Loads HF pipeline once per app session.
    If torch/transformers is missing or broken, we raise a friendly error upstream.
    """
    from transformers import pipeline

    # Force CPU to avoid GPU issues on most student setups / Render free tiers
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
        truncation=True,
    )


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: sentiment (Positive/Negative), confidence (float)
    """
    if df.empty:
        return df

    nlp = get_sentiment_pipeline()

    texts = df["text"].fillna("").astype(str).tolist()
    results = nlp(texts)

    # transformers returns e.g. [{'label': 'POSITIVE', 'score': 0.999}, ...]
    sentiments = []
    confidences = []
    for r in results:
        lbl = str(r.get("label", "")).upper()
        score = float(r.get("score", 0.0))
        sentiments.append("Positive" if "POS" in lbl else "Negative")
        confidences.append(score)

    out = df.copy()
    out["sentiment"] = sentiments
    out["confidence"] = confidences
    return out


# -----------------------------
# UI
# -----------------------------
st.title("Brand Reputation Monitor (2023)")

page = st.sidebar.radio("Navigation", ["Products", "Testimonials", "Reviews"], index=2)

if page == "Products":
    st.subheader("Products")
    df_products = load_products_df()
    if df_products.empty:
        st.warning("products.json not found or empty. Run: python scrape.py")
    else:
        st.dataframe(df_products, use_container_width=True)

elif page == "Testimonials":
    st.subheader("Testimonials")
    df_testimonials = load_testimonials_df()
    if df_testimonials.empty:
        st.warning("testimonials.json not found or empty. Run: python scrape.py")
    else:
        st.dataframe(df_testimonials, use_container_width=True)

else:
    st.subheader("Reviews (2023)")

    df_reviews = load_reviews_df()
    if df_reviews.empty:
        st.warning("reviews.json not found or empty. Run: python scrape.py")
        st.stop()

    labels, months, label_to_month = build_month_options_2023()

    # Default month: the latest month present in data (business-friendly)
    available_months = sorted(df_reviews["month"].dropna().unique().tolist())
    default_month = available_months[-1] if available_months else "2023-01"
    default_label = month_label(default_month) if default_month in months else "January 2023"
    default_index = labels.index(default_label) if default_label in labels else 0

    selected_label = st.selectbox("Select month", labels, index=default_index)
    selected_month = label_to_month[selected_label]

    df_month = df_reviews[df_reviews["month"] == selected_month].copy()

    st.write(f"Reviews in **{selected_label}**: **{len(df_month)}**")

    # Show base table first (without sentiment)
    base_cols = [c for c in ["rid", "date", "rating", "text", "month"] if c in df_month.columns]
    st.dataframe(df_month[base_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Sentiment analysis")

    col_a, col_b = st.columns([1, 2], vertical_alignment="center")
    with col_a:
        run_btn = st.button("Run Sentiment", type="primary")

    # We cache per month in session_state so you don’t recompute every time
    cache_key = f"sentiment_{selected_month}"
    if run_btn:
        try:
            with st.spinner("Running transformer sentiment analysis..."):
                st.session_state[cache_key] = run_sentiment(df_month)
        except Exception as e:
            st.error(
                "Sentiment model failed to load/run. "
                "This is usually caused by missing/incorrect packages (torch/transformers) "
                "or a broken environment.\n\n"
                f"Technical details: {e}"
            )

    if cache_key in st.session_state:
        df_sent = st.session_state[cache_key]

        # Show enriched table
        sent_cols = [c for c in ["rid", "date", "rating", "text", "sentiment", "confidence"] if c in df_sent.columns]
        st.dataframe(df_sent[sent_cols], use_container_width=True, hide_index=True)

        # Visualization
        st.divider()
        st.subheader("Monthly sentiment summary")

        counts = df_sent["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
        st.bar_chart(counts.set_index("sentiment"))

        avg_conf = float(df_sent["confidence"].mean()) if "confidence" in df_sent.columns and len(df_sent) else 0.0
        st.metric("Average confidence", f"{avg_conf:.3f}")
    else:
        st.info("Click **Run Sentiment** to compute sentiment and show the chart.")

#.\.venv\Scripts\activate
# streamlit run app.py
