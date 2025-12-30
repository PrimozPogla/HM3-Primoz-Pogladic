import json
import pandas as pd
import streamlit as st

@st.cache_data
def load_reviews(path="data/reviews_with_sentiment.json"):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return pd.DataFrame(payload["reviews"]), payload.get("exported_at"), payload.get("count")

st.title("Reviews Dashboard")

df, exported_at, total = load_reviews()
st.caption(f"Loaded {len(df)} reviews (exported_at: {exported_at})")

# Osnovni prikaz
if df.empty:
    st.warning("No reviews found in JSON.")
    st.stop()

# (Če imaš datume v tekstu, poskusi pretvorit)
date_col = None
for cand in ["date", "created_at", "createdAt", "timestamp"]:
    if cand in df.columns:
        date_col = cand
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# KPI
pos = (df["sentiment_label"] == "POSITIVE").sum()
neg = (df["sentiment_label"] == "NEGATIVE").sum()
c1, c2, c3 = st.columns(3)
c1.metric("Total", len(df))
c2.metric("Positive", pos)
c3.metric("Negative", neg)

# Chart
st.subheader("Sentiment distribution")
st.bar_chart(df["sentiment_label"].value_counts())

# Table
st.subheader("Reviews")
cols = [c for c in ["text", "sentiment_label", "sentiment_score", date_col] if c and c in df.columns]
st.dataframe(df[cols].sort_values(by=(date_col or "sentiment_score"), ascending=False), use_container_width=True)
