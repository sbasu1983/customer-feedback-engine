from fastapi import FastAPI, Query, HTTPException
from typing import Optional
import requests
import pandas as pd
from textblob import TextBlob
import os
import time
from threading import Lock
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI(title="Customer Feedback Insights API")

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# ENV CONFIG
# -------------------------------------------------
SHOP_DOMAIN = os.getenv("SHOP_DOMAIN", "reviewtestingsb.myshopify.com")
JUDGEME_API_TOKEN = os.getenv("JUDGEME_API_TOKEN")
SHOPIFY_SHOP_DOMAIN = os.getenv("SHOPIFY_SHOP_DOMAIN", SHOP_DOMAIN)
SHOPIFY_ADMIN_TOKEN = os.getenv("SHOPIFY_ADMIN_TOKEN")
SHOPIFY_API_VERSION = "2025-04"

# -------------------------------------------------
# CACHE CONFIG
# -------------------------------------------------
PRODUCT_CACHE_TTL = 300
REVIEW_CACHE_TTL = 300

product_cache = {"handles": [], "last_updated": 0}
review_cache = {"reviews": [], "last_updated": 0}

product_lock = Lock()
review_lock = Lock()

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
def safe_review_datetime(val):
    try:
        return pd.to_datetime(val, errors="coerce")
    except Exception:
        return None

def analyze_sentiment(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    return "Neutral"

def sentiment_score(text: str) -> float:
    return round(TextBlob(text).sentiment.polarity, 2)

def detect_emotions(text: str):
    score = sentiment_score(text)
    return {
        "joy": 1 if score > 0.1 else 0,
        "anger": 1 if score < -0.1 else 0,
        "sadness": 1 if score < -0.1 else 0,
        "surprise": 0,
        "disgust": 0
    }

def sentiment_percentages(df):
    total = len(df)
    if total == 0:
        return {"positive_pct": 0, "negative_pct": 0, "neutral_pct": 0}
    return {
        "positive_pct": round((df["sentiment"] == "Positive").sum() / total * 100, 2),
        "negative_pct": round((df["sentiment"] == "Negative").sum() / total * 100, 2),
        "neutral_pct": round((df["sentiment"] == "Neutral").sum() / total * 100, 2),
    }

# -------------------------------------------------
# COMPLAINT THEMES
# -------------------------------------------------
COMPLAINT_KEYWORDS = {
    "quality": ["quality", "broken", "damaged", "defective"],
    "delivery": ["late", "delay", "shipping"],
    "price": ["price", "expensive", "cost"],
    "size_fit": ["size", "fit", "small", "large"],
    "support": ["support", "customer service"]
}

def extract_themes(reviews, keywords_map):
    themes = {}
    for r in reviews:
        text = r.get("body", "").lower()
        for theme, keywords in keywords_map.items():
            if any(k in text for k in keywords):
                themes[theme] = themes.get(theme, 0) + 1
    return dict(sorted(themes.items(), key=lambda x: x[1], reverse=True))

# -------------------------------------------------
# SHOPIFY FETCH
# -------------------------------------------------
def fetch_all_product_handles():
    url = f"https://{SHOPIFY_SHOP_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/products.json"
    headers = {"X-Shopify-Access-Token": SHOPIFY_ADMIN_TOKEN}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return [p["handle"] for p in r.json().get("products", [])]

def get_product_handles_cached():
    now = time.time()
    with product_lock:
        if not product_cache["handles"] or now - product_cache["last_updated"] > PRODUCT_CACHE_TTL:
            product_cache["handles"] = fetch_all_product_handles()
            product_cache["last_updated"] = now
    return product_cache["handles"]

# -------------------------------------------------
# JUDGE.ME FETCH
# -------------------------------------------------
def fetch_all_reviews(per_page=100):
    url = "https://judge.me/api/v1/reviews"
    params = {
        "shop_domain": SHOP_DOMAIN,
        "api_token": JUDGEME_API_TOKEN,
        "per_page": per_page,
        "page": 1,
        "published": "true"
    }
    all_reviews = []
    while True:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        batch = r.json().get("reviews", [])
        if not batch:
            break
        all_reviews.extend(batch)
        params["page"] += 1
    return all_reviews

def get_reviews_cached():
    now = time.time()
    with review_lock:
        if not review_cache["reviews"] or now - review_cache["last_updated"] > REVIEW_CACHE_TTL:
            review_cache["reviews"] = fetch_all_reviews()
            review_cache["last_updated"] = now
    return review_cache["reviews"]

# -------------------------------------------------
# HEALTH
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# SUMMARY ENGINE
# -------------------------------------------------
def summarize_reviews(product_reviews):
    rows = []
    emotion_summary = {"joy": 0, "anger": 0, "sadness": 0, "surprise": 0, "disgust": 0}

    for r in product_reviews:
        text = r.get("body")
        rating = r.get("rating")
        if not text or rating is None:
            continue

        sentiment = analyze_sentiment(text)
        emotions = detect_emotions(text)

        for k in emotion_summary:
            emotion_summary[k] += emotions[k]

        rows.append({
            "review": text,
            "rating": rating,
            "sentiment": sentiment
        })

    df = pd.DataFrame(rows)
    avg_rating = round(df["rating"].mean(), 2) if not df.empty else 0

    return {
        "total_reviews": len(df),
        "average_rating": avg_rating,
        **sentiment_percentages(df),
        "emotion_summary": emotion_summary
    }

# -------------------------------------------------
# ⭐ RATINGS – SINGLE PRODUCT
# -------------------------------------------------
@app.get("/ratings")
def ratings_summary(product_handle: str = Query(...)):
    reviews = get_reviews_cached()
    product_reviews = [
        r for r in reviews
        if r.get("product_handle") == product_handle
        and r.get("body")
        and r.get("rating") is not None
    ]

    if not product_reviews:
        raise HTTPException(status_code=404, detail="No reviews found")

    return {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "product_handle": product_handle,
        "summary": summarize_reviews(product_reviews)
    }

# -------------------------------------------------
# ⭐ RATINGS – ALL PRODUCTS
# -------------------------------------------------
@app.get("/ratings/all")
def ratings_all():
    reviews = get_reviews_cached()
    grouped = {}

    for r in reviews:
        handle = r.get("product_handle")
        if not handle or not r.get("body") or r.get("rating") is None:
            continue
        grouped.setdefault(handle, []).append(r)

    return {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "products": {
            h: summarize_reviews(rs)
            for h, rs in grouped.items()
        }
    }

# -------------------------------------------------
# ⚠️ RATINGS – AT RISK
# -------------------------------------------------
@app.get("/ratings/at-risk")
def ratings_at_risk(threshold: float = Query(0.2)):
    reviews = get_reviews_cached()
    grouped = {}

    for r in reviews:
        handle = r.get("product_handle")
        if not handle or not r.get("body") or r.get("rating") is None:
            continue
        grouped.setdefault(handle, []).append(r)

    at_risk = []

    for handle, product_reviews in grouped.items():
        summary = summarize_reviews(product_reviews)
        if summary["negative_pct"] >= threshold * 100:
            themes = extract_themes(product_reviews, COMPLAINT_KEYWORDS)
            at_risk.append({
                "product_handle": handle,
                "average_rating": summary["average_rating"],
                "negative_pct": summary["negative_pct"],
                "top_complaints": list(themes.keys())[:3]
            })

    return {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "threshold": threshold,
        "total_products_scanned": len(grouped),
        "at_risk_products": at_risk
    }

# -------------------------------------------------
# 📈 RATINGS – TRENDS
# -------------------------------------------------
@app.get("/ratings/trends")
def ratings_trends(
    product_handle: Optional[str] = Query(None),
    days: int = Query(30),
    window: int = Query(7)
):
    reviews = get_reviews_cached()
    now = pd.Timestamp.utcnow()
    cutoff_total = now - pd.Timedelta(days=days)
    cutoff_recent = now - pd.Timedelta(days=window)

    cleaned = []

    for r in reviews:
        dt = safe_review_datetime(r.get("created_at"))
        if dt is None or pd.isna(dt):
            continue
        if not (cutoff_total <= dt <= now):
            continue
        if not r.get("product_handle") or not r.get("rating") or not r.get("body"):
            continue
        cleaned.append({**r, "_dt": dt})

    if product_handle:
        cleaned = [r for r in cleaned if r["product_handle"] == product_handle]

    results = []

    for handle in {r["product_handle"] for r in cleaned}:
        product_reviews = [r for r in cleaned if r["product_handle"] == handle]

        recent = [r for r in product_reviews if r["_dt"] >= cutoff_recent]
        previous = [r for r in product_reviews if r["_dt"] < cutoff_recent]

        if not recent or not previous:
            continue

        recent_avg = round(sum(r["rating"] for r in recent) / len(recent), 2)
        prev_avg = round(sum(r["rating"] for r in previous) / len(previous), 2)
        delta = round(recent_avg - prev_avg, 2)

        trend = "stable"
        if delta > 0.2:
            trend = "improving"
        elif delta < -0.2:
            trend = "declining"

        results.append({
            "product_handle": handle,
            "current_avg_rating": recent_avg,
            "previous_avg_rating": prev_avg,
            "rating_delta": delta,
            "trend": trend,
            "risk_level": "warning" if trend == "declining" else "ok"
        })

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_window_days": days,
        "recent_window_days": window,
        "products": results
    }
