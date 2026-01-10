from fastapi import FastAPI, Query, HTTPException, Request
from typing import Optional
import requests
import pandas as pd
from textblob import TextBlob
import os
import time
from threading import Lock
from datetime import datetime, timedelta
import hmac
import hashlib
import base64
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI(title="Customer Feedback Insights API")

# -------------------------------------------------
# CORS MIDDLEWARE
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

SHOPIFY_WEBHOOK_SECRET = os.getenv("SHOPIFY_WEBHOOK_SECRET")
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
# SAFE DATETIME PARSER (FIX)
# -------------------------------------------------
def safe_review_datetime(val):
    if not val:
        return None
    try:
        return pd.to_datetime(val, errors="coerce")
    except Exception:
        return None

# -------------------------------------------------
# SENTIMENT & EMOTION
# -------------------------------------------------
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
# COMPLAINT KEYWORDS
# -------------------------------------------------
COMPLAINT_KEYWORDS = {
    "quality": ["quality", "broken", "damaged", "defective", "poor"],
    "delivery": ["late", "delay", "shipping", "delivery"],
    "price": ["price", "expensive", "cost", "overpriced"],
    "size_fit": ["size", "fit", "small", "large", "tight"],
    "packaging": ["packaging", "box", "packed"],
    "support": ["support", "customer service", "help"]
}

def extract_themes(reviews, keywords_map):
    themes = {}
    for r in reviews:
        text = r.get("review", "").lower()
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
        reviews = r.json().get("reviews", [])
        if not reviews:
            break
        all_reviews.extend(reviews)
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
# SUMMARY ENGINE (SAFE)
# -------------------------------------------------
def summarize_reviews(product_reviews):
    rows = []
    emotion_summary = {"joy": 0, "anger": 0, "sadness": 0, "surprise": 0, "disgust": 0}

    for r in product_reviews:
        text = r.get("body")
        rating = r.get("rating")

        if not text or rating is None:
            continue

        row = {
            "review": text,
            "rating": rating,
            "sentiment": analyze_sentiment(text),
            "sentiment_score": sentiment_score(text),
            "emotions": detect_emotions(text),
            "length": len(text),
            "timestamp": r.get("created_at", "")
        }

        for k in emotion_summary:
            emotion_summary[k] += row["emotions"][k]

        rows.append(row)

    df = pd.DataFrame(rows)
    avg_rating = round(df["rating"].mean(), 2) if not df.empty else 0

    return {
        "total_reviews": len(df),
        "average_rating": avg_rating,
        **sentiment_percentages(df),
        "emotion_summary": emotion_summary
    }

# -------------------------------------------------
# 📈 RATINGS – TRENDS (FIXED)
# -------------------------------------------------
@app.get("/ratings/trends")
def ratings_trends(
    product_handle: Optional[str] = Query(None),
    days: int = Query(30),
    window: int = Query(7)
):
    all_reviews = get_reviews_cached()
    now = pd.Timestamp.utcnow()

    cutoff_total = now - pd.Timedelta(days=days)
    cutoff_recent = now - pd.Timedelta(days=window)

    cleaned = []

    for r in all_reviews:
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

        recent_summary = summarize_reviews(recent)
        previous_summary = summarize_reviews(previous)

        rating_delta = round(
            recent_summary["average_rating"] - previous_summary["average_rating"], 2
        )

        negative_trend = (
            "increasing"
            if recent_summary["negative_pct"] > previous_summary["negative_pct"]
            else "stable"
        )

        risk = "healthy"
        action = "No action needed"

        if rating_delta <= -0.3 and negative_trend == "increasing":
            risk = "critical"
            action = "Immediate investigation required"
        elif rating_delta < 0:
            risk = "warning"
            action = "Monitor recent complaints"

        results.append({
            "product_handle": handle,
            "current_avg_rating": recent_summary["average_rating"],
            "previous_avg_rating": previous_summary["average_rating"],
            "rating_delta": rating_delta,
            "negative_review_trend": negative_trend,
            "risk_level": risk,
            "recommended_action": action
        })

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_window_days": days,
        "recent_window_days": window,
        "products": results
    }
