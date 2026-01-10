from fastapi import FastAPI, Query, HTTPException, Request
import requests
import pandas as pd
from textblob import TextBlob
import os
import time
from threading import Lock
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
    allow_origins=["*"],  # later replace "*" with your Wix domain
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
# COMPLAINT & PRAISE KEYWORDS
# -------------------------------------------------
COMPLAINT_KEYWORDS = {
    "quality": ["quality", "broken", "damaged", "defective", "poor"],
    "delivery": ["late", "delay", "shipping", "delivery"],
    "price": ["price", "expensive", "cost", "overpriced"],
    "size_fit": ["size", "fit", "small", "large", "tight"],
    "packaging": ["packaging", "box", "packed"],
    "support": ["support", "customer service", "help"]
}

PRAISE_KEYWORDS = {
    "quality": ["good quality", "excellent", "durable", "well made"],
    "delivery": ["fast delivery", "quick shipping", "on time"],
    "price": ["affordable", "cheap", "worth it"],
    "fit": ["perfect fit", "comfortable", "nice fit"]
}

def extract_themes(reviews, keywords_map):
    themes = {}
    for r in reviews:
        text = r["review"].lower()
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
# SUMMARY ENGINE
# -------------------------------------------------
def summarize_reviews(product_reviews):
    rows = []
    emotion_summary = {"joy": 0, "anger": 0, "sadness": 0, "surprise": 0, "disgust": 0}

    for r in product_reviews:
        text = r["body"]
        row = {
            "review": text,
            "rating": r["rating"],
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
# ⭐ RATINGS – ALL PRODUCTS
# -------------------------------------------------
@app.get("/ratings/all")
def ratings_all():
    product_handles = get_product_handles_cached()
    all_reviews = get_reviews_cached()

    result = {}
    for handle in product_handles:
        product_reviews = [r for r in all_reviews if r.get("product_handle") == handle]
        if product_reviews:
            result[handle] = summarize_reviews(product_reviews)

    return result

# -------------------------------------------------
# ⭐ RATINGS – SINGLE PRODUCT
# -------------------------------------------------
@app.get("/ratings")
def ratings(product_handle: str = Query(...)):
    if product_handle not in get_product_handles_cached():
        raise HTTPException(status_code=404, detail="Product not found")

    all_reviews = get_reviews_cached()
    product_reviews = [r for r in all_reviews if r.get("product_handle") == product_handle]

    return {
        "product_handle": product_handle,
        "summary": summarize_reviews(product_reviews),
        "reviews": product_reviews
    }

# -------------------------------------------------
# ⭐ RATINGS – AT-RISK PRODUCTS
# -------------------------------------------------
@app.get("/ratings/at-risk")
def ratings_at_risk(threshold: float = Query(0.6, description="Weighted risk score threshold, 0-1")):
    """
    Returns products flagged as at-risk based on weighted risk score or negative sentiment.
    """
    product_handles = get_product_handles_cached()
    all_reviews = get_reviews_cached()

    at_risk_products = []

    for handle in product_handles:
        product_reviews = [r for r in all_reviews if r.get("product_handle") == handle]
        if not product_reviews:
            continue

        summary = summarize_reviews(product_reviews)
        weighted_risk_score = round(
            ((1 - summary["average_rating"]/5) * 0.7) + (summary["negative_pct"]/100 * 0.3), 2
        )

        if weighted_risk_score >= threshold:
            # Top complaints
            negative_reviews = [{"review": r["body"]} for r in product_reviews if analyze_sentiment(r["body"]) == "Negative"]
            top_complaints = list(extract_themes(negative_reviews, COMPLAINT_KEYWORDS).keys())

            at_risk_products.append({
                "product_handle": handle,
                "average_rating": summary["average_rating"],
                "negative_pct": summary["negative_pct"],
                "weighted_risk_score": weighted_risk_score,
                "top_complaints": top_complaints
            })

    # Sort by risk descending
    at_risk_products.sort(key=lambda x: x["weighted_risk_score"], reverse=True)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_products_scanned": len(product_handles),
        "at_risk_products": at_risk_products
    }
