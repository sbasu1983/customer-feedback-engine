from fastapi import FastAPI, Query, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from textblob import TextBlob
from datetime import datetime, timedelta
from supabase import create_client
from threading import Lock
import requests
import os
import time
import secrets
import hashlib

# -------------------------------------------------
# APP INITIALIZATION (FIXES NameError)
# -------------------------------------------------
app = FastAPI(title="Customer Feedback Insights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials not set")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------
# PER-CUSTOMER CACHES (MULTI-TENANT SAFE)
# -------------------------------------------------
review_cache: Dict[str, List[dict]] = {}
cache_timestamp: Dict[str, float] = {}
locks: Dict[str, Lock] = {}

CACHE_TTL = 3600  # 1 hour

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
def get_lock(shop_domain: str) -> Lock:
    if shop_domain not in locks:
        locks[shop_domain] = Lock()
    return locks[shop_domain]

def generate_api_key():
    raw_key = secrets.token_urlsafe(32)
    hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()
    return raw_key, hashed_key

def get_current_customer(x_api_key: str = Header(...)):
    hashed_key = hashlib.sha256(x_api_key.encode()).hexdigest()

    res = supabase.table("customers") \
        .select("id, name") \
        .eq("api_key_hash", hashed_key) \
        .execute()

    if not res.data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return res.data[0]
    
def get_customer(x_api_key: str = Header(...)):
    hashed = hashlib.sha256(x_api_key.encode()).hexdigest()

    res = supabase.table("customers").select("*")\
        .eq("api_key_hash", hashed)\
        .single().execute()

    if not res.data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return res.data
    
def sentiment(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    if polarity < -0.1:
        return "negative"
    return "neutral"

def fetch_judgeme_reviews(shop_domain: str, token: str) -> List[dict]:
    url = f"https://judge.me/api/v1/reviews?shop_domain={shop_domain}&api_token={token}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json().get("reviews", [])

def get_reviews(shop_domain: str, token: str) -> List[dict]:
    with get_lock(shop_domain):
        now = time.time()
        if (
            shop_domain in review_cache
            and now - cache_timestamp.get(shop_domain, 0) < CACHE_TTL
        ):
            return review_cache[shop_domain]

        reviews = fetch_judgeme_reviews(shop_domain, token)
        review_cache[shop_domain] = reviews
        cache_timestamp[shop_domain] = now
        return reviews

# -------------------------------------------------
# CORE PROCESSING
# -------------------------------------------------
def normalize_reviews(reviews: List[dict]) -> List[dict]:
    normalized = []
    for r in reviews:
        body = r.get("body", "") or ""
        rating = int(r.get("rating", 0))
        created = r.get("created_at", "")
        normalized.append({
            "text": body,
            "rating": rating,
            "sentiment": sentiment(body),
            "created_at": created
        })
    return normalized

# -------------------------------------------------
# ENDPOINTS (ALL PRESERVED & MULTI-TENANT)
# -------------------------------------------------

@app.post("/fetch-reviews")
def fetch_reviews(customer=Depends(get_customer)):
    customer_id = customer["id"]

@app.post("/account/api-key")
def create_api_key(customer_id: int):
    raw_key, hashed_key = generate_api_key()

    supabase.table("customers").update({
        "api_key_hash": hashed_key
    }).eq("id", customer_id).execute()

    return {
        "api_key": raw_key,
        "warning": "Store this key securely. It will not be shown again."
    }

@app.get("/ratings")
def ratings(
    shop_domain: str = Query(...),
    judgeme_token: str = Query(...)
):
    reviews = normalize_reviews(get_reviews(shop_domain, judgeme_token))
    if not reviews:
        return {"average_rating": 0}

    avg = sum(r["rating"] for r in reviews) / len(reviews)
    return {"average_rating": round(avg, 2)}

@app.get("/ratings/all")
def ratings_all(shop_domain: str, judgeme_token: str):
    return normalize_reviews(get_reviews(shop_domain, judgeme_token))

@app.get("/ratings/summary")
def ratings_summary(shop_domain: str, judgeme_token: str):
    reviews = normalize_reviews(get_reviews(shop_domain, judgeme_token))
    total = len(reviews)

    if total == 0:
        return {"summary": "No reviews"}

    positives = sum(1 for r in reviews if r["sentiment"] == "positive")
    negatives = sum(1 for r in reviews if r["sentiment"] == "negative")

    return {
        "total_reviews": total,
        "positive_pct": round(positives / total * 100, 2),
        "negative_pct": round(negatives / total * 100, 2),
    }

@app.get("/ratings/at-risk")
def ratings_at_risk(shop_domain: str, judgeme_token: str):
    reviews = normalize_reviews(get_reviews(shop_domain, judgeme_token))
    return [r for r in reviews if r["rating"] <= 2]

@app.get("/ratings/trends")
def ratings_trends(shop_domain: str, judgeme_token: str):
    reviews = normalize_reviews(get_reviews(shop_domain, judgeme_token))
    buckets = {}

    for r in reviews:
        date = r["created_at"][:10]
        buckets.setdefault(date, []).append(r["rating"])

    return {
        date: round(sum(vals) / len(vals), 2)
        for date, vals in buckets.items()
    }

@app.get("/ratings/alerts")
def ratings_alerts(shop_domain: str, judgeme_token: str):
    reviews = normalize_reviews(get_reviews(shop_domain, judgeme_token))
    alerts = []

    for r in reviews:
        if r["rating"] <= 2:
            alerts.append("Low rating detected")
        if "refund" in r["text"].lower():
            alerts.append("Refund mentioned")

    return list(set(alerts))

@app.get("/ratings/themes")
def ratings_themes(shop_domain: str, judgeme_token: str):
    reviews = normalize_reviews(get_reviews(shop_domain, judgeme_token))
    negative = {}
    positive = {}

    keywords = {
        "quality": ["quality", "stitch", "fabric"],
        "delivery": ["delivery", "late", "delay"],
        "price": ["price", "cost", "expensive"],
        "fit": ["fit", "size"]
    }

    for r in reviews:
        for theme, words in keywords.items():
            if any(w in r["text"].lower() for w in words):
                if r["sentiment"] == "negative":
                    negative[theme] = negative.get(theme, 0) + 1
                elif r["sentiment"] == "positive":
                    positive[theme] = positive.get(theme, 0) + 1

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "product_handle": "all",
        "negative_themes": negative,
        "positive_themes": positive,
    }

@app.get("/ratings/insights")
def ratings_insights(shop_domain: str, judgeme_token: str):
    reviews = normalize_reviews(get_reviews(shop_domain, judgeme_token))
    texts = [r["text"].lower() for r in reviews]

    return {
        "top_complaints": list(set(t for t in texts if "bad" in t))[:3],
        "top_praises": list(set(t for t in texts if "good" in t))[:3],
    }

@app.get("/ratings/actionable")
def ratings_actionable(shop_domain: str, judgeme_token: str):
    summary = ratings_summary(shop_domain, judgeme_token)

    if summary.get("negative_pct", 0) > 30:
        return {"action": "Investigate product quality issues"}

    return {"action": "No immediate action required"}

@app.get("/ratings/actionable-themes")
def ratings_actionable_themes(shop_domain: str, judgeme_token: str):
    themes = ratings_themes(shop_domain, judgeme_token)
    actions = []

    if themes["negative_themes"].get("quality", 0) > 5:
        actions.append("Audit manufacturing quality")

    if themes["negative_themes"].get("delivery", 0) > 3:
        actions.append("Review logistics partner")

    return {"recommended_actions": actions}
