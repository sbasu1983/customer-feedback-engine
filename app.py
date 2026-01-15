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

def generate_theme_insights(customer_id: int):
    res = supabase.table("reviews") \
        .select("product_handle, body, sentiment") \
        .eq("customer_id", customer_id) \
        .execute()

    if not res.data:
        return {"status": "no reviews"}

    keywords = {
        "quality": ["quality", "fabric", "stitch"],
        "delivery": ["delivery", "late", "delay"],
        "price": ["price", "expensive", "cheap"],
        "fit": ["fit", "size"]
    }

    aggregated = {}

    for r in res.data:
        product = r["product_handle"] or "all"
        text = (r["body"] or "").lower()
        sentiment = r["sentiment"]

        for theme, words in keywords.items():
            if any(w in text for w in words):
                key = (product, theme, sentiment)
                aggregated[key] = aggregated.get(key, 0) + 1

    rows = []
    for (product, theme, sentiment), count in aggregated.items():
        rows.append({
            "customer_id": customer_id,
            "product_handle": product,
            "type": sentiment,          # positive / negative / neutral
            "theme": theme,
            "count": count,
            "last_updated": datetime.utcnow().isoformat()
        })

    if rows:
        supabase.table("themes").upsert(
            rows,
            on_conflict="customer_id,product_handle,type,theme"
        ).execute()

    return {"themes_generated": len(rows)}

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

def generate_rating_metrics(customer_id: int):
    res = supabase.table("reviews") \
        .select("product_handle, rating, sentiment, body") \
        .eq("customer_id", customer_id) \
        .execute()

    if not res.data:
        return {"status": "no reviews"}

    buckets = {}

    for r in res.data:
        product = r["product_handle"] or "all"
        buckets.setdefault(product, []).append(r)

    rows = []

    for product, reviews in buckets.items():
        total = len(reviews)
        avg = sum(r["rating"] for r in reviews) / total

        pos = sum(1 for r in reviews if r["sentiment"] == "positive")
        neg = sum(1 for r in reviews if r["sentiment"] == "negative")
        neu = total - pos - neg

        alerts = []
        if neg / total > 0.3:
            alerts.append("High negative sentiment")

        if any("refund" in (r["body"] or "").lower() for r in reviews):
            alerts.append("Refund mentioned")

        rows.append({
            "customer_id": customer_id,
            "product_handle": product,
            "total_reviews": total,
            "avg_rating": round(avg, 2),
            "positive_pct": round(pos / total * 100, 2),
            "negative_pct": round(neg / total * 100, 2),
            "neutral_pct": round(neu / total * 100, 2),
            "at_risk": avg < 3.5 or neg / total > 0.3,
            "alerts": alerts,
            "last_updated": datetime.utcnow().isoformat()
        })

    supabase.table("rating_metrics").upsert(
        rows,
        on_conflict="customer_id,product_handle"
    ).execute()

    return {"metrics_generated": len(rows)}

# -------------------------------------------------
# ENDPOINTS (ALL PRESERVED & MULTI-TENANT)
# -------------------------------------------------


@app.post("/fetch-reviews")
def fetch_reviews(customer=Depends(get_customer)):
    customer_id = customer["id"]

    integrations = supabase.table("integrations") \
        .select("*") \
        .eq("customer_id", customer_id) \
        .execute()

    if not integrations.data:
        raise HTTPException(status_code=400, detail="No integrations configured")

    total_inserted = 0

    for integ in integrations.data:
        if integ["platform"] != "judgeme":
            continue

        shop_domain = integ["shop_domain"]
        token = integ["api_token"]

        raw_reviews = fetch_judgeme_reviews(shop_domain, token)

        rows = []
        for r in raw_reviews:
            body = r.get("body", "") or ""
            rows.append({
                "customer_id": customer_id,
                "product_handle": r.get("product_handle"),
                "rating": r.get("rating"),
                "body": body,
                "sentiment": sentiment(body),
                "sentiment_score": TextBlob(body).sentiment.polarity,
                "json_raw": r,
                "fetched_at": datetime.utcnow().isoformat()
            })

        if rows:
            supabase.table("reviews").upsert(rows).execute()
            total_inserted += len(rows)

    return {
        "status": "success",
        "reviews_processed": total_inserted
    }

@app.post("/insights/generate")
def generate_insights(customer=Depends(get_customer)):
    return generate_theme_insights(customer["id"])

@app.post("/generate-themes")
def generate_themes(customer=Depends(get_customer)):
    customer_id = customer["id"]

    reviews_res = supabase.table("reviews") \
        .select("product_handle, body, sentiment") \
        .eq("customer_id", customer_id) \
        .execute()

    if not reviews_res.data:
        return {"status": "no_reviews"}

    THEME_KEYWORDS = {
        "quality": ["quality", "stitch", "fabric", "material"],
        "delivery": ["delivery", "late", "delay", "shipping"],
        "price": ["price", "cost", "expensive", "cheap"],
        "fit": ["fit", "size", "tight", "loose"]
    }

    theme_counts = {}

    for r in reviews_res.data:
        text = (r["body"] or "").lower()
        product = r["product_handle"] or "all"
        sentiment = r["sentiment"]

        for theme, words in THEME_KEYWORDS.items():
            if any(w in text for w in words):
                key = (product, sentiment, theme)
                theme_counts[key] = theme_counts.get(key, 0) + 1

    rows = []
    now = datetime.utcnow().isoformat()

    for (product, sentiment, theme), count in theme_counts.items():
        rows.append({
            "customer_id": customer_id,
            "product_handle": product,
            "type": sentiment,
            "theme": theme,
            "count": count,
            "last_updated": now
        })

    if rows:
        supabase.table("themes") \
            .upsert(
                rows,
                on_conflict="customer_id,product_handle,type,theme"
            ) \
            .execute()

    return {
        "status": "success",
        "themes_generated": len(rows)
    }

@app.get("/insights/ratings")
def get_rating_metrics(customer=Depends(get_customer)):
    res = supabase.table("rating_metrics") \
        .select("*") \
        .eq("customer_id", customer["id"]) \
        .execute()

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "data": res.data
    }

@app.post("/insights/ratings/generate")
def generate_ratings(customer=Depends(get_customer)):
    return generate_rating_metrics(customer["id"])

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
def ratings_summary(customer=Depends(get_customer)):
    customer_id = customer["id"]

    res = supabase.table("reviews") \
        .select("sentiment") \
        .eq("customer_id", customer_id) \
        .execute()

    total = len(res.data)
    if total == 0:
        return {"summary": "No reviews"}

    positives = sum(1 for r in res.data if r["sentiment"] == "positive")
    negatives = sum(1 for r in res.data if r["sentiment"] == "negative")

    return {
        "total_reviews": total,
        "positive_pct": round((positives / total) * 100, 2),
        "negative_pct": round((negatives / total) * 100, 2),
    }

@app.get("/ratings/at-risk")
def ratings_at_risk(customer=Depends(get_customer)):
    customer_id = customer["id"]

    res = supabase.table("reviews") \
        .select("product_handle, rating, body, created_at") \
        .eq("customer_id", customer_id) \
        .lte("rating", 2) \
        .execute()

    return res.data

@app.get("/ratings/trends")
def ratings_trends(customer=Depends(get_customer)):
    customer_id = customer["id"]

    res = supabase.table("reviews") \
        .select("rating, created_at") \
        .eq("customer_id", customer_id) \
        .execute()

    buckets = {}
    for r in res.data:
        date = r["created_at"][:10]
        buckets.setdefault(date, []).append(r["rating"])

    return {
        d: round(sum(vals) / len(vals), 2)
        for d, vals in buckets.items()
    }

@app.get("/ratings/alerts")
def ratings_alerts(customer=Depends(get_customer)):
    customer_id = customer["id"]

    res = supabase.table("reviews") \
        .select("rating, body") \
        .eq("customer_id", customer_id) \
        .execute()

    alerts = set()

    for r in res.data:
        if r["rating"] <= 2:
            alerts.add("Low rating detected")
        if "refund" in (r["body"] or "").lower():
            alerts.add("Refund mentioned")

    return list(alerts)

@app.get("/insights/themes")
def get_themes(customer=Depends(get_customer)):
    res = supabase.table("themes") \
        .select("product_handle, type, theme, count") \
        .eq("customer_id", customer["id"]) \
        .execute()

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "data": res.data
    }


@app.get("/ratings/insights")
def ratings_insights(customer=Depends(get_customer)):
    customer_id = customer["id"]

    res = supabase.table("reviews") \
        .select("body, sentiment") \
        .eq("customer_id", customer_id) \
        .execute()

    complaints = [
        r["body"] for r in res.data
        if r["sentiment"] == "negative"
    ][:3]

    praises = [
        r["body"] for r in res.data
        if r["sentiment"] == "positive"
    ][:3]

    return {
        "top_complaints": complaints,
        "top_praises": praises
    }

@app.get("/ratings/actionable")
def ratings_actionable(customer=Depends(get_customer)):
    res = supabase.table("rating_metrics") \
        .select("negative_pct") \
        .eq("customer_id", customer["id"]) \
        .execute()

    if any(r["negative_pct"] > 30 for r in res.data):
        return {"action": "Investigate product quality issues"}

    return {"action": "No immediate action required"}

@app.get("/ratings/actionable-themes")
def ratings_actionable_themes(customer=Depends(get_customer)):
    customer_id = customer["id"]

    res = supabase.table("themes") \
        .select("theme, count") \
        .eq("customer_id", customer_id) \
        .eq("type", "negative") \
        .execute()

    actions = []

    for t in res.data:
        if t["theme"] == "quality" and t["count"] > 5:
            actions.append("Audit manufacturing quality")
        if t["theme"] == "delivery" and t["count"] > 3:
            actions.append("Review logistics partner")

    return {"recommended_actions": actions}
