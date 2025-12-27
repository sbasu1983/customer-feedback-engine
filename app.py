from fastapi import FastAPI, Query, HTTPException
import requests
import pandas as pd
from textblob import TextBlob
import os
import time
from threading import Lock

# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI(title="Customer Feedback Insights API")

# -------------------------------------------------
# ENV CONFIG (SET IN RENDER)
# -------------------------------------------------
SHOP_DOMAIN = os.getenv("SHOP_DOMAIN", "reviewtestingsb.myshopify.com")
JUDGEME_API_TOKEN = os.getenv("JUDGEME_API_TOKEN")

SHOPIFY_SHOP_DOMAIN = os.getenv("SHOPIFY_SHOP_DOMAIN", SHOP_DOMAIN)
SHOPIFY_ADMIN_TOKEN = os.getenv("SHOPIFY_ADMIN_TOKEN")

# -------------------------------------------------
# CACHE CONFIG
# -------------------------------------------------
PRODUCT_CACHE_TTL = 300       # 5 minutes
REVIEW_CACHE_TTL = 300        # 5 minutes

product_cache = {"handles": [], "last_updated": 0}
review_cache = {"reviews": [], "last_updated": 0}

product_lock = Lock()
review_lock = Lock()

# -------------------------------------------------
# SENTIMENT
# -------------------------------------------------
def analyze_sentiment(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    return "Neutral"

# -------------------------------------------------
# SHOPIFY
# -------------------------------------------------
def fetch_all_product_handles():
    url = f"https://{SHOPIFY_SHOP_DOMAIN}/admin/api/2023-10/products.json"
    headers = {"X-Shopify-Access-Token": SHOPIFY_ADMIN_TOKEN}

    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Shopify API error: {e}")

    products = r.json().get("products", [])
    return [p["handle"] for p in products]

def get_product_handles_cached():
    now = time.time()
    with product_lock:
        if (
            not product_cache["handles"]
            or now - product_cache["last_updated"] > PRODUCT_CACHE_TTL
        ):
            product_cache["handles"] = fetch_all_product_handles()
            product_cache["last_updated"] = now

    return product_cache["handles"]

# -------------------------------------------------
# JUDGE.ME
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
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Judge.me API error: {e}")

        data = r.json()
        reviews = data.get("reviews", [])
        if not reviews:
            break

        all_reviews.extend(reviews)
        params["page"] += 1

    return all_reviews

def get_reviews_cached():
    now = time.time()
    with review_lock:
        if (
            not review_cache["reviews"]
            or now - review_cache["last_updated"] > REVIEW_CACHE_TTL
        ):
            review_cache["reviews"] = fetch_all_reviews()
            review_cache["last_updated"] = now

    return review_cache["reviews"]

# -------------------------------------------------
# STARTUP (PRE-WARM CACHE)
# -------------------------------------------------
@app.on_event("startup")
def startup_event():
    get_product_handles_cached()
    get_reviews_cached()

# -------------------------------------------------
# HEALTH
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# ANALYZE ALL PRODUCTS
# -------------------------------------------------
@app.get("/analyze/all")
def analyze_all():
    product_handles = get_product_handles_cached()
    all_reviews = get_reviews_cached()

    result = {}

    for handle in product_handles:
        product_reviews = [
            r for r in all_reviews
            if r.get("product_handle") == handle
        ]

        if not product_reviews:
            continue

        rows = [{
            "review": r["body"],
            "rating": r["rating"],
            "sentiment": analyze_sentiment(r["body"])
        } for r in product_reviews]

        df = pd.DataFrame(rows)

        result[handle] = {
            "total_reviews": len(df),
            "positive": int((df["sentiment"] == "Positive").sum()),
            "negative": int((df["sentiment"] == "Negative").sum()),
            "neutral": int((df["sentiment"] == "Neutral").sum()),
            "average_rating": round(df["rating"].mean(), 2)
        }

    return result

# -------------------------------------------------
# ANALYZE SINGLE PRODUCT
# -------------------------------------------------
@app.get("/analyze")
def analyze(product_handle: str = Query(...)):
    product_handles = get_product_handles_cached()

    if product_handle not in product_handles:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_handle}' not found in Shopify"
        )

    all_reviews = get_reviews_cached()

    filtered_reviews = [
        r for r in all_reviews
        if r.get("product_handle") == product_handle
    ]

    if not filtered_reviews:
        return {
            "product_handle": product_handle,
            "summary": {},
            "reviews": []
        }

    rows = [{
        "review": r["body"],
        "rating": r["rating"],
        "sentiment": analyze_sentiment(r["body"])
    } for r in filtered_reviews]

    df = pd.DataFrame(rows)

    return {
        "product_handle": product_handle,
        "summary": {
            "total_reviews": len(df),
            "positive": int((df["sentiment"] == "Positive").sum()),
            "negative": int((df["sentiment"] == "Negative").sum()),
            "neutral": int((df["sentiment"] == "Neutral").sum()),
            "average_rating": round(df["rating"].mean(), 2)
        },
        "reviews": rows
    }
