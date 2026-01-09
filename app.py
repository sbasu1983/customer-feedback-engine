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
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Judge.me API error: {e}")
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
# SHOPIFY WEBHOOK VERIFICATION
# -------------------------------------------------
def verify_shopify_webhook(request: Request, body: bytes):
    hmac_header = request.headers.get("X-Shopify-Hmac-Sha256")
    digest = hmac.new(
        SHOPIFY_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).digest()
    computed_hmac = base64.b64encode(digest).decode()
    if not hmac.compare_digest(computed_hmac, hmac_header):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

@app.post("/webhooks/products")
async def product_webhook(request: Request):
    body = await request.body()
    verify_shopify_webhook(request, body)
    with product_lock:
        product_cache["handles"] = []
        product_cache["last_updated"] = 0
    return {"status": "ok"}

@app.on_event("startup")
def startup_event():
    get_product_handles_cached()
    get_reviews_cached()

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# ANALYSIS HELPERS
# -------------------------------------------------
def summarize_reviews(product_reviews):
    rows = []
    emotion_summary = {"joy": 0, "anger": 0, "sadness": 0, "surprise": 0, "disgust": 0}
    for r in product_reviews:
        review_text = r["body"]
        timestamp = r.get("created_at", "")
        row = {
            "review": review_text,
            "rating": r["rating"],
            "sentiment": analyze_sentiment(review_text),
            "sentiment_score": sentiment_score(review_text),
            "emotions": detect_emotions(review_text),
            "length": len(review_text),
            "timestamp": timestamp
        }
        for k, v in row["emotions"].items():
            emotion_summary[k] += v
        rows.append(row)

    df = pd.DataFrame(rows)
    negative_reviews = [r for r in rows if r["sentiment"] == "Negative"]
    positive_reviews = [r for r in rows if r["sentiment"] == "Positive"]

    avg_rating = round(df["rating"].mean(), 2)
    negative_pct = (len(negative_reviews) / len(df)) if df.shape[0] else 0
    weighted_risk_score = round((1 - avg_rating/5) * 0.7 + negative_pct * 0.3, 2)

    # 🟢 Monthly trends
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['year_month'] = df['timestamp'].dt.to_period('M')
    monthly_trends = []
    if not df['year_month'].isnull().all():
        grouped = df.groupby('year_month')
        for period, g in grouped:
            monthly_trends.append({
                "month": str(period),
                "average_rating": round(g["rating"].mean(), 2),
                "average_sentiment_score": round(g["sentiment_score"].mean(), 2),
                "review_count": len(g)
            })

    return {
        "total_reviews": len(df),
        "average_rating": avg_rating,
        "positive": int((df["sentiment"] == "Positive").sum()),
        "negative": int((df["sentiment"] == "Negative").sum()),
        "neutral": int((df["sentiment"] == "Neutral").sum()),
        **sentiment_percentages(df),
        "common_complaints": extract_themes(negative_reviews, COMPLAINT_KEYWORDS),
        "common_praises": extract_themes(positive_reviews, PRAISE_KEYWORDS),
        "average_review_length": round(df["length"].mean(), 2),
        "at_risk_flag": avg_rating < 3 or negative_pct > 0.5,
        "weighted_risk_score": weighted_risk_score,
        "emotion_summary": emotion_summary,
        "review_trends": [{"rating": r["rating"], "sentiment_score": r["sentiment_score"], "timestamp": r["timestamp"]} for r in rows],
        "top_keywords": {
            "positive": list(extract_themes(positive_reviews, PRAISE_KEYWORDS).keys()),
            "negative": list(extract_themes(negative_reviews, COMPLAINT_KEYWORDS).keys())
        },
        "monthly_trends": monthly_trends
    }

# -------------------------------------------------
# ANALYZE ALL PRODUCTS
# -------------------------------------------------
@app.get("/analyze/all")
def analyze_all():
    product_handles = get_product_handles_cached()
    all_reviews = get_reviews_cached()
    result = {}
    for handle in product_handles:
        product_reviews = [r for r in all_reviews if r.get("product_handle") == handle]
        if not product_reviews:
            continue
        result[handle] = summarize_reviews(product_reviews)
    return result

# -------------------------------------------------
# ANALYZE SINGLE PRODUCT
# -------------------------------------------------
@app.get("/analyze")
def analyze(product_handle: str = Query(...)):
    if product_handle not in get_product_handles_cached():
        raise HTTPException(status_code=404, detail="Product not found")
    all_reviews = get_reviews_cached()
    filtered_reviews = [r for r in all_reviews if r.get("product_handle") == product_handle]
    if not filtered_reviews:
        return {"product_handle": product_handle, "summary": {}, "reviews": []}

    summary = summarize_reviews(filtered_reviews)
    reviews = [{"review": r["body"], "rating": r["rating"], "sentiment": analyze_sentiment(r["body"]), "timestamp": r.get("created_at", "")} for r in filtered_reviews]

    return {
        "product_handle": product_handle,
        "summary": summary,
        "reviews": reviews
    }
