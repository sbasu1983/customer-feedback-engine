from fastapi import FastAPI, Query, HTTPException
import requests
import pandas as pd
from textblob import TextBlob

app = FastAPI(title="Customer Feedback Insights API")

# ----------------------------
# Configuration
# ----------------------------
SHOP_DOMAIN = "reviewtestingsb.myshopify.com"
JUDGEME_API_TOKEN = "Lofma_QAgJdAMRLoyEGtQ8yo91U"
SHOPIFY_SHOP_DOMAIN = "reviewtestingsb.myshopify.com"
SHOPIFY_ADMIN_TOKEN = "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"

# ----------------------------
# Utility Functions
# ----------------------------
def analyze_sentiment(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    return "Neutral"

def fetch_all_product_handles():
    """Fetch all product handles from Shopify"""
    url = f"https://{SHOPIFY_SHOP_DOMAIN}/admin/api/2023-10/products.json"
    headers = {"X-Shopify-Access-Token": SHOPIFY_ADMIN_TOKEN}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Shopify API error: {e}")

    products = response.json().get("products", [])
    return [{"handle": p["handle"]} for p in products]

def fetch_reviews_by_product_handle(product_handle: str, per_page: int = 100):
    """Fetch reviews from Judge.me for a specific product handle"""
    url = "https://judge.me/api/v1/reviews"
    params = {
        "shop_domain": SHOP_DOMAIN,
        "api_token": JUDGEME_API_TOKEN,
        "per_page": per_page,
        "page": 1,
        "product_handle": product_handle,
        "published": "true"
    }

    all_reviews = []
    while True:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Judge.me API error: {e}")

        data = response.json()
        reviews = data.get("reviews", [])
        if not reviews:
            break

        all_reviews.extend(reviews)
        params["page"] += 1

    return all_reviews

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/analyze/all")
def analyze_all():
    """Analyze all products and return sentiment summaries"""
    products = fetch_all_product_handles()
    result = {}

    for p in products:
        reviews = fetch_reviews_by_product_handle(p["handle"])
        if not reviews:
            continue

        rows = [{
            "review": r["body"],
            "rating": r["rating"],
            "sentiment": analyze_sentiment(r["body"])
        } for r in reviews]

        df = pd.DataFrame(rows)

        result[p["handle"]] = {
            "total_reviews": len(df),
            "positive": int((df["sentiment"] == "Positive").sum()),
            "negative": int((df["sentiment"] == "Negative").sum()),
            "neutral": int((df["sentiment"] == "Neutral").sum()),
            "average_rating": round(df["rating"].mean(), 2)
        }

    return result

@app.get("/analyze")
def analyze(product_handle: str = Query(...)):
    """Analyze a single product by handle"""
    products = fetch_all_product_handles()
    product = next((p for p in products if p["handle"] == product_handle), None)

    if not product:
        raise HTTPException(status_code=404, detail=f"Product '{product_handle}' not found")

    reviews = fetch_reviews_by_product_handle(product_handle)
    if not reviews:
        return {"product_handle": product_handle, "summary": {}, "reviews": []}

    rows = [{
        "review": r["body"],
        "rating": r["rating"],
        "sentiment": analyze_sentiment(r["body"])
    } for r in reviews]

    df = pd.DataFrame(rows)

    summary = {
        "total_reviews": len(df),
        "positive": int((df["sentiment"] == "Positive").sum()),
        "negative": int((df["sentiment"] == "Negative").sum()),
        "neutral": int((df["sentiment"] == "Neutral").sum()),
        "average_rating": round(df["rating"].mean(), 2)
    }

    return {"product_handle": product_handle, "summary": summary, "reviews": rows}
