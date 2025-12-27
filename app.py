from fastapi import FastAPI, Query, HTTPException
import requests
import pandas as pd
from textblob import TextBlob

app = FastAPI(title="Customer Feedback Insights API")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
SHOP_DOMAIN = "reviewtestingsb.myshopify.com"
JUDGEME_API_TOKEN = "Lofma_QAgJdAMRLoyEGtQ8yo91U"

SHOPIFY_SHOP_DOMAIN = "reviewtestingsb.myshopify.com"
SHOPIFY_ADMIN_TOKEN = "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"

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
        r = requests.get(url, headers=headers)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Shopify API error: {e}")

    products = r.json().get("products", [])
    return [p["handle"] for p in products]

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
            r = requests.get(url, params=params)
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

# -------------------------------------------------
# ANALYZE ALL PRODUCTS
# -------------------------------------------------
@app.get("/analyze/all")
def analyze_all():
    product_handles = fetch_all_product_handles()
    all_reviews = fetch_all_reviews()

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
# ANALYZE SINGLE PRODUCT (FIXED)
# -------------------------------------------------
@app.get("/analyze")
def analyze(product_handle: str = Query(...)):
    product_handles = fetch_all_product_handles()

    if product_handle not in product_handles:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_handle}' not found in Shopify"
        )

    all_reviews = fetch_all_reviews()

    # ✅ HARD FILTER — THIS IS THE FIX
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

    summary = {
        "total_reviews": len(df),
        "positive": int((df["sentiment"] == "Positive").sum()),
        "negative": int((df["sentiment"] == "Negative").sum()),
        "neutral": int((df["sentiment"] == "Neutral").sum()),
        "average_rating": round(df["rating"].mean(), 2)
    }

    return {
        "product_handle": product_handle,
        "summary": summary,
        "reviews": rows
    }
