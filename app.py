from fastapi import FastAPI, Query
import requests
import pandas as pd
from textblob import TextBlob

app = FastAPI(title="Customer Feedback Insights API")

SHOP_DOMAIN = "reviewtestingsb.myshopify.com"
JUDGEME_API_TOKEN = "Lofma_QAgJdAMRLoyEGtQ8yo91U"

SHOPIFY_SHOP_DOMAIN = "reviewtestingsb.myshopify.com"
SHOPIFY_ADMIN_TOKEN = "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"


@app.get("/analyze/all")
def analyze_all():
    product_handles = fetch_all_product_handles()

    result = {}

    for handle in product_handles:
        reviews = fetch_reviews(handle)
        if not reviews:
            continue

        rows = []
        for r in reviews:
            rows.append({
                "review": r["body"],
                "rating": r["rating"],
                "sentiment": analyze_sentiment(r["body"])
            })

        df = pd.DataFrame(rows)

        result[handle] = {
            "total_reviews": len(df),
            "positive": int((df["sentiment"] == "Positive").sum()),
            "negative": int((df["sentiment"] == "Negative").sum()),
            "neutral": int((df["sentiment"] == "Neutral").sum())
        }

    return result

def fetch_all_product_handles():
    url = f"https://{SHOPIFY_SHOP_DOMAIN}/admin/api/2023-10/products.json"
    headers = {
        "X-Shopify-Access-Token": SHOPIFY_ADMIN_TOKEN
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    products = response.json().get("products", [])
    return [p["handle"] for p in products]

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
        response = requests.get(url, params=params)
        data = response.json()

        reviews = data.get("reviews", [])
        if not reviews:
            break

        all_reviews.extend(reviews)
        params["page"] += 1

    return all_reviews


def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


@app.get("/analyze")
def analyze(product_handle: str = Query(...)):
    all_reviews = fetch_all_reviews()

    # ✅ SAFE FILTERING
    filtered_reviews = [
        r for r in all_reviews
        if r.get("product_handle") == product_handle
    ]

    if not filtered_reviews:
        return {
            "error": f"No reviews found for product_handle: {product_handle}"
        }

    rows = []
    for r in filtered_reviews:
        rows.append({
            "review": r["body"],
            "rating": r["rating"],
            "sentiment": analyze_sentiment(r["body"])
        })

    df = pd.DataFrame(rows)

    summary = {
        "total_reviews": len(df),
        "positive": int((df["sentiment"] == "Positive").sum()),
        "negative": int((df["sentiment"] == "Negative").sum()),
        "neutral": int((df["sentiment"] == "Neutral").sum())
    }

    return {
        "product_handle": product_handle,
        "summary": summary,
        "reviews": rows
    }
