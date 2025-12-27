from fastapi import FastAPI, Query
import requests
import os
import pandas as pd
from textblob import TextBlob

app = FastAPI(title="Customer Feedback Insights API")

# HARD-CODED FOR DEBUGGING
SHOP_DOMAIN = os.getenv("SHOP_DOMAIN")
JUDGEME_API_TOKEN = os.getenv("JUDGEME_API_TOKEN")


def fetch_reviews(product_handle: str, per_page: int = 100):
    url = "https://judge.me/api/v1/reviews"

    params = {
        "shop_domain": SHOP_DOMAIN,
        "api_token": JUDGEME_API_TOKEN,
        "product_external_id": 8952367284398,
        "per_page": per_page,
        "page": 1,
        "published": "true"
    }

    all_reviews = []

    while True:
        response = requests.get(url, params=params)

        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)

        data = response.json()
        reviews = data.get("reviews", [])

        if not reviews:
            break

        all_reviews.extend(reviews)
        params["page"] += 1

    return all_reviews


def analyze_sentiment(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


@app.get("/analyze")
def analyze(product_handle: str = Query(...)):
    reviews = fetch_reviews(product_handle)

    if not reviews:
        return {"error": "No reviews found"}

    rows = []

    for r in reviews:
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
        "summary": summary,
        "reviews": rows
    }
