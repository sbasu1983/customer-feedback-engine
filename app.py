from fastapi import FastAPI, Query
import os
import requests
import pandas as pd
from textblob import TextBlob

app = FastAPI(title="Customer Feedback Insights API")

SHOP_DOMAIN = os.getenv("SHOP_DOMAIN")
JUDGEME_API_TOKEN = os.getenv("JUDGEME_API_TOKEN")


def fetch_reviews(product_handle: str, per_page=100):
    url = f"https://judge.me/api/v1/reviews"
    headers = {
        "Authorization": f"Bearer {JUDGEME_API_TOKEN}"
    }

    params = {
        "shop_domain": SHOP_DOMAIN,
        "product_handle": product_handle,
        "per_page": per_page,
        "page": 1
    }

    all_reviews = []

    while True:
        response = requests.get(url, headers=headers, params=params)

        print("STATUS:", response.status_code)
        print("RESPONSE TEXT:", response.text)

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
    reviews = fetch_reviews(product_handle)

    if not reviews:
        return {"error": "No reviews found"}

    rows = []
    for r in reviews:
        sentiment = analyze_sentiment(r["body"])
        rows.append({
            "review": r["body"],
            "rating": r["rating"],
            "sentiment": sentiment
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
