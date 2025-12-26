import os
import json
import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime

# Read secrets from environment (production-safe)
API_TOKEN = os.getenv("JUDGEME_API_TOKEN")
SHOP_DOMAIN = os.getenv("SHOP_DOMAIN")

if not API_TOKEN or not SHOP_DOMAIN:
    raise Exception("Missing API credentials")

def fetch_reviews():
    url = "https://judge.me/api/v1/reviews"
    params = {
        "api_token": API_TOKEN,
        "shop_domain": SHOP_DOMAIN,
        "per_page": 100
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("reviews", [])

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    return "Neutral"

def process_reviews(reviews):
    df = pd.DataFrame([
        {
            "review_text": r["body"],
            "rating": r["rating"],
            "product": r["product_handle"],
            "created_at": r["created_at"]
        }
        for r in reviews
    ])

    if df.empty:
        return {}

    df["sentiment"] = df["review_text"].apply(get_sentiment)

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_reviews": int(len(df)),
        "average_rating": round(df["rating"].mean(), 2),
        "sentiment_counts": df["sentiment"].value_counts().to_dict(),
        "latest_reviews": df.sort_values(
            "created_at", ascending=False
        ).head(5).to_dict(orient="records")
    }

    return {
        "summary": summary,
        "reviews": df.to_dict(orient="records")
    }

def main():
    reviews = fetch_reviews()
    result = process_reviews(reviews)

    os.makedirs("data", exist_ok=True)
    with open("data/reviews.json", "w") as f:
        json.dump(result, f, indent=2)

    print("reviews.json generated")

if __name__ == "__main__":
    main()
