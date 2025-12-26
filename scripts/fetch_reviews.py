import os
import json
import requests
from pathlib import Path

BASE_URL = "https://judge.me/api/v1/reviews"
OUTPUT_DIR = Path("data")
CUSTOMER_CONFIG = Path("config/customers.json")

OUTPUT_DIR.mkdir(exist_ok=True)

def fetch_reviews(shop_domain, api_token):
    params = {
        "shop_domain": shop_domain,
        "api_token": api_token,
        "per_page": 100
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("reviews", [])

def main():
    customers = json.loads(CUSTOMER_CONFIG.read_text())

    for customer in customers:
        customer_id = customer["customer_id"]
        shop_domain = customer["shop_domain"]
        token_secret = customer["api_token_secret"]

        api_token = os.getenv(token_secret)
        if not api_token:
            print(f"❌ Missing token for {customer_id}")
            continue

        print(f"📥 Fetching reviews for {customer_id}")

        reviews = fetch_reviews(shop_domain, api_token)

        output_file = OUTPUT_DIR / f"{customer_id}_reviews.json"
        output_file.write_text(json.dumps(reviews, indent=2))

        print(f"✅ Saved {len(reviews)} reviews → {output_file}")

if __name__ == "__main__":
    main()
