from fastapi import FastAPI, HTTPException
from pathlib import Path
import json

app = FastAPI(
    title="Customer Feedback Engine",
    version="1.0.0"
)

DATA_DIR = Path("data")
CONFIG_FILE = Path("config/customers.json")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/customers")
def list_customers():
    customers = json.loads(CONFIG_FILE.read_text())
    return [c["customer_id"] for c in customers]

@app.get("/reviews/{customer_id}")
def get_reviews(customer_id: str):
    file = DATA_DIR / f"{customer_id}_reviews.json"
    if not file.exists():
        raise HTTPException(status_code=404, detail="Customer not found")

    return json.loads(file.read_text())

@app.get("/stats/{customer_id}")
def review_stats(customer_id: str):
    file = DATA_DIR / f"{customer_id}_reviews.json"
    if not file.exists():
        raise HTTPException(status_code=404, detail="Customer not found")

    reviews = json.loads(file.read_text())
    if not reviews:
        return {"count": 0, "average_rating": 0}

    avg = sum(r["rating"] for r in reviews) / len(reviews)

    return {
        "count": len(reviews),
        "average_rating": round(avg, 2)
    }
