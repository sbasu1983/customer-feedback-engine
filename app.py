# -------------------------------------------------
# /ratings/summary
# -------------------------------------------------
@app.get("/ratings/summary")
def ratings_summary(
    product_handle: Optional[str] = Query(None),
    days: int = Query(30),
    recent_window: Optional[int] = Query(None),
    min_avg_rating: Optional[float] = Query(None),
    max_avg_rating: Optional[float] = Query(None),
    min_negative_pct: Optional[float] = Query(None),
    max_negative_pct: Optional[float] = Query(None)
):
    all_reviews = get_reviews_cached()
    now = pd.Timestamp.utcnow()
    cutoff_recent = now - pd.Timedelta(days=recent_window) if recent_window else None

    cleaned = []
    for r in all_reviews:
        dt = safe_review_datetime(r.get("created_at"))
        if dt is None or pd.isna(dt):
            continue
        if not r.get("product_handle") or not r.get("rating") or not r.get("body"):
            continue
        cleaned.append({**r, "_dt": dt})

    if product_handle:
        cleaned = [r for r in cleaned if r["product_handle"] == product_handle]

    product_handles = {r["product_handle"] for r in cleaned}
    results = []

    for handle in product_handles:
        product_reviews = [r for r in cleaned if r["product_handle"] == handle]
        reviews_to_summarize = [r for r in product_reviews if not recent_window or r["_dt"] >= cutoff_recent]
        if not reviews_to_summarize:
            reviews_to_summarize = product_reviews

        summary = summarize_reviews(reviews_to_summarize)
        negative_reviews = [{"body": r["body"]} for r in reviews_to_summarize if analyze_sentiment(r["body"]) == "Negative"]

        # Filters
        if min_avg_rating is not None and summary["average_rating"] < min_avg_rating:
            continue
        if max_avg_rating is not None and summary["average_rating"] > max_avg_rating:
            continue
        if min_negative_pct is not None and summary["negative_pct"] < min_negative_pct:
            continue
        if max_negative_pct is not None and summary["negative_pct"] > max_negative_pct:
            continue

        results.append({
            "product_handle": handle,
            "total_reviews": summary["total_reviews"],
            "average_rating": summary["average_rating"],
            "positive_pct": summary["positive_pct"],
            "negative_pct": summary["negative_pct"],
            "neutral_pct": summary["neutral_pct"],
            "top_complaints": list(extract_themes(negative_reviews, COMPLAINT_KEYWORDS).keys())
        })

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_window_days": days,
        "recent_window_days": recent_window if recent_window else "all",
        "products": results
    }

# -------------------------------------------------
# /ratings/alerts
# -------------------------------------------------
@app.get("/ratings/alerts")
def ratings_alerts(
    product_handle: Optional[str] = Query(None),
    days: int = Query(30),
    recent_window: int = Query(7),
    rating_drop: float = Query(0.5),
    negative_spike: float = Query(20.0)
):
    all_reviews = get_reviews_cached()
    now = pd.Timestamp.utcnow()
    cutoff_recent = now - pd.Timedelta(days=recent_window)
    cleaned = []

    for r in all_reviews:
        dt = safe_review_datetime(r.get("created_at"))
        if dt is None or pd.isna(dt):
            continue
        handle = resolve_product_handle(r)
        if not handle or not r.get("rating") or not r.get("body"):
            continue
        cleaned.append({**r, "_dt": dt, "product_handle": handle})

    if product_handle:
        cleaned = [r for r in cleaned if r["product_handle"] == product_handle]

    results = []
    for handle in {r["product_handle"] for r in cleaned}:
        product_reviews = [r for r in cleaned if r["product_handle"] == handle]
        recent = [r for r in product_reviews if r["_dt"] >= cutoff_recent]
        historical = [r for r in product_reviews if r["_dt"] < cutoff_recent]
        if not recent:
            recent = product_reviews[-5:]
        if not historical:
            historical = product_reviews[:5]
        if not recent or not historical:
            continue

        hist_summary = summarize_reviews(historical)
        recent_summary = summarize_reviews(recent)

        rating_diff = hist_summary["average_rating"] - recent_summary["average_rating"]
        negative_diff = recent_summary["negative_pct"] - hist_summary["negative_pct"]

        alerts = []
        if rating_diff >= rating_drop:
            alerts.append("Average rating dropping")
        if negative_diff >= negative_spike:
            alerts.append("Spike in negative reviews")
        if recent_summary["average_rating"] <= 3.0:
            alerts.append("Critically low recent rating")
        if not alerts:
            alerts.append("No critical alerts yet – monitoring recommended")

        results.append({
            "product_handle": handle,
            "historical_avg_rating": hist_summary["average_rating"],
            "recent_avg_rating": recent_summary["average_rating"],
            "rating_drop": round(rating_diff, 2),
            "negative_pct_change": round(negative_diff, 2),
            "alerts": alerts,
            "severity": (
                "high" if len(alerts) >= 2 else
                "medium" if any(a in alerts for a in ["Average rating dropping", "Spike in negative reviews"]) else
                "low"
            )
        })

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_window_days": days,
        "recent_window_days": recent_window,
        "products": results
    }

# -------------------------------------------------
# /ratings/themes
# -------------------------------------------------
@app.get("/ratings/themes")
def ratings_themes(
    product_handle: Optional[str] = Query("all"),
    days: int = Query(30)
):
    all_reviews = get_reviews_cached()
    reviews = []

    for r in all_reviews:
        handle = resolve_product_handle(r)
        body = r.get("body") or r.get("review") or r.get("body_html") or r.get("title") or ""
        if not body.strip():
            continue
        if product_handle != "all" and handle != product_handle:
            continue
        reviews.append({"body": body})

    return {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "product_handle": product_handle,
        "negative_themes": extract_themes(reviews, COMPLAINT_KEYWORDS),
        "positive_themes": extract_themes(reviews, PRAISE_KEYWORDS)
    }

# -------------------------------------------------
# /ratings/insights
# -------------------------------------------------
@app.get("/ratings/insights")
def get_insights(product_handle: str = "all"):
    all_reviews = fetch_all_reviews()
    complaints = Counter()
    praises = Counter()
    for r in all_reviews:
        text = (r.get("review") or r.get("body") or r.get("comment") or "").strip().lower()
        if not text:
            continue
        polarity = TextBlob(text).sentiment.polarity
        words = re.findall(r"\b[a-z]{3,}\b", text)
        phrases = [" ".join(words[i:i+2]) for i in range(len(words)-1)] + [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        if polarity < -0.1:
            for p in phrases:
                complaints[p] += 1
        elif polarity > 0.1:
            for p in phrases:
                praises[p] += 1
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "product_handle": product_handle,
        "top_complaints": [p for p, _ in complaints.most_common(3)],
        "top_praises": [p for p, _ in praises.most_common(3)],
    }

# -------------------------------------------------
# /ratings/actionable
# -------------------------------------------------
@app.get("/ratings/actionable")
def ratings_actionable(
    product_handle: Optional[str] = Query(None),
    days: int = Query(30),
    recent_window: int = Query(7),
    min_avg_rating: Optional[float] = Query(None),
    max_negative_pct: Optional[float] = Query(None),
):
    all_reviews = get_reviews_cached()
    now = pd.Timestamp.utcnow()
    cutoff_recent = now - pd.Timedelta(days=recent_window)
    cleaned = []
    for r in all_reviews:
        dt = safe_review_datetime(r.get("created_at")) or now
        handle = r.get("product_handle") or "unknown_product"
        rating = r.get("rating") or 0
        body = r.get("body") or ""
        cleaned.append({**r, "_dt": dt, "product_handle": handle, "rating": rating, "body": body})

    if product_handle:
        cleaned = [r for r in cleaned if r["product_handle"] == product_handle]

    results = []
    for handle in {r["product_handle"] for r in cleaned}:
        product_reviews = [r for r in cleaned if r["product_handle"] == handle]
        recent = [r for r in product_reviews if r["_dt"] >= cutoff_recent] or product_reviews[-5:]
        if not recent:
            continue
        summary = summarize_reviews(recent)
        avg_rating = summary["average_rating"]
        negative_pct = summary["negative_pct"]
        if avg_rating <= 3.0 or negative_pct >= 40:
            priority = "high"
            action = "Investigate recurring customer complaints immediately"
        elif avg_rating < 4.0 or negative_pct >= 25:
            priority = "medium"
            action = "Monitor feedback and address emerging issues"
        else:
            priority = "low"
            action = "No immediate action needed"
        if min_avg_rating is not None and avg_rating < min_avg_rating:
            continue
        if max_negative_pct is not None and negative_pct > max_negative_pct:
            continue
        results.append({
            "product_handle": handle,
            "average_rating": avg_rating,
            "negative_pct": negative_pct,
            "priority": priority,
            "recommended_action": action
        })

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_window_days": days,
        "recent_window_days": recent_window,
        "products": results
    }

# -------------------------------------------------
# /ratings/actionable-themes
# -------------------------------------------------
@app.get("/ratings/actionable-themes")
def ratings_actionable_themes(
    product_handle: Optional[str] = Query(None),
    days: int = Query(30),
    recent_window: int = Query(7),
    min_avg_rating: Optional[float] = Query(None),
    max_negative_pct: Optional[float] = Query(None),
):
    all_reviews = get_reviews_cached()
    now = pd.Timestamp.utcnow()
    cutoff_recent = now - pd.Timedelta(days=recent_window)
    cleaned = []
    for r in all_reviews:
        dt = safe_review_datetime(r.get("created_at")) or now
        handle = r.get("product_handle") or "unknown_product"
        rating = r.get("rating") or 0
        body = r.get("body") or ""
        cleaned.append({**r, "_dt": dt, "product_handle": handle, "rating": rating, "body": body})

    if product_handle:
        cleaned = [r for r in cleaned if r["product_handle"] == product_handle]

    results = []
    for handle in {r["product_handle"] for r in cleaned}:
        product_reviews = [r for r in cleaned if r["product_handle"] == handle]
        recent = [r for r in product_reviews if r["_dt"] >= cutoff_recent] or product_reviews[-5:]
        if not recent:
            continue
        summary = summarize_reviews(recent)
        avg_rating = summary["average_rating"]
        negative_pct = summary["negative_pct"]
        if avg_rating <= 3.0 or negative_pct >= 40:
            priority = "high"
            action = "Investigate recurring customer complaints immediately"
        elif avg_rating < 4.0 or negative_pct >= 25:
            priority = "medium"
            action = "Monitor feedback and address emerging issues"
        else:
            priority = "low"
            action = "No immediate action needed"
        if min_avg_rating is not None and avg_rating < min_avg_rating:
            continue
        if max_negative_pct is not None and negative_pct > max_negative_pct:
            continue
        negative_reviews = [{"body": r["body"].lower()} for r in recent if analyze_sentiment(r["body"]) == "Negative"]
        positive_reviews = [{"body": r["body"].lower()} for r in recent if analyze_sentiment(r["body"]) == "Positive"]
        if not negative_reviews:
            negative_reviews = [{"body": r["body"].lower()} for r in recent]
        if not positive_reviews:
            positive_reviews = [{"body": r["body"].lower()} for r in recent]
        results.append({
            "product_handle": handle,
            "average_rating": avg_rating,
            "negative_pct": negative_pct,
            "priority": priority,
            "recommended_action": action,
            "top_complaints": list(extract_themes(negative_reviews, COMPLAINT_KEYWORDS).keys())[:5],
            "top_praises": list(extract_themes(positive_reviews, PRAISE_KEYWORDS).keys())[:5]
        })

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_window_days": days,
        "recent_window_days": recent_window,
        "products": results
    }
