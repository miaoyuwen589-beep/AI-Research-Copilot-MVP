"""Download verifiable AAPL headlines from Alpha Vantage.

Only publication metadata is retained. Alpha Vantage's own sentiment labels are
deliberately discarded so that the experiment evaluates FinBERT.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


API_URL = "https://www.alphavantage.co/query"
DEFAULT_MIN_RELEVANCE = 0.60
APPLE_TITLE_PATTERN = re.compile(
    r"\b(?:apple|aapl|iphone|ipad|macbook|mac|ios|siri|tim\s+cook|wwdc|"
    r"app\s+store|vision\s+pro|airpods)\b",
    flags=re.IGNORECASE,
)


def load_local_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without exposing or printing secrets."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _normalise_title(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_alpha_vantage_feed(
    payload: dict,
    start_date: str,
    end_date: str,
    max_per_day: int = 5,
    min_relevance: float = DEFAULT_MIN_RELEVANCE,
) -> tuple[pd.DataFrame, dict]:
    """Validate, filter and deterministically deduplicate an API response."""
    if not 0 <= min_relevance <= 1:
        raise ValueError("min_relevance must be between 0 and 1")
    for key in ("Error Message", "Information", "Note"):
        if payload.get(key):
            raise RuntimeError(str(payload[key]))
    feed = payload.get("feed")
    if not isinstance(feed, list):
        raise RuntimeError("Alpha Vantage response did not contain a news feed")

    rows: list[dict[str, str | float]] = []
    rejected = {
        "missing_aapl_tag": 0,
        "missing_or_low_relevance": 0,
        "no_direct_apple_reference": 0,
    }
    for item in feed:
        title = _normalise_title(str(item.get("title", "")))
        raw_time = str(item.get("time_published", ""))
        source = _normalise_title(str(item.get("source", ""))) or "Unknown source"
        url = str(item.get("url", "")).strip()
        if not title or not raw_time or not url.startswith(("http://", "https://")):
            continue
        published = pd.to_datetime(raw_time, format="%Y%m%dT%H%M%S", errors="coerce", utc=True)
        if pd.isna(published):
            continue
        day = published.date().isoformat()
        if day < start_date or day > end_date:
            continue
        ticker_items = item.get("ticker_sentiment", [])
        aapl_item = next(
            (
                entry
                for entry in ticker_items
                if str(entry.get("ticker", "")).strip().upper() == "AAPL"
            ),
            None,
        )
        if aapl_item is None:
            rejected["missing_aapl_tag"] += 1
            continue
        try:
            relevance = float(aapl_item.get("relevance_score", ""))
        except (TypeError, ValueError):
            rejected["missing_or_low_relevance"] += 1
            continue
        if not 0 <= relevance <= 1 or relevance < min_relevance:
            rejected["missing_or_low_relevance"] += 1
            continue
        if not APPLE_TITLE_PATTERN.search(title):
            rejected["no_direct_apple_reference"] += 1
            continue
        rows.append(
            {
                "date": day,
                "time_published_utc": published.isoformat(),
                "headline": title,
                "source": source,
                "url": url,
                "aapl_relevance_score": relevance,
            }
        )

    if not rows:
        raise RuntimeError("No usable AAPL news was returned for the requested period")
    frame = pd.DataFrame(rows)
    raw_usable_count = len(frame)
    frame["title_key"] = frame["headline"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    frame = frame.sort_values(
        ["date", "aapl_relevance_score", "time_published_utc", "source", "headline"],
        ascending=[True, False, True, True, True],
    )
    frame = frame.drop_duplicates(subset=["url"]).drop_duplicates(subset=["date", "title_key"])
    deduplicated_count = len(frame)
    frame = frame.groupby("date", group_keys=False).head(max_per_day)
    frame = frame.drop(columns="title_key").reset_index(drop=True)
    metadata = {
        "provider": "Alpha Vantage NEWS_SENTIMENT",
        "ticker": "AAPL",
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_feed_count": len(feed),
        "usable_before_deduplication": raw_usable_count,
        "after_deduplication": deduplicated_count,
        "final_headline_count": len(frame),
        "max_headlines_per_day": max_per_day,
        "minimum_aapl_relevance_score": min_relevance,
        "rejected_counts": rejected,
        "selection_rule": (
            "Require an explicit AAPL ticker tag, AAPL relevance at or above the "
            "threshold, and a direct Apple entity in the title; then keep the five "
            "highest-relevance unique headlines per publication date"
        ),
        "provider_sentiment_used": False,
    }
    return frame, metadata


def download_aapl_news(
    api_key: str,
    start_date: str,
    end_date: str,
    output_path: str | Path,
    max_per_day: int = 5,
    min_relevance: float = DEFAULT_MIN_RELEVANCE,
    timeout: int = 45,
) -> tuple[pd.DataFrame, dict]:
    if not api_key or api_key.lower().startswith(("replace", "your_")):
        raise ValueError("A valid ALPHA_VANTAGE_API_KEY is required")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("Start date must not be after end date")
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "AAPL",
        "time_from": start.strftime("%Y%m%dT0000"),
        "time_to": end.strftime("%Y%m%dT2359"),
        "sort": "EARLIEST",
        "limit": "1000",
        "apikey": api_key,
    }
    request = Request(
        f"{API_URL}?{urlencode(params)}",
        headers={"User-Agent": "PE6201-AAPL-Educational-MVP/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    frame, metadata = parse_alpha_vantage_feed(
        payload, start_date, end_date, max_per_day, min_relevance
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    destination.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return frame, metadata
