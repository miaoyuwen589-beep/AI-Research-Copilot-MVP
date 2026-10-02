"""One-command download of the fixed AAPL evaluation-news dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from news_source import DEFAULT_MIN_RELEVANCE, download_aapl_news, load_local_env


ROOT = Path(__file__).parent
DEFAULT_OUTPUT = ROOT / "data" / "aapl_news_real.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real, dated AAPL headlines")
    parser.add_argument("--start", default="2026-05-20")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-per-day", type=int, default=5)
    parser.add_argument(
        "--min-relevance", type=float, default=DEFAULT_MIN_RELEVANCE,
        help="Minimum Alpha Vantage AAPL relevance score (0 to 1)",
    )
    args = parser.parse_args()
    load_local_env(ROOT / ".env")
    key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not key:
        raise SystemExit(
            "ALPHA_VANTAGE_API_KEY was not found. Create .env in this folder and add "
            "ALPHA_VANTAGE_API_KEY=your_key"
        )
    frame, _metadata = download_aapl_news(
        key, args.start, args.end, args.output, args.max_per_day, args.min_relevance
    )
    print(f"Saved {len(frame)} verified AAPL headlines to {args.output}")
    print(f"Dates: {frame['date'].min()} to {frame['date'].max()}")
    print(f"Sources: {frame['source'].nunique()}")
    print(f"Minimum AAPL relevance: {args.min_relevance:.2f}")
    print("Alpha Vantage sentiment was discarded; FinBERT will analyse the headlines.")
    print("Important: regenerate human labels after every news download.")
    print(f"Metadata: {Path(args.output).with_suffix('.metadata.json')}")


if __name__ == "__main__":
    main()
