# Data Explainer

This project checks in the data required to inspect the reported results. No
personal financial information or real brokerage data is used.

## Primary news corpus

- File: `data/aapl_news_real.csv`
- Metadata: `data/aapl_news_real.metadata.json`
- Source: Alpha Vantage `NEWS_SENTIMENT` endpoint
- Window: 20 May to 30 September 2026
- Final size: 134 headlines from 28 sources
- Selection: explicit AAPL ticker tag, direct Apple reference in the title,
  AAPL relevance score of at least 0.60, at most five unique headlines per day
- Cleaning: 15 duplicate records removed

Alpha Vantage's supplied sentiment label is discarded. The checked-in headline
text is classified again by `ProsusAI/finbert`, preventing the provider label
from becoming both input and evaluation target.

## Balanced challenge pool

- Files: `data/aapl_news_challenge_pool.csv` and its metadata JSON
- Window: 10 January to 13 October 2025
- Size: 162 AAPL headlines from 85 sources
- Minimum AAPL relevance: 0.35

Thirty items were selected before viewing FinBERT predictions to create a
challenge set with 10 negative, 10 neutral and 10 positive examples. This set
tests class discrimination; its artificial balance does not represent real
news prevalence.

## Label and prediction files

- `evaluation/human_labels.csv`: fixed-seed random 30-headline sample. Labels
  were AI-assisted; it contains 25 neutral, five positive and no negative items.
- `evaluation/balanced_human_labels.csv`: 30-item balanced challenge set. All
  labels were manually reviewed by one human annotator on 4 October 2026.
- `evaluation/sentiment_predictions.csv`: random-sample predictions.
- `evaluation/balanced_sentiment_predictions.csv`: balanced-set predictions.

The label files include a dataset identifier so evaluation stops if a label
file belongs to a different downloaded corpus. There is no second annotator or
inter-annotator agreement statistic.

## Price and decision records

Price data is obtained at runtime from `yfinance`. The checked-in
`backtest_results.csv` and `evaluation/backtest_summary.json` preserve the
reported 90-trading-day results. `evaluation/decision_record.json` records one
end-to-end simulated decision. No real trade was placed.

## Reproduction

Run `download_news.py` to refresh the primary corpus. Refreshing data changes
the dataset ID, so labels must then be prepared and reviewed again before
reporting new evaluation results.
