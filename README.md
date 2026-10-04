# PE6201 AAPL Investment Co-Pilot MVP

This is a deliberately small, human-in-the-loop Agent for one stock (`AAPL`).
It is an educational simulation, not investment advice.

## What the Agent does

1. Downloads real, dated AAPL headlines from Alpha Vantage and saves source URLs.
2. Fetches AAPL price data with `yfinance` and validates all input.
3. Analyses at most five real headlines per date with FinBERT.
4. Calculates 14-day RSI in deterministic Python.
5. Applies a fixed Python rule to return `BUY`, `HOLD`, or `SELL`.
6. Retrieves the three most relevant real headlines using a local TF-IDF RAG layer.
7. Uses a constrained OpenRouter LLM to word the retrieved evidence, then validates every citation and blocks attempts to alter the signal or introduce numbers.
8. Applies non-AI position-size and loss guardrails.
9. Requires `Y`, `N`, or `V` human input before simulated execution.
10. Evaluates the rule over 90 trading days against buy-and-hold and a
    500-run seeded fair-coin baseline (heads = AAPL, tails = cash).

FinBERT, retrieval and the constrained explanation model are the AI components.
RSI, signal thresholds, portfolio arithmetic, guardrails, and execution are not
delegated to the explanation LLM.

## Windows setup

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_news.py
.\run_app.ps1
```

Before downloading, copy `.env.example` to a new file named `.env` and replace
the placeholders with your Alpha Vantage and OpenRouter keys:

```text
ALPHA_VANTAGE_API_KEY=your_real_key
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Never share or commit `.env`. The downloader makes one API request for AAPL
headlines from 20 May through 30 September 2026. It requires an explicit AAPL
ticker tag, an AAPL relevance score of at least `0.60`, and a direct Apple entity
in the title. It then keeps the five highest-relevance unique headlines per
publication date and creates:

- `data/aapl_news_real.csv` containing date, UTC time, headline, source and URL;
- `data/aapl_news_real.metadata.json` documenting counts and selection rules.

Alpha Vantage's supplied sentiment is discarded. The saved real headlines are
analysed again by this project's FinBERT model.

The first full run downloads `ProsusAI/finbert`; later runs use the local model
cache. The constrained explanation is requested through OpenRouter using the
fixed model in `OPENROUTER_MODEL`, so a working internet connection is required. The explanation LLM
receives only retrieved headline text and source names. It cannot calculate RSI,
select BUY/HOLD/SELL, change risk limits or execute an action. Its output is
accepted only when all citations refer to retrieved items, every sentence is
cited, no signal word or numerical claim is introduced, and the output remains
within the length limit. Invalid output falls back to a deterministic template.

An OpenRouter/network/model failure also falls back safely to the deterministic
template. To demonstrate the non-LLM mode explicitly:

```powershell
.\.venv\Scripts\python.exe mvp.py --template-explanation
```

To use another fixed OpenRouter model, set `OPENROUTER_MODEL` in `.env` or pass
`--llm-model`. Avoid automatic model routing in reported experiments so that the
model identity remains reproducible.

The original local explanation model remains available when an API is not
appropriate:

```powershell
.\.venv\Scripts\python.exe mvp.py --local-llm
```

Set `LOCAL_LLM_MODEL` in `.env` or pass `--local-llm-model` to change it.

For a fast offline demonstration of the workflow only:

```powershell
.\.venv\Scripts\python.exe mvp.py --demo-sentiment
```

The program labels this mode as a keyword fallback. Do not report its output as
a FinBERT result.

## Input data

The default input is the downloaded `data/aapl_news_real.csv`:

```csv
date,time_published_utc,headline,source,url,aapl_relevance_score
2026-09-01,2026-09-01T08:00:00+00:00,"Real AAPL headline","Publisher",https://...,0.91
```

The MVP refuses to run when this real-news file is missing and tells the user to
run `download_news.py`. To evaluate another verified dataset:

```powershell
.\.venv\Scripts\python.exe mvp.py --news data\my_aapl_news.csv
```

User-controlled caps can be changed without changing source code:

```powershell
.\.venv\Scripts\python.exe mvp.py --max-position-pct 15 --max-loss-pct 2
```

The report should state the exact price window, valid trading-day count,
headline count, sources, missing records, and duplicates removed.

## Fixed signal rule

- `BUY`: mean sentiment score >= `+0.15` and RSI < `70`
- `SELL`: mean sentiment score <= `-0.15` and RSI > `30`
- `HOLD`: all other cases

On days with no headline, sentiment is neutral (`0.0`). It is never
forward-filled. Weekend news is assigned to the next trading day. Backtest
execution uses the previous day's position for the current day's return and
applies a 0.1% transaction cost when the position changes.

## Baselines and outputs

The fixed 90-trading-day evaluation reports:

- cumulative strategy return;
- buy-and-hold cumulative return;
- mean cumulative return across 500 seeded fair-coin runs;
- maximum drawdown for all three approaches;
- signal counts; and
- `backtest_results.csv` containing daily details.

Outperforming buy-and-hold is not a success requirement. Transparent,
reproducible comparison is the requirement.

## Automated checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover deterministic RSI, fixed signal thresholds, guardrail blocking,
human confirmation, traceable RAG retrieval, constrained LLM validation and
fallback, and the 90-day baseline comparison.

## Quality evaluation

The Agent does not label its own performance. A fixed random seed selects 30
real headlines for human review:

```powershell
.\.venv\Scripts\python.exe evaluate_quality.py --prepare-labels
.\.venv\Scripts\python.exe evaluate_quality.py --label
.\.venv\Scripts\python.exe evaluate_quality.py --run
```

For each headline, enter `P`, `U`, or `N` for positive, neutral, or negative.
Use the criteria in `LABEL_GUIDE.md` and do not inspect model predictions first.
Run `--prepare-labels` again after every news download. Each label file contains
a dataset ID, and evaluation stops if labels belong to an older news file.
The included 30-headline random-sample labels were AI-assisted rather than
produced by an independent human reviewer. That sample reflects the collected
news mix, but it contains no negative labels and therefore cannot measure
negative-class recall.

To address that coverage gap without changing the representative sample, the
repository also includes a separate 30-headline balanced challenge set with 10
negative, 10 neutral and 10 positive examples. It was selected from a broader
2025 Alpha Vantage AAPL news pool without consulting FinBERT or the provider's
sentiment field. Because the classes were deliberately balanced, its results
measure class discrimination rather than real-world prevalence. To repeat or
audit the label review interactively:

```powershell
.\.venv\Scripts\python.exe evaluate_quality.py --review-balanced-labels
.\.venv\Scripts\python.exe evaluate_quality.py --run
```

The included balanced labels have now been manually reviewed by one human
annotator. They are therefore reported as single-annotator human-reviewed
labels, not as independently double-annotated ground truth. No inter-annotator
agreement score is available.

The final command evaluates both datasets against a transparent keyword
baseline and a majority-class baseline using accuracy and macro F1. It also
reports citation integrity and six deterministic guardrail tests. Outputs are
written to:

- `evaluation/evaluation_report.json`;
- `evaluation/sentiment_predictions.csv`;
- `evaluation/balanced_sentiment_predictions.csv`;
- `evaluation/human_labels.csv`;
- `evaluation/balanced_human_labels.csv`.

## Structure

```text
AI_Research_Copilot/
├── mvp.py
├── download_news.py
├── news_source.py
├── evaluate_quality.py
├── LABEL_GUIDE.md
├── investment_agent.py
├── backtest.py
├── data/
│   ├── aapl_news_real.csv       # created locally; not included in the ZIP
│   └── aapl_news_real.metadata.json
├── tests/
│   └── test_investment_agent.py
├── requirements.txt
└── run_app.ps1
```

## Deliberate exclusions

- no real-money trading;
- no multi-stock portfolio;
- no LightGBM or price-prediction claim;
- no synthetic investor profiles;
- no web UI;
- no automatic risk-limit changes; and
- no promised return or 30% drawdown-reduction target.
