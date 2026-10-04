# Evaluation Explainer

The evaluation asks four separate questions: whether sentiment predictions
match reference labels, whether retrieved citations are traceable, whether
guardrails block prohibited actions, and how the deterministic strategy behaved
historically relative to simple baselines.

## Reproduce the evaluation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe evaluate_quality.py --run
.\.venv\Scripts\python.exe backtest.py
```

Machine-readable results are stored in `evaluation/evaluation_report.json`,
`evaluation/backtest_summary.json`, the two prediction CSV files, and
`backtest_results.csv`.

## Sentiment evaluation

| Dataset | FinBERT accuracy | FinBERT macro F1 | Keyword accuracy | Keyword macro F1 |
|---|---:|---:|---:|---:|
| Fixed-seed random sample | 73.33% | 35.41% | 86.67% | 41.98% |
| Balanced challenge set | 73.33% | 71.36% | 36.67% | 23.15% |

The random sample reflects the collected corpus but has no negative labels, so
it cannot measure negative recall and rewards neutral prediction. The balanced
set deliberately contains 10 examples per class. FinBERT correctly classified
8/10 negative, 10/10 neutral and 4/10 positive items. It demonstrates targeted
three-class discrimination, not expected production accuracy.

The balanced labels were reviewed by one human annotator. The random labels
remain AI-assisted. Neither dataset is an independent multi-annotator benchmark.

## Retrieval and safety

- Citation integrity: 100% (3/3 cited identifiers had a source and URL).
- Guardrail scenarios: 100% (6/6 passed).
- Automated tests: 12/12 passed.

Citation integrity confirms traceability only. It does not prove that retrieval
selected the best evidence, that each sentence is fully entailed, or that users
find the explanation useful. The constrained LLM is outside the signal path and
therefore cannot be credited for investment return.

## Backtest

| Approach | Cumulative return | Maximum drawdown |
|---|---:|---:|
| FinBERT plus RSI rules | 15.09% | -11.05% |
| Buy-and-hold | 9.28% | -12.71% |
| Mean of 500 seeded coin-flip runs | -1.17% | -11.26% |

The test covers 90 trading days and applies a 0.1% cost when exposure changes.
Signals were 82 HOLD, five BUY and three SELL. The result is descriptive: one
stock, one period, one completed round trip and one open position cannot
establish statistical significance or future profitability. The coin-flip
baseline is binary AAPL/cash exposure, not a random three-way signal policy.

## Highest-priority future evaluation

1. Add a second blinded annotator and report agreement.
2. Run rolling walk-forward periods and additional stocks.
3. Score retrieval relevance and entailment on a predeclared sample.
4. Compare the constrained LLM with the deterministic template using
   groundedness, fallback rate, latency and human preference.
5. Conduct a small usability study without introducing real-money execution.
