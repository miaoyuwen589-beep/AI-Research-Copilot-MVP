# Five Minute Demo Guide

The submitted video must show the presenter's face and the computer screen at
the same time. Aim for five minutes; the marking instructions allow roughly two
to eight minutes but only the first eight minutes will be reviewed.

## Recording sequence

1. **Problem and contribution (0:00-0:35).** Introduce the AAPL research
   co-pilot and state that it is an educational simulation, not financial
   advice. Explain that the contribution is traceable AI assistance with
   deterministic controls, not a claim of market prediction.
2. **Architecture (0:35-1:15).** Show the diagram in `README.md`. Identify real
   news and prices as inputs; FinBERT and RSI as analysis; fixed Python rules as
   the signal; TF-IDF plus OpenRouter as the explanation path; and `Y/N/V` as
   the human control.
3. **Live operation (1:15-2:45).** Run `run_app.ps1`. Read the price, RSI,
   sentiment and resulting signal. Use `V` to inspect evidence, then `Y` or `N`
   to demonstrate that no simulated action occurs without a person.
4. **Safety boundary (2:45-3:20).** Explain that the LLM sees only retrieved
   evidence, cannot alter the signal or risk limits, and falls back to a
   deterministic explanation if validation fails.
5. **Evaluation (3:20-4:25).** Show `evaluation/evaluation_report.json` and the
   test result. State balanced-set FinBERT accuracy 73.33%, macro F1 71.36%,
   keyword macro F1 23.15%, 12/12 tests, and the 90-day backtest comparison.
6. **Critique and close (4:25-5:00).** State that the study covers one stock,
   one short period and one human annotator. Mention weak positive recall and
   the need for walk-forward tests and a second annotator.

## Before recording

- Keep `.env` closed and ensure no API key appears on screen.
- Increase terminal font size and close unrelated windows.
- Start with the repository and required files already open.
- Rehearse once and keep explanations precise rather than reading every line.
- Upload the final video using the course submission method, then add its link
  here only if the link is accessible to the instructor or TA.
