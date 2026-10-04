"""Command-line entry point for the PE6201 AAPL MVP."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from backtest import fetch_aapl_prices, load_dated_news, run_backtest
from investment_agent import (
    FinBERTSentiment,
    HuggingFaceGroundedExplainer,
    InvestmentAgent,
    KeywordSentiment,
    OpenRouterGroundedExplainer,
    RiskLimits,
)
from news_source import load_local_env


ROOT = Path(__file__).parent
DEFAULT_NEWS = ROOT / "data" / "aapl_news_real.csv"
EVALUATION_END = "2026-09-30"


def _headlines(news):
    latest = news.sort_values("date").tail(5)
    return [
        {
            "headline": row.headline,
            "source": row.source,
            "url": getattr(row, "url", ""),
        }
        for row in latest.itertuples(index=False)
    ]


def main() -> None:
    load_local_env(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Small human-in-the-loop AAPL decision agent")
    parser.add_argument("--demo-sentiment", action="store_true", help="Use transparent offline fallback")
    parser.add_argument(
        "--template-explanation",
        action="store_true",
        help="Disable the LLM and use the deterministic evidence template",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        help="OpenRouter model used only for evidence commentary",
    )
    parser.add_argument(
        "--local-llm",
        action="store_true",
        help="Use the local Hugging Face explanation model instead of OpenRouter",
    )
    parser.add_argument(
        "--local-llm-model",
        default=os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        help="Local Hugging Face model used with --local-llm",
    )
    parser.add_argument("--news", default=str(DEFAULT_NEWS), help="Dated AAPL news CSV")
    parser.add_argument("--no-backtest", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Confirm simulation non-interactively")
    parser.add_argument("--account-value", type=float, default=10_000.0)
    parser.add_argument("--requested-position-pct", type=float, default=10.0)
    parser.add_argument("--max-position-pct", type=float, default=20.0)
    parser.add_argument("--estimated-loss-pct", type=float, default=2.0)
    parser.add_argument("--max-loss-pct", type=float, default=3.0)
    args = parser.parse_args()

    print("PE6201 AAPL INVESTMENT CO-PILOT — educational simulation only")
    print("Loading AAPL market data...")
    prices = fetch_aapl_prices(evaluation_end=EVALUATION_END)
    if not Path(args.news).exists():
        raise SystemExit(
            f"Real-news file not found: {args.news}\n"
            "Run this first: .\\.venv\\Scripts\\python.exe download_news.py"
        )
    news = load_dated_news(args.news)
    print(
        f"Loaded {len(news)} real AAPL headlines from "
        f"{news['source'].nunique()} sources."
    )
    model = KeywordSentiment() if args.demo_sentiment else FinBERTSentiment()
    if args.demo_sentiment:
        print("WARNING: using keyword demo fallback, not FinBERT.")

    limits = RiskLimits(
        account_value=args.account_value,
        requested_position_pct=args.requested_position_pct,
        max_position_pct=args.max_position_pct,
        estimated_loss_pct=args.estimated_loss_pct,
        max_loss_pct=args.max_loss_pct,
    )
    if args.template_explanation:
        explainer = None
    elif args.local_llm:
        explainer = HuggingFaceGroundedExplainer(args.local_llm_model)
    else:
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not openrouter_key:
            raise SystemExit(
                "OPENROUTER_API_KEY is missing. Add it to .env, or run with "
                "--local-llm / --template-explanation."
            )
        explainer = OpenRouterGroundedExplainer(openrouter_key, args.llm_model)
    agent = InvestmentAgent(model, explainer=explainer)
    decision = agent.run(prices.rename(columns={"Date": "Date"}), _headlines(news), limits)

    print(f"\nSignal: {decision.signal}")
    print(f"Price: ${decision.price:.2f} | RSI(14): {decision.rsi:.1f}")
    print(f"Aggregate sentiment: {decision.sentiment_score:+.3f}")
    print(f"Explanation mode: {decision.explanation_mode}")
    print(decision.rationale)
    if decision.guardrail_reasons:
        print("Guardrail blocked the action:", "; ".join(decision.guardrail_reasons))

    confirmation = "Y" if args.yes else input("\nY=simulate, N=cancel, V=view evidence: ").strip().upper()
    while confirmation == "V":
        for item in decision.evidence:
            print(f"[{item['document_id']}] {item['text']} — {item['source']}")
            if item.get("url"):
                print(f"    {item['url']}")
        confirmation = input("\nY=simulate, N=cancel: ").strip().upper()
    agent.simulate_execute(decision, confirmation == "Y")
    print("Simulated action recorded." if decision.executed else "No action executed.")
    decision_path = ROOT / "evaluation" / "decision_record.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(
        json.dumps(decision.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Decision record: {decision_path}")

    if not args.no_backtest:
        print("\nRunning fixed 90-trading-day comparison...")
        result, detail = run_backtest(prices, news, model)
        summary = asdict(result)
        print(json.dumps(summary, indent=2))
        detail.to_csv(ROOT / "backtest_results.csv", index=False)
        (ROOT / "evaluation" / "backtest_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print("Detailed results: backtest_results.csv")


if __name__ == "__main__":
    main()
