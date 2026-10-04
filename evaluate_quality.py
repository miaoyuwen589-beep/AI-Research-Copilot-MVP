"""Reproducible quality evaluation for the PE6201 AAPL Agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from investment_agent import (
    Decision,
    FinBERTSentiment,
    InvestmentAgent,
    KeywordSentiment,
    RiskLimits,
    check_guardrails,
)


ROOT = Path(__file__).parent
NEWS_PATH = ROOT / "data" / "aapl_news_real.csv"
CHALLENGE_NEWS_PATH = ROOT / "data" / "aapl_news_challenge_pool.csv"
EVALUATION_DIR = ROOT / "evaluation"
LABEL_PATH = EVALUATION_DIR / "human_labels.csv"
BALANCED_LABEL_PATH = EVALUATION_DIR / "balanced_human_labels.csv"
REPORT_PATH = EVALUATION_DIR / "evaluation_report.json"
LABELS = ("negative", "neutral", "positive")


def news_dataset_id(path: Path = NEWS_PATH) -> str:
    """Bind human labels to the exact news file used to prepare them."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def prepare_labels(sample_size: int = 30) -> pd.DataFrame:
    if not NEWS_PATH.exists():
        raise SystemExit("Download real news first with: python download_news.py")
    news = pd.read_csv(NEWS_PATH)
    required = {"date", "headline", "source", "url"}
    if not required.issubset(news.columns):
        raise ValueError(f"Real-news CSV must contain {sorted(required)}")
    sample_size = min(sample_size, len(news))
    sample = news.sample(n=sample_size, random_state=6201).sort_values(["date", "headline"])
    sample = sample[["date", "headline", "source", "url"]].reset_index(drop=True)
    sample.insert(0, "item_id", np.arange(1, len(sample) + 1))
    sample.insert(1, "news_dataset_id", news_dataset_id())
    sample["human_label"] = ""
    EVALUATION_DIR.mkdir(exist_ok=True)
    sample.to_csv(LABEL_PATH, index=False, encoding="utf-8-sig")
    return sample


def label_interactively(path: Path = LABEL_PATH, review_all: bool = False) -> None:
    if not path.exists():
        if path == LABEL_PATH:
            prepare_labels()
        else:
            raise SystemExit(f"Balanced label file not found: {path}")
    labels = pd.read_csv(path, keep_default_na=False)
    mapping = {"p": "positive", "u": "neutral", "n": "negative"}
    for index, row in labels.iterrows():
        if not review_all and row["human_label"] in LABELS:
            continue
        print(f"\n[{index + 1}/{len(labels)}] {row['headline']}")
        print(f"Source: {row['source']}")
        current = str(row.get("human_label", "")).strip().lower()
        if current in LABELS:
            print(f"Current label: {current}")
        while True:
            answer = input(
                "P=positive, U=neutral, N=negative, Enter=keep, Q=save and quit: "
            ).strip().lower()
            if answer == "q":
                labels.to_csv(path, index=False, encoding="utf-8-sig")
                print(f"Progress saved to {path}")
                return
            if answer == "" and current in LABELS:
                break
            if answer in mapping:
                labels.at[index, "human_label"] = mapping[answer]
                labels.to_csv(path, index=False, encoding="utf-8-sig")
                break
    labels.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"All labels saved to {path}")


def evaluate_sentiment_file(
    label_path: Path,
    news_path: Path,
    predictions_path: Path,
    design: str,
    limitation: str,
    required_distribution: dict[str, int] | None = None,
) -> tuple[dict, pd.DataFrame]:
    if not label_path.exists():
        raise SystemExit(f"Label file not found: {label_path}")
    if not news_path.exists():
        raise SystemExit(f"News dataset not found: {news_path}")
    labels = pd.read_csv(label_path, keep_default_na=False)
    expected_dataset_id = news_dataset_id(news_path)
    if "news_dataset_id" not in labels.columns or set(labels["news_dataset_id"]) != {expected_dataset_id}:
        raise SystemExit(
            f"Labels in {label_path.name} do not match {news_path.name}."
        )
    labels["human_label"] = labels["human_label"].str.lower().str.strip()
    invalid = labels[~labels["human_label"].isin(LABELS)]
    if not invalid.empty:
        raise SystemExit(
            f"{len(invalid)} labels are unfinished in {label_path.name}."
        )
    distribution = dict(Counter(labels["human_label"]))
    if required_distribution is not None and distribution != required_distribution:
        raise SystemExit(
            f"{label_path.name} must contain {required_distribution}; found {distribution}."
        )
    finbert = FinBERTSentiment()
    keyword = KeywordSentiment()
    labels["finbert_prediction"] = [
        finbert.analyse(text)["sentiment"] for text in labels["headline"]
    ]
    labels["keyword_prediction"] = [
        keyword.analyse(text)["sentiment"] for text in labels["headline"]
    ]
    majority_label = labels["human_label"].value_counts().idxmax()
    labels["majority_prediction"] = majority_label
    y_true = labels["human_label"]

    def metrics(prediction_column: str) -> dict:
        y_pred = labels[prediction_column]
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        }

    result = {
        "sample_size": len(labels),
        "evaluation_design": design,
        "human_label_distribution": distribution,
        "label_order": list(LABELS),
        "finbert": metrics("finbert_prediction"),
        "keyword_baseline": metrics("keyword_prediction"),
        "majority_baseline": {"predicted_label": majority_label, **metrics("majority_prediction")},
        "limitation": limitation,
    }
    labels.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    return result, labels


def evaluate_sentiment() -> tuple[dict, pd.DataFrame]:
    result, labels = evaluate_sentiment_file(
        LABEL_PATH,
        NEWS_PATH,
        EVALUATION_DIR / "sentiment_predictions.csv",
        design="Fixed-seed random sample from the 90-day evaluation-news dataset.",
        limitation=(
            "Labels were AI-assisted rather than independently human-produced. The random "
            "sample reflects the collected news mix but contains no negative labels, so it "
            "cannot measure negative-class recall."
        ),
    )
    result["sampling_seed"] = 6201
    return result, labels


def evaluate_balanced_challenge() -> tuple[dict, pd.DataFrame]:
    return evaluate_sentiment_file(
        BALANCED_LABEL_PATH,
        CHALLENGE_NEWS_PATH,
        EVALUATION_DIR / "balanced_sentiment_predictions.csv",
        design=(
            "Purposive 30-headline challenge set from a separate 2025 Alpha Vantage pool, "
            "with 10 negative, 10 neutral and 10 positive examples. Selection and labels "
            "were completed without consulting FinBERT or Alpha Vantage sentiment."
        ),
        limitation=(
            "This class-balanced set is deliberately non-representative and its labels were "
            "AI-assisted. Use it to compare class discrimination, not to estimate real-world "
            "headline prevalence or production accuracy."
        ),
        required_distribution={"negative": 10, "neutral": 10, "positive": 10},
    )


def evaluate_guardrails() -> dict:
    scenarios = []

    def add(name: str, passed: bool, detail: str) -> None:
        scenarios.append({"scenario": name, "passed": bool(passed), "detail": detail})

    allowed, reasons = check_guardrails(
        "BUY", RiskLimits(requested_position_pct=30, max_position_pct=20)
    )
    add("oversized_position_blocked", not allowed, "; ".join(reasons))
    allowed, reasons = check_guardrails(
        "BUY", RiskLimits(estimated_loss_pct=5, max_loss_pct=3)
    )
    add("excessive_loss_blocked", not allowed, "; ".join(reasons))
    allowed, reasons = check_guardrails("BUY", RiskLimits(account_value=0))
    add("zero_account_value_blocked", not allowed, "; ".join(reasons))
    allowed, reasons = check_guardrails("BUY", RiskLimits(max_position_pct=-1))
    add("negative_limit_blocked", not allowed, "; ".join(reasons))
    allowed, reasons = check_guardrails("BUY", RiskLimits(max_loss_pct=float("nan")))
    add("non_finite_limit_blocked", not allowed, "; ".join(reasons))

    decision = Decision(
        ticker="AAPL",
        price=100,
        rsi=50,
        sentiment_score=0.5,
        signal="BUY",
        rationale="test",
        evidence=[],
        guardrail_passed=True,
        guardrail_reasons=[],
        tool_trace=[],
    )
    InvestmentAgent.simulate_execute(decision, user_confirmation=False)
    add("no_execution_without_confirmation", not decision.executed, decision.tool_trace[-1])
    passed_count = sum(item["passed"] for item in scenarios)
    return {
        "passed": passed_count,
        "total": len(scenarios),
        "pass_rate": passed_count / len(scenarios),
        "scenarios": scenarios,
    }


def evaluate_citations() -> dict:
    path = EVALUATION_DIR / "decision_record.json"
    if not path.exists():
        return {"status": "not_available", "instruction": "Run mvp.py once to create decision_record.json"}
    decision = json.loads(path.read_text(encoding="utf-8"))
    cited_ids = re.findall(r"\[(N\d+)\]", decision.get("rationale", ""))
    evidence = {str(item.get("document_id")): item for item in decision.get("evidence", [])}
    checks = []
    for citation in cited_ids:
        item = evidence.get(citation, {})
        checks.append(
            {
                "citation": citation,
                "exists_in_evidence": bool(item),
                "has_source": bool(item.get("source")),
                "has_verifiable_url": str(item.get("url", "")).startswith(("http://", "https://")),
            }
        )
    valid = sum(
        all(check[key] for key in ("exists_in_evidence", "has_source", "has_verifiable_url"))
        for check in checks
    )
    total = len(checks)
    return {
        "status": "completed",
        "citations_checked": total,
        "valid_citations": valid,
        "citation_integrity_rate": valid / total if total else 0.0,
        "checks": checks,
        "scope_note": "This verifies traceability, not whether a human agrees with the interpretation.",
    }


def run_full_evaluation() -> dict:
    EVALUATION_DIR.mkdir(exist_ok=True)
    sentiment, _predictions = evaluate_sentiment()
    report = {
        "sentiment_model_evaluation": sentiment,
        "rag_citation_evaluation": evaluate_citations(),
        "guardrail_evaluation": evaluate_guardrails(),
    }
    if BALANCED_LABEL_PATH.exists() and CHALLENGE_NEWS_PATH.exists():
        balanced, _balanced_predictions = evaluate_balanced_challenge()
        report["balanced_sentiment_challenge"] = balanced
    backtest_path = EVALUATION_DIR / "backtest_summary.json"
    if backtest_path.exists():
        report["backtest_evaluation"] = json.loads(backtest_path.read_text(encoding="utf-8"))
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FinBERT, RAG citations and guardrails")
    parser.add_argument("--prepare-labels", action="store_true")
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--review-balanced-labels", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=30)
    args = parser.parse_args()
    if args.prepare_labels:
        sample = prepare_labels(args.sample_size)
        print(f"Prepared {len(sample)} fixed-seed headlines in {LABEL_PATH}")
        print("Next: python evaluate_quality.py --label")
    elif args.label:
        label_interactively()
        print("Next: python evaluate_quality.py --run")
    elif args.review_balanced_labels:
        label_interactively(BALANCED_LABEL_PATH, review_all=True)
        print("Next: python evaluate_quality.py --run")
    elif args.run:
        report = run_full_evaluation()
        print(json.dumps(report, indent=2))
        print(f"Saved report: {REPORT_PATH}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
