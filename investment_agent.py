"""Small, auditable AAPL investment-decision agent.

AI is limited to headline sentiment and optional grounded wording.  RSI, the
signal, guardrails and simulated execution are deterministic Python code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class SentimentModel(Protocol):
    name: str

    def analyse(self, text: str) -> dict[str, Any]: ...


class FinBERTSentiment:
    """Lazy FinBERT adapter so importing the project stays fast."""

    name = "ProsusAI/finbert"

    def __init__(self) -> None:
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "FinBERT needs transformers and torch. Run: pip install -r requirements.txt"
                ) from exc
            self._pipeline = pipeline(
                "text-classification", model=self.name, tokenizer=self.name, top_k=None
            )
        return self._pipeline

    def analyse(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Headline cannot be empty")
        raw = self._load()(cleaned, truncation=True)[0]
        probabilities = {item["label"].lower(): float(item["score"]) for item in raw}
        label = max(probabilities, key=probabilities.get)
        return {
            "sentiment": label,
            "confidence": probabilities[label],
            "score": probabilities.get("positive", 0.0) - probabilities.get("negative", 0.0),
            "probabilities": probabilities,
            "model": self.name,
        }


class KeywordSentiment:
    """Transparent offline fallback for tests/demos; it is not FinBERT."""

    name = "keyword-demo-fallback"
    POSITIVE = {"growth", "grew", "gain", "profit", "record", "increase", "strong", "repurchase"}
    NEGATIVE = {"loss", "fell", "decline", "risk", "weak", "decrease", "charge", "uncertain"}

    def analyse(self, text: str) -> dict[str, Any]:
        words = {token.strip(".,:;!?()[]\"").lower() for token in text.split()}
        raw = len(words & self.POSITIVE) - len(words & self.NEGATIVE)
        score = float(np.tanh(raw / 2))
        sentiment = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        return {
            "sentiment": sentiment,
            "confidence": abs(score) if sentiment != "neutral" else 1 - abs(score),
            "score": score,
            "probabilities": None,
            "model": self.name,
        }


@dataclass(frozen=True)
class RiskLimits:
    account_value: float = 10_000.0
    requested_position_pct: float = 10.0
    max_position_pct: float = 20.0
    estimated_loss_pct: float = 2.0
    max_loss_pct: float = 3.0


@dataclass
class Decision:
    ticker: str
    price: float
    rsi: float
    sentiment_score: float
    signal: str
    rationale: str
    evidence: list[dict[str, Any]]
    guardrail_passed: bool
    guardrail_reasons: list[str]
    tool_trace: list[str]
    executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder RSI without using an LLM."""
    values = pd.to_numeric(close, errors="coerce")
    delta = values.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return rsi


def aggregate_sentiment(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return float(np.mean([float(item["score"]) for item in results]))


def rule_signal(sentiment_score: float, rsi: float) -> str:
    """Pre-registered, inspectable decision rule."""
    if not 0 <= rsi <= 100:
        raise ValueError("RSI must be between 0 and 100")
    if sentiment_score >= 0.15 and rsi < 70:
        return "BUY"
    if sentiment_score <= -0.15 and rsi > 30:
        return "SELL"
    return "HOLD"


def check_guardrails(signal: str, limits: RiskLimits) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    numeric = [
        limits.account_value,
        limits.requested_position_pct,
        limits.max_position_pct,
        limits.estimated_loss_pct,
        limits.max_loss_pct,
    ]
    if any(not np.isfinite(value) or value < 0 for value in numeric):
        reasons.append("Risk settings must be finite non-negative numbers.")
    if limits.account_value <= 0:
        reasons.append("Account value must be greater than zero.")
    if signal == "BUY" and limits.requested_position_pct > limits.max_position_pct:
        reasons.append(
            f"Requested position {limits.requested_position_pct:.1f}% exceeds "
            f"the {limits.max_position_pct:.1f}% cap."
        )
    if signal == "BUY" and limits.estimated_loss_pct > limits.max_loss_pct:
        reasons.append(
            f"Estimated loss {limits.estimated_loss_pct:.1f}% exceeds "
            f"the {limits.max_loss_pct:.1f}% cap."
        )
    return not reasons, reasons


def retrieve_news_evidence(
    headlines: list[dict[str, str]],
    sentiment_results: list[dict[str, Any]],
    signal: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rank real headlines using lexical relevance plus signal direction."""
    query_by_signal = {
        "BUY": "Apple growth profit demand launch gains positive performance",
        "SELL": "Apple risk decline loss concern weak negative performance",
        "HOLD": "Apple mixed neutral update outlook performance",
    }
    texts = [item["headline"] for item in headlines]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts + [query_by_signal[signal]])
    lexical = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    if lexical.max(initial=0) > 0:
        lexical = lexical / lexical.max()
    scores: list[float] = []
    for lexical_score, result in zip(lexical, sentiment_results):
        sentiment = float(result["score"])
        if signal == "BUY":
            directional = max(sentiment, 0.0)
        elif signal == "SELL":
            directional = max(-sentiment, 0.0)
        else:
            directional = max(0.0, 1.0 - abs(sentiment))
        scores.append(0.4 * float(lexical_score) + 0.6 * directional)
    ranked = sorted(
        [
            {
                "document_id": f"N{index}",
                "source": item.get("source", "Unknown source"),
                "text": item["headline"],
                "url": item.get("url", ""),
                "retrieval_score": float(score),
            }
            for index, (item, score) in enumerate(zip(headlines, scores), start=1)
        ],
        key=lambda item: item["retrieval_score"],
        reverse=True,
    )
    return ranked[: min(top_k, len(ranked))]


class InvestmentAgent:
    """Orchestrates data validation, tools, retrieval and human approval."""

    def __init__(self, sentiment_model: SentimentModel):
        self.sentiment_model = sentiment_model

    def run(
        self,
        prices: pd.DataFrame,
        headlines: list[dict[str, str]],
        limits: RiskLimits,
        ticker: str = "AAPL",
    ) -> Decision:
        trace = ["validate_market_data"]
        if ticker != "AAPL":
            raise ValueError("This MVP is intentionally limited to AAPL")
        if "Close" not in prices or len(prices) < 15:
            raise ValueError("At least 15 valid AAPL closing prices are required")
        clean_close = pd.to_numeric(prices["Close"], errors="coerce").dropna()
        if len(clean_close) < 15 or (clean_close <= 0).any():
            raise ValueError("Price data contain missing or invalid closing prices")
        if not 1 <= len(headlines) <= 5:
            raise ValueError("Provide between one and five AAPL headlines")

        trace.append("calculate_rsi_python")
        rsi = float(calculate_rsi(clean_close).dropna().iloc[-1])
        price = float(clean_close.iloc[-1])

        trace.append(f"analyse_headlines_{self.sentiment_model.name}")
        sentiment_results = [self.sentiment_model.analyse(item["headline"]) for item in headlines]
        sentiment_score = aggregate_sentiment(sentiment_results)

        trace.append("apply_deterministic_signal_rule")
        signal = rule_signal(sentiment_score, rsi)

        trace.append("retrieve_grounding_evidence")
        evidence = retrieve_news_evidence(headlines, sentiment_results, signal, top_k=3)

        trace.append("generate_grounded_template_explanation")
        cited = ", ".join(f"[{item['document_id']}]" for item in evidence[:3])
        rationale = (
            f"The rule produced {signal}: aggregate {self.sentiment_model.name} sentiment score "
            f"{sentiment_score:+.3f}, 14-day RSI {rsi:.1f}. Supporting retrieved "
            f"evidence: {cited}. This is a simulated educational signal, not financial advice."
        )

        trace.append("apply_non_ai_guardrails")
        passed, reasons = check_guardrails(signal, limits)
        return Decision(
            ticker=ticker,
            price=price,
            rsi=rsi,
            sentiment_score=sentiment_score,
            signal=signal,
            rationale=rationale,
            evidence=evidence,
            guardrail_passed=passed,
            guardrail_reasons=reasons,
            tool_trace=trace,
        )

    @staticmethod
    def simulate_execute(decision: Decision, user_confirmation: bool) -> Decision:
        decision.tool_trace.append("request_human_confirmation")
        decision.executed = bool(user_confirmation and decision.guardrail_passed)
        decision.tool_trace.append("simulate_execution" if decision.executed else "stop_without_execution")
        return decision
