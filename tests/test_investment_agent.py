import numpy as np
import pandas as pd
import json
from unittest.mock import patch

from backtest import run_backtest
from investment_agent import (
    InvestmentAgent,
    KeywordSentiment,
    OpenRouterGroundedExplainer,
    RiskLimits,
    build_grounded_rationale,
    calculate_rsi,
    check_guardrails,
    retrieve_news_evidence,
    rule_signal,
    validate_grounded_commentary,
)
from news_source import parse_alpha_vantage_feed


def price_frame(rows=120):
    dates = pd.bdate_range("2026-01-02", periods=rows)
    close = 180 + np.linspace(0, 20, rows) + np.sin(np.arange(rows) / 3) * 3
    return pd.DataFrame({"Date": dates, "Close": close})


def test_rsi_is_deterministic_and_bounded():
    result = calculate_rsi(price_frame()["Close"]).dropna()
    assert len(result) > 0
    assert result.between(0, 100).all()


def test_signal_rules_are_transparent():
    assert rule_signal(0.5, 55) == "BUY"
    assert rule_signal(-0.5, 55) == "SELL"
    assert rule_signal(0.5, 75) == "HOLD"


def test_guardrail_blocks_oversized_buy():
    passed, reasons = check_guardrails(
        "BUY", RiskLimits(requested_position_pct=30, max_position_pct=20)
    )
    assert not passed
    assert "exceeds" in reasons[0]


def test_agent_requires_human_confirmation():
    headlines = [{"headline": "Apple reported strong profit growth.", "source": "test"}]
    agent = InvestmentAgent(KeywordSentiment())
    decision = agent.run(price_frame(), headlines, RiskLimits())
    assert not decision.executed
    agent.simulate_execute(decision, False)
    assert not decision.executed


def test_90_day_backtest_has_two_baselines():
    news = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-02", "2026-04-02"]),
            "headline": ["Apple reported strong profit growth", "Apple reported a sharp loss"],
            "source": ["test", "test"],
        }
    )
    result, detail = run_backtest(
        price_frame(), news, KeywordSentiment(), coin_flip_runs=20
    )
    assert result.observations == 90
    assert isinstance(result.buy_hold_return, float)
    assert isinstance(result.coin_flip_mean_return, float)
    assert len(detail) == 90


def test_real_news_parser_keeps_sources_but_discards_provider_sentiment():
    payload = {
        "feed": [
            {
                "title": "Apple launches a new product",
                "time_published": "20260901T120000",
                "source": "Example Publisher",
                "url": "https://example.com/aapl-story",
                "overall_sentiment_score": 0.99,
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.88",
                        "ticker_sentiment_score": "0.88",
                    }
                ],
            }
        ]
    }
    news, metadata = parse_alpha_vantage_feed(payload, "2026-05-20", "2026-09-30")
    assert len(news) == 1
    assert news.iloc[0]["source"] == "Example Publisher"
    assert news.iloc[0]["url"] == "https://example.com/aapl-story"
    assert news.iloc[0]["aapl_relevance_score"] == 0.88
    assert "overall_sentiment_score" not in news.columns
    assert metadata["provider_sentiment_used"] is False


def test_real_news_parser_rejects_tangential_and_low_relevance_items():
    payload = {
        "feed": [
            {
                "title": "Nvidia lifts a broad technology ETF",
                "time_published": "20260901T080000",
                "source": "Publisher",
                "url": "https://example.com/tangential",
                "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.80"}],
            },
            {
                "title": "Apple mentioned briefly in a market roundup",
                "time_published": "20260901T090000",
                "source": "Publisher",
                "url": "https://example.com/low-relevance",
                "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.20"}],
            },
            {
                "title": "Apple reports strong iPhone demand",
                "time_published": "20260901T100000",
                "source": "Publisher",
                "url": "https://example.com/direct",
                "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.90"}],
            },
        ]
    }
    news, metadata = parse_alpha_vantage_feed(payload, "2026-05-20", "2026-09-30")
    assert list(news["url"]) == ["https://example.com/direct"]
    assert metadata["rejected_counts"]["no_direct_apple_reference"] == 1
    assert metadata["rejected_counts"]["missing_or_low_relevance"] == 1


def test_real_news_parser_ranks_each_day_by_aapl_relevance():
    feed = []
    times = ["20260901T080000", "20260901T090000", "20260901T100000"]
    for index, relevance in enumerate([0.61, 0.75, 0.95]):
        feed.append(
            {
                "title": f"Apple product update number {index}",
                "time_published": times[index],
                "source": "Publisher",
                "url": f"https://example.com/{index}",
                "ticker_sentiment": [
                    {"ticker": "AAPL", "relevance_score": str(relevance)}
                ],
            }
        )
    news, _metadata = parse_alpha_vantage_feed(
        {"feed": feed}, "2026-05-20", "2026-09-30", max_per_day=2
    )
    assert list(news["aapl_relevance_score"]) == [0.95, 0.75]


def test_rag_retrieval_returns_traceable_real_news():
    headlines = [
        {"headline": "Apple reports a decline and weak demand", "source": "Publisher A", "url": "https://example.com/a"},
        {"headline": "Apple announces a routine meeting", "source": "Publisher B", "url": "https://example.com/b"},
        {"headline": "Apple posts strong profit growth", "source": "Publisher C", "url": "https://example.com/c"},
    ]
    model = KeywordSentiment()
    results = [model.analyse(item["headline"]) for item in headlines]
    evidence = retrieve_news_evidence(headlines, results, "SELL", top_k=2)
    assert len(evidence) == 2
    assert evidence[0]["document_id"] == "N1"
    assert all(item["url"].startswith("https://") for item in evidence)


class ValidFakeExplainer:
    name = "test-grounded-llm"

    def generate(self, evidence):
        return "The retrieved headlines describe mixed Apple updates [N1] and routine coverage [N2]."


class UnsafeFakeExplainer:
    name = "test-unsafe-llm"

    def generate(self, evidence):
        return "BUY will rise 10 percent with guaranteed profit [N1]."


def test_constrained_llm_explanation_is_validated_and_cannot_change_signal():
    evidence = [
        {"document_id": "N1", "source": "A", "text": "Apple update", "url": "https://example.com/a"},
        {"document_id": "N2", "source": "B", "text": "Apple coverage", "url": "https://example.com/b"},
    ]
    rationale, mode, trace = build_grounded_rationale(
        "HOLD", "test-sentiment", 0.0, 50.0, evidence, ValidFakeExplainer()
    )
    assert mode == "llm_validated"
    assert rationale.startswith("The rule produced HOLD")
    assert "[N1]" in rationale and "[N2]" in rationale
    assert "validate_llm_explanation" in trace


def test_unsafe_llm_output_falls_back_to_deterministic_template():
    evidence = [
        {"document_id": "N1", "source": "A", "text": "Apple update", "url": "https://example.com/a"}
    ]
    valid, reasons = validate_grounded_commentary(UnsafeFakeExplainer().generate(evidence), evidence)
    assert not valid
    assert reasons
    rationale, mode, trace = build_grounded_rationale(
        "HOLD", "test-sentiment", 0.0, 50.0, evidence, UnsafeFakeExplainer()
    )
    assert mode == "template_fallback"
    assert "fallback_to_grounded_template" in trace
    assert rationale.startswith("The rule produced HOLD")


class FakeOpenRouterResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": "Apple issued an update [N1]."}}]}
        ).encode("utf-8")


def test_openrouter_explainer_sends_constrained_deterministic_request():
    evidence = [
        {"document_id": "N1", "source": "A", "text": "Apple issued an update"}
    ]
    explainer = OpenRouterGroundedExplainer("secret-test-key", "provider/test-model")
    with patch("investment_agent.urlopen", return_value=FakeOpenRouterResponse()) as mocked:
        result = explainer.generate(evidence)
    request = mocked.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert result == "Apple issued an update [N1]."
    assert payload["model"] == "provider/test-model"
    assert payload["temperature"] == 0
    assert "secret-test-key" not in request.data.decode("utf-8")
    assert request.get_header("Authorization") == "Bearer secret-test-key"
