import numpy as np
import pandas as pd

from backtest import run_backtest
from investment_agent import (
    InvestmentAgent,
    KeywordSentiment,
    RiskLimits,
    calculate_rsi,
    check_guardrails,
    retrieve_news_evidence,
    rule_signal,
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
