"""
================================================================================
CRYPTO TRADING DECISION ENGINE (Step 3: Multi-Factor Quantitative Strategy)
================================================================================
Author: Senior Quantitative Python Developer
Dependencies: sentiment_engine (FinBERT), market_data (CCXT / Binance)
Description:
    Synthesizes NLP Sentiment Alpha (ProsusAI/finbert) with Live Market
    Microstructure (Binance Ticker & Momentum) to generate quantitative
    BUY, SELL, or HOLD execution signals with risk-managed rationales.
================================================================================
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

# Import modules from Steps 1 and 2
try:
    from sentiment_engine import (
        AnalyzedHeadline,
        CryptoNewsFetcher,
        FinBertSentimentEngine,
        MarketSentimentAggregator,
        MarketSentimentSummary,
    )
    from market_data import fetch_live_ticker, fetch_ohlcv_dataframe, init_exchange
except ImportError as exc:
    print(f"[!] Error importing project components: {exc}")
    print("    Ensure sentiment_engine.py and market_data.py are in the working directory.")
    sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ==============================================================================
# LOGGING & CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DecisionEngine")

# Standard token mapping from ticker symbol to news search keyword
SYMBOL_TO_TOKEN_MAP = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "XRP": "Ripple",
    "ADA": "Cardano",
    "DOGE": "Dogecoin",
    "AVAX": "Avalanche",
    "BNB": "Binance",
}


_SENTIMENT_ENGINE: Optional[FinBertSentimentEngine] = None


def _get_sentiment_engine() -> FinBertSentimentEngine:
    """Loads FinBERT once per process; reloading it every cycle is slow and wasteful."""
    global _SENTIMENT_ENGINE
    if _SENTIMENT_ENGINE is None:
        _SENTIMENT_ENGINE = FinBertSentimentEngine()
    return _SENTIMENT_ENGINE


# ==============================================================================
# DATA MODELS & ENUMS
# ==============================================================================
class ActionSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class QuantitativeFactors:
    """Individual factor readings contributing to the decision."""
    sentiment_score: float         # [-1.0 to +1.0]
    sentiment_label: str           # Bullish, Bearish, Neutral
    headline_count: int
    price_change_24h_pct: float    # e.g., +2.45%
    momentum_positive: bool        # True if 24h change > 0
    drawdown_breached: bool        # True if price fell past stop threshold
    high_low_spread_pct: float     # (High - Low) / Low * 100
    volume_24h_usd: float          # 24h quote volume


@dataclass
class PaperTradingContext:
    """Readiness container for Step 4 Paper Trading execution."""
    entry_price: float
    suggested_stop_loss: float     # E.g. -2.0% below entry
    suggested_take_profit: float   # E.g. +4.0% above entry (2.0 R:R)
    risk_reward_ratio: float = 2.0


@dataclass
class TradingDecision:
    """Final synthesized multi-factor trading decision."""
    symbol: str
    token_name: str
    timestamp_utc: str
    current_price: float
    action: ActionSignal
    confidence: float              # Overall signal conviction [0.0 to 1.0]
    rationale: str
    factors: QuantitativeFactors
    paper_order: Optional[PaperTradingContext] = None


# ==============================================================================
# STRATEGY RULES & MULTI-FACTOR DECISION ENGINE
# ==============================================================================
class MultiFactorDecisionEngine:
    """
    Combines NLP News Sentiment (Factor 1) and Price Action / Momentum (Factor 2)
    to generate algorithmic trading signals.

    QUANTITATIVE RULES:
    1. BUY SIGNAL:
       - Sentiment Score > sentiment_bull_thresh (e.g., +0.30)
       - 24-hour price change is positive (> 0.0%)
       Confluence: Bullish news sentiment backed by upward technical momentum.

    2. SELL / EXIT SIGNAL:
       - Sentiment Score < sentiment_bear_thresh (e.g., -0.30)
         OR
       - 24-hour price change < max_drawdown_stop_pct (e.g., -3.5%)
       Confluence: Macro/news panic OR severe downside violation requiring capital exit.

    3. HOLD SIGNAL:
       - Conflicting signals (e.g. Bullish sentiment but falling price)
       - Neutral sentiment range [-0.30 to +0.30]
       - Low conviction consolidation.
    """

    def __init__(
        self,
        sentiment_bull_thresh: float = 0.30,
        sentiment_bear_thresh: float = -0.30,
        drawdown_stop_pct: float = -3.50,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
    ):
        self.bull_thresh = sentiment_bull_thresh
        self.bear_thresh = sentiment_bear_thresh
        self.drawdown_stop_pct = drawdown_stop_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def evaluate(
        self,
        symbol: str,
        token_name: str,
        ticker: Dict[str, Any],
        sentiment_summary: MarketSentimentSummary,
    ) -> TradingDecision:
        """
        Synthesizes market data and sentiment into a decisive ActionSignal.
        """
        current_price = float(ticker.get("last_price") or 0.0)
        price_change_24h = float(ticker.get("percentage_24h") or 0.0)
        high_24h = float(ticker.get("high_24h") or current_price)
        low_24h = float(ticker.get("low_24h") or current_price)
        quote_vol = float(ticker.get("quote_volume") or 0.0)

        sentiment_score = sentiment_summary.market_sentiment_index
        momentum_pos = price_change_24h > 0.0
        drawdown_violated = price_change_24h <= self.drawdown_stop_pct

        spread_pct = ((high_24h - low_24h) / low_24h * 100.0) if low_24h > 0 else 0.0

        # Classify sentiment regime
        if sentiment_score >= self.bull_thresh:
            sentiment_regime = "Bullish"
        elif sentiment_score <= self.bear_thresh:
            sentiment_regime = "Bearish"
        else:
            sentiment_regime = "Neutral"

        factors = QuantitativeFactors(
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_regime,
            headline_count=sentiment_summary.total_headlines,
            price_change_24h_pct=price_change_24h,
            momentum_positive=momentum_pos,
            drawdown_breached=drawdown_violated,
            high_low_spread_pct=spread_pct,
            volume_24h_usd=quote_vol,
        )

        # ----------------------------------------------------------------------
        # Strategy Rule Execution
        # ----------------------------------------------------------------------
        # Case 1: BUY
        if (sentiment_score > self.bull_thresh) and momentum_pos:
            action = ActionSignal.BUY
            confidence = min(1.0, (abs(sentiment_score) * 0.6) + (min(abs(price_change_24h), 10) / 10 * 0.4))
            rationale = (
                f"BULLISH CONFLUENCE: Sentiment index ({sentiment_score:+.2f}) exceeds +{self.bull_thresh:.2f} "
                f"threshold and is confirmed by positive 24h price momentum ({price_change_24h:+.2f}%)."
            )

        # Case 2: SELL / EXIT
        elif (sentiment_score < self.bear_thresh) or drawdown_violated:
            action = ActionSignal.SELL
            confidence = min(1.0, (abs(sentiment_score) * 0.6) + (abs(price_change_24h) / 10 * 0.4))
            if sentiment_score < self.bear_thresh and drawdown_violated:
                rationale = (
                    f"DUAL RISK EXIT: Bearish sentiment ({sentiment_score:+.2f}) AND price breakdown "
                    f"({price_change_24h:+.2f}% <= {self.drawdown_stop_pct:.2f}% stop threshold). Liquidate/Avoid."
                )
            elif sentiment_score < self.bear_thresh:
                rationale = (
                    f"SENTIMENT BREAKDOWN: Heavy negative sentiment ({sentiment_score:+.2f} < {self.bear_thresh:.2f}) "
                    f"signals macro/institutional news risk. Recommend derisking."
                )
            else:
                rationale = (
                    f"TECHNICAL STOP-LOSS HIT: 24h drop of {price_change_24h:+.2f}% breached maximum risk "
                    f"threshold ({self.drawdown_stop_pct:.2f}%). Capital protection exit."
                )

        # Case 3: HOLD (Divergence or Neutrality)
        else:
            action = ActionSignal.HOLD
            confidence = 0.50
            if sentiment_score > self.bull_thresh and not momentum_pos:
                rationale = (
                    f"BULLISH DIVERGENCE: Sentiment is bullish ({sentiment_score:+.2f}), but price action is "
                    f"lagging/negative ({price_change_24h:+.2f}%). Wait for price confirmation before entry."
                )
            elif sentiment_score < self.bear_thresh and momentum_pos:
                rationale = (
                    f"BEARISH DIVERGENCE: Price is positive ({price_change_24h:+.2f}%), but underlying news "
                    f"sentiment is hostile ({sentiment_score:+.2f}). High fakeout risk."
                )
            else:
                rationale = (
                    f"CONSOLIDATION / NEUTRAL: Sentiment ({sentiment_score:+.2f}) and price action ({price_change_24h:+.2f}%) "
                    f"are in the neutral band. No clear edge."
                )

        # ----------------------------------------------------------------------
        # Paper Trading Plan (Preparation for Step 4)
        # ----------------------------------------------------------------------
        paper_context = PaperTradingContext(
            entry_price=current_price,
            suggested_stop_loss=round(current_price * (1 - (self.stop_loss_pct / 100)), 4),
            suggested_take_profit=round(current_price * (1 + (self.take_profit_pct / 100)), 4),
            risk_reward_ratio=self.take_profit_pct / self.stop_loss_pct,
        )

        return TradingDecision(
            symbol=symbol,
            token_name=token_name,
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            current_price=current_price,
            action=action,
            confidence=confidence,
            rationale=rationale,
            factors=factors,
            paper_order=paper_context,
        )


# ==============================================================================
# TERMINAL DECISION DASHBOARD
# ==============================================================================
class DecisionDashboard:
    """Renders a polished, professional quantitative trading dashboard."""

    @classmethod
    def render(cls, decision: TradingDecision, headlines: list):
        """Displays the full decision dashboard."""
        f = decision.factors
        quote_currency = decision.symbol.split("/")[1] if "/" in decision.symbol else "USDT"

        # Action formatting badge
        if decision.action == ActionSignal.BUY:
            badge = ">>> [ BUY SIGNAL ] <<< (LONG ENTRY)"
            action_color = "[+]"
        elif decision.action == ActionSignal.SELL:
            badge = ">>> [ SELL / EXIT ] <<< (FLATTEN / DE-RISK)"
            action_color = "[-]"
        else:
            badge = ">>> [ HOLD / WAIT ] <<< (NO ACTION)"
            action_color = "[=]"

        print("\n" + "=" * 90)
        print(f"   QUANTITATIVE TRADING DECISION DASHBOARD (Step 3: Multi-Factor Engine)")
        print(f"   Target: {decision.symbol} ({decision.token_name})  |  Time: {decision.timestamp_utc}")
        print("=" * 90)

        # 1. Executive Summary Table
        summary_table = [
            ["Asset Pair", decision.symbol],
            ["Target Asset Name", decision.token_name],
            ["Current Market Price", f"${decision.current_price:,.4f} {quote_currency}"],
            ["24-Hour Price Change", f"{'+' if f.price_change_24h_pct >= 0 else ''}{f.price_change_24h_pct:.2f}%"],
            ["24-Hour Trading Volume", f"${f.volume_24h_usd:,.2f} {quote_currency}"],
            ["24-Hour High / Low Spread", f"{f.high_low_spread_pct:.2f}% volatility range"],
            ["NLP Sentiment Index", f"{f.sentiment_score:+.4f} ({f.sentiment_label})"],
            ["News Sample Size", f"{f.headline_count} analyzed headlines"],
            ["Decision Conviction", f"{decision.confidence * 100:.1f}%"],
            ["FINAL BOT DECISION", f"{action_color} {decision.action.value}"],
        ]

        if HAS_TABULATE:
            print(tabulate(summary_table, headers=["Metric", "Reading"], tablefmt="grid"))
        else:
            for row in summary_table:
                print(f"  {row[0]:<26}: {row[1]}")

        # 2. Strategy Verdict & Rationale
        print("\n" + "-" * 90)
        print(f"   STRATEGY VERDICT: {badge}")
        print("-" * 90)
        print(f"  * Rationale: {decision.rationale}")

        # 3. Paper Trading Execution Framework (Step 4 Hook)
        if decision.paper_order and decision.action == ActionSignal.BUY:
            po = decision.paper_order
            print("\n" + "-" * 90)
            print("   PRE-TRADE EXECUTION RISK BRACKET (Step 4 Ready)")
            print("-" * 90)
            print(f"  * Limit Entry Price : ${po.entry_price:,.4f}")
            sl_pct = (po.suggested_stop_loss / po.entry_price - 1) * 100 if po.entry_price else 0.0
            tp_pct = (po.suggested_take_profit / po.entry_price - 1) * 100 if po.entry_price else 0.0
            print(f"  * Stop-Loss Target  : ${po.suggested_stop_loss:,.4f} ({sl_pct:+.1f}%)")
            print(f"  * Take-Profit Target: ${po.suggested_take_profit:,.4f} ({tp_pct:+.1f}%)")
            print(f"  * Planned R:R Ratio : {po.risk_reward_ratio:.1f} : 1.0 (Positive Expectancy)")

        # 4. Recent Headlines Sample
        if headlines:
            print("\n" + "-" * 90)
            print(f"   KEY HEADLINES DRIVING SENTIMENT FACTOR")
            print("-" * 90)
            for idx, item in enumerate(headlines[:4], 1):
                sign = "+" if item.sentiment.polarity_score > 0 else ""
                print(f"  {idx}. [{item.sentiment.label.upper():<8}] ({sign}{item.sentiment.polarity_score:.2f}) {item.article.title[:70]}...")

        print("=" * 90 + "\n")


# ==============================================================================
# PIPELINE ORCHESTRATION & CLI
# ==============================================================================
def run_trading_decision_pipeline(
    symbol: str = "SOL/USDT",
    token: Optional[str] = None,
    news_limit: int = 10,
    bull_thresh: float = 0.30,
    bear_thresh: float = -0.30,
    stop_loss_pct: float = -3.50,
    render_dashboard: bool = True,
) -> TradingDecision:
    """
    Executes the end-to-end multi-factor trading decision pipeline:
    1. Determines token keyword from trading pair.
    2. Ingests news & runs FinBERT sentiment evaluation (Step 1).
    3. Ingests Binance live ticker & order book spread (Step 2).
    4. Evaluates multi-factor rules & outputs actionable decision.
    """
    clean_symbol = symbol.upper().strip()
    base_asset = clean_symbol.split("/")[0]

    # Resolve token name
    token_name = token or SYMBOL_TO_TOKEN_MAP.get(base_asset, base_asset)
    logger.info("Executing Decision Engine for %s (%s)...", clean_symbol, token_name)

    # --------------------------------------------------------------------------
    # Step 1: Ingest Sentiment (FinBERT)
    # --------------------------------------------------------------------------
    logger.info("[Factor 1] Ingesting RSS headlines for '%s'...", token_name)
    news_fetcher = CryptoNewsFetcher()
    articles = news_fetcher.fetch_headlines(token=token_name, max_articles=news_limit)

    if articles:
        engine = _get_sentiment_engine()
        sent_results = engine.analyze_batch([a.title for a in articles])
        analyzed_items = [
            AnalyzedHeadline(article=art, sentiment=res)
            for art, res in zip(articles, sent_results)
        ]
        sentiment_summary = MarketSentimentAggregator.aggregate(
            analyzed_items, target_token=token_name
        )
    else:
        logger.warning("No recent news found for '%s'. Falling back to neutral sentiment.", token_name)
        analyzed_items = []
        sentiment_summary = MarketSentimentAggregator.aggregate([], target_token=token_name)

    # --------------------------------------------------------------------------
    # Step 2: Ingest Live Market Data (Binance / CCXT)
    # --------------------------------------------------------------------------
    logger.info("[Factor 2] Fetching live ticker for '%s'...", clean_symbol)
    exchange = init_exchange()
    ticker = fetch_live_ticker(exchange, clean_symbol)

    if not ticker or not ticker.get("last_price"):
        # Raise (not sys.exit) so the 24/7 loop in main.py can catch it and keep running
        raise RuntimeError(f"Failed to fetch live market data for '{clean_symbol}'.")

    # --------------------------------------------------------------------------
    # Step 3: Multi-Factor Evaluation & Dashboard
    # --------------------------------------------------------------------------
    strategy = MultiFactorDecisionEngine(
        sentiment_bull_thresh=bull_thresh,
        sentiment_bear_thresh=bear_thresh,
        drawdown_stop_pct=stop_loss_pct,
    )

    decision = strategy.evaluate(
        symbol=clean_symbol,
        token_name=token_name,
        ticker=ticker,
        sentiment_summary=sentiment_summary,
    )

    if render_dashboard:
        DecisionDashboard.render(decision, analyzed_items)
    return decision


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3: Multi-Factor Crypto Trading Decision Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SOL/USDT",
        help="Trading pair symbol (e.g. 'SOL/USDT', 'BTC/USDT', 'ETH/USDT')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token name for news search (auto-detected if omitted)",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=8,
        help="Number of headlines to analyze with FinBERT",
    )
    parser.add_argument(
        "--bull-thresh",
        type=float,
        default=0.30,
        help="Sentiment score threshold for BUY (> +0.30)",
    )
    parser.add_argument(
        "--bear-thresh",
        type=float,
        default=-0.30,
        help="Sentiment score threshold for SELL (< -0.30)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=-3.50,
        help="24h drawdown % that triggers automatic risk exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_arguments()
    try:
        run_trading_decision_pipeline(
            symbol=cli_args.symbol,
            token=cli_args.token,
            news_limit=cli_args.news_limit,
            bull_thresh=cli_args.bull_thresh,
            bear_thresh=cli_args.bear_thresh,
            stop_loss_pct=cli_args.stop_loss_pct,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
