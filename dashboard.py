"""
================================================================================
CRYPTO TRADING BOT: REAL-TIME WEB DASHBOARD (Streamlit)
================================================================================
Author: Senior Python Quantitative Developer
Dependencies: streamlit, ccxt, pandas, transformers, torch, requests, feedparser
Description:
    Real-time interactive dashboard visualizing:
    1. Wallet Balances, 24h PnL, and Live Token Prices.
    2. Interactive CCXT OHLCV Candlestick & Price Line Charts.
    3. NLP FinBERT Sentiment News Feed with color-coded badges.
    4. Multi-Factor Bot Decision & Position Status.
    5. Configurable Sidebar controls & auto-refresh settings.
================================================================================
"""

import os
import sys

# Suppress verbose warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Import CBT modules
try:
    from decision_engine import (
        ActionSignal,
        MultiFactorDecisionEngine,
        SYMBOL_TO_TOKEN_MAP,
    )
    from market_data import fetch_live_ticker, fetch_ohlcv_dataframe, init_exchange
    from paper_trading import BinanceTestnetBroker
    from sentiment_engine import (
        AnalyzedHeadline,
        CryptoNewsFetcher,
        FinBertSentimentEngine,
        MarketSentimentAggregator,
        MarketSentimentSummary,
        NewsArticle,
        SentimentResult,
    )
except ImportError as err:
    st.error(f"Failed to import CBT core modules: {err}")
    st.stop()


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="CBT | AI Crypto Trading Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Quantitative Terminal CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
    }
    /* Hide default Streamlit Deploy button (local quantitative terminal mode) */
    .stDeployButton, [data-testid="stToolbar"] .stDeployButton {
        display: none !important;
    }
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }
    /* News Card Styling */
    .news-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .news-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .news-title {
        font-size: 1rem;
        font-weight: 600;
        color: #f1f5f9;
        text-decoration: none;
    }
    .news-title:hover {
        color: #38bdf8;
    }
    .news-meta {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
    }
    /* Sentiment Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 8px;
    }
    .badge-pos {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .badge-neg {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-neu {
        background: rgba(148, 163, 184, 0.15);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
    .badge-buy {
        background: rgba(34, 197, 94, 0.25);
        color: #22c55e;
        border: 1px solid #22c55e;
        padding: 6px 14px;
        font-size: 0.9rem;
    }
    .badge-sell {
        background: rgba(239, 68, 68, 0.25);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        font-size: 0.9rem;
    }
    .badge-hold {
        background: rgba(234, 179, 8, 0.25);
        color: #eab308;
        border: 1px solid #eab308;
        padding: 6px 14px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# FAST RESOURCE CACHING (OFFLINE-FIRST FOR ZERO-LATENCY LOAD)
# ==============================================================================
@st.cache_resource(show_spinner="Loading FinBERT AI engine...")
def get_cached_finbert():
    """
    Caches FinBERT in memory. Loads from local disk cache first
    to eliminate slow Hugging Face API network round-trips.
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", local_files_only=True)
        return FinBertSentimentEngine(tokenizer=tokenizer, model=model, local_files_only=True)
    except Exception:
        # Fallback to standard offline loader
        return FinBertSentimentEngine(local_files_only=True)

@st.cache_resource(show_spinner=False)
def get_cached_exchange():
    """Caches the CCXT Binance exchange connection with vision mirror fallback."""
    return init_exchange()

@st.cache_resource(show_spinner=False)
def get_cached_broker():
    """Maintains a persistent in-memory paper trading broker across UI reruns."""
    return BinanceTestnetBroker(initial_usdt_balance=1000.0)


# ==============================================================================
# DATA INGESTION HELPERS WITH TTL CACHE (60s CACHE PREVENTS UI HANGS)
# ==============================================================================
@st.cache_data(ttl=30, show_spinner=False)
def load_market_data(
    symbol: str, timeframe: str = "1h", limit: int = 40
) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """Fetches real-time Binance ticker and recent OHLCV records (cached for 30s)."""
    try:
        exchange = get_cached_exchange()
        ticker = fetch_live_ticker(exchange, symbol)
        df = fetch_ohlcv_dataframe(exchange, symbol, timeframe=timeframe, limit=limit)
        return ticker, df
    except Exception as exc:
        st.warning(f"Market data fetch issue: {exc}")
        return None, None

@st.cache_data(ttl=60, show_spinner=False)
def load_sentiment_data(
    token_name: str, max_headlines: int = 8
) -> Tuple[List[AnalyzedHeadline], MarketSentimentSummary]:
    """Fetches RSS news headlines and evaluates sentiment with FinBERT (cached for 60s)."""
    try:
        fetcher = CryptoNewsFetcher()
        # Fast 4-second timeout per feed prevents blocking
        articles = fetcher.fetch_headlines(token=token_name, max_articles=max_headlines, timeout=4)

        if not articles:
            # Fallback mock headlines if offline
            empty_summary = MarketSentimentAggregator.aggregate([], target_token=token_name)
            return [], empty_summary

        engine = get_cached_finbert()
        titles = [a.title for a in articles]
        results = engine.analyze_batch(titles)

        analyzed = [
            AnalyzedHeadline(article=art, sentiment=res)
            for art, res in zip(articles, results)
        ]
        summary = MarketSentimentAggregator.aggregate(analyzed, target_token=token_name)
        return analyzed, summary
    except Exception as exc:
        st.warning(f"Sentiment ingestion issue: {exc}")
        empty_summary = MarketSentimentAggregator.aggregate([], target_token=token_name)
        return [], empty_summary


# ==============================================================================
# SIDEBAR CONTROLS & SETTINGS
# ==============================================================================
with st.sidebar:
    st.title("⚡ CBT Controls")
    st.caption("AI-Powered Algorithmic Crypto Terminal")

    st.markdown("---")
    st.subheader("🎯 Market Selection")

    selected_symbol = st.selectbox(
        "Trading Pair",
        options=["SOL/USDT", "BTC/USDT", "ETH/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT"],
        index=0,
    )

    base_asset = selected_symbol.split("/")[0]
    token_keyword = SYMBOL_TO_TOKEN_MAP.get(base_asset, base_asset)

    col_tf, col_limit = st.columns(2)
    with col_tf:
        selected_timeframe = st.selectbox(
            "Timeframe",
            options=["15m", "1h", "4h", "1d"],
            index=1,
        )
    with col_limit:
        candle_limit = st.slider("Candles", min_value=15, max_value=100, value=35, step=5)

    news_count = st.slider("Headlines Analyzed", min_value=3, max_value=15, value=6)

    st.markdown("---")
    st.subheader("⚙️ Strategy Thresholds")
    bull_thresh = st.slider("Bullish BUY Threshold", min_value=0.10, max_value=0.60, value=0.30, step=0.05)
    bear_thresh = st.slider("Bearish SELL Threshold", min_value=-0.60, max_value=-0.10, value=-0.30, step=0.05)
    stop_loss_pct = st.slider("24h Drawdown Stop (%)", min_value=-10.0, max_value=-1.0, value=-3.5, step=0.5)

    st.markdown("---")
    st.subheader("🔄 Live Refresh Settings")
    auto_refresh = st.checkbox("Auto-Refresh Dashboard", value=False)
    refresh_rate = st.selectbox("Interval", options=[15, 30, 60, 120, 300], index=1, format_func=lambda s: f"{s} seconds")

    refresh_clicked = st.button("🔄 Refresh Data Now", use_container_width=True, type="primary")

    if refresh_clicked:
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Environment: **Binance Testnet / Paper Sandbox**")
    st.caption("Network: **data-api.binance.vision** (Active)")


# ==============================================================================
# HEADER (RENDERED IMMEDIATELY SO SCREEN IS NEVER BLANK)
# ==============================================================================
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800;">
                <span style="color: #38bdf8;">CBT</span> Quantitative Dashboard
            </h1>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Real-Time Multi-Factor Sentiment & Order Execution Terminal
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(34, 197, 94, 0.2); color: #4ade80; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid rgba(34, 197, 94, 0.4);">
                ● SYSTEM OPERATIONAL
            </span>
            <p style="margin: 6px 0 0 0; color: #64748b; font-size: 0.8rem;">
                Last Sync: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# DATA LOAD & REFRESH LOGIC (WITH RESPONSIVE STATUS BAR)
# ==============================================================================
with st.status(f"Ingesting live market data and FinBERT sentiment for {selected_symbol}...", expanded=False) as status:
    st.write("Fetching Binance Ticker & Candlesticks...")
    ticker, ohlcv_df = load_market_data(
        symbol=selected_symbol,
        timeframe=selected_timeframe,
        limit=candle_limit,
    )
    st.write(f"Querying crypto news & analyzing FinBERT sentiment for {token_keyword}...")
    headlines, sentiment_summary = load_sentiment_data(
        token_name=token_keyword,
        max_headlines=news_count,
    )
    if hasattr(status, "update"):
        status.update(label="Market Data & AI Sentiment Synced!", state="complete", expanded=False)


# Instantiate strategy decision engine
broker = get_cached_broker()
strategy = MultiFactorDecisionEngine(
    sentiment_bull_thresh=bull_thresh,
    sentiment_bear_thresh=bear_thresh,
    drawdown_stop_pct=stop_loss_pct,
)

if ticker:
    decision = strategy.evaluate(
        symbol=selected_symbol,
        token_name=token_keyword,
        ticker=ticker,
        sentiment_summary=sentiment_summary,
    )
else:
    decision = None


# ==============================================================================
# TOP METRIC CARDS
# ==============================================================================
cash_balance = broker.get_balance("USDT")
initial_capital = 1000.0
total_pnl_usd = cash_balance - initial_capital
total_pnl_pct = (total_pnl_usd / initial_capital) * 100.0

active_pos = broker.get_position(selected_symbol)
last_price = ticker.get("last_price", 0.0) if ticker else 0.0
chg_24h = ticker.get("percentage_24h", 0.0) if ticker else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Wallet Balance (USDT)",
        value=f"${cash_balance:,.2f}",
        delta=f"{'+' if total_pnl_usd >= 0 else ''}${total_pnl_usd:,.2f} ({total_pnl_pct:+.2f}%)",
    )

with col2:
    st.metric(
        label="24h Profit / Loss",
        value=f"{'+' if total_pnl_usd >= 0 else ''}${total_pnl_usd:,.2f}",
        delta=f"{total_pnl_pct:+.2f}% vs initial",
    )

with col3:
    st.metric(
        label=f"{selected_symbol} Last Price",
        value=f"${last_price:,.4f}" if last_price < 1.0 else f"${last_price:,.2f}",
        delta=f"{chg_24h:+.2f}% (24h)",
    )

with col4:
    sent_score = sentiment_summary.market_sentiment_index
    st.metric(
        label="Market Sentiment Index",
        value=f"{sent_score:+.4f}",
        delta=f"{sentiment_summary.trading_signal.split()[0]} ({sentiment_summary.total_headlines} headlines)",
    )

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# STRATEGY ACTION BANNER
# ==============================================================================
if decision:
    if decision.action == ActionSignal.BUY:
        badge_html = '<span class="badge badge-buy">🚀 BUY SIGNAL (LONG ENTRY)</span>'
        border_color = "#22c55e"
    elif decision.action == ActionSignal.SELL:
        badge_html = '<span class="badge badge-sell">🔻 SELL / EXIT SIGNAL</span>'
        border_color = "#ef4444"
    else:
        badge_html = '<span class="badge badge-hold">⚖️ HOLD / WAIT (CONSOLIDATION)</span>'
        border_color = "#eab308"

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.7); border-left: 5px solid {border_color}; border-radius: 8px; padding: 14px 20px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Current Quantitative Strategy Verdict</div>
                <div>{badge_html}</div>
            </div>
            <div style="font-size: 1rem; color: #f8fafc; font-weight: 500;">
                {decision.rationale}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN TABS: CHARTS, SENTIMENT NEWS, POSITION AUDIT
# ==============================================================================
tab_charts, tab_news, tab_position = st.tabs([
    "📈 Price Action & Candlesticks",
    "📰 Live Sentiment News Feed",
    "💼 Portfolio & Active Position",
])

# ------------------------------------------------------------------------------
# TAB 1: INTERACTIVE PRICE CHART
# ------------------------------------------------------------------------------
with tab_charts:
    st.subheader(f"{selected_symbol} — Recent Price History ({selected_timeframe} Candles)")

    if ohlcv_df is not None and not ohlcv_df.empty:
        chart_df = ohlcv_df.set_index("timestamp")

        st.line_chart(
            chart_df[["close"]],
            use_container_width=True,
            height=380,
            color="#38bdf8",
        )

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.caption(f"**Period High:** ${ohlcv_df['high'].max():,.2f}")
        with col_c2:
            st.caption(f"**Period Low:** ${ohlcv_df['low'].min():,.2f}")
        with col_c3:
            st.caption(f"**Average Bar Volume:** {ohlcv_df['volume'].mean():,.2f}")
        with col_c4:
            st.caption(f"**Return Volatility:** {ohlcv_df['bar_return_pct'].std():.2f}%")

        with st.expander("📊 View Raw OHLCV Candlestick Data"):
            st.dataframe(ohlcv_df, use_container_width=True, height=220)
    else:
        st.warning("Could not load candlestick history from Binance. Retrying...")

# ------------------------------------------------------------------------------
# TAB 2: LIVE SENTIMENT NEWS FEED (WITH BADGES)
# ------------------------------------------------------------------------------
with tab_news:
    st.subheader(f"Latest News Headlines & FinBERT Sentiment for {token_keyword}")

    if headlines:
        pos_n = sentiment_summary.positive_count
        neg_n = sentiment_summary.negative_count
        neu_n = sentiment_summary.neutral_count

        st.markdown(
            f"""
            <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                <span class="badge badge-pos" style="padding: 6px 14px; font-size: 0.85rem;">Bullish: {pos_n}</span>
                <span class="badge badge-neg" style="padding: 6px 14px; font-size: 0.85rem;">Bearish: {neg_n}</span>
                <span class="badge badge-neu" style="padding: 6px 14px; font-size: 0.85rem;">Neutral: {neu_n}</span>
                <span style="color: #94a3b8; font-size: 0.85rem; align-self: center;">
                    Average Confidence: <b>{sentiment_summary.avg_confidence * 100:.1f}%</b>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for idx, item in enumerate(headlines, 1):
            lbl = item.sentiment.label
            score = item.sentiment.polarity_score
            conf = item.sentiment.confidence * 100

            if lbl == "Positive":
                badge_class = "badge-pos"
                badge_text = f"[+] POSITIVE ({score:+.2f})"
            elif lbl == "Negative":
                badge_class = "badge-neg"
                badge_text = f"[-] NEGATIVE ({score:+.2f})"
            else:
                badge_class = "badge-neu"
                badge_text = f"[~] NEUTRAL ({score:+.2f})"

            link_html = (
                f'<a href="{item.article.link}" target="_blank" class="news-title">{item.article.title}</a>'
                if item.article.link
                else f'<span class="news-title">{item.article.title}</span>'
            )

            st.markdown(
                f"""
                <div class="news-card">
                    <div style="margin-bottom: 6px;">
                        <span class="badge {badge_class}">{badge_text}</span>
                        <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">{conf:.1f}% Confidence</span>
                    </div>
                    <div>{link_html}</div>
                    <div class="news-meta">
                        Source: <b>{item.article.source}</b> | Published: {item.article.published}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info(f"No recent news articles detected for '{token_keyword}'. Showing neutral baseline.")

# ------------------------------------------------------------------------------
# TAB 3: PORTFOLIO & ACTIVE POSITIONS
# ------------------------------------------------------------------------------
with tab_position:
    st.subheader(f"Active Position & Execution Audit ({selected_symbol})")

    col_pos, col_act = st.columns([2, 1])

    with col_pos:
        if active_pos and active_pos.amount > 0:
            unreal_usd, unreal_pct = active_pos.calculate_unrealized_pnl(last_price)
            sign = "+" if unreal_usd >= 0 else ""

            st.success(f"**ACTIVE LONG POSITION:** {active_pos.amount:.6f} {base_asset}")
            st.write(f"• **Entry Price:** ${active_pos.entry_price:,.4f}")
            st.write(f"• **Current Price:** ${last_price:,.4f}")
            st.write(f"• **Total Position Cost:** ${active_pos.total_cost:,.2f} USDT")
            st.write(f"• **Unrealized PnL:** {sign}${unreal_usd:,.2f} ({sign}{unreal_pct:.2f}%)")
            st.write(f"• **Opened At:** {active_pos.opened_at_utc}")
        else:
            st.info("No active open positions. Portfolio is currently **100% in USDT Cash**.")

    with col_act:
        st.markdown("**Simulate Paper Trade**")
        sim_buy = st.button("Simulate Market BUY ($10)", use_container_width=True)
        sim_sell = st.button("Simulate Market SELL", use_container_width=True)

        if sim_buy:
            receipt = broker.execute_market_buy(selected_symbol, 10.0, last_price)
            if receipt:
                st.toast(f"Buy filled! {receipt.amount:.4f} {base_asset} @ ${receipt.fill_price:,.2f}", icon="🚀")
                st.rerun()

        if sim_sell:
            receipt = broker.execute_market_sell(selected_symbol, last_price)
            if receipt:
                st.toast(f"Sell filled! PnL: ${receipt.realized_pnl:+,.2f}", icon="💰")
                st.rerun()

    st.markdown("---")
    st.subheader("📜 Session Trade History")
    if broker.trade_history:
        history_rows = []
        for r in broker.trade_history:
            history_rows.append({
                "Order ID": r.order_id,
                "Time (UTC)": r.timestamp_utc,
                "Pair": r.symbol,
                "Side": r.side,
                "Fill Price": f"${r.fill_price:,.4f}",
                "Amount": f"{r.amount:.6f}",
                "Cost": f"${r.cost:,.2f}",
                "Fee (0.1%)": f"${r.fee:,.4f}",
                "Realized PnL": f"${r.realized_pnl:+,.2f}" if r.realized_pnl is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True)
    else:
        st.caption("No trades executed in this session yet.")


# ==============================================================================
# AUTO-REFRESH SCRIPT
# ==============================================================================
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
