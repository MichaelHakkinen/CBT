# Project Context & System Architecture (CONTEXT.md)

## 📌 Executive Summary
**CBT (Crypto Trading Bot)** is a modular quantitative algorithmic trading system engineered in Python. It evaluates cryptocurrency markets through a multi-factor investment framework combining **Natural Language Processing (NLP) News Sentiment** with **Live Exchange Microstructure and Technical Momentum**.

---

## 🏛️ High-Level System Architecture

```
                       ┌────────────────────────────────────────┐
                       │          LIVE CRYPTO SOURCES           │
                       └───────────────────┬────────────────────┘
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             ▼                                                           ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│     5+ Live RSS News Feeds    │                       │  Binance Spot Market Data     │
│   (CoinDesk, Cointelegraph,   │                       │  (CCXT + Binance Vision       │
│    Decrypt, BeInCrypto)       │                       │   Public Mirror)              │
└──────────────┬────────────────┘                       └───────────────┬───────────────┘
               │                                                        │
               ▼                                                        ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│       STEP 1 ENGINE           │                       │       STEP 2 ENGINE           │
│   Token Filter & FinBERT NLP  │                       │   Real-Time Ticker & OHLCV    │
│   (ProsusAI/finbert)          │                       │   (Pandas Cleaned DataFrames) │
└──────────────┬────────────────┘                       └───────────────┬───────────────┘
               │                                                        │
               │  • Continuous Polarity: P(Pos) - P(Neg)                │  • Current Last Price
               │  • Aggregated Index: [-1.0 to +1.0]                    │  • 24h High, Low, Volume
               │  • Discrete Labels & Confidence                        │  • 24h Momentum (% Change)
               │                                                        │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │            STEP 3 ENGINE               │
                       │     Multi-Factor Decision Engine       │
                       │     (decision_engine.py)               │
                       └───────────────────┬────────────────────┘
                                           │
                       ┌───────────────────┴────────────────────┐
                       │   Confluence Rules:                    │
                       │   • BUY: Sentiment > +0.3 & Chg > 0%   │
                       │   • SELL: Sentiment < -0.3 OR Drop<-3.5│
                       │   • HOLD: Divergence / Neutral         │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │     STEP 4 (UPCOMING MILESTONE)        │
                       │   Paper Trading & Order Execution      │
                       │   (Virtual Balance, Slippage, PnL)     │
                       └────────────────────────────────────────┘
```

---

## 🧱 Module Specifications

### Step 1: AI Sentiment Analysis Engine (`sentiment_engine.py`)
- **News Ingestion:** Ingests live XML/RSS from CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine, and BeInCrypto with browser-grade headers to defeat Cloudflare/bot-detection.
- **Token Filtering:** Word-boundary regular expressions (`\bbtc\b`, `\beth\b`, `\bsol\b`) avoid spurious substring matches.
- **Deep Learning Model:** Hugging Face `transformers` running `ProsusAI/finbert` (BERT pre-trained on Financial PhraseBank).
- **Quantitative Polarity Metric:**
  $$s_i = P(\text{Positive}) - P(\text{Negative}) \in [-1.0, +1.0]$$
- **Market Sentiment Index:**
  $$S_{\text{market}} = \frac{1}{N} \sum_{i=1}^{N} s_i \in [-1.0, +1.0]$$

---

### Step 2: Live Market Data Connection (`market_data.py`)
- **Exchange:** Binance Spot Market via `ccxt`.
- **Rate-Limiting Protection:** Strict rate-limiting enabled (`enableRateLimit: True`) adhering to exchange weight ceilings.
- **Firewall & Geo-Bypass:** Public endpoint routing redirected to `https://data-api.binance.vision/api/v3`, guaranteeing uninterrupted access in regions where standard `api.binance.com` is ISP-blocked.
- **Candlestick Pipeline:** Fetches OHLCV records and constructs typed Pandas DataFrames with UTC timestamps and derived volatility metrics (`bar_return_pct`, `range`).

---

### Step 3: Multi-Factor Trading Decision Engine (`decision_engine.py`)
- **Role:** Merges news alpha and price action into actionable execution signals.
- **Confluence Decision Matrix:**

| Sentiment Score ($S$) | 24h Price Change ($\Delta P$) | Decision | Quantitative Rationale |
|---|---|---|---|
| $S > +0.30$ (Bullish) | $\Delta P > 0.0\%$ (Positive) | **BUY** | Bullish sentiment aligned with upward technical momentum. |
| $S < -0.30$ (Bearish) | Any | **SELL** | Severe news/macro hostility signals downside institutional risk. |
| Any | $\Delta P \le -3.50\%$ (Drawdown) | **SELL** | Technical stop-loss triggered for capital preservation. |
| $S > +0.30$ (Bullish) | $\Delta P \le 0.0\%$ (Negative) | **HOLD** | Bullish divergence: price lagging, awaiting upward confirmation. |
| $S < -0.30$ (Bearish) | $\Delta P > 0.0\%$ (Positive) | **HOLD** | Bearish divergence: price rallying into negative news (fakeout risk). |
| $-0.30 \le S \le +0.30$ | Any | **HOLD** | Range-bound neutral sentiment; no statistical edge. |

- **Step 4 Pre-trade Context:**
  When a **BUY** signal triggers, `PaperTradingContext` dynamically calculates:
  - Exact limit entry price.
  - 2.0% stop-loss target.
  - 4.0% take-profit target (2.0 : 1.0 positive expectancy risk-reward ratio).

---

## 💻 Environment & Infrastructure

- **Operating System:** Windows 11 (64-bit).
- **Environment Isolation:** `.venv/` running CPython 3.12.15 via `uv`.
- **Installed Stack:**
  - `torch==2.14.1`
  - `transformers==5.20.2`
  - `ccxt==4.5.89`
  - `pandas==3.1.0`
  - `feedparser==6.0.14`
  - `requests==2.34.2`
  - `tabulate==0.10.0`

---

## 🗺️ Product Roadmap

- [x] **Step 1:** AI Sentiment Analysis Engine (FinBERT + RSS).
- [x] **Step 2:** Market Data Connection (CCXT + Binance Spot + Pandas).
- [x] **Step 3:** Multi-Factor Trading Decision Engine (Confluence rules + Dashboard).
- [x] **Step 4:** Paper Trading Simulator (Virtual balances, order execution, slippage, PnL accounting).
- [x] **Step 5:** 24/7 Automated Production Loop (`main.py` with persistent `trading_bot.log` and clean CMD).
- [x] **Step 6 (UI):** Real-Time Web Dashboard (`dashboard.py` in Streamlit with metric cards, charts, and news badges).
- [ ] **Step 7:** Automation & Alerts (Discord/Telegram notification webhooks, live trading safety gates).
