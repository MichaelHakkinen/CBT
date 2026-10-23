# AI Developer Guide (CLAUDE.md)

## 📌 Project Overview
**CBT (Crypto Trading Bot)** is a modular, production-grade algorithmic cryptocurrency trading bot written in Python. It integrates NLP Sentiment Analysis, live exchange order-book/ticker feeds, and quantitative multi-factor strategy execution.

---

## 🛠️ Environment & Runtime Rules

### Python Interpreter & Virtual Environment
- **Root Python on host:** Python 3.15.1 (note: PyPI has limited pre-built binary wheels for PyTorch on 3.15).
- **Project Virtual Environment:** Located at `.venv/` running **CPython 3.12.15** managed via `uv`.
- **Always execute scripts using the virtual environment:**
  - **Windows PowerShell / CMD:**
    ```powershell
    .\.venv\Scripts\python.exe <script_name>.py [args]
    ```
  - **Package Management:** Use `python -m uv pip install --python .venv <package>` or `pip install` inside activated environment.

---

## 📂 Core Architecture & Scripts

| File | Step | Description | Key Technologies |
|---|---|---|---|
| `sentiment_engine.py` | Step 1 | AI Crypto Sentiment Ingestion & Inference | Hugging Face `transformers`, `ProsusAI/finbert`, `feedparser`, `requests` |
| `market_data.py` | Step 2 | Live Binance Ticker & Candlestick Ingestion | `ccxt`, `pandas`, Binance Vision Mirror |
| `decision_engine.py` | Step 3 | Multi-Factor Quantitative Strategy Evaluator | Confluence scoring, `tabulate`, Decision Dashboard |
| `paper_trading.py` | Step 4 | Paper Trading Execution & Binance Testnet Loop | CCXT Sandbox, Position Tracking, Taker Fees |
| `main.py` | Step 5 | 24/7 Automated Orchestrator with Persistent File Logging | Non-crashing loop, rate-limit backoff, clean CMD |
| `dashboard.py` | UI | Real-Time Interactive Web Dashboard | Streamlit, Altair, Caching, Live News Badges |
| `trading_bot.log` | Core | Persistent, detailed production log file | Timestamped audit records, tracebacks, raw payloads |
| `requirements.txt` | Core | Pinned dependency manifest | PyTorch, Transformers, CCXT, Pandas, etc. |
| `README.md` | Docs | Quickstart & CLI documentation | Markdown |

---

## ⚡ Common CLI Commands

### 1. 24/7 Automated Bot Main Loop (Step 5)
```powershell
# Run the 24/7 loop (5-minute interval, clean CMD, detailed logs in trading_bot.log)
.\.venv\Scripts\python.exe main.py --symbol SOL/USDT --interval 300

# Run multi-asset basket
.\.venv\Scripts\python.exe main.py --symbols "SOL/USDT,BTC/USDT" --interval 300
```

### 2. Standalone Component Runs
```powershell
# Step 1: Sentiment analysis only
.\.venv\Scripts\python.exe sentiment_engine.py --token Solana

# Step 2: Live ticker and candles only
.\.venv\Scripts\python.exe market_data.py --symbol SOL/USDT --limit 5

# Step 3: Decision engine evaluation
.\.venv\Scripts\python.exe decision_engine.py --symbol SOL/USDT

# Step 4: Paper trading execution test
.\.venv\Scripts\python.exe paper_trading.py --symbol SOL/USDT --amount 10.0 --force-signal BUY
```

---

## 📐 Quantitative Design Principles

1. **FinBERT Polarity Normalization:**
   Continuous headline polarity is defined as:
   $$s_i = P(\text{Positive}) - P(\text{Negative}) \in [-1.0, +1.0]$$
   High neutral probability naturally dampens the signal toward $0.0$, preventing false bullish/bearish inflation.

2. **Network Resilience & Exchange Quotas:**
   - Always instantiate `ccxt.binance` with `enableRateLimit: True`.
   - Use the official public mirror `https://data-api.binance.vision/api/v3` to circumvent regional ISP blocks without needing VPN/proxies.
   - For RSS feeds, always pass browser `User-Agent` headers in `requests` to prevent 403 Forbidden responses.

3. **Multi-Factor Confluence Rule:**
   - **BUY:** Sentiment $> +0.30$ **AND** 24h Price Change $> 0.0\%$.
   - **SELL:** Sentiment $< -0.30$ **OR** 24h Price Drop $\le -3.50\%$.
   - **HOLD:** Conflicting signals (bullish/bearish divergence) or neutral band.

---

## 🔮 Roadmap & Next Steps
- **Step 4:** Paper Trading Simulator (Virtual wallet, slippage model, trade fill logging, active position tracker).
- **Step 5:** Technical Indicators Overlay (RSI, EMA 20/50 Ribbon, MACD, ATR-based volatility stops).
- **Step 6:** Automation & Notifications (Continuous 5-minute event loop, Telegram / Discord Webhook alerts).
