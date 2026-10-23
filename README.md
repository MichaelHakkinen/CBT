# Crypto Trading Bot — Step 1: AI Sentiment Analysis Engine

A production-grade, quantitative AI sentiment analysis engine for cryptocurrency assets (Bitcoin, Ethereum, Solana, and more). Built with Python, Hugging Face `transformers`, and the financial NLP model `ProsusAI/finbert`.

---

## 🚀 Quick Start

### 1. Installation

Install all required dependencies:

```bash
pip install torch transformers feedparser requests tabulate ccxt pandas
```

> **Note for Windows users:** If `pip` is not in your global `PATH`, run:
> ```powershell
> python -m pip install torch transformers feedparser requests tabulate ccxt pandas
> ```

Alternatively, install via the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Step 1: AI Sentiment Analysis Engine
```bash
# Analyze Bitcoin headlines & sentiment
python sentiment_engine.py --token Bitcoin

# Analyze Ethereum
python sentiment_engine.py --token Ethereum --limit 15
```

### Step 2: Live Market Data Connection (Binance / CCXT)
```bash
# Fetch live BTC/USDT ticker & 1h candlestick DataFrame
python market_data.py --symbol BTC/USDT

# Fetch live SOL/USDT with 15m timeframe and 5 candles
python market_data.py --symbol SOL/USDT --timeframe 15m --limit 5

### Real-Time Web Dashboard (Streamlit)
```bash
# Launch the interactive web dashboard in your browser
streamlit run dashboard.py
```

### Step 5: 24/7 Automated Production Loop (`main.py`)
```bash
# Launch the 24/7 automated bot (runs every 5 minutes / 300 seconds)
python main.py --symbol SOL/USDT --interval 300

# Run multi-asset monitoring basket
python main.py --symbols "SOL/USDT,BTC/USDT,ETH/USDT" --interval 300

# Run with custom trade budget ($25 USDT) and custom log file
python main.py --symbol SOL/USDT --amount 25.0 --log-file trading_bot.log

# Optional: Display full dashboard tables on CMD (by default, tables go strictly to log file)
python main.py --symbol SOL/USDT --show-dashboard
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--token` | `str` | `Bitcoin` | Crypto token to target (`Bitcoin`, `Ethereum`, `Solana`, or `all`) |
| `--limit` | `int` | `10` | Max number of headlines to fetch and score |
| `--device` | `str` | `auto` | Inference device: `cpu`, `cuda`, or auto-detected |

---

## 📐 Quantitative Methodology

### 1. FinBERT Polarity Scoring
For each headline $i$, FinBERT outputs probability distribution across three classes:
$$\{P(\text{Positive})_i, P(\text{Negative})_i, P(\text{Neutral})_i\}$$

We compute the continuous polarity score $s_i \in [-1.0, +1.0]$:
$$s_i = P(\text{Positive})_i - P(\text{Negative})_i$$

- **Strong Bullish:** $P(\text{Pos}) = 0.95, P(\text{Neg}) = 0.01 \implies s_i = +0.94$
- **Strong Bearish:** $P(\text{Neg}) = 0.90, P(\text{Pos}) = 0.02 \implies s_i = -0.88$
- **High Neutrality:** $P(\text{Neu}) = 0.90, P(\text{Pos}) = 0.05, P(\text{Neg}) = 0.05 \implies s_i = 0.00$

### 2. Market Sentiment Index
The overall Market Sentiment Index $S_{\text{market}} \in [-1.0, +1.0]$ is the arithmetic mean of all headline polarities:
$$S_{\text{market}} = \frac{1}{N} \sum_{i=1}^{N} s_i$$

### 3. Quantitative Decision Rules

| Sentiment Index ($S_{\text{market}}$) | Market State | Quantitative Action |
|---|---|---|
| $\ge +0.35$ | Extreme Bullish | **STRONG BUY 🚀** |
| $+0.15 \le S < +0.35$ | Bullish | **BUY 📈** |
| $-0.15 \le S < +0.15$ | Consolidation / Indecision | **HOLD / NEUTRAL ⚖️** |
| $-0.35 < S \le -0.15$ | Bearish | **SELL 📉** |
| $\le -0.35$ | Extreme Bearish | **STRONG SELL 🔻** |

---

## 📊 Sample Terminal Output

```text
==========================================================================================
   CRYPTO SENTIMENT ANALYSIS REPORT  --  Target: BITCOIN
   Timestamp: 2026-10-23 09:30:00 UTC
==========================================================================================
+----+---------------+----------------------------------------------+---------------+------------+------------+
|  # | Source        | Headline Text                                | Sentiment     | Confidence | Polarity   |
+====+===============+==============================================+===============+============+============+
|  1 | CoinDesk      | Bitcoin Crosses New High as Institutional    | [+] POSITIVE  | 94.2%      | +0.92      |
|    |               | Inflows Surge into Spot ETFs                 |               |            |            |
+----+---------------+----------------------------------------------+---------------+------------+------------+
|  2 | Cointelegraph | Federal Reserve Signals Prolonged Pause      | [~] NEUTRAL   | 88.5%      | -0.04      |
|    |               | on Interest Rate Decisions                   |               |            |            |
+----+---------------+----------------------------------------------+---------------+------------+------------+
|  3 | Decrypt       | Regulatory Scrutiny Tightens on Offshore     | [-] NEGATIVE  | 91.1%      | -0.89      |
|    |               | Derivatives Exchanges                        |               |            |            |
+----+---------------+----------------------------------------------+---------------+------------+------------+

------------------------------------------------------------------------------------------
   EXECUTIVE QUANTITATIVE SUMMARY
------------------------------------------------------------------------------------------
  Headlines Analyzed    : 10
  Sentiment Breakdown   : Positive: 6 | Negative: 2 | Neutral: 2
  Average Confidence    : 91.24%
  Market Sentiment Index: +0.4120 (Scale: -1.0000 to +1.0000)
  Visual Sentiment Meter: [-1.0 BEARISH] -------------|------O--- [+1.0 BULLISH]
  Algorithmic Signal    : >>> STRONG BUY (Extreme Bullish) <<<
==========================================================================================
```
