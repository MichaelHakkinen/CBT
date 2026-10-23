"""
================================================================================
CRYPTO MARKET DATA CONNECTOR (Step 2: CCXT & Pandas Market Ingestion)
================================================================================
Author: Senior Quantitative Python Developer
Target Exchange: Binance (Spot Market Data via CCXT)
Description:
    Connects to the Binance exchange with rate-limiting and geo-fallback.
    Fetches real-time ticker data (Last, High, Low, Volume) and historical
    OHLCV candlestick records into Pandas DataFrames for quantitative analysis.
================================================================================
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Third-party quantitative libraries
try:
    import ccxt
    import pandas as pd
except ImportError:
    print(
        "[!] Missing market data dependencies. Please install:\n"
        "    pip install ccxt pandas"
    )
    sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ==============================================================================
# CONFIGURATION & LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MarketData")

# Binance Official Public Data Mirror (Prevents ISP/Geo-blocking in certain regions)
BINANCE_PUBLIC_MIRROR = "https://data-api.binance.vision/api/v3"


# ==============================================================================
# EXCHANGE INITIALIZATION
# ==============================================================================
def init_exchange(
    timeout_ms: int = 10000,
    use_mirror_fallback: bool = True,
) -> ccxt.binance:
    """
    Initializes the Binance exchange connection using CCXT with:
    1. Built-in rate-limiting enabled (strictly adheres to Binance API weight limits).
    2. Spot-only market configuration (prevents unsolicited futures endpoints).
    3. Public mirror routing fallback (ensures connectivity even if api.binance.com
       is ISP-filtered or geoblocked in the user's jurisdiction).

    Args:
        timeout_ms: HTTP request timeout in milliseconds.
        use_mirror_fallback: If True, uses the official Binance data mirror.

    Returns:
        Configured ccxt.binance exchange instance.
    """
    config: Dict[str, Any] = {
        "enableRateLimit": True,  # Prevent IP bans and HTTP 429 errors
        "timeout": timeout_ms,
        "options": {
            "defaultType": "spot",
            "fetchMarkets": ["spot"],
        },
    }

    if use_mirror_fallback:
        # Route public endpoints through Binance's dedicated market data mirror
        config["urls"] = {
            "api": {
                "public": BINANCE_PUBLIC_MIRROR,
            }
        }

    exchange = ccxt.binance(config)
    logger.info(
        "Binance exchange initialized (RateLimit: ENABLED, Timeout: %dms, Mirror: %s)",
        timeout_ms,
        "Active" if use_mirror_fallback else "Default",
    )
    return exchange


# ==============================================================================
# TICKER DATA FETCHER
# ==============================================================================
def fetch_live_ticker(exchange: ccxt.Exchange, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetches real-time 24-hour ticker statistics for a given cryptocurrency pair.

    Args:
        exchange: Initialized CCXT exchange instance.
        symbol: Trading pair symbol (e.g., 'BTC/USDT', 'SOL/USDT', 'ETH/USDT').

    Returns:
        Parsed ticker dictionary or None if fetch fails.
    """
    formatted_symbol = symbol.upper().strip()
    logger.info("Fetching live ticker for '%s'...", formatted_symbol)

    try:
        raw_ticker = exchange.fetch_ticker(formatted_symbol)

        # Standardize extracted metrics
        ticker_data = {
            "symbol": formatted_symbol,
            "last_price": raw_ticker.get("last"),
            "high_24h": raw_ticker.get("high"),
            "low_24h": raw_ticker.get("low"),
            "base_volume": raw_ticker.get("baseVolume"),    # e.g., Volume in BTC
            "quote_volume": raw_ticker.get("quoteVolume"),  # e.g., Volume in USDT
            "percentage_24h": raw_ticker.get("percentage"), # 24h Price change in %
            "bid": raw_ticker.get("bid"),
            "ask": raw_ticker.get("ask"),
            "timestamp": raw_ticker.get("timestamp"),
            "datetime_utc": (
                datetime.fromtimestamp(raw_ticker["timestamp"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                if raw_ticker.get("timestamp")
                else "N/A"
            ),
        }
        return ticker_data

    except ccxt.BadSymbol:
        logger.error("Symbol '%s' is not recognized or not traded on Binance.", formatted_symbol)
    except ccxt.RateLimitExceeded:
        # Must precede NetworkError (its parent class), otherwise this branch is dead code
        logger.error("Binance rate limit exceeded. Backing off request.")
    except ccxt.NetworkError as exc:
        logger.error("Network error while connecting to Binance: %s", str(exc))
    except ccxt.ExchangeError as exc:
        logger.error("Binance exchange error for '%s': %s", formatted_symbol, str(exc))
    except Exception as exc:
        logger.error("Unexpected error fetching ticker: %s", str(exc))

    return None


# ==============================================================================
# OHLCV CANDLESTICK FETCHER (PANDAS DATAFRAME)
# ==============================================================================
def fetch_ohlcv_dataframe(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 24,
) -> Optional[pd.DataFrame]:
    """
    Fetches recent OHLCV (Open, High, Low, Close, Volume) candlestick records
    for a chosen timeframe and structures them into a Pandas DataFrame.

    Args:
        exchange: Initialized CCXT exchange instance.
        symbol: Trading pair symbol (e.g., 'BTC/USDT', 'SOL/USDT').
        timeframe: Candle interval (e.g., '1m', '5m', '15m', '1h', '4h', '1d').
        limit: Number of candlestick periods to fetch (max 1000 per request).

    Returns:
        pd.DataFrame with UTC timestamps and float OHLCV values, or None on error.
    """
    formatted_symbol = symbol.upper().strip()
    logger.info(
        "Fetching %d '%s' candles for '%s'...", limit, timeframe, formatted_symbol
    )

    try:
        # Validate timeframe against exchange capabilities
        if timeframe not in exchange.timeframes:
            logger.error(
                "Invalid timeframe '%s'. Supported timeframes: %s",
                timeframe,
                list(exchange.timeframes.keys()),
            )
            return None

        # Fetch raw list of candles: [[timestamp, open, high, low, close, volume], ...]
        raw_ohlcv = exchange.fetch_ohlcv(
            symbol=formatted_symbol,
            timeframe=timeframe,
            limit=limit,
        )

        if not raw_ohlcv:
            logger.warning("No OHLCV records returned for '%s'.", formatted_symbol)
            return None

        # Build Pandas DataFrame
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(raw_ohlcv, columns=columns)

        # Convert millisecond epoch to timezone-aware UTC datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Ensure numeric precision
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Compute helpful quantitative derivatives
        df["bar_return_pct"] = ((df["close"] - df["open"]) / df["open"]) * 100.0
        df["range"] = df["high"] - df["low"]

        return df

    except ccxt.BadSymbol:
        logger.error("Symbol '%s' is not recognized on Binance.", formatted_symbol)
    except ccxt.RateLimitExceeded:
        logger.error("Binance rate limit exceeded during OHLCV query.")
    except ccxt.NetworkError as exc:
        logger.error("Network error while fetching OHLCV: %s", str(exc))
    except ccxt.ExchangeError as exc:
        logger.error("Binance error for '%s' (%s): %s", formatted_symbol, timeframe, str(exc))
    except Exception as exc:
        logger.error("Unexpected error building OHLCV DataFrame: %s", str(exc))

    return None


# ==============================================================================
# TERMINAL VISUALIZATION & REPORTING
# ==============================================================================
def display_market_report(ticker: Optional[Dict[str, Any]], df: Optional[pd.DataFrame]):
    """Renders formatted ticker statistics and the OHLCV candlestick table."""
    print("\n" + "=" * 90)
    print("   CRYPTO MARKET DATA REPORT (Step 2: Binance Market Ingestion)")
    print(f"   Execution Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 90)

    # 1. Real-Time Ticker Overview
    if ticker:
        pct_change = ticker.get("percentage_24h") or 0.0
        pct_sign = "+" if pct_change >= 0 else ""
        pct_str = f"{pct_sign}{pct_change:.2f}%"

        base_curr = ticker["symbol"].split("/")[0]
        quote_curr = ticker["symbol"].split("/")[1] if "/" in ticker["symbol"] else "USDT"

        print("\n" + "-" * 90)
        print(f"   LIVE TICKER: {ticker['symbol']} ({ticker['datetime_utc']})")
        print("-" * 90)
        print(f"  * Current Last Price : ${ticker['last_price']:,.2f} {quote_curr}")
        print(f"  * 24-Hour Price High : ${ticker['high_24h']:,.2f} {quote_curr}")
        print(f"  * 24-Hour Price Low  : ${ticker['low_24h']:,.2f} {quote_curr}")
        print(f"  * 24-Hour Base Volume: {ticker['base_volume']:,.4f} {base_curr}")
        print(f"  * 24-Hour Quote Vol  : ${ticker['quote_volume']:,.2f} {quote_curr}")
        print(f"  * 24-Hour Change     : {pct_str}")
        print(f"  * Order Book Spread  : Bid: ${ticker.get('bid', 0):,.2f} | Ask: ${ticker.get('ask', 0):,.2f}")
    else:
        print("[!] Ticker data could not be retrieved.")

    # 2. OHLCV Candlestick DataFrame Table
    if df is not None and not df.empty:
        print("\n" + "-" * 90)
        print(f"   RECENT OHLCV CANDLESTICKS (Last {len(df)} Periods)")
        print("-" * 90)

        # Prepare pretty-printed display copy
        display_df = df.copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["open"] = display_df["open"].map(lambda x: f"${x:,.2f}")
        display_df["high"] = display_df["high"].map(lambda x: f"${x:,.2f}")
        display_df["low"] = display_df["low"].map(lambda x: f"${x:,.2f}")
        display_df["close"] = display_df["close"].map(lambda x: f"${x:,.2f}")
        display_df["volume"] = display_df["volume"].map(lambda x: f"{x:,.2f}")
        display_df["return"] = display_df["bar_return_pct"].map(
            lambda x: f"{'+' if x >= 0 else ''}{x:.2f}%"
        )

        formatted_cols = ["timestamp", "open", "high", "low", "close", "volume", "return"]
        headers = ["Candle Time (UTC)", "Open", "High", "Low", "Close", "Volume", "Return"]

        if HAS_TABULATE:
            print(tabulate(display_df[formatted_cols].values, headers=headers, tablefmt="grid"))
        else:
            print(display_df[formatted_cols].to_string(index=False))

        # 3. Quantitative Candle Statistics
        highest_high = df["high"].max()
        lowest_low = df["low"].min()
        volatility = df["bar_return_pct"].std()
        avg_volume = df["volume"].mean()

        print("\n" + "-" * 90)
        print("   CANDLESTICK STATISTICAL SUMMARY")
        print("-" * 90)
        print(f"  * Window High / Low : ${highest_high:,.2f} / ${lowest_low:,.2f}")
        print(f"  * Bar Return Vol (StdDev): {volatility:.2f}% per candle")
        print(f"  * Mean Bar Volume   : {avg_volume:,.2f}")
    else:
        print("[!] Candlestick DataFrame could not be constructed.")

    print("=" * 90 + "\n")


# ==============================================================================
# PIPELINE EXECUTION & CLI
# ==============================================================================
def main():
    """Command-line entrypoint for market data connector."""
    parser = argparse.ArgumentParser(
        description="Step 2: Real-time Market Data Connection (Binance / CCXT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Crypto trading pair to fetch (e.g. 'BTC/USDT', 'SOL/USDT', 'ETH/USDT')",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candlestick interval (e.g. '1m', '5m', '15m', '1h', '4h', '1d')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of recent candlestick bars to retrieve",
    )
    args = parser.parse_args()

    # 1. Initialize exchange with rate-limiting & fallback
    exchange = init_exchange()

    # 2. Fetch live ticker statistics
    ticker = fetch_live_ticker(exchange, args.symbol)

    # 3. Fetch OHLCV candlestick records into Pandas DataFrame
    df = fetch_ohlcv_dataframe(
        exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
    )

    # 4. Render clean terminal tables and summaries
    display_market_report(ticker, df)


if __name__ == "__main__":
    main()
