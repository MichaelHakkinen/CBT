"""
================================================================================
CRYPTO TRADING BOT: 24/7 AUTOMATED MAIN ORCHESTRATION LOOP (Step 5)
================================================================================
Author: Senior Quantitative Python Developer
Description:
    Runs the complete 24/7 automated algorithmic trading lifecycle:
    - Detailed, verbose logs directed strictly into `trading_bot.log`
    - Clean, concise continuous status updates printed to the CMD terminal
    - 5-minute loop cadence with robust exception catching and auto-recovery
================================================================================
"""

import os
import sys

# Suppress verbose Hugging Face & symlink warnings from terminal
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import signal
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Third-party quantitative libraries
try:
    import ccxt
    import requests
except ImportError:
    print("[!] Missing web/exchange libraries. Run: pip install ccxt requests")
    sys.exit(1)

# Import internal modular components
try:
    from decision_engine import ActionSignal, run_trading_decision_pipeline
    from paper_trading import (
        DEFAULT_TESTNET_API_KEY,
        DEFAULT_TESTNET_SECRET,
        BinanceTestnetBroker,
        ExecutionReporter,
        OrderReceipt,
    )
except ImportError as exc:
    print(f"[!] Critical Import Failure in CBT Architecture: {exc}")
    sys.exit(1)


# ==============================================================================
# DEDICATED LOGGING: VERBOSE TO FILE, CLEAN TO CMD
# ==============================================================================
LOG_FILE_NAME = "trading_bot.log"

def configure_bot_logging(log_file_path: str = LOG_FILE_NAME) -> logging.Logger:
    """
    Configures dual-channel logging:
    1. File (`trading_bot.log`): Receives ALL verbose debug messages, raw payloads,
       model outputs, trade confirmations, and complete tracebacks.
    2. Terminal (CMD): Displays ONLY clean, high-level status updates (no clutter).
    """
    # Root logger captures everything to file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear pre-existing handlers
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # 1. File Handler (Captures everything)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S UTC",
    )
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 2. Console Handler (Clean, minimal status only)
    console_formatter = logging.Formatter(
        "%(asctime)s %(message)s",
        datefmt="[%H:%M:%S]",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Dedicated clean logger for terminal presentation
    cmd_logger = logging.getLogger("CBT_CMD")
    cmd_logger.setLevel(logging.INFO)
    cmd_logger.propagate = False
    cmd_logger.handlers.clear()
    cmd_logger.addHandler(file_handler)     # Also write CMD status to file
    cmd_logger.addHandler(console_handler)  # Print clean line to CMD

    # Silence third-party spam on console
    for lib in ["urllib3", "huggingface_hub", "transformers", "requests", "feedparser", "ccxt"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return cmd_logger

cmd_logger = configure_bot_logging()
file_logger = logging.getLogger("CBT_Internal")


# ==============================================================================
# 24/7 AUTOMATED ENGINE ORCHESTRATOR
# ==============================================================================
class CBTContinuousTradingEngine:
    """
    24/7 Automated Trading Engine.
    Executes: Fetch News -> FinBERT Sentiment -> Market Ticker -> Decision -> Paper Trade.
    """

    RETRY_BACKOFF_SECONDS = 60  # Wait time after API/Network error before resuming
    STOP_LOSS_PCT = 2.0         # Mirrors MultiFactorDecisionEngine defaults
    TAKE_PROFIT_PCT = 4.0

    def __init__(
        self,
        symbols: List[str],
        trade_amount_usd: float = 10.0,
        interval_seconds: int = 300,
        initial_balance_usd: float = 1000.0,
        api_key: str = DEFAULT_TESTNET_API_KEY,
        secret: str = DEFAULT_TESTNET_SECRET,
        log_file: str = LOG_FILE_NAME,
        show_dashboard: bool = False,
    ):
        self.symbols = [s.upper().strip() for s in symbols]
        self.trade_amount = trade_amount_usd
        self.interval = interval_seconds
        self.log_file = log_file
        self.is_running = False
        self.show_dashboard = show_dashboard

        # Unified Paper / Testnet Broker
        self.broker = BinanceTestnetBroker(
            api_key=api_key,
            secret=secret,
            initial_usdt_balance=initial_balance_usd,
        )

        # Operational Statistics
        self.session_start = datetime.now(timezone.utc)
        self.cycle_count = 0
        self.successful_cycles = 0
        self.error_count = 0

        file_logger.info(
            "CBT 24/7 Engine initialized. Symbols=%s | Cadence=%ds | LogFile=%s",
            self.symbols, self.interval, self.log_file,
        )

    def execute_symbol_pipeline(self, symbol: str):
        """
        Runs the full 4-step pipeline for a single asset.
        Writes all details to log file and prints a clean status line to CMD.
        """
        file_logger.info("Starting pipeline for %s...", symbol)

        # Run Step 1, 2, 3: Decision Engine (suppress verbose table on CMD unless requested)
        decision = run_trading_decision_pipeline(
            symbol=symbol,
            news_limit=6,
            render_dashboard=self.show_dashboard,
        )

        current_price = decision.current_price
        signal = decision.action
        f = decision.factors

        # Enforce the stop-loss / take-profit bracket on any open position. The decision
        # engine only plans these levels; without this check nothing ever acts on them.
        pos = self.broker.get_position(symbol)
        if pos and pos.amount > 0 and decision.paper_order:
            bracket_floor = pos.entry_price * (1 - self.STOP_LOSS_PCT / 100)
            bracket_ceiling = pos.entry_price * (1 + self.TAKE_PROFIT_PCT / 100)
            if current_price <= bracket_floor:
                signal = ActionSignal.SELL
                file_logger.info("[%s] Stop-loss hit (%.4f <= %.4f). Forcing SELL.", symbol, current_price, bracket_floor)
            elif current_price >= bracket_ceiling:
                signal = ActionSignal.SELL
                file_logger.info("[%s] Take-profit hit (%.4f >= %.4f). Forcing SELL.", symbol, current_price, bracket_ceiling)
        base_asset = symbol.split("/")[0]
        quote_asset = symbol.split("/")[1] if "/" in symbol else "USDT"

        # Log detailed metrics to trading_bot.log
        file_logger.info(
            "[%s] Price=$%.4f | 24h Chg=%+.2f%% | Vol=$%.2f | Sentiment=%+.4f (%s) | Verdict=%s | Rationale='%s'",
            symbol, current_price, f.price_change_24h_pct, f.volume_24h_usd,
            f.sentiment_score, f.sentiment_label, signal.value, decision.rationale,
        )

        existing_pos = self.broker.get_position(symbol)
        action_summary = f"{signal.value}"

        # Step 4: Paper Order Execution Logic
        if signal == ActionSignal.BUY:
            if existing_pos and existing_pos.amount > 0:
                action_summary = "BUY (SKIPPED: Already in position)"
                file_logger.info("[%s] Already holding %.6f %s. Skipping duplicate buy.", symbol, existing_pos.amount, base_asset)
            else:
                receipt = self.broker.execute_market_buy(
                    symbol=symbol,
                    usd_budget=self.trade_amount,
                    current_price=current_price,
                )
                if receipt:
                    action_summary = f"BUY EXECUTED (Filled: {receipt.amount:.4f} {base_asset} @ ${receipt.fill_price:,.2f})"
                    file_logger.info(
                        "[ORDER RECEIPT] BUY %s | OrderID: %s | Qty: %.6f | Fill: $%.4f | Cost: $%.2f | Fee: $%.4f",
                        symbol, receipt.order_id, receipt.amount, receipt.fill_price, receipt.cost, receipt.fee,
                    )
                    if self.show_dashboard:
                        ExecutionReporter.print_order_receipt(receipt, self.broker)

        elif signal == ActionSignal.SELL:
            if existing_pos and existing_pos.amount > 0:
                receipt = self.broker.execute_market_sell(
                    symbol=symbol,
                    current_price=current_price,
                )
                if receipt:
                    sign = "+" if (receipt.realized_pnl or 0) >= 0 else ""
                    action_summary = f"SELL EXECUTED (Closed {receipt.amount:.4f} {base_asset} | PnL: {sign}${receipt.realized_pnl:,.2f})"
                    file_logger.info(
                        "[ORDER RECEIPT] SELL %s | OrderID: %s | Qty: %.6f | Fill: $%.4f | PnL: %s$%.2f (%s%.2f%%)",
                        symbol, receipt.order_id, receipt.amount, receipt.fill_price,
                        sign, receipt.realized_pnl or 0.0, sign, receipt.realized_pnl_pct or 0.0,
                    )
                    if self.show_dashboard:
                        ExecutionReporter.print_order_receipt(receipt, self.broker)
            else:
                action_summary = "SELL (SKIPPED: No position held)"
                file_logger.info("[%s] Sell signal active but no active position held.", symbol)

        else:
            # ActionSignal.HOLD
            if existing_pos and existing_pos.amount > 0:
                unreal, unreal_pct = existing_pos.calculate_unrealized_pnl(current_price)
                sign = "+" if unreal >= 0 else ""
                action_summary = f"HOLD (Active Pos: {existing_pos.amount:.4f} {base_asset} | uPnL: {sign}${unreal:.2f})"
            else:
                action_summary = "HOLD (In Cash)"

        # Balances
        cash_usdt = self.broker.get_balance("USDT")

        # ----------------------------------------------------------------------
        # CLEAN STATUS LOG TO CMD (Compact & Human Readable)
        # ----------------------------------------------------------------------
        status_line = (
            f"[CYCLE #{self.cycle_count}] {symbol:<8} | "
            f"Price: ${current_price:>9,.2f} ({'+' if f.price_change_24h_pct >= 0 else ''}{f.price_change_24h_pct:.2f}%) | "
            f"Sent: {f.sentiment_score:+.2f} ({f.sentiment_label:<7}) | "
            f"Action: {action_summary} | "
            f"Cash: ${cash_usdt:,.2f} USDT"
        )
        cmd_logger.info(status_line)

        if self.show_dashboard:
            ExecutionReporter.print_portfolio_snapshot(self.broker, symbol, current_price)

    def run_single_cycle(self):
        """Executes one pass across all target cryptocurrency symbols."""
        self.cycle_count += 1
        cycle_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        file_logger.info("-------------------- CYCLE #%d [%s] --------------------", self.cycle_count, cycle_timestamp)

        for sym in self.symbols:
            try:
                self.execute_symbol_pipeline(sym)
                self.successful_cycles += 1

            except ccxt.RateLimitExceeded as rle:
                self.error_count += 1
                msg = f"[RATE LIMIT] Hit rate limit on {sym}: {rle}. Pausing {self.RETRY_BACKOFF_SECONDS}s..."
                cmd_logger.warning(msg)
                file_logger.warning(msg + "\n" + traceback.format_exc())
                time.sleep(self.RETRY_BACKOFF_SECONDS)

            except (ccxt.NetworkError, requests.exceptions.RequestException, ConnectionError, TimeoutError) as net_err:
                self.error_count += 1
                msg = f"[NETWORK ISSUE] Network error on {sym}: {net_err}. Retrying in {self.RETRY_BACKOFF_SECONDS}s..."
                cmd_logger.warning(msg)
                file_logger.warning(msg + "\n" + traceback.format_exc())
                time.sleep(self.RETRY_BACKOFF_SECONDS)

            except Exception as exc:
                self.error_count += 1
                msg = f"[ERROR CAUGHT] Unexpected exception on {sym}: {exc}. Recovering in {self.RETRY_BACKOFF_SECONDS}s..."
                cmd_logger.error(msg)
                file_logger.error(msg + "\n" + traceback.format_exc())
                time.sleep(self.RETRY_BACKOFF_SECONDS)

    def start(self, max_cycles: Optional[int] = None):
        """Starts the continuous 24/7 loop."""
        self.is_running = True

        banner = (
            f"\n{'=' * 85}\n"
            f"   CBT 24/7 AUTOMATED CRYPTO TRADING BOT IS RUNNING\n"
            f"   Symbols: {', '.join(self.symbols)}  |  Cadence: {self.interval}s (every {self.interval / 60.0:.1f} min)\n"
            f"   Terminal: Clean Status Mode  |  Verbose Log: {os.path.abspath(self.log_file)}\n"
            f"   Press Ctrl+C in terminal to stop safely.\n"
            f"{'=' * 85}\n"
        )
        print(banner)
        file_logger.info("24/7 Loop Started. Interval=%ds, MaxCycles=%s", self.interval, max_cycles)

        # Graceful shutdown handler
        def shutdown_handler(signum, frame):
            print("\n[!] Graceful shutdown requested. Wrapping up...")
            file_logger.info("Shutdown signal received (Signal: %s)", signum)
            self.is_running = False

        signal.signal(signal.SIGINT, shutdown_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, shutdown_handler)

        try:
            while self.is_running:
                cycle_start_time = time.time()

                # Execute cycle
                self.run_single_cycle()

                if max_cycles and self.cycle_count >= max_cycles:
                    cmd_logger.info("[COMPLETE] Finished target cycle count (%d).", max_cycles)
                    break

                if not self.is_running:
                    break

                # Sleep until next scheduled interval
                elapsed = time.time() - cycle_start_time
                sleep_secs = max(1.0, self.interval - elapsed)
                next_time_str = datetime.fromtimestamp(time.time() + sleep_secs, tz=timezone.utc).strftime("%H:%M:%S UTC")

                cmd_logger.info("[STANDBY] Next cycle scheduled at %s (Sleeping %.0fs)...", next_time_str, sleep_secs)
                file_logger.info("Cycle %d complete in %.2fs. Sleeping %.2fs until %s.", self.cycle_count, elapsed, sleep_secs, next_time_str)

                # Responsive sleep in 1-second chunks
                slept = 0.0
                while self.is_running and slept < sleep_secs:
                    time.sleep(min(1.0, sleep_secs - slept))
                    slept += 1.0

        finally:
            self._print_session_summary()

    def _print_session_summary(self):
        """Prints session summary to CMD and writes full audit to log file."""
        uptime = datetime.now(timezone.utc) - self.session_start
        history = self.broker.trade_history
        buys = sum(1 for o in history if o.side == "BUY")
        sells = sum(1 for o in history if o.side == "SELL")
        total_pnl = sum((o.realized_pnl or 0.0) for o in history if o.side == "SELL")

        summary = (
            f"\n{'=' * 85}\n"
            f"   CBT 24/7 SESSION AUDIT SUMMARY\n"
            f"{'=' * 85}\n"
            f"  * Uptime                : {str(uptime).split('.')[0]}\n"
            f"  * Cycles Executed       : {self.cycle_count} (Success: {self.successful_cycles} | Errors Handled: {self.error_count})\n"
            f"  * Total Trades          : {len(history)} (Buys: {buys} | Sells: {sells})\n"
            f"  * Realized Session PnL  : ${total_pnl:+,.2f} USDT\n"
            f"  * Final Cash Balance    : ${self.broker.get_balance('USDT'):,.2f} USDT\n"
            f"  * Full Audit Log Saved  : {os.path.abspath(self.log_file)}\n"
            f"{'=' * 85}\n"
        )
        print(summary)
        file_logger.info("Session ended safely. Realized PnL=$%.2f. Total trades=%d.", total_pnl, len(history))


# ==============================================================================
# CLI PARSER & MAIN ENTRYPOINT
# ==============================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CBT Step 5: 24/7 Automated Crypto Trading Bot Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SOL/USDT",
        help="Primary trading pair (e.g. 'SOL/USDT', 'BTC/USDT', 'ETH/USDT')",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of pairs (e.g. 'SOL/USDT,BTC/USDT')",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval between cycles in seconds (default: 300s = 5 minutes)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=10.0,
        help="Fixed USDT order budget per trade",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=1000.0,
        help="Initial simulated USDT balance",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Max number of cycles to execute (default: None = continuous infinite loop)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=LOG_FILE_NAME,
        help="Path to persistent log file",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_TESTNET_API_KEY,
        help="Binance Testnet API Key",
    )
    parser.add_argument(
        "--secret",
        type=str,
        default=DEFAULT_TESTNET_SECRET,
        help="Binance Testnet Secret Key",
    )
    parser.add_argument(
        "--show-dashboard",
        action="store_true",
        help="Print full tabular dashboards to CMD (default: clean single-line status, detailed tables go to log file)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_arguments()

    # Reconfigure loggers with user-specified log-file path
    cmd_logger = configure_bot_logging(log_file_path=cli_args.log_file)
    file_logger = logging.getLogger("CBT_Internal")

    # Determine symbols list
    if cli_args.symbols:
        target_symbols = [s.strip() for s in cli_args.symbols.split(",") if s.strip()]
    else:
        target_symbols = [cli_args.symbol]

    engine = CBTContinuousTradingEngine(
        symbols=target_symbols,
        trade_amount_usd=cli_args.amount,
        interval_seconds=cli_args.interval,
        initial_balance_usd=cli_args.initial_balance,
        api_key=cli_args.api_key,
        secret=cli_args.secret,
        log_file=cli_args.log_file,
        show_dashboard=cli_args.show_dashboard,
    )

    # Launch 24/7 automated loop
    engine.start(max_cycles=cli_args.cycles)
