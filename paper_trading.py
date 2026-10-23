"""
================================================================================
CRYPTO PAPER TRADING EXECUTION ENGINE (Step 4: CCXT & Binance Testnet)
================================================================================
Author: Senior Quantitative Python Developer
Dependencies: ccxt, pandas, decision_engine, market_data, sentiment_engine
Description:
    Connects to the Binance Testnet (Sandbox) with API key authentication.
    Integrates Step 3's Multi-Factor Decision Engine into an automated trading
    execution loop with fixed risk budgeting, position tracking, transaction
    logging, and automatic failover to local paper simulation.
================================================================================
"""

import argparse
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

# Third-party quantitative libraries
try:
    import ccxt
    import pandas as pd
except ImportError:
    print(
        "[!] Missing trading dependencies. Please install:\n"
        "    pip install ccxt pandas"
    )
    sys.exit(1)

# Import internal modules from Steps 1, 2, and 3
try:
    from decision_engine import (
        ActionSignal,
        MultiFactorDecisionEngine,
        TradingDecision,
        run_trading_decision_pipeline,
    )
    from market_data import fetch_live_ticker, init_exchange
    from sentiment_engine import (
        AnalyzedHeadline,
        CryptoNewsFetcher,
        FinBertSentimentEngine,
        MarketSentimentAggregator,
    )
except ImportError as exc:
    print(f"[!] Error importing project components: {exc}")
    print("    Ensure sentiment_engine.py, market_data.py, and decision_engine.py exist.")
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
logger = logging.getLogger("PaperTrader")

# Default Binance Testnet API Key Placeholders
# Users can set real testnet keys via environment variables or CLI flags
DEFAULT_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "YOUR_BINANCE_TESTNET_API_KEY")
DEFAULT_TESTNET_SECRET = os.getenv("BINANCE_TESTNET_SECRET", "YOUR_BINANCE_TESTNET_SECRET_KEY")

BINANCE_TESTNET_BASE_URL = "https://testnet.binance.vision/api/v3"


# ==============================================================================
# DATA MODELS: ORDERS & POSITIONS
# ==============================================================================
@dataclass
class OrderReceipt:
    """Represents an executed market order receipt."""
    order_id: str
    symbol: str
    side: str                 # "BUY" or "SELL"
    fill_price: float         # Execution price in quote currency
    amount: float             # Base crypto amount filled
    cost: float               # Quote currency (USDT) spent or received
    fee: float                # Transaction fee (e.g., 0.1% taker)
    timestamp_utc: str
    status: str               # "FILLED", "REJECTED", "EXPIRED"
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None


@dataclass
class ActivePosition:
    """Tracks an open cryptocurrency inventory position."""
    symbol: str
    amount: float             # Base asset quantity (e.g., SOL)
    entry_price: float        # Weighted average entry price
    total_cost: float         # Total USDT invested
    opened_at_utc: str

    def calculate_unrealized_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calculates (unrealized_pnl_usd, unrealized_pnl_pct)."""
        current_value = self.amount * current_price
        pnl_usd = current_value - self.total_cost
        pnl_pct = (pnl_usd / self.total_cost * 100.0) if self.total_cost > 0 else 0.0
        return pnl_usd, pnl_pct


# ==============================================================================
# HYBRID TESTNET & PAPER TRADING BROKER
# ==============================================================================
class BinanceTestnetBroker:
    """
    Unified execution broker that interfaces with the live Binance Spot Testnet
    (via CCXT sandbox) or seamlessly provides a local simulated ledger if testnet
    API keys are placeholders.
    """

    TAKER_FEE_RATE = 0.001  # Standard 0.10% Binance Spot trading fee

    def __init__(
        self,
        api_key: str = DEFAULT_TESTNET_API_KEY,
        secret: str = DEFAULT_TESTNET_SECRET,
        initial_usdt_balance: float = 1000.0,
    ):
        self.api_key = api_key
        self.secret = secret
        self.is_placeholder_key = (
            "YOUR_BINANCE" in api_key.upper() or len(api_key.strip()) < 10
        )

        # Local simulated ledger (always available as fallback/paper balance)
        self.simulated_balances: Dict[str, float] = {
            "USDT": initial_usdt_balance,
        }
        self.active_positions: Dict[str, ActivePosition] = {}
        self.trade_history: list[OrderReceipt] = []

        # CCXT Testnet initialization
        self.exchange = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.secret,
            "enableRateLimit": True,
            "timeout": 10000,
            "options": {
                "defaultType": "spot",
                "fetchMarkets": ["spot"],
            },
            "urls": {
                "api": {
                    "public": BINANCE_TESTNET_BASE_URL,
                    "private": BINANCE_TESTNET_BASE_URL,
                }
            },
        })
        self.exchange.set_sandbox_mode(True)

        if self.is_placeholder_key:
            logger.info(
                "No live Binance Testnet API key provided. "
                "Running in HIGH-FIDELITY LOCAL PAPER TRADING mode (Starting Balance: $%.2f USDT).",
                initial_usdt_balance,
            )
        else:
            logger.info("Connecting to Binance Spot Testnet Sandbox (%s)...", BINANCE_TESTNET_BASE_URL)

    def get_balance(self, currency: str = "USDT") -> float:
        """
        Retrieves available balance from Binance Testnet or local paper ledger.
        """
        curr = currency.upper().strip()

        # If real keys are provided, query Binance Testnet sandbox
        if not self.is_placeholder_key:
            try:
                raw_bal = self.exchange.fetch_balance()
                free_amt = float(raw_bal.get("free", {}).get(curr, 0.0))
                return free_amt
            except ccxt.AuthenticationError:
                logger.warning(
                    "Binance Testnet rejected API key. Falling back to local paper balance."
                )
            except ccxt.NetworkError as exc:
                logger.warning("Network issue connecting to Testnet balance: %s", str(exc))
            except Exception as exc:
                logger.warning("Error fetching Testnet balance: %s", str(exc))

        # Local paper ledger balance
        return self.simulated_balances.get(curr, 0.0)

    def get_position(self, symbol: str) -> Optional[ActivePosition]:
        """Returns the active open position for a trading pair."""
        return self.active_positions.get(symbol.upper().strip())

    def execute_market_buy(self, symbol: str, usd_budget: float, current_price: float) -> Optional[OrderReceipt]:
        """
        Executes a simulated market BUY order for a fixed USD budget.

        Args:
            symbol: Trading pair (e.g., 'SOL/USDT').
            usd_budget: Amount in USDT to spend (e.g., $10.00).
            current_price: Current market price of the base asset.

        Returns:
            OrderReceipt with filled transaction details, or None if rejected.
        """
        clean_symbol = symbol.upper().strip()
        base_asset = clean_symbol.split("/")[0]
        quote_asset = clean_symbol.split("/")[1] if "/" in clean_symbol else "USDT"

        if current_price <= 0:
            logger.error("Invalid price %.8f for '%s'. Order rejected.", current_price, clean_symbol)
            return None

        available_usdt = self.get_balance(quote_asset)
        if available_usdt < usd_budget:
            logger.error(
                "INSUFFICIENT FUNDS: Required $%.2f %s, but only $%.2f %s available.",
                usd_budget, quote_asset, available_usdt, quote_asset
            )
            return None

        # Calculate base crypto quantity to purchase
        fee_cost = usd_budget * self.TAKER_FEE_RATE
        net_budget = usd_budget - fee_cost
        fill_amount = round(net_budget / current_price, 6)

        if fill_amount <= 0:
            logger.error("Calculated buy quantity is 0. Increase trade budget.")
            return None

        order_id = f"ORD-BUY-{uuid.uuid4().hex[:8].upper()}"
        timestamp_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Attempt remote testnet execution if keys are configured
        if not self.is_placeholder_key:
            try:
                remote_order = self.exchange.create_market_buy_order_with_cost(
                    clean_symbol, usd_budget
                )
                # Market orders often return price=None; float(None) would raise AFTER the
                # order was already placed and make us double-book it on the paper ledger.
                order_id = str(remote_order.get("id") or order_id)
                current_price = float(remote_order.get("average") or remote_order.get("price") or current_price)
                fill_amount = float(remote_order.get("filled") or remote_order.get("amount") or fill_amount)
                logger.info("Binance Testnet Market Buy executed successfully.")
            except ccxt.InsufficientFunds:
                logger.error("Binance Testnet rejected order: Insufficient Testnet balance.")
                return None
            except ccxt.RateLimitExceeded:
                logger.error("Binance Testnet rate limit exceeded.")
                return None
            except Exception as exc:
                logger.warning(
                    "Testnet API execution skipped (%s). Filling via paper ledger.", str(exc)
                )

        # Update local ledger balances
        self.simulated_balances[quote_asset] = self.simulated_balances.get(quote_asset, 0.0) - usd_budget
        self.simulated_balances[base_asset] = self.simulated_balances.get(base_asset, 0.0) + fill_amount

        # Update active position
        self.active_positions[clean_symbol] = ActivePosition(
            symbol=clean_symbol,
            amount=fill_amount,
            entry_price=current_price,
            total_cost=usd_budget,
            opened_at_utc=timestamp_now,
        )

        receipt = OrderReceipt(
            order_id=order_id,
            symbol=clean_symbol,
            side="BUY",
            fill_price=current_price,
            amount=fill_amount,
            cost=usd_budget,
            fee=fee_cost,
            timestamp_utc=timestamp_now,
            status="FILLED",
        )
        self.trade_history.append(receipt)
        return receipt

    def execute_market_sell(self, symbol: str, current_price: float) -> Optional[OrderReceipt]:
        """
        Executes a simulated market SELL order to close an open position.

        Args:
            symbol: Trading pair (e.g., 'SOL/USDT').
            current_price: Current market price of the base asset.

        Returns:
            OrderReceipt with realized PnL details, or None if rejected.
        """
        clean_symbol = symbol.upper().strip()
        base_asset = clean_symbol.split("/")[0]
        quote_asset = clean_symbol.split("/")[1] if "/" in clean_symbol else "USDT"

        position = self.active_positions.get(clean_symbol)
        if not position or position.amount <= 0:
            logger.info("No open position in '%s' to sell. Standing by.", clean_symbol)
            return None

        sell_amount = position.amount
        gross_proceeds = sell_amount * current_price
        fee_cost = gross_proceeds * self.TAKER_FEE_RATE
        net_proceeds = gross_proceeds - fee_cost

        realized_pnl_usd = net_proceeds - position.total_cost
        realized_pnl_pct = (realized_pnl_usd / position.total_cost * 100.0) if position.total_cost > 0 else 0.0

        order_id = f"ORD-SELL-{uuid.uuid4().hex[:8].upper()}"
        timestamp_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Attempt remote testnet execution if keys are configured
        if not self.is_placeholder_key:
            try:
                remote_order = self.exchange.create_market_sell_order(clean_symbol, sell_amount)
                order_id = str(remote_order.get("id") or order_id)
                current_price = float(remote_order.get("average") or remote_order.get("price") or current_price)
                logger.info("Binance Testnet Market Sell executed successfully.")
            except Exception as exc:
                logger.warning(
                    "Testnet API execution skipped (%s). Liquidating via paper ledger.", str(exc)
                )

        # Update local ledger balances
        self.simulated_balances[quote_asset] = self.simulated_balances.get(quote_asset, 0.0) + net_proceeds
        self.simulated_balances[base_asset] = max(0.0, self.simulated_balances.get(base_asset, 0.0) - sell_amount)

        # Close active position
        del self.active_positions[clean_symbol]

        receipt = OrderReceipt(
            order_id=order_id,
            symbol=clean_symbol,
            side="SELL",
            fill_price=current_price,
            amount=sell_amount,
            cost=net_proceeds,
            fee=fee_cost,
            timestamp_utc=timestamp_now,
            status="FILLED",
            realized_pnl=realized_pnl_usd,
            realized_pnl_pct=realized_pnl_pct,
        )
        self.trade_history.append(receipt)
        return receipt


# ==============================================================================
# TRANSACTION LOGGING & PORTFOLIO DASHBOARD
# ==============================================================================
class ExecutionReporter:
    """Formats order execution receipts and portfolio state in the terminal."""

    @staticmethod
    def print_order_receipt(receipt: OrderReceipt, broker: BinanceTestnetBroker):
        """Displays a clean transaction confirmation card."""
        base_asset = receipt.symbol.split("/")[0]
        quote_asset = receipt.symbol.split("/")[1] if "/" in receipt.symbol else "USDT"

        action_badge = "[+] BUY EXECUTED" if receipt.side == "BUY" else "[-] SELL EXECUTED"

        print("\n" + "=" * 90)
        print(f"   TRANSACTION CONFIRMATION: {action_badge}")
        print(f"   Order ID: {receipt.order_id}  |  Time: {receipt.timestamp_utc}")
        print("=" * 90)

        rows = [
            ["Trading Pair", receipt.symbol],
            ["Order Type", f"MARKET {receipt.side}"],
            ["Execution Fill Price", f"${receipt.fill_price:,.4f} {quote_asset}"],
            ["Filled Quantity", f"{receipt.amount:,.6f} {base_asset}"],
            ["Gross Value", f"${receipt.cost:,.2f} {quote_asset}"],
            ["Estimated Taker Fee", f"${receipt.fee:,.4f} {quote_asset} (0.10%)"],
            ["Order Status", f"CONFIRMED [{receipt.status}]"],
        ]

        if receipt.realized_pnl is not None:
            sign = "+" if receipt.realized_pnl >= 0 else ""
            pnl_str = f"{sign}${receipt.realized_pnl:,.2f} ({sign}{receipt.realized_pnl_pct:.2f}%)"
            rows.append(["Realized Profit / Loss", pnl_str])

        # Updated Balances
        updated_quote = broker.get_balance(quote_asset)
        updated_base = broker.get_balance(base_asset)
        rows.append([f"Updated {quote_asset} Balance", f"${updated_quote:,.2f} {quote_asset}"])
        rows.append([f"Updated {base_asset} Balance", f"{updated_base:,.6f} {base_asset}"])

        if HAS_TABULATE:
            print(tabulate(rows, headers=["Field", "Details"], tablefmt="grid"))
        else:
            for r in rows:
                print(f"  {r[0]:<26}: {r[1]}")

        print("=" * 90 + "\n")

    @staticmethod
    def print_portfolio_snapshot(broker: BinanceTestnetBroker, symbol: str, current_price: float):
        """Displays current position and cash balance status."""
        clean_symbol = symbol.upper().strip()
        base_asset = clean_symbol.split("/")[0]
        quote_asset = clean_symbol.split("/")[1] if "/" in clean_symbol else "USDT"

        usdt_bal = broker.get_balance(quote_asset)
        base_bal = broker.get_balance(base_asset)
        pos = broker.get_position(clean_symbol)

        print("-" * 90)
        print(f"   PORTFOLIO STATUS  |  {clean_symbol}  |  Cash: ${usdt_bal:,.2f} {quote_asset}")
        print("-" * 90)

        if pos and pos.amount > 0:
            unreal_pnl, unreal_pnl_pct = pos.calculate_unrealized_pnl(current_price)
            sign = "+" if unreal_pnl >= 0 else ""
            print(
                f"  * ACTIVE POSITION   : {pos.amount:,.6f} {base_asset} "
                f"(Entry: ${pos.entry_price:,.2f} | Current: ${current_price:,.2f})"
            )
            print(f"  * UNREALIZED PnL    : {sign}${unreal_pnl:,.2f} ({sign}{unreal_pnl_pct:.2f}%)")
            print(f"  * POSITION COST     : ${pos.total_cost:,.2f} {quote_asset}")
        else:
            print(f"  * ACTIVE POSITION   : NONE (100% Cash: ${usdt_bal:,.2f} {quote_asset})")

        print("-" * 90 + "\n")


# ==============================================================================
# AUTOMATED TRADING LOOP PIPELINE
# ==============================================================================
class AutomatedPaperTradingBot:
    """
    Orchestrates the continuous trading execution loop:
    1. Evaluates market via MultiFactorDecisionEngine (Steps 1 & 2).
    2. Enforces position risk sizing ($10 USDT fixed size).
    3. Executes market BUY/SELL orders or HOLDs.
    4. Logs transaction receipts and maintains audit ledger.
    """

    def __init__(
        self,
        symbol: str = "SOL/USDT",
        trade_amount_usd: float = 10.0,
        api_key: str = DEFAULT_TESTNET_API_KEY,
        secret: str = DEFAULT_TESTNET_SECRET,
        initial_balance: float = 1000.0,
        bull_thresh: float = 0.30,
        bear_thresh: float = -0.30,
        stop_loss_pct: float = -3.50,
    ):
        self.symbol = symbol.upper().strip()
        self.trade_amount = trade_amount_usd
        self.broker = BinanceTestnetBroker(
            api_key=api_key,
            secret=secret,
            initial_usdt_balance=initial_balance,
        )
        self.decision_engine = MultiFactorDecisionEngine(
            sentiment_bull_thresh=bull_thresh,
            sentiment_bear_thresh=bear_thresh,
            drawdown_stop_pct=stop_loss_pct,
        )

    def execute_cycle(self, override_signal: Optional[str] = None) -> Optional[OrderReceipt]:
        """
        Executes a single end-to-end decision & paper execution cycle.
        """
        logger.info(">>> Starting Automated Trading Cycle for %s <<<", self.symbol)

        # 1. Run Strategy Decision Engine (Steps 1, 2, 3)
        decision = run_trading_decision_pipeline(
            symbol=self.symbol,
            news_limit=6,
        )

        current_price = decision.current_price
        if override_signal:
            signal_name = override_signal.upper().strip()
            signal = ActionSignal[signal_name]
            logger.info("FORCED SIGNAL OVERRIDE ACTIVE: %s", signal.value)
        else:
            signal = decision.action
        receipt: Optional[OrderReceipt] = None

        # 2. Risk Management & Order Execution Logic
        existing_pos = self.broker.get_position(self.symbol)

        if signal == ActionSignal.BUY:
            if existing_pos and existing_pos.amount > 0:
                logger.info(
                    "BUY signal detected, but already holding active position (%s). "
                    "Skipping duplicate entry to enforce maximum single-position risk.",
                    self.symbol,
                )
            else:
                logger.info(
                    "BUY Signal confirmed. Placing simulated MARKET BUY ($%.2f USDT)...",
                    self.trade_amount,
                )
                receipt = self.broker.execute_market_buy(
                    symbol=self.symbol,
                    usd_budget=self.trade_amount,
                    current_price=current_price,
                )
                if receipt:
                    ExecutionReporter.print_order_receipt(receipt, self.broker)

        elif signal == ActionSignal.SELL:
            if existing_pos and existing_pos.amount > 0:
                logger.info("SELL / EXIT Signal confirmed. Placing simulated MARKET SELL to close position...")
                receipt = self.broker.execute_market_sell(
                    symbol=self.symbol,
                    current_price=current_price,
                )
                if receipt:
                    ExecutionReporter.print_order_receipt(receipt, self.broker)
            else:
                logger.info("SELL signal generated, but no active position to liquidate. Standing by in cash.")

        else:
            # ActionSignal.HOLD
            logger.info("HOLD Signal active. No order dispatched. Maintaining risk status.")

        # 3. Print Portfolio Snapshot
        ExecutionReporter.print_portfolio_snapshot(self.broker, self.symbol, current_price)
        return receipt

    def run_loop(self, interval_seconds: int = 60):
        """Runs the automated trading loop continuously with sleep intervals."""
        logger.info(
            "Starting continuous automated loop for %s (Polling Interval: %d seconds)...",
            self.symbol, interval_seconds
        )
        print(f"[*] Automated Paper Trading Loop active. Press Ctrl+C to terminate.\n")

        cycle_count = 1
        try:
            while True:
                print(f"\n{'=' * 90}")
                print(f"   AUTOMATED CYCLE #{cycle_count}  --  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"{'=' * 90}")
                try:
                    self.execute_cycle()
                except Exception as exc:
                    logger.error("Error encountered during execution cycle: %s", str(exc))

                cycle_count += 1
                logger.info("Cycle complete. Sleeping for %d seconds...", interval_seconds)
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n[!] Trading bot loop stopped by user. Generating final session summary...")
            self.print_session_summary()

    def print_session_summary(self):
        """Prints cumulative trading statistics."""
        history = self.broker.trade_history
        buys = sum(1 for o in history if o.side == "BUY")
        sells = sum(1 for o in history if o.side == "SELL")
        total_pnl = sum(o.realized_pnl for o in history if o.realized_pnl is not None)

        print("\n" + "=" * 90)
        print("   SESSION PERFORMANCE AUDIT SUMMARY")
        print("=" * 90)
        print(f"  * Total Trades Executed : {len(history)} (Buys: {buys} | Sells: {sells})")
        print(f"  * Realized Session PnL  : ${total_pnl:+,.2f} USDT")
        print(f"  * Final Cash Balance    : ${self.broker.get_balance('USDT'):,.2f} USDT")
        print("=" * 90 + "\n")


# ==============================================================================
# CLI ARGUMENT PARSER & ENTRYPOINT
# ==============================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: Automated Crypto Paper Trading Engine (Binance Testnet / CCXT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SOL/USDT",
        help="Crypto trading pair (e.g. 'SOL/USDT', 'BTC/USDT', 'ETH/USDT')",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=10.0,
        help="Fixed USDT order size per trade",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=1000.0,
        help="Simulated initial starting balance in USDT",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously in an automated periodic loop",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval between cycles in seconds when running in --loop mode",
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
        "--force-signal",
        type=str,
        choices=["BUY", "SELL", "HOLD"],
        default=None,
        help="Simulate an explicit signal execution (BUY, SELL, HOLD) for test verification",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    bot = AutomatedPaperTradingBot(
        symbol=args.symbol,
        trade_amount_usd=args.amount,
        api_key=args.api_key,
        secret=args.secret,
        initial_balance=args.initial_balance,
    )

    if args.loop:
        bot.run_loop(interval_seconds=args.interval)
    else:
        # Default: Single execution cycle
        bot.execute_cycle(override_signal=args.force_signal)
