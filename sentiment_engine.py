"""
================================================================================
CRYPTO SENTIMENT ANALYSIS ENGINE (Step 1: Quantitative AI Engine)
================================================================================
Author: Senior Quantitative Python Developer
Target Model: ProsusAI/finbert (Financial PhraseBank BERT)
Description:
    Fetches real-time crypto headlines from multiple RSS feeds, filters by
    token (Bitcoin, Ethereum, Solana, etc.), runs transformer-based financial
    sentiment inference using FinBERT, and computes an aggregated Market
    Sentiment Index [-1.0 to +1.0] with quantitative trading signals.
================================================================================
"""

import concurrent.futures
import argparse
import logging
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Third-party imports with clear guidance if missing
try:
    import feedparser
    import requests
except ImportError:
    print(
        "[!] Missing web dependencies. Please install: "
        "pip install requests feedparser"
    )
    sys.exit(1)

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print(
        "[!] Missing deep learning dependencies. Please install: "
        "pip install torch transformers"
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
logger = logging.getLogger("CryptoSentiment")

# Default RSS feeds providing free, live crypto news coverage
DEFAULT_RSS_FEEDS = [
    {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    },
    {
        "name": "Cointelegraph",
        "url": "https://cointelegraph.com/rss",
    },
    {
        "name": "Decrypt",
        "url": "https://decrypt.co/feed",
    },
    {
        "name": "Bitcoin Magazine",
        "url": "https://bitcoinmagazine.com/.rss/full/",
    },
    {
        "name": "BeInCrypto",
        "url": "https://beincrypto.com/feed/",
    },
]

# Keyword aliases for high-precision token filtering
TOKEN_ALIASES = {
    "bitcoin": ["bitcoin", r"\bbtc\b", r"\bxbt\b", "satoshi"],
    # "ether" must be word-bounded or it matches "together", "whether", "either"
    "ethereum": ["ethereum", r"\beth\b", r"\bether\b", "vitalik"],
    "solana": ["solana", r"\bsol\b"],
    "ripple": ["ripple", r"\bxrp\b"],
    "cardano": ["cardano", r"\bada\b"],
    "dogecoin": ["dogecoin", r"\bdoge\b"],
}

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


# ==============================================================================
# DATA MODELS
# ==============================================================================
@dataclass
class NewsArticle:
    """Represents an ingested crypto news headline."""
    title: str
    source: str
    link: str
    published: str
    token_matched: Optional[str] = None


@dataclass
class SentimentResult:
    """Encapsulates the FinBERT model inference results."""
    label: str               # "Positive", "Negative", or "Neutral"
    confidence: float        # Max softmax probability [0.0 to 1.0]
    polarity_score: float    # Signed polarity: P(Pos) - P(Neg) in [-1.0, 1.0]
    probabilities: Dict[str, float]  # Full probability distribution


@dataclass
class AnalyzedHeadline:
    """Pairs raw article data with its quantitative sentiment output."""
    article: NewsArticle
    sentiment: SentimentResult


@dataclass
class MarketSentimentSummary:
    """Aggregated quantitative metrics across all analyzed headlines."""
    target_token: str
    total_headlines: int
    positive_count: int
    negative_count: int
    neutral_count: int
    avg_confidence: float
    market_sentiment_index: float  # Scale: -1.0 (Bearish) to +1.0 (Bullish)
    trading_signal: str            # STRONG BUY, BUY, HOLD, SELL, STRONG SELL


# ==============================================================================
# NEWS INGESTION ENGINE
# ==============================================================================
class CryptoNewsFetcher:
    """
    Robust RSS feed aggregator that queries top cryptocurrency media outlets,
    applies user-agent headers to prevent bot-blocking, and filters by token.
    """

    def __init__(self, feeds: Optional[List[Dict[str, str]]] = None):
        self.feeds = feeds or DEFAULT_RSS_FEEDS

    def _matches_token(self, text: str, token: str) -> bool:
        """
        Check if text mentions target token using word boundary regex.
        Prevents false positive matches (e.g. 'sol' in 'absolute').
        """
        token_key = token.lower().strip()
        patterns = TOKEN_ALIASES.get(token_key, [rf"\b{re.escape(token_key)}\b"])
        text_lower = text.lower()
        for pat in patterns:
            if re.search(pat, text_lower):
                return True
        return False

    def _fetch_feed_articles(self, feed_info: Dict[str, str], timeout: int) -> List[Dict]:
        """Fetches and parses a single feed with strict timeout."""
        feed_name = feed_info["name"]
        feed_url = feed_info["url"]
        try:
            resp = requests.get(feed_url, headers=HTTP_HEADERS, timeout=timeout)
            if resp.status_code != 200:
                return []
            feed_data = feedparser.parse(resp.content)
            if feed_data.bozo and not feed_data.entries:
                return []
            items = []
            for entry in feed_data.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                items.append({
                    "title": title,
                    "summary": entry.get("summary", ""),
                    "source": feed_name,
                    "link": entry.get("link", ""),
                    "published": entry.get("published", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                })
            return items
        except Exception:
            return []

    def fetch_headlines(
        self,
        token: Optional[str] = None,
        max_articles: int = 15,
        timeout: int = 4,
    ) -> List[NewsArticle]:
        """
        Fetches headlines concurrently from all configured feeds, filters and deduplicates.
        """
        collected: List[NewsArticle] = []
        seen_titles = set()

        filter_enabled = token is not None and token.lower() != "all"
        target_display = token.title() if filter_enabled else "All Crypto"

        # Concurrently fetch all RSS feeds to eliminate UI loading delays
        all_entries: List[Dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.feeds)) as executor:
            future_to_feed = {
                executor.submit(self._fetch_feed_articles, f, timeout): f["name"]
                for f in self.feeds
            }
            for future in concurrent.futures.as_completed(future_to_feed):
                try:
                    entries = future.result()
                    all_entries.extend(entries)
                except Exception:
                    pass

        for entry in all_entries:
            raw_title = entry["title"]
            norm_key = re.sub(r"\W+", "", raw_title.lower())
            if norm_key in seen_titles:
                continue

            summary = entry.get("summary", "")
            full_context = f"{raw_title} {summary}"

            if filter_enabled:
                if not self._matches_token(full_context, token):
                    continue
                matched_token = target_display
            else:
                matched_token = "Crypto"

            seen_titles.add(norm_key)
            collected.append(
                NewsArticle(
                    title=raw_title,
                    source=entry["source"],
                    link=entry.get("link", ""),
                    published=entry.get("published", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                    token_matched=matched_token,
                )
            )

            if len(collected) >= max_articles:
                break

        return collected


# ==============================================================================
# HUGGING FACE FINBERT SENTIMENT ANALYZER
# ==============================================================================
class FinBertSentimentEngine:
    """
    Evaluates financial headlines using ProsusAI/finbert.
    Extracts class probabilities, discrete labels, and continuous polarity.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(
        self,
        device: Optional[str] = None,
        local_files_only: bool = False,
        tokenizer=None,
        model=None,
    ):
        """
        Initializes tokenizer and model on CPU or CUDA GPU.
        Supports offline loading to eliminate slow remote Hugging Face API queries.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            logger.info("Loading '%s' on %s (offline_first=%s)...", self.MODEL_NAME, self.device, local_files_only)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, local_files_only=local_files_only)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME, local_files_only=local_files_only)
            except Exception:
                # Fallback to online fetch if local files not found
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()

        # FinBERT labels: usually ['positive', 'negative', 'neutral']
        # Dynamically map id to label name from model config
        self.id2label = {idx: label.lower() for idx, label in self.model.config.id2label.items()}
        logger.info("FinBERT initialized successfully with labels: %s", self.id2label)

    def analyze_headline(self, text: str) -> SentimentResult:
        """
        Infers sentiment for a single headline.

        Returns:
            SentimentResult containing label, confidence, and signed polarity.
        """
        return self.analyze_batch([text])[0]

    def analyze_batch(self, headlines: List[str], batch_size: int = 16) -> List[SentimentResult]:
        """
        Runs batched GPU/CPU inference for maximum quantitative pipeline efficiency.
        """
        results: List[SentimentResult] = []

        for i in range(0, len(headlines), batch_size):
            chunk = headlines[i : i + batch_size]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            for probs in probabilities:
                # Map probabilities to classes
                prob_dict = {
                    self.id2label[idx]: float(probs[idx])
                    for idx in range(len(probs))
                }

                p_pos = prob_dict.get("positive", 0.0)
                p_neg = prob_dict.get("negative", 0.0)
                p_neu = prob_dict.get("neutral", 0.0)

                # Determine winning label & confidence
                max_idx = int(probs.argmax())
                raw_label = self.id2label[max_idx]
                confidence = float(probs[max_idx])

                # Quantitative Polarity:
                # Polarity = P(Positive) - P(Negative) in [-1.0, +1.0]
                # High neutral probability naturally dampens the polarity toward 0.0
                polarity = p_pos - p_neg

                results.append(
                    SentimentResult(
                        label=raw_label.capitalize(),
                        confidence=confidence,
                        polarity_score=polarity,
                        probabilities={
                            "Positive": p_pos,
                            "Negative": p_neg,
                            "Neutral": p_neu,
                        },
                    )
                )

        return results


# ==============================================================================
# MARKET SENTIMENT AGGREGATOR & SIGNAL GENERATOR
# ==============================================================================
class MarketSentimentAggregator:
    """
    Computes overall Market Sentiment Index [-1.0 to +1.0] and generates
    algorithmic trading signals based on quantitative thresholds.
    """

    # Signal Thresholds for Sentiment Index
    THRESH_STRONG_BUY = 0.35
    THRESH_BUY = 0.15
    THRESH_SELL = -0.15
    THRESH_STRONG_SELL = -0.35

    @classmethod
    def aggregate(
        cls,
        analyzed_items: List[AnalyzedHeadline],
        target_token: str,
    ) -> MarketSentimentSummary:
        """
        Aggregates individual headline scores into an overall index & signal.
        """
        if not analyzed_items:
            return MarketSentimentSummary(
                target_token=target_token,
                total_headlines=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                avg_confidence=0.0,
                market_sentiment_index=0.0,
                trading_signal="NEUTRAL / NO DATA",
            )

        pos_count = sum(1 for item in analyzed_items if item.sentiment.label == "Positive")
        neg_count = sum(1 for item in analyzed_items if item.sentiment.label == "Negative")
        neu_count = sum(1 for item in analyzed_items if item.sentiment.label == "Neutral")
        total = len(analyzed_items)

        # Average polarity score across all headlines clamped to [-1.0, +1.0]
        mean_polarity = sum(item.sentiment.polarity_score for item in analyzed_items) / total
        sentiment_index = max(-1.0, min(1.0, round(mean_polarity, 4)))

        avg_conf = sum(item.sentiment.confidence for item in analyzed_items) / total

        # Algorithmic Trading Signal derivation
        if sentiment_index >= cls.THRESH_STRONG_BUY:
            signal = "STRONG BUY (Extreme Bullish)"
        elif sentiment_index >= cls.THRESH_BUY:
            signal = "BUY (Bullish)"
        elif sentiment_index <= cls.THRESH_STRONG_SELL:
            signal = "STRONG SELL (Extreme Bearish)"
        elif sentiment_index <= cls.THRESH_SELL:
            signal = "SELL (Bearish)"
        else:
            signal = "HOLD / NEUTRAL (Consolidation)"

        return MarketSentimentSummary(
            target_token=target_token,
            total_headlines=total,
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neu_count,
            avg_confidence=avg_conf,
            market_sentiment_index=sentiment_index,
            trading_signal=signal,
        )


# ==============================================================================
# TERMINAL VISUALIZATION & REPORTING
# ==============================================================================
class TerminalReporter:
    """Renders formatted tables, sentiment meters, and executive summaries."""

    @staticmethod
    def render_gauge(score: float, width: int = 30) -> str:
        """
        Creates an ASCII needle gauge for sentiment visualization.
        Range: -1.0 (Bearish) to +1.0 (Bullish).
        """
        # Map score from [-1.0, 1.0] to [0, width - 1]
        clamped = max(-1.0, min(1.0, score))
        norm_pos = int(((clamped + 1.0) / 2.0) * (width - 1))

        bar = []
        for i in range(width):
            if i == norm_pos:
                bar.append("O")
            elif i == width // 2:
                bar.append("|")
            else:
                bar.append("-")

        bar_str = "".join(bar)
        return f"[-1.0 BEARISH] {bar_str} [+1.0 BULLISH]"

    @classmethod
    def display_report(
        cls,
        analyzed_items: List[AnalyzedHeadline],
        summary: MarketSentimentSummary,
    ):
        """Displays the full terminal report with tables and metrics."""
        print("\n" + "=" * 90)
        print(f"   CRYPTO SENTIMENT ANALYSIS REPORT  --  Target: {summary.target_token.upper()}")
        print(f"   Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 90)

        # 1. Headline Breakdown Table
        table_rows = []
        for idx, item in enumerate(analyzed_items, 1):
            # Wrap long headlines to keep output cleanly aligned in terminals
            wrapped_title = "\n".join(textwrap.wrap(item.article.title, width=44))

            # Sign for polarity display
            sign = "+" if item.sentiment.polarity_score > 0 else ""
            polarity_fmt = f"{sign}{item.sentiment.polarity_score:.2f}"

            label_str = item.sentiment.label
            if label_str == "Positive":
                label_fmt = "[+] POSITIVE"
            elif label_str == "Negative":
                label_fmt = "[-] NEGATIVE"
            else:
                label_fmt = "[~] NEUTRAL"

            table_rows.append([
                idx,
                item.article.source[:12],
                wrapped_title,
                label_fmt,
                f"{item.sentiment.confidence * 100:.1f}%",
                polarity_fmt,
            ])

        headers = ["#", "Source", "Headline Text", "Sentiment", "Confidence", "Polarity"]

        if HAS_TABULATE:
            print(tabulate(table_rows, headers=headers, tablefmt="grid"))
        else:
            # Fallback simple table if tabulate is not installed
            print(f"{'#':<3} | {'Source':<12} | {'Headline':<44} | {'Sentiment':<12} | {'Conf':<7} | {'Score'}")
            print("-" * 90)
            for r in table_rows:
                first_line = r[2].split("\n")[0]
                print(f"{r[0]:<3} | {r[1]:<12} | {first_line:<44} | {r[3]:<12} | {r[4]:<7} | {r[5]}")

        # 2. Executive Sentiment Metrics
        print("\n" + "-" * 90)
        print("   EXECUTIVE QUANTITATIVE SUMMARY")
        print("-" * 90)
        print(f"  Headlines Analyzed    : {summary.total_headlines}")
        print(
            f"  Sentiment Breakdown   : Positive: {summary.positive_count} | "
            f"Negative: {summary.negative_count} | "
            f"Neutral: {summary.neutral_count}"
        )
        print(f"  Average Confidence    : {summary.avg_confidence * 100:.2f}%")
        print(f"  Market Sentiment Index: {summary.market_sentiment_index:+.4f} (Scale: -1.0000 to +1.0000)")
        print(f"  Visual Sentiment Meter: {cls.render_gauge(summary.market_sentiment_index)}")
        print(f"  Algorithmic Signal    : >>> {summary.trading_signal} <<<")
        print("=" * 90 + "\n")


# ==============================================================================
# PIPELINE ORCHESTRATION & CLI
# ==============================================================================
def run_pipeline(
    token: str = "Bitcoin",
    limit: int = 10,
    device: Optional[str] = None,
) -> MarketSentimentSummary:
    """
    Executes the end-to-end sentiment engine pipeline:
    1. Ingests news from RSS feeds.
    2. Runs FinBERT inference on each headline.
    3. Computes the aggregate Sentiment Index and Trading Signal.
    4. Outputs formatted terminal reports.
    """
    logger.info("Initializing Crypto Sentiment Analysis Pipeline for '%s'...", token)

    # Step 1: Ingest news headlines
    fetcher = CryptoNewsFetcher()
    articles = fetcher.fetch_headlines(token=token, max_articles=limit)

    if not articles:
        logger.warning(
            "No recent headlines found matching '%s'. "
            "Try increasing the limit or using 'all' to inspect broad crypto feeds.",
            token,
        )
        empty_summary = MarketSentimentAggregator.aggregate([], token)
        TerminalReporter.display_report([], empty_summary)
        return empty_summary

    # Step 2: Initialize FinBERT and infer sentiment
    engine = FinBertSentimentEngine(device=device)
    titles = [a.title for a in articles]
    sentiment_results = engine.analyze_batch(titles)

    analyzed_items = [
        AnalyzedHeadline(article=art, sentiment=sent)
        for art, sent in zip(articles, sentiment_results)
    ]

    # Step 3: Aggregate scores & compute index
    summary = MarketSentimentAggregator.aggregate(analyzed_items, target_token=token)

    # Step 4: Render clean terminal report
    TerminalReporter.display_report(analyzed_items, summary)

    return summary


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Step 1: AI Crypto Sentiment Analysis Engine (FinBERT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--token",
        type=str,
        default="Bitcoin",
        help="Crypto token to analyze (e.g. 'Bitcoin', 'Ethereum', 'Solana', or 'all')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of headlines to fetch and score",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu', 'cuda', or auto-detect if None",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(token=args.token, limit=args.limit, device=args.device)
