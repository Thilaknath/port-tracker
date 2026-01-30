"""
Correlation Tracker for Port-Tracker.
Tracks correlated assets and detects divergences.
"""
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

import yfinance as yf


class CorrelationType(Enum):
    """Type of correlation between assets."""
    POSITIVE = "positive"  # Assets move together
    NEGATIVE = "negative"  # Assets move inversely


@dataclass
class AssetPrice:
    """Current price data for an asset."""
    ticker: str
    price: float
    change_pct: float
    change_5d_pct: float = 0.0
    at_high: bool = False  # At or near 52-week high
    at_low: bool = False   # At or near 52-week low
    consecutive_up_days: int = 0
    consecutive_down_days: int = 0


@dataclass
class Divergence:
    """A detected divergence between correlated assets."""
    asset1: str
    asset2: str
    correlation_type: CorrelationType
    expected_behavior: str
    actual_behavior: str
    severity: str  # HIGH, MEDIUM, LOW
    description: str


# Known correlations between assets
ASSET_CORRELATIONS = {
    # Precious metals vs Dollar (negative correlation)
    ("GLD", "DXY"): CorrelationType.NEGATIVE,
    ("SLV", "DXY"): CorrelationType.NEGATIVE,
    ("GLD", "UUP"): CorrelationType.NEGATIVE,
    ("SLV", "UUP"): CorrelationType.NEGATIVE,

    # Gold and Silver (positive correlation)
    ("GLD", "SLV"): CorrelationType.POSITIVE,

    # Tech and Growth (positive correlation)
    ("QQQ", "SPY"): CorrelationType.POSITIVE,
    ("QQQ", "AAPL"): CorrelationType.POSITIVE,
    ("QQQ", "NVDA"): CorrelationType.POSITIVE,

    # Precious metals and real rates/bonds (complex)
    ("GLD", "TLT"): CorrelationType.POSITIVE,  # Both benefit from rate cuts
    ("SLV", "TLT"): CorrelationType.POSITIVE,
}


class CorrelationTracker:
    """Tracks correlated assets and detects divergences."""

    def __init__(self):
        self._price_cache: Dict[str, AssetPrice] = {}
        self._cache_time: Optional[datetime] = None

    def get_price_data(self, ticker: str) -> Optional[AssetPrice]:
        """
        Get current price data for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            AssetPrice object or None
        """
        # Check cache (5 minute validity)
        if self._cache_time and (datetime.now() - self._cache_time).seconds < 300:
            if ticker in self._price_cache:
                return self._price_cache[ticker]

        try:
            yf_ticker = yf.Ticker(ticker)

            # Get recent history
            hist = yf_ticker.history(period='10d')
            if len(hist) < 2:
                return None

            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change_pct = ((current_price - prev_close) / prev_close) * 100

            # 5-day change
            if len(hist) >= 5:
                five_day_ago = hist['Close'].iloc[-5]
                change_5d_pct = ((current_price - five_day_ago) / five_day_ago) * 100
            else:
                change_5d_pct = 0

            # Count consecutive up/down days
            consecutive_up = 0
            consecutive_down = 0
            for i in range(len(hist) - 1, 0, -1):
                if hist['Close'].iloc[i] > hist['Close'].iloc[i-1]:
                    if consecutive_down == 0:
                        consecutive_up += 1
                    else:
                        break
                else:
                    if consecutive_up == 0:
                        consecutive_down += 1
                    else:
                        break

            # Check if at high/low
            try:
                info = yf_ticker.info
                high_52w = info.get('fiftyTwoWeekHigh', current_price * 1.1)
                low_52w = info.get('fiftyTwoWeekLow', current_price * 0.9)
                at_high = current_price >= high_52w * 0.98
                at_low = current_price <= low_52w * 1.02
            except:
                at_high = False
                at_low = False

            price_data = AssetPrice(
                ticker=ticker,
                price=current_price,
                change_pct=change_pct,
                change_5d_pct=change_5d_pct,
                at_high=at_high,
                at_low=at_low,
                consecutive_up_days=consecutive_up,
                consecutive_down_days=consecutive_down
            )

            self._price_cache[ticker] = price_data
            self._cache_time = datetime.now()

            return price_data

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None

    def detect_divergences(self, tickers: List[str]) -> List[Divergence]:
        """
        Detect divergences between correlated assets.

        Args:
            tickers: List of tickers to analyze

        Returns:
            List of detected divergences
        """
        divergences = []

        # Get price data for all tickers
        prices = {}
        for ticker in tickers:
            price_data = self.get_price_data(ticker)
            if price_data:
                prices[ticker] = price_data

        # Check known correlations
        for (t1, t2), corr_type in ASSET_CORRELATIONS.items():
            if t1 in prices and t2 in prices:
                p1 = prices[t1]
                p2 = prices[t2]

                divergence = self._check_divergence(p1, p2, corr_type)
                if divergence:
                    divergences.append(divergence)

        return divergences

    def _check_divergence(
        self,
        p1: AssetPrice,
        p2: AssetPrice,
        corr_type: CorrelationType
    ) -> Optional[Divergence]:
        """Check for divergence between two assets."""

        if corr_type == CorrelationType.NEGATIVE:
            # Negative correlation: expect opposite movements
            # Divergence if both moving same direction significantly
            if (p1.change_pct > 1 and p2.change_pct > 1) or \
               (p1.change_pct < -1 and p2.change_pct < -1):
                severity = "HIGH" if abs(p1.change_pct) > 2 or abs(p2.change_pct) > 2 else "MEDIUM"
                return Divergence(
                    asset1=p1.ticker,
                    asset2=p2.ticker,
                    correlation_type=corr_type,
                    expected_behavior=f"{p1.ticker} and {p2.ticker} should move opposite",
                    actual_behavior=f"Both moving {'up' if p1.change_pct > 0 else 'down'}",
                    severity=severity,
                    description=f"{p1.ticker} {p1.change_pct:+.1f}% while {p2.ticker} {p2.change_pct:+.1f}% - "
                               f"historically negative correlation suggests reversion"
                )

            # Also flag if one is moving strongly while other is flat
            if abs(p1.change_pct) > 2 and abs(p2.change_pct) < 0.5:
                return Divergence(
                    asset1=p1.ticker,
                    asset2=p2.ticker,
                    correlation_type=corr_type,
                    expected_behavior=f"{p2.ticker} should react to {p1.ticker} move",
                    actual_behavior=f"{p1.ticker} moving {p1.change_pct:+.1f}% but {p2.ticker} flat",
                    severity="MEDIUM",
                    description=f"Lagged correlation - {p2.ticker} may catch up to {p1.ticker}'s move"
                )

        elif corr_type == CorrelationType.POSITIVE:
            # Positive correlation: expect same direction
            # Divergence if moving opposite directions significantly
            if (p1.change_pct > 1 and p2.change_pct < -1) or \
               (p1.change_pct < -1 and p2.change_pct > 1):
                severity = "HIGH" if abs(p1.change_pct - p2.change_pct) > 3 else "MEDIUM"
                return Divergence(
                    asset1=p1.ticker,
                    asset2=p2.ticker,
                    correlation_type=corr_type,
                    expected_behavior=f"{p1.ticker} and {p2.ticker} should move together",
                    actual_behavior=f"Moving opposite: {p1.ticker} {p1.change_pct:+.1f}%, {p2.ticker} {p2.change_pct:+.1f}%",
                    severity=severity,
                    description=f"Unusual divergence - one may be mispriced or leading indicator"
                )

        return None

    def detect_streak_risk(self, ticker: str) -> Optional[str]:
        """
        Detect if an asset has an unusual streak (potential reversal risk).

        Args:
            ticker: Ticker to check

        Returns:
            Risk description or None
        """
        price_data = self.get_price_data(ticker)
        if not price_data:
            return None

        risks = []

        # Check consecutive days
        if price_data.consecutive_up_days >= 5:
            risks.append(
                f"{ticker} has risen {price_data.consecutive_up_days} consecutive days. "
                f"Historical data shows 70%+ chance of pullback within 5 days after 5+ day streaks."
            )

        if price_data.consecutive_down_days >= 5:
            risks.append(
                f"{ticker} has fallen {price_data.consecutive_down_days} consecutive days. "
                f"May be oversold - watch for bounce."
            )

        # Check if at 52-week extremes
        if price_data.at_high:
            risks.append(
                f"{ticker} is at/near 52-week high at ${price_data.price:.2f}. "
                f"Extended rallies at highs often see profit-taking."
            )

        if price_data.at_low:
            risks.append(
                f"{ticker} is at/near 52-week low. "
                f"May indicate fundamental problems or capitulation."
            )

        return " | ".join(risks) if risks else None

    def format_for_llm(self, tickers: List[str]) -> str:
        """Format correlation data for LLM prompt."""
        lines = []
        lines.append("## Correlated Asset Analysis")
        lines.append("")

        # Get divergences
        divergences = self.detect_divergences(tickers)
        if divergences:
            lines.append("### Detected Divergences")
            for d in divergences:
                lines.append(f"- [{d.severity}] {d.asset1} vs {d.asset2}: {d.description}")
            lines.append("")

        # Check streaks for key assets
        lines.append("### Streak Analysis")
        for ticker in tickers[:5]:
            streak_risk = self.detect_streak_risk(ticker)
            if streak_risk:
                lines.append(f"- {streak_risk}")

        if len(lines) == 3:  # Only headers
            lines.append("- No significant patterns detected")

        return "\n".join(lines)
