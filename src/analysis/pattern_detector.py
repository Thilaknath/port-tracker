"""
Pattern Detector for Port-Tracker.
Detects historical patterns that often precede price moves.
"""
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

import yfinance as yf


class PatternType(Enum):
    """Types of detectable patterns."""
    STREAK = "streak"                # Consecutive up/down days
    EXTREME = "extreme"              # At 52-week high/low
    DIVERGENCE = "divergence"        # Correlated assets diverging
    VOLATILITY = "volatility"        # Unusual volatility
    MOMENTUM = "momentum"            # Strong momentum that may exhaust


@dataclass
class DetectedPattern:
    """A detected pattern that may indicate risk."""
    ticker: str
    pattern_type: PatternType
    description: str
    historical_outcome: str
    risk_level: str  # HIGH, MEDIUM, LOW
    probability: float  # Historical probability of outcome


# Historical pattern definitions
PATTERNS = {
    "consecutive_highs": {
        "type": PatternType.STREAK,
        "threshold": 5,
        "description": "Asset hit {count} consecutive record highs",
        "historical_outcome": "70% chance of 5%+ pullback within 5 trading days",
        "risk_level": "HIGH"
    },
    "consecutive_up_days": {
        "type": PatternType.STREAK,
        "threshold": 7,
        "description": "Asset rose {count} consecutive days",
        "historical_outcome": "75% chance of at least 1 down day within 3 trading days",
        "risk_level": "MEDIUM"
    },
    "consecutive_down_days": {
        "type": PatternType.STREAK,
        "threshold": 5,
        "description": "Asset fell {count} consecutive days",
        "historical_outcome": "65% chance of bounce within 2 trading days",
        "risk_level": "MEDIUM"
    },
    "52_week_high": {
        "type": PatternType.EXTREME,
        "description": "Asset at or near 52-week high",
        "historical_outcome": "Increased volatility; 60% see 3%+ pullback within 2 weeks",
        "risk_level": "MEDIUM"
    },
    "52_week_low": {
        "type": PatternType.EXTREME,
        "description": "Asset at or near 52-week low",
        "historical_outcome": "May indicate fundamental issues; 55% continue lower",
        "risk_level": "HIGH"
    },
    "parabolic_move": {
        "type": PatternType.MOMENTUM,
        "threshold_pct": 15,
        "threshold_days": 5,
        "description": "Asset gained {pct}%+ in {days} days (parabolic)",
        "historical_outcome": "80% see mean reversion of 30-50% of gains within 2 weeks",
        "risk_level": "HIGH"
    }
}


class PatternDetector:
    """Detects historical patterns in asset prices."""

    def __init__(self):
        self._cache: Dict[str, dict] = {}

    def detect_patterns(self, ticker: str) -> List[DetectedPattern]:
        """
        Detect all patterns for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            List of detected patterns
        """
        patterns = []

        # Get price history
        try:
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period='30d')
            if len(hist) < 5:
                return []
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return []

        # Check consecutive up/down days
        streak_pattern = self._check_streak(ticker, hist)
        if streak_pattern:
            patterns.append(streak_pattern)

        # Check 52-week extremes
        extreme_pattern = self._check_extremes(ticker, yf_ticker, hist)
        if extreme_pattern:
            patterns.append(extreme_pattern)

        # Check for parabolic move
        parabolic_pattern = self._check_parabolic(ticker, hist)
        if parabolic_pattern:
            patterns.append(parabolic_pattern)

        return patterns

    def _check_streak(self, ticker: str, hist) -> Optional[DetectedPattern]:
        """Check for consecutive up/down day streaks."""
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

        if consecutive_up >= 7:
            pattern_def = PATTERNS["consecutive_up_days"]
            return DetectedPattern(
                ticker=ticker,
                pattern_type=pattern_def["type"],
                description=pattern_def["description"].format(count=consecutive_up),
                historical_outcome=pattern_def["historical_outcome"],
                risk_level=pattern_def["risk_level"],
                probability=0.75
            )

        if consecutive_down >= 5:
            pattern_def = PATTERNS["consecutive_down_days"]
            return DetectedPattern(
                ticker=ticker,
                pattern_type=pattern_def["type"],
                description=pattern_def["description"].format(count=consecutive_down),
                historical_outcome=pattern_def["historical_outcome"],
                risk_level=pattern_def["risk_level"],
                probability=0.65
            )

        return None

    def _check_extremes(self, ticker: str, yf_ticker, hist) -> Optional[DetectedPattern]:
        """Check if at 52-week high/low."""
        try:
            info = yf_ticker.info
            current_price = hist['Close'].iloc[-1]
            high_52w = info.get('fiftyTwoWeekHigh', current_price * 1.1)
            low_52w = info.get('fiftyTwoWeekLow', current_price * 0.9)

            # Within 2% of 52-week high
            if current_price >= high_52w * 0.98:
                pattern_def = PATTERNS["52_week_high"]
                return DetectedPattern(
                    ticker=ticker,
                    pattern_type=pattern_def["type"],
                    description=f"{ticker} at ${current_price:.2f}, near 52-week high of ${high_52w:.2f}",
                    historical_outcome=pattern_def["historical_outcome"],
                    risk_level=pattern_def["risk_level"],
                    probability=0.60
                )

            # Within 2% of 52-week low
            if current_price <= low_52w * 1.02:
                pattern_def = PATTERNS["52_week_low"]
                return DetectedPattern(
                    ticker=ticker,
                    pattern_type=pattern_def["type"],
                    description=f"{ticker} at ${current_price:.2f}, near 52-week low of ${low_52w:.2f}",
                    historical_outcome=pattern_def["historical_outcome"],
                    risk_level=pattern_def["risk_level"],
                    probability=0.55
                )

        except Exception:
            pass

        return None

    def _check_parabolic(self, ticker: str, hist) -> Optional[DetectedPattern]:
        """Check for parabolic (unsustainable) price move."""
        if len(hist) < 5:
            return None

        current_price = hist['Close'].iloc[-1]
        price_5d_ago = hist['Close'].iloc[-5]

        pct_change = ((current_price - price_5d_ago) / price_5d_ago) * 100

        pattern_def = PATTERNS["parabolic_move"]
        threshold = pattern_def["threshold_pct"]

        if pct_change >= threshold:
            return DetectedPattern(
                ticker=ticker,
                pattern_type=pattern_def["type"],
                description=pattern_def["description"].format(pct=f"{pct_change:.1f}", days=5),
                historical_outcome=pattern_def["historical_outcome"],
                risk_level=pattern_def["risk_level"],
                probability=0.80
            )

        if pct_change <= -threshold:
            return DetectedPattern(
                ticker=ticker,
                pattern_type=PatternType.MOMENTUM,
                description=f"Asset fell {abs(pct_change):.1f}%+ in 5 days (capitulation)",
                historical_outcome="May indicate oversold conditions; watch for bounce or further breakdown",
                risk_level="HIGH",
                probability=0.60
            )

        return None

    def detect_all(self, tickers: List[str]) -> Dict[str, List[DetectedPattern]]:
        """
        Detect patterns for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dict mapping ticker to list of patterns
        """
        results = {}
        for ticker in tickers:
            patterns = self.detect_patterns(ticker)
            if patterns:
                results[ticker] = patterns
        return results

    def format_for_llm(self, patterns: Dict[str, List[DetectedPattern]]) -> str:
        """Format detected patterns for LLM prompt."""
        if not patterns:
            return "No significant technical patterns detected."

        lines = []
        lines.append("## Technical Pattern Analysis")
        lines.append("")

        for ticker, ticker_patterns in patterns.items():
            for p in ticker_patterns:
                risk_icon = {"HIGH": "[!!!]", "MEDIUM": "[!!]", "LOW": "[!]"}.get(p.risk_level, "[?]")
                lines.append(f"### {risk_icon} {ticker}: {p.pattern_type.value.upper()}")
                lines.append(f"    {p.description}")
                lines.append(f"    Historical: {p.historical_outcome}")
                lines.append(f"    Probability: {p.probability*100:.0f}%")
                lines.append("")

        return "\n".join(lines)
