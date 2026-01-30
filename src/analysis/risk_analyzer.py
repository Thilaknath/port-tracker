"""
LLM-Powered Risk Analyzer for Port-Tracker.
The core predictive analysis engine that identifies portfolio risks.
"""
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

from src.llm.providers import get_llm
from src.portfolio.holdings import Portfolio
from src.monitors.news_scanner import NewsScanner, NewsItem
from src.monitors.event_calendar import EventCalendar, ScheduledEvent
from src.monitors.correlation_tracker import CorrelationTracker


class RiskSeverity(Enum):
    """Severity levels for identified risks."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskType(Enum):
    """Types of portfolio risks."""
    MACRO = "MACRO"           # Fed, rates, dollar, geopolitical
    SECTOR = "SECTOR"         # Sector-wide risk
    COMPANY = "COMPANY"       # Company-specific risk
    TECHNICAL = "TECHNICAL"   # Price patterns, streaks, extremes
    CORRELATION = "CORRELATION"  # Divergence/correlation risk


class RecommendedAction(Enum):
    """Recommended actions for risks."""
    WATCH = "WATCH"
    HEDGE = "HEDGE"
    REDUCE = "REDUCE"
    EXIT = "EXIT"


@dataclass
class Risk:
    """An identified portfolio risk."""
    risk_id: str
    affected_holdings: List[str]
    risk_type: RiskType
    severity: RiskSeverity
    time_horizon: str  # IMMEDIATE, SHORT, MEDIUM
    title: str
    description: str
    historical_pattern: str
    leading_indicators: List[str]
    recommended_action: RecommendedAction
    confidence: str  # HIGH, MEDIUM, LOW

    def to_dict(self) -> dict:
        return {
            "risk_id": self.risk_id,
            "affected_holdings": self.affected_holdings,
            "risk_type": self.risk_type.value,
            "severity": self.severity.value,
            "time_horizon": self.time_horizon,
            "title": self.title,
            "description": self.description,
            "historical_pattern": self.historical_pattern,
            "leading_indicators": self.leading_indicators,
            "recommended_action": self.recommended_action.value,
            "confidence": self.confidence
        }


@dataclass
class RiskAssessment:
    """Complete risk assessment for a portfolio."""
    timestamp: datetime
    market_regime: str  # RISK_ON, RISK_OFF, UNCERTAIN
    overall_risk: str  # LOW, MODERATE, ELEVATED, HIGH
    risks: List[Risk]
    safe_holdings: List[str]
    summary: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "market_regime": self.market_regime,
            "overall_risk": self.overall_risk,
            "risks": [r.to_dict() for r in self.risks],
            "safe_holdings": self.safe_holdings,
            "summary": self.summary
        }


RISK_ANALYSIS_PROMPT = """You are a PREDICTIVE RISK ANALYST monitoring a portfolio for early warning signs.
Your job is to identify POTENTIAL RISKS BEFORE they impact prices - not react to drops that already happened.

## YOUR PORTFOLIO
{portfolio_summary}

## CURRENT NEWS & EVENTS
{news_summary}

## UPCOMING SCHEDULED EVENTS
{calendar_summary}

## CORRELATED ASSET ANALYSIS
{correlation_summary}

## YOUR TASK
Analyze the current market environment and identify POTENTIAL RISKS to the portfolio.
Focus on LEADING INDICATORS - events or patterns that historically precede price declines.

CRITICAL GUIDELINES:
1. **Be Predictive, Not Reactive**: Focus on what COULD happen, not what already happened
2. **Use Historical Patterns**: Leverage your knowledge of how similar situations played out
3. **Watch Correlations**: Dollar strength → precious metals weakness, rising rates → growth stock pressure
4. **Flag Streaks/Extremes**: 5+ consecutive up days, at 52-week highs = reversal risk
5. **Policy Shifts Matter**: Fed personnel changes, hawkish pivots affect asset classes
6. **Divergences Are Signals**: If correlated assets diverge, one will likely catch up

For each risk identified, provide:
1. **Affected Holdings**: Which tickers in the portfolio
2. **Risk Type**: MACRO / SECTOR / COMPANY / TECHNICAL / CORRELATION
3. **Severity**: CRITICAL (act now) / HIGH (act soon) / MEDIUM (watch closely) / LOW (be aware)
4. **Time Horizon**: IMMEDIATE (today-tomorrow) / SHORT (1-5 days) / MEDIUM (1-2 weeks)
5. **Historical Pattern**: What similar situations in history led to
6. **Leading Indicators**: Specific signals suggesting this risk
7. **Recommended Action**: WATCH / HEDGE / REDUCE / EXIT

Output VALID JSON only (no markdown code blocks):
{{
  "analysis_timestamp": "{timestamp}",
  "market_regime": "RISK_ON or RISK_OFF or UNCERTAIN",
  "overall_portfolio_risk": "LOW or MODERATE or ELEVATED or HIGH",
  "risks": [
    {{
      "risk_id": "risk_001",
      "affected_holdings": ["SLV", "GLD"],
      "risk_type": "MACRO",
      "severity": "HIGH",
      "time_horizon": "SHORT",
      "title": "Short descriptive title",
      "description": "Detailed description of the risk",
      "historical_pattern": "What happened in similar past situations",
      "leading_indicators": ["Indicator 1", "Indicator 2"],
      "recommended_action": "REDUCE",
      "confidence": "MEDIUM"
    }}
  ],
  "safe_holdings": ["QQQ"],
  "summary": "One sentence overall assessment"
}}

IMPORTANT: If no significant risks are found, return an empty risks array.
Do not manufacture risks - only report genuine concerns based on the data provided.
"""


class RiskAnalyzer:
    """LLM-powered risk analysis engine."""

    def __init__(self, model: str = "claude-4-sonnet"):
        self.model = model
        self.news_scanner = NewsScanner()
        self.event_calendar = EventCalendar()
        self.correlation_tracker = CorrelationTracker()

    def analyze(self, portfolio: Portfolio, use_perplexity: bool = True) -> RiskAssessment:
        """
        Perform comprehensive risk analysis on portfolio.

        Args:
            portfolio: User's portfolio
            use_perplexity: Whether to use Perplexity for enhanced search

        Returns:
            Complete risk assessment
        """
        # Gather data from all monitors
        print("  Scanning news...")
        portfolio_news = self.news_scanner.scan_portfolio_news(portfolio)
        macro_news = self.news_scanner.scan_macro_events()
        all_news = portfolio_news + macro_news

        # Enhanced search with Perplexity
        perplexity_intel = ""
        if use_perplexity:
            print("  Getting real-time intelligence (Perplexity)...")
            plex_result = self.news_scanner.scan_with_perplexity(portfolio)
            if plex_result:
                perplexity_intel = f"\n\n## REAL-TIME MARKET INTELLIGENCE\n{plex_result}"

        print("  Checking economic calendar...")
        events = self.event_calendar.get_upcoming_events(days_ahead=7)
        events = self.event_calendar.match_events_to_holdings(events, portfolio)

        print("  Analyzing correlations...")
        all_tickers = list(portfolio.get_all_watch_tickers())
        correlation_summary = self.correlation_tracker.format_for_llm(all_tickers)

        # Format for LLM
        portfolio_summary = portfolio.format_for_llm()
        news_summary = self.news_scanner.format_news_for_llm(all_news) + perplexity_intel
        calendar_summary = self.event_calendar.format_events_for_llm(events)

        # Build prompt
        prompt = RISK_ANALYSIS_PROMPT.format(
            portfolio_summary=portfolio_summary,
            news_summary=news_summary,
            calendar_summary=calendar_summary,
            correlation_summary=correlation_summary,
            timestamp=datetime.now().isoformat()
        )

        # Call LLM
        print(f"  Analyzing with {self.model}...")
        llm = get_llm(self.model, temperature=0.3, max_tokens=3000)
        response = llm.invoke(prompt)

        # Parse response
        return self._parse_response(response)

    def _parse_response(self, response) -> RiskAssessment:
        """Parse LLM response into RiskAssessment."""
        try:
            content = response.content if hasattr(response, 'content') else str(response)
            content = content.strip()

            # Clean up markdown if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]

            result = json.loads(content)

            # Parse risks
            risks = []
            for r in result.get('risks', []):
                try:
                    risks.append(Risk(
                        risk_id=r.get('risk_id', f"risk_{len(risks)+1}"),
                        affected_holdings=r.get('affected_holdings', []),
                        risk_type=RiskType(r.get('risk_type', 'MACRO')),
                        severity=RiskSeverity(r.get('severity', 'MEDIUM')),
                        time_horizon=r.get('time_horizon', 'SHORT'),
                        title=r.get('title', 'Unknown Risk'),
                        description=r.get('description', ''),
                        historical_pattern=r.get('historical_pattern', ''),
                        leading_indicators=r.get('leading_indicators', []),
                        recommended_action=RecommendedAction(r.get('recommended_action', 'WATCH')),
                        confidence=r.get('confidence', 'MEDIUM')
                    ))
                except Exception as e:
                    print(f"  Warning: Could not parse risk: {e}")

            return RiskAssessment(
                timestamp=datetime.now(),
                market_regime=result.get('market_regime', 'UNCERTAIN'),
                overall_risk=result.get('overall_portfolio_risk', 'MODERATE'),
                risks=risks,
                safe_holdings=result.get('safe_holdings', []),
                summary=result.get('summary', 'Analysis complete.')
            )

        except json.JSONDecodeError as e:
            print(f"  Warning: Could not parse LLM response: {e}")
            return RiskAssessment(
                timestamp=datetime.now(),
                market_regime="UNCERTAIN",
                overall_risk="UNKNOWN",
                risks=[],
                safe_holdings=[],
                summary="Analysis could not be completed - parsing error."
            )

    def format_report(self, assessment: RiskAssessment, portfolio: Portfolio) -> str:
        """Format risk assessment as a readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append("PORT-TRACKER RISK ANALYSIS")
        lines.append(f"Timestamp: {assessment.timestamp.strftime('%Y-%m-%d %H:%M:%S ET')}")
        lines.append(f"Portfolio: {len(portfolio.holdings)} holdings ({', '.join(portfolio.get_tickers())})")
        lines.append("=" * 70)

        # Overall assessment
        risk_emoji = {
            "LOW": "[OK]",
            "MODERATE": "[~]",
            "ELEVATED": "[!]",
            "HIGH": "[!!]"
        }
        risk_icon = risk_emoji.get(assessment.overall_risk, "[?]")

        lines.append(f"\nOVERALL RISK LEVEL: {risk_icon} {assessment.overall_risk}")
        lines.append(f"Market Regime: {assessment.market_regime}")
        lines.append(f"Summary: {assessment.summary}")

        # Critical and High risks
        critical_risks = [r for r in assessment.risks if r.severity in [RiskSeverity.CRITICAL, RiskSeverity.HIGH]]
        if critical_risks:
            lines.append("\n" + "-" * 70)
            lines.append("## CRITICAL / HIGH PRIORITY ALERTS")
            lines.append("-" * 70)

            for risk in critical_risks:
                severity_icon = "[!!!]" if risk.severity == RiskSeverity.CRITICAL else "[!!]"
                lines.append(f"\n{severity_icon} {risk.title}")
                lines.append(f"Affected: {', '.join(risk.affected_holdings)}")
                lines.append(f"Time Horizon: {risk.time_horizon}")
                lines.append("")
                lines.append(f"{risk.description}")
                lines.append("")
                lines.append("Leading Indicators Detected:")
                for indicator in risk.leading_indicators:
                    lines.append(f"  - {indicator}")
                lines.append("")
                lines.append(f"Historical Pattern:")
                lines.append(f"  {risk.historical_pattern}")
                lines.append("")
                lines.append(f"Recommended Action: {risk.recommended_action.value}")
                lines.append(f"Confidence: {risk.confidence}")

        # Medium and Low risks
        other_risks = [r for r in assessment.risks if r.severity in [RiskSeverity.MEDIUM, RiskSeverity.LOW]]
        if other_risks:
            lines.append("\n" + "-" * 70)
            lines.append("## WATCH LIST")
            lines.append("-" * 70)

            for risk in other_risks:
                severity_icon = "[!]" if risk.severity == RiskSeverity.MEDIUM else "[i]"
                lines.append(f"\n{severity_icon} {risk.title}")
                lines.append(f"    Affected: {', '.join(risk.affected_holdings)}")
                lines.append(f"    {risk.description[:200]}...")
                lines.append(f"    Action: {risk.recommended_action.value}")

        # Safe holdings
        if assessment.safe_holdings:
            lines.append("\n" + "-" * 70)
            lines.append("## SAFE HOLDINGS (No immediate risks detected)")
            lines.append("-" * 70)
            for ticker in assessment.safe_holdings:
                lines.append(f"  - {ticker}")

        # No risks found
        if not assessment.risks:
            lines.append("\n" + "-" * 70)
            lines.append("## NO SIGNIFICANT RISKS DETECTED")
            lines.append("-" * 70)
            lines.append("Your portfolio appears stable. Continue monitoring.")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)
