"""
Event Calendar Monitor for Port-Tracker.
Tracks scheduled economic events that could impact holdings.
"""
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

from src.portfolio.holdings import Portfolio


class EventType(Enum):
    """Types of market events."""
    FED = "fed"              # FOMC, rate decisions, Fed speeches
    ECONOMIC = "economic"    # CPI, jobs, GDP, retail sales
    EARNINGS = "earnings"    # Company earnings reports
    GEOPOLITICAL = "geopolitical"  # Elections, trade, sanctions
    OTHER = "other"


class EventImpact(Enum):
    """Expected impact level of event."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ScheduledEvent:
    """A scheduled market event."""
    name: str
    event_type: EventType
    date: datetime
    impact: EventImpact
    description: str = ""
    affected_sectors: List[str] = field(default_factory=list)
    affected_tickers: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "event_type": self.event_type.value,
            "date": self.date.isoformat(),
            "impact": self.impact.value,
            "description": self.description,
            "affected_sectors": self.affected_sectors,
            "affected_tickers": self.affected_tickers
        }


# Known recurring events and their typical impacts
KNOWN_EVENTS = {
    "FOMC": {
        "type": EventType.FED,
        "impact": EventImpact.HIGH,
        "affected_sectors": ["precious_metals", "tech", "financials"],
        "description": "Federal Reserve interest rate decision"
    },
    "CPI": {
        "type": EventType.ECONOMIC,
        "impact": EventImpact.HIGH,
        "affected_sectors": ["precious_metals", "tech"],
        "description": "Consumer Price Index inflation data"
    },
    "NFP": {
        "type": EventType.ECONOMIC,
        "impact": EventImpact.HIGH,
        "affected_sectors": ["all"],
        "description": "Non-Farm Payrolls jobs report"
    },
    "GDP": {
        "type": EventType.ECONOMIC,
        "impact": EventImpact.MEDIUM,
        "affected_sectors": ["all"],
        "description": "Gross Domestic Product report"
    },
    "Fed Chair Speech": {
        "type": EventType.FED,
        "impact": EventImpact.MEDIUM,
        "affected_sectors": ["precious_metals", "tech", "financials"],
        "description": "Federal Reserve Chair public remarks"
    },
    "Retail Sales": {
        "type": EventType.ECONOMIC,
        "impact": EventImpact.MEDIUM,
        "affected_sectors": ["consumer", "tech"],
        "description": "Monthly retail sales data"
    }
}


class EventCalendar:
    """Monitors economic calendar for upcoming events."""

    def __init__(self):
        if TAVILY_AVAILABLE:
            api_key = os.getenv('TAVILY_API_KEY')
            if api_key:
                self.tavily = TavilyClient(api_key=api_key)
            else:
                self.tavily = None
        else:
            self.tavily = None

    def get_upcoming_events(self, days_ahead: int = 7) -> List[ScheduledEvent]:
        """
        Get upcoming economic events.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of scheduled events
        """
        events = []

        # Search for economic calendar events
        if self.tavily:
            events.extend(self._search_economic_calendar(days_ahead))

        # Search for Fed events specifically
        if self.tavily:
            events.extend(self._search_fed_events(days_ahead))

        # Deduplicate by name
        seen = set()
        unique_events = []
        for event in events:
            if event.name.lower() not in seen:
                seen.add(event.name.lower())
                unique_events.append(event)

        # Sort by date
        unique_events.sort(key=lambda e: e.date)

        return unique_events

    def _search_economic_calendar(self, days_ahead: int) -> List[ScheduledEvent]:
        """Search for economic calendar events."""
        if not self.tavily:
            return []

        today = datetime.now().strftime('%B %d, %Y')
        query = f"""
        US economic calendar this week {today}:
        - FOMC Federal Reserve meeting
        - CPI inflation report
        - Jobs report NFP
        - GDP data
        - Fed Chair Powell speech
        - Retail sales
        """

        try:
            result = self.tavily.search(query, max_results=5)
            events = []

            for r in result.get('results', []):
                title = r.get('title', '')
                content = r.get('content', '')

                # Try to identify event type from content
                event = self._parse_event_from_search(title, content)
                if event:
                    events.append(event)

            return events
        except Exception as e:
            print(f"Calendar search error: {e}")
            return []

    def _search_fed_events(self, days_ahead: int) -> List[ScheduledEvent]:
        """Search specifically for Fed-related events."""
        if not self.tavily:
            return []

        query = "Federal Reserve FOMC meeting schedule Powell speech this week"

        try:
            result = self.tavily.search(query, max_results=3)
            events = []

            for r in result.get('results', []):
                content = r.get('content', '')

                if any(kw in content.lower() for kw in ['fomc', 'federal reserve', 'powell', 'rate decision']):
                    events.append(ScheduledEvent(
                        name="Fed Event Detected",
                        event_type=EventType.FED,
                        date=datetime.now() + timedelta(days=1),  # Approximate
                        impact=EventImpact.HIGH,
                        description=content[:200],
                        affected_sectors=["precious_metals", "tech", "financials"]
                    ))

            return events
        except Exception as e:
            print(f"Fed events search error: {e}")
            return []

    def _parse_event_from_search(self, title: str, content: str) -> Optional[ScheduledEvent]:
        """Parse a scheduled event from search result."""
        combined = (title + " " + content).lower()

        for event_name, event_info in KNOWN_EVENTS.items():
            keywords = event_name.lower().split()
            if all(kw in combined for kw in keywords):
                return ScheduledEvent(
                    name=event_name,
                    event_type=event_info["type"],
                    date=datetime.now() + timedelta(days=1),  # Approximate
                    impact=event_info["impact"],
                    description=event_info["description"],
                    affected_sectors=event_info["affected_sectors"]
                )

        return None

    def match_events_to_holdings(
        self,
        events: List[ScheduledEvent],
        portfolio: Portfolio
    ) -> List[ScheduledEvent]:
        """
        Match events to portfolio holdings.

        Args:
            events: List of upcoming events
            portfolio: User's portfolio

        Returns:
            Events with affected_tickers populated
        """
        portfolio_sectors = portfolio.get_sectors()

        for event in events:
            affected_tickers = []

            # Match by sector
            for sector in event.affected_sectors:
                if sector == "all":
                    affected_tickers.extend(portfolio.get_tickers())
                elif sector in portfolio_sectors:
                    for h in portfolio.get_holdings_by_sector(sector):
                        affected_tickers.append(h.ticker)

            event.affected_tickers = list(set(affected_tickers))

        return events

    def format_events_for_llm(self, events: List[ScheduledEvent]) -> str:
        """Format events for LLM prompt."""
        if not events:
            return "No significant upcoming events detected."

        lines = []
        lines.append(f"## Upcoming Events ({len(events)} found)")
        lines.append("")

        for event in events[:5]:  # Limit to 5
            impact_icon = {"high": "[!!!]", "medium": "[!!]", "low": "[!]"}
            icon = impact_icon.get(event.impact.value, "[?]")

            lines.append(f"### {icon} {event.name}")
            lines.append(f"    Type: {event.event_type.value.upper()}")
            lines.append(f"    Impact: {event.impact.value.upper()}")
            lines.append(f"    Sectors: {', '.join(event.affected_sectors)}")
            if event.affected_tickers:
                lines.append(f"    Your Holdings: {', '.join(event.affected_tickers)}")
            lines.append(f"    {event.description}")
            lines.append("")

        return "\n".join(lines)
