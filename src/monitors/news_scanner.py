"""
News Scanner for Port-Tracker.
Monitors news for portfolio-relevant events using Tavily/Perplexity.
"""
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv

load_dotenv()

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

from src.portfolio.holdings import Portfolio
from src.monitors.perplexity_search import PerplexitySearch


@dataclass
class NewsItem:
    """A single news item."""
    title: str
    content: str
    url: str
    source: str
    published: Optional[datetime] = None
    relevance_score: float = 0.0
    affected_tickers: List[str] = field(default_factory=list)
    sentiment: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "published": self.published.isoformat() if self.published else None,
            "relevance_score": self.relevance_score,
            "affected_tickers": self.affected_tickers,
            "sentiment": self.sentiment
        }


class NewsScanner:
    """Scans news sources for portfolio-relevant events."""

    def __init__(self, use_perplexity: bool = True):
        if TAVILY_AVAILABLE:
            api_key = os.getenv('TAVILY_API_KEY')
            if api_key:
                self.tavily = TavilyClient(api_key=api_key)
            else:
                self.tavily = None
        else:
            self.tavily = None

        # Initialize Perplexity for enhanced search
        self.perplexity = PerplexitySearch() if use_perplexity else None
        self._perplexity_cache: Dict[str, str] = {}

    def scan_portfolio_news(self, portfolio: Portfolio) -> List[NewsItem]:
        """
        Scan news for all holdings, correlated assets, and sectors.

        Args:
            portfolio: User's portfolio

        Returns:
            List of relevant news items
        """
        all_news = []

        # 1. Scan direct holdings
        for holding in portfolio.holdings:
            news = self._search_ticker_news(holding.ticker, holding.name)
            for item in news:
                item.affected_tickers.append(holding.ticker)
            all_news.extend(news)

        # 2. Scan sectors
        for sector in portfolio.get_sectors():
            news = self._search_sector_news(sector)
            # Map sector news to affected holdings
            for item in news:
                for h in portfolio.get_holdings_by_sector(sector):
                    if h.ticker not in item.affected_tickers:
                        item.affected_tickers.append(h.ticker)
            all_news.extend(news)

        # 3. Scan correlated assets
        correlated = portfolio.get_correlated_tickers()
        for ticker in list(correlated)[:5]:  # Limit to top 5 correlated
            news = self._search_ticker_news(ticker)
            all_news.extend(news)

        # Deduplicate by URL
        seen_urls = set()
        unique_news = []
        for item in all_news:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                unique_news.append(item)

        return unique_news

    def scan_macro_events(self) -> List[NewsItem]:
        """
        Scan for macro economic events.

        Returns:
            List of macro news items
        """
        queries = [
            "Federal Reserve FOMC interest rate decision today",
            "US dollar DXY strength currency markets",
            "Treasury yields bonds market today",
            "inflation CPI economic data",
            "geopolitical risk market impact"
        ]

        all_news = []
        for query in queries:
            news = self._search_general(query, max_results=3)
            all_news.extend(news)

        # Deduplicate
        seen_urls = set()
        unique_news = []
        for item in all_news:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                unique_news.append(item)

        return unique_news

    def scan_risk_factors(self, risk_factors: Set[str]) -> List[NewsItem]:
        """
        Scan news for specific risk factors.

        Args:
            risk_factors: Set of risk factor keywords

        Returns:
            List of relevant news items
        """
        all_news = []

        for factor in risk_factors:
            query = f"{factor} market news today impact"
            news = self._search_general(query, max_results=2)
            for item in news:
                item.relevance_score = 0.8  # High relevance for risk factors
            all_news.extend(news)

        return all_news

    def _search_ticker_news(self, ticker: str, name: str = "") -> List[NewsItem]:
        """Search news for a specific ticker."""
        if not self.tavily:
            return []

        query = f"{ticker} {name} stock ETF news today market"
        return self._search_general(query, max_results=3)

    def _search_sector_news(self, sector: str) -> List[NewsItem]:
        """Search news for a sector."""
        if not self.tavily:
            return []

        sector_queries = {
            "precious_metals": "gold silver precious metals market news today",
            "tech": "technology stocks Nasdaq tech sector news today",
            "energy": "oil energy sector stocks news today",
            "financials": "financial banks stocks sector news today",
            "healthcare": "healthcare pharma biotech stocks news today"
        }

        query = sector_queries.get(sector.lower(), f"{sector} sector stocks news today")
        return self._search_general(query, max_results=3)

    def _search_general(self, query: str, max_results: int = 5) -> List[NewsItem]:
        """Execute a general news search."""
        if not self.tavily:
            return []

        try:
            result = self.tavily.search(query, max_results=max_results)
            news = []

            for r in result.get('results', []):
                news.append(NewsItem(
                    title=r.get('title', ''),
                    content=r.get('content', '')[:500],
                    url=r.get('url', ''),
                    source='tavily',
                    relevance_score=r.get('score', 0.5)
                ))

            return news
        except Exception as e:
            print(f"News search error: {e}")
            return []

    def scan_with_perplexity(self, portfolio: Portfolio) -> Optional[str]:
        """
        Get enhanced risk analysis using Perplexity.

        Args:
            portfolio: User's portfolio

        Returns:
            Formatted Perplexity analysis or None
        """
        if not self.perplexity:
            return None

        tickers = portfolio.get_tickers()
        sectors = list(portfolio.get_sectors())

        result = self.perplexity.search_portfolio_risks(tickers, sectors)
        if result:
            self._perplexity_cache['portfolio'] = self.perplexity.format_for_llm(result)
            return self._perplexity_cache['portfolio']

        return None

    def get_perplexity_macro_risks(self) -> Optional[str]:
        """Get macro risk analysis from Perplexity."""
        if not self.perplexity:
            return None

        result = self.perplexity.search_macro_risks()
        if result:
            self._perplexity_cache['macro'] = self.perplexity.format_for_llm(result)
            return self._perplexity_cache['macro']

        return None

    def format_news_for_llm(self, news_items: List[NewsItem]) -> str:
        """Format news items for LLM prompt."""
        if not news_items:
            return "No relevant news found."

        lines = []
        lines.append(f"## Recent News ({len(news_items)} items)")
        lines.append("")

        for i, item in enumerate(news_items[:10], 1):  # Limit to 10
            tickers = ", ".join(item.affected_tickers) if item.affected_tickers else "General"
            lines.append(f"### {i}. {item.title}")
            lines.append(f"    Tickers: {tickers}")
            lines.append(f"    {item.content[:300]}...")
            lines.append("")

        # Add Perplexity insights if available
        if 'portfolio' in self._perplexity_cache:
            lines.append("\n" + self._perplexity_cache['portfolio'])

        return "\n".join(lines)
