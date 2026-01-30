"""
Perplexity Search Integration for Port-Tracker.
Provides real-time news and market information using Perplexity API.
"""
import os
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PerplexityResult:
    """A search result from Perplexity."""
    content: str
    citations: List[str]
    query: str
    timestamp: datetime


class PerplexitySearch:
    """Client for Perplexity API searches."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"

        if not self.api_key:
            print("Warning: PERPLEXITY_API_KEY not set. Perplexity search disabled.")

    def search(self, query: str, focus: str = "news") -> Optional[PerplexityResult]:
        """
        Execute a search query using Perplexity.

        Args:
            query: Search query
            focus: Search focus - "news" for recent news, "general" for broader search

        Returns:
            PerplexityResult or None if failed
        """
        if not self.api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Use sonar model for search
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial news analyst. Provide concise, factual summaries of market news and events. Focus on information that could impact investment portfolios."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "search_context_size": "high",
            "return_citations": True
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = data.get('citations', [])

            return PerplexityResult(
                content=content,
                citations=citations,
                query=query,
                timestamp=datetime.now()
            )

        except requests.exceptions.RequestException as e:
            print(f"Perplexity search error: {e}")
            return None

    def search_portfolio_risks(self, tickers: List[str], sectors: List[str]) -> Optional[PerplexityResult]:
        """
        Search for risks related to portfolio holdings.

        Args:
            tickers: List of ticker symbols
            sectors: List of sectors in portfolio

        Returns:
            PerplexityResult with risk analysis
        """
        ticker_str = ", ".join(tickers[:5])  # Limit to 5 tickers
        sector_str = ", ".join(sectors)

        query = f"""
        What are the current market risks and news that could negatively impact these investments?

        Holdings: {ticker_str}
        Sectors: {sector_str}

        Focus on:
        1. Breaking news that could affect these assets
        2. Upcoming economic events (Fed, CPI, earnings)
        3. Technical patterns suggesting potential reversals
        4. Sector-specific risks
        5. Macro risks (dollar, rates, geopolitical)

        Provide specific, actionable risk warnings with time horizons.
        """

        return self.search(query)

    def search_macro_risks(self) -> Optional[PerplexityResult]:
        """Search for current macro-economic risks."""
        query = """
        What are the top 5 macro-economic risks for US equity and precious metals investors today?

        Consider:
        - Federal Reserve policy and rate expectations
        - US Dollar strength/weakness
        - Inflation data and expectations
        - Geopolitical tensions
        - Market sentiment extremes

        For each risk, indicate:
        - Severity (Critical/High/Medium/Low)
        - Time horizon (Immediate/Short-term/Medium-term)
        - Which assets are most affected
        """

        return self.search(query)

    def search_asset_news(self, ticker: str, asset_name: str) -> Optional[PerplexityResult]:
        """Search for news about a specific asset."""
        query = f"""
        What is the latest news and analysis for {ticker} ({asset_name})?

        Include:
        - Price action and trend
        - Recent news affecting the asset
        - Analyst opinions and price targets
        - Upcoming catalysts or risks
        - Technical levels to watch
        """

        return self.search(query)

    def search_correlation_risks(self, asset1: str, asset2: str) -> Optional[PerplexityResult]:
        """Search for correlation analysis between two assets."""
        query = f"""
        Analyze the current correlation between {asset1} and {asset2}.

        - Are they currently diverging from historical correlation?
        - What could explain any divergence?
        - Which asset is likely to "catch up" to the other?
        - What does this divergence signal for investors?
        """

        return self.search(query)

    def format_for_llm(self, result: PerplexityResult) -> str:
        """Format Perplexity result for LLM prompt."""
        if not result:
            return "No Perplexity search results available."

        lines = []
        lines.append("## Real-Time Market Intelligence (via Perplexity)")
        lines.append(f"Query: {result.query}")
        lines.append(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append(result.content)

        if result.citations:
            lines.append("")
            lines.append("### Sources:")
            for i, cite in enumerate(result.citations[:5], 1):
                lines.append(f"  {i}. {cite}")

        return "\n".join(lines)
