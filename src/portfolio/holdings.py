"""
Portfolio Holdings Management for Port-Tracker.
Defines holdings, portfolios, and loading from JSON configuration.
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from pathlib import Path


@dataclass
class Holding:
    """A single holding in the portfolio."""
    ticker: str
    name: str
    asset_type: str  # stock, etf, commodity, crypto
    sector: str  # tech, precious_metals, energy, healthcare, etc.
    correlated_assets: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    quantity: Optional[float] = None
    avg_price: Optional[float] = None

    def __post_init__(self):
        self.ticker = self.ticker.upper()
        self.correlated_assets = [t.upper() for t in self.correlated_assets]

    def to_dict(self) -> dict:
        result = {
            "ticker": self.ticker,
            "name": self.name,
            "asset_type": self.asset_type,
            "sector": self.sector,
            "correlated_assets": self.correlated_assets,
            "risk_factors": self.risk_factors
        }
        if self.quantity is not None:
            result["quantity"] = self.quantity
        if self.avg_price is not None:
            result["avg_price"] = self.avg_price
        return result


@dataclass
class Portfolio:
    """User's portfolio of holdings."""
    holdings: List[Holding]
    name: str = "My Portfolio"

    def get_tickers(self) -> List[str]:
        """Get all ticker symbols in the portfolio."""
        return [h.ticker for h in self.holdings]

    def get_sectors(self) -> Set[str]:
        """Get unique sectors represented in the portfolio."""
        return {h.sector for h in self.holdings}

    def get_correlated_tickers(self) -> Set[str]:
        """Get all correlated tickers to monitor (excluding held tickers)."""
        held = set(self.get_tickers())
        correlated = set()
        for h in self.holdings:
            correlated.update(h.correlated_assets)
        return correlated - held

    def get_all_watch_tickers(self) -> Set[str]:
        """Get all tickers to watch (held + correlated)."""
        tickers = set(self.get_tickers())
        for h in self.holdings:
            tickers.update(h.correlated_assets)
        return tickers

    def get_all_risk_factors(self) -> Set[str]:
        """Get all risk factors across holdings."""
        factors = set()
        for h in self.holdings:
            factors.update(h.risk_factors)
        return factors

    def get_holdings_by_sector(self, sector: str) -> List[Holding]:
        """Get all holdings in a specific sector."""
        return [h for h in self.holdings if h.sector.lower() == sector.lower()]

    def get_holding(self, ticker: str) -> Optional[Holding]:
        """Get a specific holding by ticker."""
        ticker = ticker.upper()
        for h in self.holdings:
            if h.ticker == ticker:
                return h
        return None

    def format_for_llm(self) -> str:
        """Format portfolio summary for LLM prompt."""
        lines = []
        lines.append(f"Portfolio: {self.name}")
        lines.append(f"Total Holdings: {len(self.holdings)}")
        lines.append(f"Sectors: {', '.join(self.get_sectors())}")
        lines.append("")

        for h in self.holdings:
            lines.append(f"### {h.ticker} - {h.name}")
            lines.append(f"    Type: {h.asset_type} | Sector: {h.sector}")
            lines.append(f"    Correlated: {', '.join(h.correlated_assets) if h.correlated_assets else 'None'}")
            lines.append(f"    Risk Factors: {', '.join(h.risk_factors) if h.risk_factors else 'None'}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "holdings": [h.to_dict() for h in self.holdings]
        }


def load_portfolio(filepath: str = "data/portfolio.json") -> Portfolio:
    """
    Load portfolio from JSON file.

    Args:
        filepath: Path to portfolio JSON file

    Returns:
        Portfolio object
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Portfolio file not found: {filepath}")

    with open(path, 'r') as f:
        data = json.load(f)

    holdings = []
    for h in data.get("holdings", []):
        holdings.append(Holding(
            ticker=h["ticker"],
            name=h["name"],
            asset_type=h.get("asset_type", "stock"),
            sector=h.get("sector", "unknown"),
            correlated_assets=h.get("correlated_assets", []),
            risk_factors=h.get("risk_factors", []),
            quantity=h.get("quantity"),
            avg_price=h.get("avg_price")
        ))

    return Portfolio(
        holdings=holdings,
        name=data.get("name", "My Portfolio")
    )


def save_portfolio(portfolio: Portfolio, filepath: str = "data/portfolio.json") -> None:
    """Save portfolio to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(portfolio.to_dict(), f, indent=2)


# Predefined sector correlations for common assets
SECTOR_CORRELATIONS = {
    "precious_metals": {
        "positive": ["GLD", "SLV", "GOLD", "NEM", "PAAS"],
        "negative": ["DXY", "UUP"],  # Dollar strength is negative for metals
        "risk_factors": ["fed policy", "dollar strength", "inflation", "real rates", "geopolitical"]
    },
    "tech": {
        "positive": ["QQQ", "SPY", "AAPL", "MSFT", "NVDA", "META", "GOOGL"],
        "negative": ["TLT"],  # Rising rates (falling bonds) pressure growth
        "risk_factors": ["rate hikes", "earnings", "AI sentiment", "growth stocks", "antitrust"]
    },
    "energy": {
        "positive": ["XLE", "XOP", "USO", "CVX", "XOM"],
        "negative": [],
        "risk_factors": ["oil prices", "OPEC", "geopolitical", "green energy", "demand"]
    },
    "financials": {
        "positive": ["XLF", "JPM", "BAC", "GS"],
        "negative": [],
        "risk_factors": ["interest rates", "yield curve", "credit risk", "regulation"]
    }
}


def get_sector_info(sector: str) -> Dict:
    """Get correlation and risk info for a sector."""
    return SECTOR_CORRELATIONS.get(sector.lower(), {
        "positive": [],
        "negative": [],
        "risk_factors": []
    })
