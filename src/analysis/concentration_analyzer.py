"""
Sector Concentration Analyzer for Port-Tracker.
Analyzes portfolio diversification and flags concentration risks.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.portfolio.holdings import Portfolio, Holding


@dataclass
class SectorConcentration:
    """Concentration data for a single sector."""
    sector: str
    holdings: List[str]  # tickers in this sector
    weight_pct: float  # % of total portfolio
    value: float  # dollar value
    risk_level: str  # LOW/MEDIUM/HIGH/CRITICAL based on thresholds


@dataclass
class ConcentrationAnalysis:
    """Complete concentration analysis results."""
    total_value: float
    sector_breakdown: List[SectorConcentration]
    concentration_risks: List[str]  # warning messages
    herfindahl_index: float  # diversification metric (0-1, lower=more diverse)


class ConcentrationAnalyzer:
    """Analyzes portfolio sector concentration and diversification."""

    # Concentration risk thresholds
    THRESHOLDS = {
        "LOW": 20.0,      # < 20%
        "MEDIUM": 35.0,   # 20-35%
        "HIGH": 50.0,     # 35-50%
        "CRITICAL": 100.0  # > 50%
    }

    def __init__(self):
        pass

    def analyze(self, portfolio: Portfolio) -> ConcentrationAnalysis:
        """
        Analyze portfolio concentration using cost basis valuation.

        Args:
            portfolio: Portfolio to analyze

        Returns:
            ConcentrationAnalysis with sector breakdown and risks
        """
        # Calculate holding values and total
        holding_values = self._calculate_holding_values(portfolio)
        total_value = sum(holding_values.values())

        if total_value == 0:
            return ConcentrationAnalysis(
                total_value=0,
                sector_breakdown=[],
                concentration_risks=["Unable to calculate concentration - no cost basis data"],
                herfindahl_index=0
            )

        # Group by sector
        sector_weights = self._get_sector_weights(portfolio, holding_values, total_value)

        # Build sector breakdown
        sector_breakdown = []
        for sector, data in sorted(sector_weights.items(), key=lambda x: x[1]["weight"], reverse=True):
            risk_level = self._get_risk_level(data["weight"])
            sector_breakdown.append(SectorConcentration(
                sector=sector,
                holdings=data["holdings"],
                weight_pct=data["weight"],
                value=data["value"],
                risk_level=risk_level
            ))

        # Calculate Herfindahl Index
        hhi = self._calculate_herfindahl_index(
            {s: d["weight"] for s, d in sector_weights.items()}
        )

        # Generate concentration warnings
        concentration_risks = self._generate_warnings(sector_breakdown, hhi)

        return ConcentrationAnalysis(
            total_value=total_value,
            sector_breakdown=sector_breakdown,
            concentration_risks=concentration_risks,
            herfindahl_index=hhi
        )

    def _calculate_holding_values(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate cost basis value for each holding."""
        values = {}
        for holding in portfolio.holdings:
            if holding.quantity is not None and holding.avg_price is not None:
                values[holding.ticker] = holding.quantity * holding.avg_price
            else:
                values[holding.ticker] = 0
        return values

    def _get_sector_weights(
        self,
        portfolio: Portfolio,
        holding_values: Dict[str, float],
        total_value: float
    ) -> Dict[str, Dict]:
        """Calculate sector weights from holding values."""
        sector_data: Dict[str, Dict] = {}

        for holding in portfolio.holdings:
            sector = holding.sector
            value = holding_values.get(holding.ticker, 0)

            if sector not in sector_data:
                sector_data[sector] = {"holdings": [], "value": 0, "weight": 0}

            sector_data[sector]["holdings"].append(holding.ticker)
            sector_data[sector]["value"] += value

        # Calculate weights
        for sector in sector_data:
            sector_data[sector]["weight"] = (
                sector_data[sector]["value"] / total_value * 100
                if total_value > 0 else 0
            )

        return sector_data

    def _get_risk_level(self, weight_pct: float) -> str:
        """Determine risk level based on concentration weight."""
        if weight_pct < self.THRESHOLDS["LOW"]:
            return "LOW"
        elif weight_pct < self.THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        elif weight_pct < self.THRESHOLDS["HIGH"]:
            return "HIGH"
        else:
            return "CRITICAL"

    def _calculate_herfindahl_index(self, weights: Dict[str, float]) -> float:
        """
        Calculate Herfindahl-Hirschman Index for diversification.

        HHI = sum of squared market shares (as decimals)
        Range: 0 to 1 (lower = more diversified)
        - < 0.15: Highly diversified
        - 0.15-0.25: Moderately diversified
        - > 0.25: Concentrated
        """
        return sum((w / 100) ** 2 for w in weights.values())

    def _generate_warnings(
        self,
        sector_breakdown: List[SectorConcentration],
        hhi: float
    ) -> List[str]:
        """Generate concentration risk warnings."""
        warnings = []

        # Check individual sector concentrations
        for sector in sector_breakdown:
            if sector.risk_level == "CRITICAL":
                warnings.append(
                    f"CRITICAL: {sector.sector} is {sector.weight_pct:.1f}% of portfolio - "
                    f"severely over-concentrated"
                )
            elif sector.risk_level == "HIGH":
                warnings.append(
                    f"HIGH: {sector.sector} at {sector.weight_pct:.1f}% - "
                    f"consider reducing exposure"
                )
            elif sector.risk_level == "MEDIUM":
                warnings.append(
                    f"MEDIUM: {sector.sector} at {sector.weight_pct:.1f}% - "
                    f"monitor concentration"
                )

        # Check overall diversification via HHI
        if hhi > 0.25:
            warnings.append(
                f"Portfolio is concentrated (HHI: {hhi:.3f}) - "
                f"consider diversifying across more sectors"
            )
        elif hhi > 0.15:
            warnings.append(
                f"Portfolio is moderately diversified (HHI: {hhi:.3f})"
            )

        return warnings

    def format_report(self, analysis: ConcentrationAnalysis) -> str:
        """Format concentration analysis as a readable report."""
        lines = []
        lines.append("-" * 70)
        lines.append("SECTOR CONCENTRATION ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"Total Portfolio Value (Cost Basis): ${analysis.total_value:,.2f}")

        # HHI interpretation
        if analysis.herfindahl_index < 0.15:
            hhi_desc = "(Highly Diversified)"
        elif analysis.herfindahl_index < 0.25:
            hhi_desc = "(Moderately Diversified)"
        else:
            hhi_desc = "(Concentrated)"
        lines.append(f"Herfindahl Index: {analysis.herfindahl_index:.3f} {hhi_desc}")

        lines.append("")
        lines.append("Sector Breakdown:")
        lines.append("-" * 50)

        # Display sectors sorted by weight
        for sector in analysis.sector_breakdown:
            risk_icon = {
                "LOW": "[OK]",
                "MEDIUM": "[~]",
                "HIGH": "[!]",
                "CRITICAL": "[!!]"
            }.get(sector.risk_level, "[?]")

            # Create bar chart
            bar_length = int(sector.weight_pct / 2)  # Scale: 2% per char
            bar = "#" * bar_length

            lines.append(
                f"  {risk_icon} {sector.sector:<22} "
                f"{sector.weight_pct:>5.1f}% "
                f"${sector.value:>12,.2f}  {bar}"
            )
            lines.append(f"       Holdings: {', '.join(sector.holdings)}")

        # Concentration warnings
        if analysis.concentration_risks:
            lines.append("")
            lines.append("Concentration Warnings:")
            lines.append("-" * 50)
            for warning in analysis.concentration_risks:
                lines.append(f"  - {warning}")

        lines.append("-" * 70)

        return "\n".join(lines)
