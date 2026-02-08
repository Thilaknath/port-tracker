#!/usr/bin/env python3
"""
Port-Tracker: One-Time Portfolio Risk Check

Usage:
    python run_check.py              # Check all holdings
    python run_check.py --verbose    # Detailed output
    python run_check.py --tickers SLV,GLD  # Check specific tickers
"""
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.portfolio.holdings import load_portfolio, Portfolio, Holding
from src.analysis.risk_analyzer import RiskAnalyzer
from src.analysis.concentration_analyzer import ConcentrationAnalyzer
from src.alerts.notifier import AlertNotifier, create_alerts_from_assessment


def main():
    parser = argparse.ArgumentParser(description='Port-Tracker: Portfolio Risk Check')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--tickers', '-t', type=str, help='Comma-separated tickers to check')
    parser.add_argument('--portfolio', '-p', type=str, default='data/portfolio.json',
                       help='Path to portfolio JSON file')
    parser.add_argument('--model', '-m', type=str, default='claude-4-sonnet',
                       help='LLM model to use (default: claude-4-sonnet)')
    parser.add_argument('--save', '-s', action='store_true', help='Save alerts to file')

    args = parser.parse_args()

    print("=" * 70)
    print("PORT-TRACKER: Predictive Portfolio Risk Monitor")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("=" * 70)

    # Load portfolio
    try:
        portfolio = load_portfolio(args.portfolio)
        print(f"\nLoaded portfolio: {portfolio.name}")
        print(f"Holdings: {', '.join(portfolio.get_tickers())}")
    except FileNotFoundError:
        print(f"\nError: Portfolio file not found at {args.portfolio}")
        print("Create a portfolio.json file in data/ directory.")
        return 1

    # Filter to specific tickers if requested
    if args.tickers:
        ticker_list = [t.strip().upper() for t in args.tickers.split(',')]
        filtered_holdings = [h for h in portfolio.holdings if h.ticker in ticker_list]
        if not filtered_holdings:
            print(f"\nError: None of the specified tickers found in portfolio")
            return 1
        portfolio = Portfolio(holdings=filtered_holdings, name=portfolio.name)
        print(f"Filtering to: {', '.join(portfolio.get_tickers())}")

    # Run analysis
    print(f"\n[1/4] Gathering market data...")
    analyzer = RiskAnalyzer(model=args.model)

    print(f"[2/4] Analyzing risks with {args.model}...")
    assessment = analyzer.analyze(portfolio)

    print(f"[3/4] Analyzing sector concentration...")
    conc_analyzer = ConcentrationAnalyzer()
    conc_analysis = conc_analyzer.analyze(portfolio)

    print(f"[4/4] Generating report...")

    # Format and display report
    report = analyzer.format_report(assessment, portfolio)
    print(report)

    # Display concentration analysis
    conc_report = conc_analyzer.format_report(conc_analysis)
    print(conc_report)

    # Create alerts
    notifier = AlertNotifier()
    create_alerts_from_assessment(assessment, notifier)

    # Show alert summary
    print(f"\n{notifier.format_summary()}")

    # Save alerts if requested
    if args.save:
        filepath = notifier.save_alerts()
        print(f"Alerts saved to: {filepath}")

    # Verbose output
    if args.verbose:
        print("\n" + "-" * 70)
        print("VERBOSE: Raw Assessment Data")
        print("-" * 70)
        import json
        print(json.dumps(assessment.to_dict(), indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
