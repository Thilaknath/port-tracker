#!/usr/bin/env python3
"""
Port-Tracker: Continuous Portfolio Monitoring

Usage:
    python run_monitor.py                    # Monitor every 30 minutes
    python run_monitor.py --interval 15      # Monitor every 15 minutes
    python run_monitor.py --market-hours     # Only during market hours (9:30 AM - 4:00 PM ET)
"""
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.portfolio.holdings import load_portfolio
from src.analysis.risk_analyzer import RiskAnalyzer
from src.alerts.notifier import AlertNotifier, AlertLevel, create_alerts_from_assessment


def is_market_hours() -> bool:
    """Check if current time is within US market hours (9:30 AM - 4:00 PM ET)."""
    now = datetime.now()
    # Simple check - assumes local time is ET
    # For production, use pytz for proper timezone handling
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    # Check if weekday (Mon=0, Sun=6)
    if now.weekday() >= 5:
        return False

    return market_open <= now <= market_close


def run_single_check(portfolio, analyzer, notifier, verbose=False):
    """Run a single portfolio check."""
    print(f"\n{'='*70}")
    print(f"PORT-TRACKER CHECK: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"{'='*70}")

    try:
        assessment = analyzer.analyze(portfolio)

        # Create alerts
        notifier.clear_alerts()
        create_alerts_from_assessment(assessment, notifier)

        # Get alert counts
        critical = len([a for a in notifier._alerts if a.level == AlertLevel.CRITICAL])
        warning = len([a for a in notifier._alerts if a.level == AlertLevel.WARNING])

        # Summary
        print(f"\nOverall Risk: {assessment.overall_risk}")
        print(f"Market Regime: {assessment.market_regime}")
        print(f"{notifier.format_summary()}")

        # Show critical/warning alerts
        if critical > 0 or warning > 0:
            print("\n" + "-"*50)
            notifier.notify_console(min_level=AlertLevel.WARNING)

            # Save critical alerts
            if critical > 0:
                filepath = notifier.save_alerts()
                print(f"\nAlerts saved to: {filepath}")

        if verbose:
            report = analyzer.format_report(assessment, portfolio)
            print(report)

        return assessment

    except Exception as e:
        print(f"\nError during check: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Port-Tracker: Continuous Monitoring')
    parser.add_argument('--interval', '-i', type=int, default=30,
                       help='Check interval in minutes (default: 30)')
    parser.add_argument('--market-hours', '-m', action='store_true',
                       help='Only monitor during market hours')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--portfolio', '-p', type=str, default='data/portfolio.json',
                       help='Path to portfolio JSON file')
    parser.add_argument('--model', type=str, default='claude-4-sonnet',
                       help='LLM model to use (default: claude-4-sonnet)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (same as run_check.py)')

    args = parser.parse_args()

    print("=" * 70)
    print("PORT-TRACKER: Continuous Portfolio Risk Monitor")
    print("=" * 70)
    print(f"Interval: {args.interval} minutes")
    print(f"Market hours only: {args.market_hours}")
    print(f"Model: {args.model}")

    # Load portfolio
    try:
        portfolio = load_portfolio(args.portfolio)
        print(f"Portfolio: {portfolio.name}")
        print(f"Holdings: {', '.join(portfolio.get_tickers())}")
    except FileNotFoundError:
        print(f"\nError: Portfolio file not found at {args.portfolio}")
        return 1

    # Initialize
    analyzer = RiskAnalyzer(model=args.model)
    notifier = AlertNotifier()

    # Run once mode
    if args.once:
        run_single_check(portfolio, analyzer, notifier, args.verbose)
        return 0

    # Continuous monitoring
    print(f"\nStarting continuous monitoring...")
    print(f"Press Ctrl+C to stop.\n")

    try:
        while True:
            # Check if we should run
            if args.market_hours and not is_market_hours():
                print(f"[{datetime.now().strftime('%H:%M')}] Outside market hours. Sleeping...")
                time.sleep(60 * 5)  # Check every 5 minutes if market is open
                continue

            # Run check
            run_single_check(portfolio, analyzer, notifier, args.verbose)

            # Wait for next interval
            next_check = datetime.now().strftime('%H:%M')
            print(f"\nNext check in {args.interval} minutes...")
            time.sleep(60 * args.interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
