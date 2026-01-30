# Port-Tracker: Predictive Portfolio Risk Monitor

An LLM-powered early warning system that monitors news, market events, and macro conditions to alert you about **potential risks to your portfolio holdings BEFORE price impacts occur**.

## Key Features

- **Predictive Alerts**: Uses LLM knowledge of historical patterns + real-time news to identify risks before they materialize
- **Multi-Source Intelligence**: Combines Tavily, Perplexity, and economic calendar data
- **Correlation Tracking**: Monitors correlated assets (e.g., dollar strength → precious metals weakness)
- **Pattern Detection**: Identifies streaks, extremes, and divergences that historically precede moves
- **Flexible Monitoring**: One-time checks or continuous monitoring with configurable intervals

## Example Use Case

The January 2026 precious metals crash saw gold drop 8% and silver 17% in a single day. **Port-Tracker would have flagged these warning signs BEFORE the crash:**

- 7 consecutive record highs → historical pullback pattern
- Fed Chair replacement rumors → policy uncertainty
- Dollar strengthening → inverse correlation with metals
- ETF outflows beginning → smart money exiting

## Quick Start

### 1. Setup

```bash
# Clone and enter directory
cd port-tracker

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configure Your Portfolio

Edit `data/portfolio.json` with your holdings:

```json
{
  "name": "My Portfolio",
  "holdings": [
    {
      "ticker": "SLV",
      "name": "iShares Silver Trust",
      "asset_type": "etf",
      "sector": "precious_metals",
      "correlated_assets": ["GLD", "DXY", "TLT"],
      "risk_factors": ["fed policy", "dollar strength", "inflation"]
    }
  ]
}
```

### 3. Run Risk Check

```bash
# One-time check
python run_check.py

# Verbose output
python run_check.py --verbose

# Check specific tickers
python run_check.py --tickers SLV,GLD
```

### 4. Continuous Monitoring

```bash
# Monitor every 30 minutes
python run_monitor.py

# Custom interval (15 minutes)
python run_monitor.py --interval 15

# Market hours only (9:30 AM - 4:00 PM ET)
python run_monitor.py --market-hours
```

## Project Structure

```
port-tracker/
├── src/
│   ├── portfolio/
│   │   └── holdings.py           # Portfolio management
│   ├── monitors/
│   │   ├── news_scanner.py       # News monitoring (Tavily)
│   │   ├── perplexity_search.py  # Real-time intelligence (Perplexity)
│   │   ├── event_calendar.py     # Economic calendar
│   │   └── correlation_tracker.py # Correlation analysis
│   ├── analysis/
│   │   ├── risk_analyzer.py      # LLM-powered risk analysis
│   │   └── pattern_detector.py   # Historical pattern detection
│   ├── alerts/
│   │   └── notifier.py           # Alert delivery
│   └── llm/
│       └── providers.py          # SAP AI Core integration
├── data/
│   ├── portfolio.json            # Your holdings
│   └── alerts/                   # Alert history
├── run_check.py                  # One-time check
├── run_monitor.py                # Continuous monitoring
└── requirements.txt
```

## Required API Keys

| API | Purpose | Required |
|-----|---------|----------|
| SAP AI Core | LLM access (GPT-4o, Claude) | Yes |
| Tavily | News search | Yes |
| Perplexity | Real-time intelligence | Recommended |
| LangSmith | Observability | Optional |

## Alert Levels

| Level | Meaning | Action |
|-------|---------|--------|
| CRITICAL | Immediate risk detected | Act now |
| WARNING | Significant risk | Consider action |
| WATCH | Monitor closely | Stay alert |
| INFO | FYI | Be aware |

## Example Output

```
======================================================================
PORT-TRACKER RISK ANALYSIS
Timestamp: 2026-01-29 09:15:00 ET
Portfolio: 3 holdings (SLV, GLD, QQQ)
======================================================================

OVERALL RISK LEVEL: [!] ELEVATED

----------------------------------------------------------------------
## CRITICAL / HIGH PRIORITY ALERTS
----------------------------------------------------------------------

[!!!] Precious Metals - Reversal Risk
Affected: SLV, GLD
Time Horizon: IMMEDIATE

Gold and silver have hit 7 consecutive record highs. Historical analysis
shows this streak length precedes a 5-10% pullback 73% of the time within
5 trading days.

Leading Indicators Detected:
  - Dollar (DXY) strengthening +0.8% this week
  - Gold ETF outflows starting ($340M)
  - Fed Chair replacement rumors (Warsh = hawkish)

Recommended Action: REDUCE
======================================================================
```

## Disclaimer

This is an educational project for exploring LLM-based portfolio monitoring. **Do NOT make investment decisions based solely on this tool.** Always do your own research and consult with a financial advisor.

## License

MIT
