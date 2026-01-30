"""
Alert Notifier for Port-Tracker.
Handles alert delivery through various channels.
"""
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional
from enum import Enum
from pathlib import Path


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"           # FYI, no action needed
    WATCH = "watch"         # Monitor closely
    WARNING = "warning"     # Consider action
    CRITICAL = "critical"   # Immediate attention required


@dataclass
class Alert:
    """An alert to be delivered."""
    level: AlertLevel
    affected_holdings: List[str]
    title: str
    summary: str
    recommended_action: str
    timestamp: datetime
    details: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "affected_holdings": self.affected_holdings,
            "title": self.title,
            "summary": self.summary,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }


class AlertNotifier:
    """Manages alert delivery."""

    def __init__(self, alerts_dir: str = "data/alerts"):
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self._alerts: List[Alert] = []

    def add_alert(self, alert: Alert) -> None:
        """Add an alert to the queue."""
        self._alerts.append(alert)

    def add_from_risk(
        self,
        title: str,
        affected_holdings: List[str],
        severity: str,
        description: str,
        recommended_action: str
    ) -> None:
        """Create and add an alert from risk data."""
        level_map = {
            "CRITICAL": AlertLevel.CRITICAL,
            "HIGH": AlertLevel.WARNING,
            "MEDIUM": AlertLevel.WATCH,
            "LOW": AlertLevel.INFO
        }

        alert = Alert(
            level=level_map.get(severity, AlertLevel.INFO),
            affected_holdings=affected_holdings,
            title=title,
            summary=description[:200] + "..." if len(description) > 200 else description,
            recommended_action=recommended_action,
            timestamp=datetime.now(),
            details=description
        )

        self.add_alert(alert)

    def get_alerts(self, min_level: AlertLevel = AlertLevel.INFO) -> List[Alert]:
        """Get alerts at or above a minimum level."""
        level_order = {
            AlertLevel.INFO: 0,
            AlertLevel.WATCH: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.CRITICAL: 3
        }

        min_order = level_order[min_level]
        return [a for a in self._alerts if level_order[a.level] >= min_order]

    def clear_alerts(self) -> None:
        """Clear all pending alerts."""
        self._alerts = []

    def notify_console(self, min_level: AlertLevel = AlertLevel.INFO) -> None:
        """Print alerts to console."""
        alerts = self.get_alerts(min_level)

        if not alerts:
            print("\n[OK] No alerts to report.")
            return

        level_icons = {
            AlertLevel.INFO: "[i]",
            AlertLevel.WATCH: "[~]",
            AlertLevel.WARNING: "[!]",
            AlertLevel.CRITICAL: "[!!!]"
        }

        level_colors = {
            AlertLevel.INFO: "",
            AlertLevel.WATCH: "",
            AlertLevel.WARNING: "",
            AlertLevel.CRITICAL: ""
        }

        print("\n" + "=" * 60)
        print("ALERTS")
        print("=" * 60)

        for alert in alerts:
            icon = level_icons[alert.level]
            print(f"\n{icon} {alert.level.value.upper()}: {alert.title}")
            print(f"    Holdings: {', '.join(alert.affected_holdings)}")
            print(f"    {alert.summary}")
            print(f"    Action: {alert.recommended_action}")

        print("\n" + "=" * 60)

    def save_alerts(self, filename: Optional[str] = None) -> str:
        """Save alerts to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"alerts_{timestamp}.json"

        filepath = self.alerts_dir / filename

        data = {
            "generated": datetime.now().isoformat(),
            "alert_count": len(self._alerts),
            "alerts": [a.to_dict() for a in self._alerts]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def load_alerts(self, filepath: str) -> List[Alert]:
        """Load alerts from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        alerts = []
        for a in data.get('alerts', []):
            alerts.append(Alert(
                level=AlertLevel(a['level']),
                affected_holdings=a['affected_holdings'],
                title=a['title'],
                summary=a['summary'],
                recommended_action=a['recommended_action'],
                timestamp=datetime.fromisoformat(a['timestamp']),
                details=a.get('details')
            ))

        return alerts

    def format_summary(self) -> str:
        """Format a summary of current alerts."""
        critical = len([a for a in self._alerts if a.level == AlertLevel.CRITICAL])
        warning = len([a for a in self._alerts if a.level == AlertLevel.WARNING])
        watch = len([a for a in self._alerts if a.level == AlertLevel.WATCH])
        info = len([a for a in self._alerts if a.level == AlertLevel.INFO])

        return f"Alerts: {critical} CRITICAL | {warning} WARNING | {watch} WATCH | {info} INFO"


def create_alerts_from_assessment(assessment, notifier: AlertNotifier) -> None:
    """
    Create alerts from a RiskAssessment.

    Args:
        assessment: RiskAssessment from risk analyzer
        notifier: AlertNotifier to add alerts to
    """
    for risk in assessment.risks:
        notifier.add_from_risk(
            title=risk.title,
            affected_holdings=risk.affected_holdings,
            severity=risk.severity.value,
            description=risk.description,
            recommended_action=risk.recommended_action.value
        )
