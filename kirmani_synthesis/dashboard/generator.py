"""
Dashboard Generator

Creates real-time HTML dashboards for the Synthesis Mesh,
showing market state, recommendations, and system health.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models import SynthesisReport, MarketRegime


class DashboardGenerator:
    """
    Generates HTML dashboards for synthesis reports

    Usage:
        generator = DashboardGenerator()
        html = generator.generate(report)
        generator.save(report, "/path/to/dashboard.html")
    """

    def generate(self, report: SynthesisReport) -> str:
        """Generate HTML dashboard from synthesis report"""
        state = report.market_state

        # Determine colors based on state
        regime_color = self._regime_color(state.regime)
        risk_color = self._risk_color(state.overall_risk_score)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kirmani Synthesis Mesh - {report.generated_at.strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-secondary: #141a27;
            --bg-card: #1a2332;
            --text-primary: #e4e8f0;
            --text-secondary: #8892a0;
            --accent-green: #00d4aa;
            --accent-red: #ff4757;
            --accent-yellow: #ffc107;
            --accent-blue: #00b4d8;
            --accent-purple: #9c27b0;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 20px;
            line-height: 1.6;
        }}

        .header {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            border-radius: 16px;
            margin-bottom: 25px;
            border: 1px solid rgba(0, 212, 170, 0.3);
        }}

        .header h1 {{
            font-size: 2rem;
            background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}

        .card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .card h3 {{
            color: var(--accent-blue);
            margin-bottom: 15px;
            font-size: 1.1rem;
        }}

        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .metric:last-child {{ border-bottom: none; }}

        .metric .label {{ color: var(--text-secondary); }}
        .metric .value {{ font-weight: bold; }}

        .regime-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.2rem;
        }}

        .regime-crisis {{ background: rgba(255, 71, 87, 0.3); color: var(--accent-red); }}
        .regime-risk-off {{ background: rgba(255, 193, 7, 0.3); color: var(--accent-yellow); }}
        .regime-neutral {{ background: rgba(136, 146, 160, 0.3); color: var(--text-secondary); }}
        .regime-risk-on {{ background: rgba(0, 212, 170, 0.3); color: var(--accent-green); }}

        .recommendation {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid var(--accent-yellow);
        }}

        .recommendation.urgent {{ border-left-color: var(--accent-red); }}
        .recommendation.bullish {{ border-left-color: var(--accent-green); }}

        .recommendation h4 {{
            margin-bottom: 8px;
        }}

        .quick-win {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid var(--accent-purple);
        }}

        .quick-win h4 {{ color: var(--accent-purple); margin-bottom: 5px; }}

        .contradiction {{
            background: rgba(255, 71, 87, 0.1);
            border: 1px solid rgba(255, 71, 87, 0.3);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
        }}

        .confirmation {{
            background: rgba(0, 212, 170, 0.1);
            border: 1px solid rgba(0, 212, 170, 0.3);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
        }}

        .progress-bar {{
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }}

        .progress-bar .fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}

        .stance {{
            text-align: center;
            padding: 20px;
            margin-bottom: 25px;
        }}

        .stance-badge {{
            display: inline-block;
            padding: 15px 40px;
            border-radius: 12px;
            font-size: 1.5rem;
            font-weight: bold;
        }}

        .stance-defensive {{ background: rgba(255, 71, 87, 0.3); color: var(--accent-red); }}
        .stance-neutral {{ background: rgba(136, 146, 160, 0.3); color: var(--text-secondary); }}
        .stance-aggressive {{ background: rgba(0, 212, 170, 0.3); color: var(--accent-green); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>KIRMANI SYNTHESIS MESH</h1>
        <p style="color: var(--text-secondary);">Cross-System Intelligence Dashboard</p>
        <p style="color: var(--text-secondary); font-size: 0.9rem;">Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>

    <div class="stance">
        <div class="stance-badge stance-{report.overall_stance.lower()}">
            {report.overall_stance} STANCE
        </div>
        <p style="margin-top: 10px; color: var(--text-secondary);">
            Recommended Exposure: {report.recommended_exposure:.0%} |
            {report.systems_reporting} Systems Reporting |
            {report.total_signals_processed} Signals Processed
        </p>
    </div>

    <div class="grid">
        <div class="card">
            <h3>Market Regime</h3>
            <div style="text-align: center; padding: 20px;">
                <span class="regime-badge regime-{state.regime.value.lower().replace('_', '-')}">{state.regime.value}</span>
                <p style="margin-top: 10px; color: var(--text-secondary);">Confidence: {state.regime_confidence:.0%}</p>
            </div>
            <div class="metric">
                <span class="label">Crash Hazard</span>
                <span class="value" style="color: {self._hazard_color(state.crash_hazard)}">{state.crash_hazard:.0%}</span>
            </div>
            <div class="progress-bar">
                <div class="fill" style="width: {state.crash_hazard * 100}%; background: {self._hazard_color(state.crash_hazard)};"></div>
            </div>
            <div class="metric">
                <span class="label">Bubble Phase</span>
                <span class="value">{state.bubble_phase}/5 ({self._phase_name(state.bubble_phase)})</span>
            </div>
            <div class="metric">
                <span class="label">Vol Regime</span>
                <span class="value">{state.vol_regime}</span>
            </div>
        </div>

        <div class="card">
            <h3>Risk Assessment</h3>
            <div class="metric">
                <span class="label">Overall Risk Score</span>
                <span class="value" style="color: {risk_color}">{state.overall_risk_score:.0f}/100</span>
            </div>
            <div class="progress-bar">
                <div class="fill" style="width: {state.overall_risk_score}%; background: {risk_color};"></div>
            </div>
            <div class="metric">
                <span class="label">Squeeze Risk</span>
                <span class="value">{state.squeeze_risk:.0%}</span>
            </div>
            <div class="metric">
                <span class="label">Intermarket Signal</span>
                <span class="value">{state.intermarket_signal}</span>
            </div>
            <div class="metric">
                <span class="label">System Agreement</span>
                <span class="value">{state.system_agreement:.0%}</span>
            </div>
        </div>

        <div class="card">
            <h3>Signal Summary</h3>
            <div class="metric">
                <span class="label">Total Signals</span>
                <span class="value">{report.total_signals_processed}</span>
            </div>
            <div class="metric">
                <span class="label">Systems Reporting</span>
                <span class="value">{report.systems_reporting}</span>
            </div>
            <div class="metric">
                <span class="label">Critical Alerts</span>
                <span class="value" style="color: {'var(--accent-red)' if report.critical_alerts > 0 else 'inherit'}">{report.critical_alerts}</span>
            </div>
            <div class="metric">
                <span class="label">Quick Wins Triggered</span>
                <span class="value">{len(report.quick_wins_triggered)}</span>
            </div>
            <div class="metric">
                <span class="label">Contradictions</span>
                <span class="value">{len(report.contradictions)}</span>
            </div>
            <div class="metric">
                <span class="label">Confirmations</span>
                <span class="value">{len(report.confirmations)}</span>
            </div>
        </div>
    </div>

    {self._render_recommendations(report.recommendations)}
    {self._render_quick_wins(report.quick_wins_triggered)}
    {self._render_contradictions(report.contradictions)}
    {self._render_confirmations(report.confirmations)}

    <div class="footer">
        <p>Kirmani Synthesis Mesh v1.0.0 | 48+ Trading Systems | Bayesian Fusion Engine</p>
        <p>Report ID: {report.id[:8]}</p>
    </div>
</body>
</html>"""

        return html

    def _regime_color(self, regime: MarketRegime) -> str:
        colors = {
            MarketRegime.CRISIS: "var(--accent-red)",
            MarketRegime.RISK_OFF: "var(--accent-yellow)",
            MarketRegime.NEUTRAL: "var(--text-secondary)",
            MarketRegime.TRANSITION: "var(--accent-blue)",
            MarketRegime.RISK_ON: "var(--accent-green)",
        }
        return colors.get(regime, "var(--text-secondary)")

    def _risk_color(self, score: float) -> str:
        if score >= 70:
            return "var(--accent-red)"
        elif score >= 50:
            return "var(--accent-yellow)"
        else:
            return "var(--accent-green)"

    def _hazard_color(self, hazard: float) -> str:
        if hazard >= 0.7:
            return "var(--accent-red)"
        elif hazard >= 0.4:
            return "var(--accent-yellow)"
        else:
            return "var(--accent-green)"

    def _phase_name(self, phase: int) -> str:
        names = {1: "Displacement", 2: "Boom", 3: "Euphoria", 4: "Distress", 5: "Panic"}
        return names.get(phase, "Unknown")

    def _render_recommendations(self, recommendations) -> str:
        if not recommendations:
            return ""

        html = '<div class="card"><h3>Position Recommendations</h3>'
        for rec in recommendations:
            urgency_class = "urgent" if rec.urgency.value in ("IMMEDIATE", "URGENT") else ""
            bullish_class = "bullish" if rec.action.value.startswith("ENTER_LONG") or rec.action.value.startswith("INCREASE_LONG") else ""

            html += f'''
            <div class="recommendation {urgency_class} {bullish_class}">
                <h4>{rec.action.value}: {rec.ticker}</h4>
                <p>Position Change: {rec.change_pct:+.0f}% | Urgency: {rec.urgency.value}</p>
                <p style="color: var(--text-secondary);">Trigger: {rec.primary_trigger}</p>
                <p style="color: var(--text-secondary);">Confidence: {rec.confidence:.0%} | Algo: {rec.execution_algo}</p>
            </div>
            '''
        html += '</div>'
        return html

    def _render_quick_wins(self, quick_wins) -> str:
        if not quick_wins:
            return ""

        html = '<div class="card"><h3>Quick Wins Triggered</h3>'
        for qw in quick_wins:
            html += f'''
            <div class="quick-win">
                <h4>{qw.rule_name}</h4>
                <p>{qw.action.value}: {qw.target_ticker} ({qw.position_change_pct:+.0f}%)</p>
                <p style="color: var(--text-secondary);">Confidence: {qw.confidence:.0%} | Sources: {", ".join(qw.source_signals)}</p>
            </div>
            '''
        html += '</div>'
        return html

    def _render_contradictions(self, contradictions) -> str:
        if not contradictions:
            return ""

        html = '<div class="card"><h3>Contradictions Detected</h3>'
        for c in contradictions:
            html += f'''
            <div class="contradiction">
                <p><strong>{c.ticker}</strong>: {c.system_a} ({c.signal_a_direction}) vs {c.system_b} ({c.signal_b_direction})</p>
                <p>Severity: {c.severity:.0%} | Recommended: {c.recommended_action}</p>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">{c.resolution_rationale}</p>
            </div>
            '''
        html += '</div>'
        return html

    def _render_confirmations(self, confirmations) -> str:
        if not confirmations:
            return ""

        html = '<div class="card"><h3>Cross-System Confirmations</h3>'
        for c in confirmations:
            html += f'''
            <div class="confirmation">
                <p><strong>{c.ticker}</strong>: {c.direction} ({c.confirmation_strength})</p>
                <p>{c.system_count} systems, {c.layer_count} layers | Combined: {c.combined_confidence:.0%}</p>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">Systems: {", ".join(c.systems)}</p>
            </div>
            '''
        html += '</div>'
        return html

    def save(self, report: SynthesisReport, filepath: str) -> None:
        """Save dashboard to file"""
        html = self.generate(report)
        Path(filepath).write_text(html)
