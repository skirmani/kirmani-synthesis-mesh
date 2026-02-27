"""
Intelligence Dashboard Generator

Creates real-time HTML dashboards for the Cross-System Intelligence output,
showing unified metrics, actions, and layer-specific insights.
"""

from datetime import datetime
from pathlib import Path

from .engine import UnifiedIntelligenceReport, MarketPosture, ActionPriority


class IntelligenceDashboardGenerator:
    """
    Generates HTML dashboards for intelligence reports

    Usage:
        generator = IntelligenceDashboardGenerator()
        html = generator.generate(report)
        generator.save(report, "/path/to/dashboard.html")
    """

    def generate(self, report: UnifiedIntelligenceReport) -> str:
        """Generate HTML dashboard from intelligence report"""

        posture_color = self._posture_color(report.market_posture)
        risk_color = self._risk_color(report.unified_risk_score)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kirmani Cross-System Intelligence - {report.generated_at.strftime('%Y-%m-%d %H:%M')}</title>
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
            --accent-gold: #ffd700;
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
            border: 1px solid rgba(255, 215, 0, 0.3);
        }}
        .header h1 {{
            font-size: 2rem;
            background: linear-gradient(90deg, var(--accent-gold), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .posture-badge {{
            display: inline-block;
            padding: 15px 40px;
            border-radius: 12px;
            font-size: 1.8rem;
            font-weight: bold;
            margin: 20px 0;
            background: {posture_color}33;
            color: {posture_color};
            border: 2px solid {posture_color};
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
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 10px;
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
        .progress-bar {{
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .progress-bar .fill {{
            height: 100%;
            border-radius: 4px;
        }}
        .action {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid var(--accent-yellow);
        }}
        .action.critical {{ border-left-color: var(--accent-red); background: rgba(255, 71, 87, 0.1); }}
        .action.high {{ border-left-color: var(--accent-yellow); }}
        .action.medium {{ border-left-color: var(--accent-blue); }}
        .action h4 {{ margin-bottom: 5px; }}
        .action p {{ color: var(--text-secondary); font-size: 0.9rem; }}
        .layer-section {{
            margin-bottom: 20px;
        }}
        .layer-header {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }}
        .layer-icon {{
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            margin-right: 12px;
        }}
        .layer-macro {{ background: rgba(156, 39, 176, 0.3); }}
        .layer-risk {{ background: rgba(255, 71, 87, 0.3); }}
        .layer-alpha {{ background: rgba(0, 212, 170, 0.3); }}
        .stats-row {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 15px 20px;
            text-align: center;
            min-width: 120px;
        }}
        .stat-box .value {{
            font-size: 1.5rem;
            font-weight: bold;
        }}
        .stat-box .label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        .signal-list {{
            max-height: 300px;
            overflow-y: auto;
        }}
        .signal-item {{
            background: var(--bg-secondary);
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .signal-item .ticker {{
            font-weight: bold;
            color: var(--accent-green);
        }}
        .signal-item.short .ticker {{
            color: var(--accent-red);
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>KIRMANI CROSS-SYSTEM INTELLIGENCE</h1>
        <p style="color: var(--text-secondary);">Phase 3: Unified Multi-Layer Analysis</p>
        <div class="posture-badge">{report.market_posture.value}</div>
        <p style="color: var(--text-secondary);">
            Confidence: {report.posture_confidence:.0%} |
            {len(report.systems_contributing)} Systems |
            {report.total_signals_processed} Signals
        </p>
        <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 10px;">
            {report.posture_rationale}
        </p>
    </div>

    <div class="stats-row">
        <div class="stat-box">
            <div class="value" style="color: {risk_color}">{report.unified_risk_score:.0f}</div>
            <div class="label">Risk Score</div>
        </div>
        <div class="stat-box">
            <div class="value" style="color: var(--accent-green)">{report.unified_opportunity_score:.0f}</div>
            <div class="label">Opportunity Score</div>
        </div>
        <div class="stat-box">
            <div class="value">{report.risk_reward_balance:+.2f}</div>
            <div class="label">R/R Balance</div>
        </div>
        <div class="stat-box">
            <div class="value">{report.full_stack_alignment:.0%}</div>
            <div class="label">Layer Alignment</div>
        </div>
        <div class="stat-box">
            <div class="value">{report.recommended_equity_exposure:.0%}</div>
            <div class="label">Equity Exposure</div>
        </div>
        <div class="stat-box">
            <div class="value">{report.hedge_ratio:.0%}</div>
            <div class="label">Hedge Ratio</div>
        </div>
    </div>

    <div class="grid">
        {self._render_actions(report)}
        {self._render_macro_layer(report)}
        {self._render_risk_layer(report)}
        {self._render_alpha_layer(report)}
    </div>

    <div class="footer">
        <p>Kirmani Synthesis Mesh v2.0.0 | Phase 3 Cross-System Intelligence</p>
        <p>Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>
</body>
</html>"""

        return html

    def _posture_color(self, posture: MarketPosture) -> str:
        return {
            MarketPosture.FULL_DEFENSE: "var(--accent-red)",
            MarketPosture.DEFENSIVE: "#ff6b6b",
            MarketPosture.CAUTIOUS: "var(--accent-yellow)",
            MarketPosture.NEUTRAL: "var(--text-secondary)",
            MarketPosture.OPPORTUNISTIC: "var(--accent-blue)",
            MarketPosture.AGGRESSIVE: "var(--accent-green)",
        }.get(posture, "var(--text-secondary)")

    def _risk_color(self, score: float) -> str:
        if score >= 70:
            return "var(--accent-red)"
        elif score >= 50:
            return "var(--accent-yellow)"
        else:
            return "var(--accent-green)"

    def _render_actions(self, report: UnifiedIntelligenceReport) -> str:
        if not report.actions:
            return ""

        html = '<div class="card"><h3>Recommended Actions</h3>'

        for action in report.actions[:8]:
            priority_class = action.priority.value.lower()
            ticker_str = f" <strong>{action.ticker}</strong>" if action.ticker else ""

            html += f'''
            <div class="action {priority_class}">
                <h4>[{action.priority.value}] {action.action_type}{ticker_str}</h4>
                <p>{action.description}</p>
                <p style="font-size: 0.8rem;">{action.rationale}</p>
            </div>
            '''

        html += '</div>'
        return html

    def _render_macro_layer(self, report: UnifiedIntelligenceReport) -> str:
        macro = report.macro_intelligence

        html = '''<div class="card">
            <div class="layer-header">
                <div class="layer-icon layer-macro">M</div>
                <h3 style="margin: 0; border: none; padding: 0;">Macro Intelligence</h3>
            </div>'''

        html += f'''
            <div class="metric">
                <span class="label">Regime</span>
                <span class="value">{macro.regime.value}</span>
            </div>
            <div class="metric">
                <span class="label">Crash Hazard</span>
                <span class="value" style="color: {self._risk_color(macro.unified_crash_hazard * 100)}">{macro.unified_crash_hazard:.0%}</span>
            </div>
            <div class="progress-bar">
                <div class="fill" style="width: {macro.unified_crash_hazard * 100}%; background: {self._risk_color(macro.unified_crash_hazard * 100)};"></div>
            </div>
            <div class="metric">
                <span class="label">Bubble Phase</span>
                <span class="value">{macro.bubble_phase}/5</span>
            </div>
            <div class="metric">
                <span class="label">Intermarket</span>
                <span class="value">{macro.intermarket_regime}</span>
            </div>
            <div class="metric">
                <span class="label">Crash Timeframe</span>
                <span class="value">{macro.crash_timeframe.value}</span>
            </div>
            <div class="metric">
                <span class="label">Equity Exposure</span>
                <span class="value">{macro.recommended_equity_exposure:.0%}</span>
            </div>
        </div>'''

        return html

    def _render_risk_layer(self, report: UnifiedIntelligenceReport) -> str:
        risk = report.risk_intelligence

        html = '''<div class="card">
            <div class="layer-header">
                <div class="layer-icon layer-risk">R</div>
                <h3 style="margin: 0; border: none; padding: 0;">Risk Intelligence</h3>
            </div>'''

        html += f'''
            <div class="metric">
                <span class="label">Risk Level</span>
                <span class="value">{risk.risk_level.value}</span>
            </div>
            <div class="metric">
                <span class="label">Risk Score</span>
                <span class="value" style="color: {self._risk_color(risk.risk_score)}">{risk.risk_score:.0f}/100</span>
            </div>
            <div class="progress-bar">
                <div class="fill" style="width: {risk.risk_score}%; background: {self._risk_color(risk.risk_score)};"></div>
            </div>
            <div class="metric">
                <span class="label">Vol Regime</span>
                <span class="value">{risk.vol_regime.value}</span>
            </div>
            <div class="metric">
                <span class="label">VIX Level</span>
                <span class="value">{risk.vix_level:.1f}</span>
            </div>
            <div class="metric">
                <span class="label">Squeeze Alerts</span>
                <span class="value">{len(risk.squeeze_alerts)}</span>
            </div>
            <div class="metric">
                <span class="label">Hedge Urgency</span>
                <span class="value">{risk.hedge_urgency}</span>
            </div>
            <div class="metric">
                <span class="label">Position Multiplier</span>
                <span class="value">{risk.position_size_multiplier:.2f}x</span>
            </div>
        </div>'''

        return html

    def _render_alpha_layer(self, report: UnifiedIntelligenceReport) -> str:
        alpha = report.alpha_intelligence

        html = '''<div class="card">
            <div class="layer-header">
                <div class="layer-icon layer-alpha">A</div>
                <h3 style="margin: 0; border: none; padding: 0;">Alpha Intelligence</h3>
            </div>'''

        html += f'''
            <div class="metric">
                <span class="label">Total Opportunities</span>
                <span class="value">{alpha.total_opportunities}</span>
            </div>
            <div class="metric">
                <span class="label">High Conviction</span>
                <span class="value" style="color: var(--accent-green)">{alpha.high_conviction_count}</span>
            </div>
            <div class="metric">
                <span class="label">Market Breadth</span>
                <span class="value">{alpha.market_breadth}</span>
            </div>
            <div class="metric">
                <span class="label">SPY Wyckoff</span>
                <span class="value">{alpha.spy_wyckoff_phase.value}</span>
            </div>
            <div class="metric">
                <span class="label">Dominant Cycle</span>
                <span class="value">{alpha.dominant_cycle} days</span>
            </div>
            <div class="metric">
                <span class="label">Cycle Phase</span>
                <span class="value">{alpha.cycle_phase}</span>
            </div>'''

        # Top signals
        if alpha.alpha_signals[:5]:
            html += '<h4 style="margin-top: 15px; color: var(--accent-green);">Top Signals</h4>'
            html += '<div class="signal-list">'
            for sig in alpha.alpha_signals[:5]:
                short_class = "short" if sig.direction == "SHORT" else ""
                html += f'''
                <div class="signal-item {short_class}">
                    <span class="ticker">{sig.direction[0]} {sig.ticker}</span>
                    <span>{sig.strength.value} ({sig.confidence:.0%})</span>
                </div>
                '''
            html += '</div>'

        html += '</div>'
        return html

    def save(self, report: UnifiedIntelligenceReport, filepath: str) -> None:
        """Save dashboard to file"""
        html = self.generate(report)
        Path(filepath).write_text(html)
