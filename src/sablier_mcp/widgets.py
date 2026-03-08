"""
Sablier MCP Widgets

Self-contained HTML generators for rich visual output in AI chat interfaces.
Each function takes parsed API data and returns a complete HTML document
with inline CSS (dark theme, no external dependencies).
"""

from __future__ import annotations

import html
from typing import Any

# ── Design tokens ──────────────────────────────────────────────

_BG = "#0f1117"
_CARD = "#1a1b2e"
_CARD_BORDER = "#2a2b3d"
_TEXT = "#e2e4e9"
_TEXT_MUTED = "#8b8fa3"
_ACCENT = "#6366f1"       # indigo
_ACCENT_LIGHT = "#818cf8"
_GREEN = "#22c55e"
_YELLOW = "#eab308"
_RED = "#ef4444"
_BLUE = "#3b82f6"
_ORANGE = "#f97316"

_BASE_CSS = f"""
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, sans-serif;
    background: {_BG};
    color: {_TEXT};
    padding: 16px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
}}
.widget-title {{
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {_TEXT_MUTED};
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.widget-title .brand {{
    color: {_ACCENT_LIGHT};
}}
.card {{
    background: {_CARD};
    border: 1px solid {_CARD_BORDER};
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}}
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.badge-high {{ background: rgba(34,197,94,0.15); color: {_GREEN}; }}
.badge-moderate {{ background: rgba(234,179,8,0.15); color: {_YELLOW}; }}
.badge-low {{ background: rgba(239,68,68,0.15); color: {_RED}; }}
.badge-minimal {{ background: rgba(139,143,163,0.15); color: {_TEXT_MUTED}; }}
"""


def _wrap(title: str, body: str, width: int = 520) -> str:
    """Wrap widget body in a full HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{_BASE_CSS}</style>
</head>
<body style="max-width:{width}px">
<div class="widget-title"><span class="brand">Sablier</span> {html.escape(title)}</div>
{body}
</body>
</html>"""


def _score_color(score: float | None) -> str:
    """Map a 0-100 score to a color."""
    if score is None:
        return _TEXT_MUTED
    if score >= 70:
        return _GREEN
    if score >= 40:
        return _YELLOW
    return _RED


def _tier_class(tier: str | int | None) -> str:
    if tier is None:
        return "badge-minimal"
    t = str(tier).lower()
    if t in ("high", "very high", "critical"):
        return "badge-high"
    if t in ("moderate", "medium"):
        return "badge-moderate"
    if t in ("low",):
        return "badge-low"
    return "badge-minimal"


def _pct(val: float | None, digits: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.{digits}f}%"


def _num(val: float | None, digits: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}"


# ═══════════════════════════════════════════════════════
# 1. GRAIN Score Card
# ═══════════════════════════════════════════════════════


def grain_score_card(data: dict[str, Any]) -> str:
    """Render GRAIN qualitative analysis results as a visual score card.

    Expects the parsed summary dict with:
      themes: [{ theme, portfolio_score, ticker_scores: [{ ticker, score, tier, top_evidence }] }]
    """
    themes = data.get("themes", [])
    if not themes:
        return _wrap("Qualitative Analysis", '<div class="card">No results</div>')

    cards_html = ""
    for theme in themes:
        name = theme.get("display_name") or theme.get("theme", "Unknown Theme")
        p_score = theme.get("portfolio_score")

        # Ticker rows
        rows = ""
        for ts in theme.get("ticker_scores", []):
            ticker = ts.get("ticker", "???")
            score = ts.get("score")
            tier = ts.get("tier")
            bar_w = min(score or 0, 100)
            color = _score_color(score)

            # Evidence preview
            ev_html = ""
            for ev in (ts.get("top_evidence") or [])[:2]:
                passage = html.escape((ev.get("passage") or "")[:180])
                src = ev.get("source_type") or ""
                period = ev.get("fiscal_period") or ""
                ev_html += f"""<div style="margin-top:6px;padding:8px;background:rgba(99,102,241,0.06);border-radius:6px;font-size:11px;color:{_TEXT_MUTED};line-height:1.4">
                    <span style="color:{_ACCENT_LIGHT};font-weight:600">{html.escape(src)} {html.escape(period)}</span><br>
                    &ldquo;{passage}&hellip;&rdquo;
                </div>"""

            rows += f"""<div style="margin-bottom:14px">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
                    <div style="display:flex;align-items:center;gap:8px">
                        <span style="font-weight:700;font-size:14px">{html.escape(ticker)}</span>
                        <span class="badge {_tier_class(tier)}">{html.escape(str(tier) if tier is not None else 'N/A')}</span>
                    </div>
                    <span style="font-weight:700;font-size:16px;color:{color}">{score if score is not None else 'N/A'}</span>
                </div>
                <div style="height:6px;background:{_CARD_BORDER};border-radius:3px;overflow:hidden">
                    <div style="height:100%;width:{bar_w}%;background:{color};border-radius:3px;transition:width 0.3s"></div>
                </div>
                {ev_html}
            </div>"""

        # Theme header
        p_color = _score_color(p_score)
        cards_html += f"""<div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
                <div style="font-size:15px;font-weight:700">{html.escape(name)}</div>
                <div style="text-align:right">
                    <div style="font-size:22px;font-weight:800;color:{p_color}">{p_score if p_score is not None else '—'}</div>
                    <div style="font-size:10px;color:{_TEXT_MUTED}">PORTFOLIO SCORE</div>
                </div>
            </div>
            {rows}
        </div>"""

    return _wrap("Qualitative Analysis", cards_html)


# ═══════════════════════════════════════════════════════
# 2. Factor Betas Heatmap
# ═══════════════════════════════════════════════════════


def betas_heatmap(data: dict[str, Any]) -> str:
    """Render factor betas as a color-coded heatmap table.

    Expects the parsed betas dict with:
      conditioning_features: [str]
      assets: { ticker: { linear_betas: { factor: value }, alpha, residual_std } }
    """
    features = data.get("conditioning_features", [])
    assets = data.get("assets", {})

    if not assets or not features:
        return _wrap("Factor Betas", '<div class="card">No betas data</div>')

    # Find min/max for color scaling
    all_vals = []
    for a_data in assets.values():
        betas = a_data.get("linear_betas", {})
        for f in features:
            v = betas.get(f)
            if v is not None:
                all_vals.append(abs(v))
    max_abs = max(all_vals) if all_vals else 1.0

    def beta_cell(val: float | None) -> str:
        if val is None:
            return f'<td style="text-align:center;padding:6px 8px;color:{_TEXT_MUTED}">—</td>'
        intensity = min(abs(val) / max_abs, 1.0) * 0.7
        if val > 0:
            bg = f"rgba(34,197,94,{intensity:.2f})"
        elif val < 0:
            bg = f"rgba(239,68,68,{intensity:.2f})"
        else:
            bg = "transparent"
        return f'<td style="text-align:center;padding:6px 8px;background:{bg};font-size:12px;font-weight:600;font-variant-numeric:tabular-nums">{val:.4f}</td>'

    # Table header
    th_style = f"padding:6px 8px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:{_TEXT_MUTED};text-align:center;border-bottom:1px solid {_CARD_BORDER}"
    headers = f'<th style="{th_style};text-align:left">Asset</th>'
    for f in features:
        label = html.escape(f.replace("_", " ")[:12])
        headers += f'<th style="{th_style}">{label}</th>'
    headers += f'<th style="{th_style}">Alpha</th>'
    headers += f'<th style="{th_style}">Resid &sigma;</th>'

    # Rows
    rows = ""
    for ticker, a_data in sorted(assets.items()):
        betas = a_data.get("linear_betas", {})
        alpha = a_data.get("alpha")
        resid = a_data.get("residual_std")
        cells = f'<td style="padding:6px 8px;font-weight:700;font-size:13px;white-space:nowrap">{html.escape(ticker)}</td>'
        for f in features:
            cells += beta_cell(betas.get(f))
        cells += beta_cell(alpha)
        cells += f'<td style="text-align:center;padding:6px 8px;font-size:12px;color:{_TEXT_MUTED}">{_num(resid)}</td>'
        rows += f'<tr style="border-bottom:1px solid {_CARD_BORDER}">{cells}</tr>'

    table = f"""<div class="card" style="overflow-x:auto;padding:0">
        <table style="width:100%;border-collapse:collapse;min-width:400px">
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>"""

    # Legend
    legend = f"""<div style="display:flex;align-items:center;gap:16px;font-size:11px;color:{_TEXT_MUTED};margin-top:4px">
        <div style="display:flex;align-items:center;gap:4px"><div style="width:12px;height:12px;border-radius:2px;background:rgba(34,197,94,0.5)"></div> Positive</div>
        <div style="display:flex;align-items:center;gap:4px"><div style="width:12px;height:12px;border-radius:2px;background:rgba(239,68,68,0.5)"></div> Negative</div>
        <div>Intensity &prop; magnitude</div>
    </div>"""

    return _wrap("Factor Betas", table + legend, width=700)


# ═══════════════════════════════════════════════════════
# 3. Risk Dashboard
# ═══════════════════════════════════════════════════════


def risk_dashboard(data: dict[str, Any]) -> str:
    """Render portfolio risk metrics as a visual dashboard.

    Expects the parsed risk dict with:
      expected_return, var_95, cvar_95, portfolio_alpha, diversification_ratio,
      portfolio_betas: { factor: value },
      risk_contribution: { factor: value },
      marginal_ctr: { asset: value },
      n_assets
    """
    exp_ret = data.get("expected_return")
    var95 = data.get("var_95")
    cvar95 = data.get("cvar_95")
    alpha = data.get("portfolio_alpha")
    div_ratio = data.get("diversification_ratio")
    n_assets = data.get("n_assets")

    # KPI row
    def kpi(label: str, value: str, color: str = _TEXT) -> str:
        return f"""<div style="flex:1;min-width:100px;text-align:center">
            <div style="font-size:22px;font-weight:800;color:{color};font-variant-numeric:tabular-nums">{value}</div>
            <div style="font-size:10px;color:{_TEXT_MUTED};text-transform:uppercase;letter-spacing:0.06em;margin-top:2px">{label}</div>
        </div>"""

    var_color = _RED if var95 is not None and var95 < -0.05 else _YELLOW if var95 is not None and var95 < -0.02 else _GREEN
    ret_color = _GREEN if exp_ret is not None and exp_ret > 0 else _RED

    kpis = f"""<div class="card" style="display:flex;flex-wrap:wrap;gap:8px">
        {kpi("Expected Return", _pct(exp_ret), ret_color)}
        {kpi("VaR 95%", _pct(var95), var_color)}
        {kpi("CVaR 95%", _pct(cvar95), _RED if cvar95 is not None and cvar95 < 0 else _TEXT)}
        {kpi("Alpha", _pct(alpha), _ACCENT_LIGHT)}
    </div>"""

    # Diversification gauge
    gauge = ""
    if div_ratio is not None:
        pct = min(div_ratio * 100, 100)
        gauge_color = _GREEN if div_ratio > 0.7 else _YELLOW if div_ratio > 0.4 else _RED
        gauge = f"""<div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-size:12px;font-weight:600">Diversification</span>
                <span style="font-size:14px;font-weight:700;color:{gauge_color}">{_num(div_ratio, 2)}</span>
            </div>
            <div style="height:6px;background:{_CARD_BORDER};border-radius:3px;overflow:hidden">
                <div style="height:100%;width:{pct:.0f}%;background:{gauge_color};border-radius:3px"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;color:{_TEXT_MUTED};margin-top:3px">
                <span>Concentrated</span><span>Diversified</span>
            </div>
        </div>"""

    # Risk contribution bars
    risk_ctr = data.get("risk_contribution", {})
    risk_bars = ""
    if risk_ctr:
        max_rc = max(abs(v) for v in risk_ctr.values()) if risk_ctr else 1.0
        rows = ""
        for factor, val in sorted(risk_ctr.items(), key=lambda x: abs(x[1]), reverse=True):
            bar_w = abs(val) / max_rc * 100 if max_rc else 0
            color = _BLUE if val >= 0 else _ORANGE
            label = html.escape(factor.replace("_", " "))
            rows += f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                <div style="width:90px;font-size:11px;font-weight:600;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{label}</div>
                <div style="flex:1;height:8px;background:{_CARD_BORDER};border-radius:4px;overflow:hidden">
                    <div style="height:100%;width:{bar_w:.0f}%;background:{color};border-radius:4px"></div>
                </div>
                <div style="width:55px;text-align:right;font-size:11px;font-variant-numeric:tabular-nums;color:{color}">{_pct(val)}</div>
            </div>"""
        risk_bars = f"""<div class="card">
            <div style="font-size:12px;font-weight:600;margin-bottom:10px">Risk Contribution by Factor</div>
            {rows}
        </div>"""

    # Marginal contribution per asset
    mctr = data.get("marginal_ctr", {})
    mctr_html = ""
    if mctr:
        rows = ""
        for asset, val in sorted(mctr.items(), key=lambda x: x[1], reverse=True):
            color = _GREEN if val >= 0 else _RED
            rows += f"""<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {_CARD_BORDER}">
                <span style="font-weight:600;font-size:13px">{html.escape(asset)}</span>
                <span style="font-variant-numeric:tabular-nums;color:{color}">{_pct(val)}</span>
            </div>"""
        mctr_html = f"""<div class="card">
            <div style="font-size:12px;font-weight:600;margin-bottom:8px">Marginal Contribution per Asset</div>
            {rows}
        </div>"""

    # Info footer
    footer = ""
    if n_assets is not None:
        footer = f'<div style="font-size:11px;color:{_TEXT_MUTED};text-align:center;margin-top:4px">{n_assets} assets in portfolio</div>'

    return _wrap("Risk Dashboard", kpis + gauge + risk_bars + mctr_html + footer)


# ═══════════════════════════════════════════════════════
# 4. Portfolio Overview
# ═══════════════════════════════════════════════════════


def portfolio_overview(data: dict[str, Any]) -> str:
    """Render a portfolio list as visual cards.

    Expects: { total, portfolios: [{ id, name, assets, status, created_at }] }
    """
    portfolios = data.get("portfolios", [])
    total = data.get("total", len(portfolios))

    if not portfolios:
        return _wrap("Portfolios", '<div class="card">No portfolios found</div>')

    cards = ""
    for p in portfolios:
        name = html.escape(p.get("name") or "Untitled")
        status = p.get("status") or "unknown"
        assets = p.get("assets", [])
        created = (p.get("created_at") or "")[:10]

        status_color = _GREEN if status in ("active", "ready") else _YELLOW if status == "pending" else _TEXT_MUTED
        status_dot = f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{status_color};margin-right:4px"></span>'

        # Asset chips (show first 6)
        chips = ""
        for a in assets[:6]:
            ticker = a.get("ticker", a) if isinstance(a, dict) else str(a)
            weight = a.get("weight") if isinstance(a, dict) else None
            w_label = f" {weight:.0%}" if weight else ""
            chips += f'<span style="display:inline-block;padding:2px 8px;background:rgba(99,102,241,0.1);color:{_ACCENT_LIGHT};border-radius:4px;font-size:11px;font-weight:600;margin:2px">{html.escape(str(ticker))}{w_label}</span>'
        if len(assets) > 6:
            chips += f'<span style="display:inline-block;padding:2px 8px;color:{_TEXT_MUTED};font-size:11px">+{len(assets)-6} more</span>'

        cards += f"""<div class="card" style="cursor:pointer">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <div style="font-size:15px;font-weight:700">{name}</div>
                <div style="display:flex;align-items:center;font-size:11px;color:{status_color}">{status_dot}{html.escape(status)}</div>
            </div>
            <div style="margin-bottom:8px">{chips}</div>
            <div style="display:flex;justify-content:space-between;font-size:11px;color:{_TEXT_MUTED}">
                <span>{len(assets)} assets</span>
                <span>{html.escape(created)}</span>
            </div>
        </div>"""

    header = f'<div style="font-size:12px;color:{_TEXT_MUTED};margin-bottom:8px">{total} portfolio{"s" if total != 1 else ""}</div>'
    return _wrap("Portfolios", header + cards)


# ═══════════════════════════════════════════════════════
# 5. Training Progress
# ═══════════════════════════════════════════════════════


def training_progress(data: dict[str, Any]) -> str:
    """Render training progress as a visual card.

    Expects: { status, current_asset, completed_models, total_models,
               progress_percent, current_epoch, max_epochs, train_loss, val_loss }
    """
    status = data.get("status", "unknown")
    completed = data.get("completed_models", 0)
    total = data.get("total_models", 0)
    progress = data.get("progress_percent", 0) or 0
    current = data.get("current_asset", "")
    epoch = data.get("current_epoch")
    max_ep = data.get("max_epochs")
    t_loss = data.get("train_loss")
    v_loss = data.get("val_loss")

    status_color = _GREEN if status == "completed" else _BLUE if status in ("running", "training") else _YELLOW
    dot = f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{status_color}"></span>'

    # Progress bar
    bar = f"""<div style="margin:12px 0">
        <div style="height:8px;background:{_CARD_BORDER};border-radius:4px;overflow:hidden">
            <div style="height:100%;width:{min(progress, 100):.0f}%;background:{_ACCENT};border-radius:4px;transition:width 0.3s"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:11px;color:{_TEXT_MUTED};margin-top:3px">
            <span>{completed}/{total} models</span>
            <span>{progress:.0f}%</span>
        </div>
    </div>"""

    # Details
    details = ""
    if current:
        details += f'<div style="font-size:12px;margin-bottom:4px">Currently training: <span style="font-weight:700;color:{_ACCENT_LIGHT}">{html.escape(str(current))}</span></div>'
    if epoch is not None:
        details += f'<div style="font-size:12px;color:{_TEXT_MUTED}">Epoch {epoch}/{max_ep or "?"}'
        if t_loss is not None:
            details += f' &middot; Train loss: {t_loss:.6f}'
        if v_loss is not None:
            details += f' &middot; Val loss: {v_loss:.6f}'
        details += '</div>'

    body = f"""<div class="card">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
            {dot}
            <span style="font-size:14px;font-weight:700;text-transform:capitalize">{html.escape(status)}</span>
        </div>
        {bar}
        {details}
    </div>"""

    return _wrap("Training Progress", body)


# ── Flow Fan Chart ────────────────────────────────────────────


def flow_fan_chart(
    summary: dict[str, Any],
    horizon: int,
    constraints: list[dict] | None = None,
) -> str | None:
    """Render inline SVG fan charts for Flow path generation results.

    Shows per-asset percentile bands (p5/p25/p50/p75/p95) as a fan chart.
    Only displays target assets (not conditioning factors). Returns None if
    no target features have timeseries data.
    """
    # Collect features with timeseries data, preferring target assets
    all_with_ts: list[tuple[str, dict]] = []
    targets_only: list[tuple[str, dict]] = []
    for name, data in summary.items():
        if not isinstance(data, dict):
            continue
        ts = data.get('timeseries')
        if not ts or 'p50' not in ts:
            continue
        all_with_ts.append((name, data))
        if data.get('feature_type') == 'target':
            targets_only.append((name, data))

    # Prefer target assets; fall back to all features if none marked as target
    panels = targets_only if targets_only else all_with_ts
    if not panels:
        return None

    # Limit to 6 panels
    panels = panels[:6]

    # Build constraint lookup: feature_name -> list of {lower, upper}
    constraint_map: dict[str, list[dict]] = {}
    for c in (constraints or []):
        fn = c.get('feature_name', '')
        constraint_map.setdefault(fn, []).append(c)

    # SVG dimensions
    svg_w, svg_h = 380, 180
    pad_l, pad_r, pad_t, pad_b = 55, 15, 30, 25  # left, right, top, bottom
    plot_w = svg_w - pad_l - pad_r
    plot_h = svg_h - pad_t - pad_b

    cards_html = []
    for name, data in panels:
        ts = data['timeseries']
        p5 = ts['p5']
        p25 = ts['p25']
        p50 = ts['p50']
        p75 = ts['p75']
        p95 = ts['p95']
        last_price = data.get('last_price', 0)
        mean_ret = data.get('mean_return', 0)
        n_pts = len(p50)

        # Y range: include all bands + last_price
        all_vals = p5 + p95 + [last_price]
        y_min = min(all_vals)
        y_max = max(all_vals)
        y_pad = (y_max - y_min) * 0.08 or 1.0
        y_min -= y_pad
        y_max += y_pad
        y_range = y_max - y_min if y_max != y_min else 1.0

        def x(i: int) -> float:
            return pad_l + (i / max(n_pts - 1, 1)) * plot_w

        def y(v: float) -> float:
            return pad_t + (1 - (v - y_min) / y_range) * plot_h

        # Build polygon points for bands
        def band_points(lo: list, hi: list) -> str:
            fwd = ' '.join(f'{x(i):.1f},{y(hi[i]):.1f}' for i in range(n_pts))
            rev = ' '.join(f'{x(i):.1f},{y(lo[i]):.1f}' for i in range(n_pts - 1, -1, -1))
            return f'{fwd} {rev}'

        # Median line points
        med_pts = ' '.join(f'{x(i):.1f},{y(p50[i]):.1f}' for i in range(n_pts))

        # Return color
        ret_color = _GREEN if mean_ret >= 0 else _RED
        ret_str = f'+{mean_ret:.1%}' if mean_ret >= 0 else f'{mean_ret:.1%}'

        # Y-axis labels (3 ticks)
        y_mid = (y_min + y_max) / 2
        y_ticks = [
            (y_max, f'${y_max:,.0f}' if y_max > 10 else f'{y_max:.2f}'),
            (y_mid, f'${y_mid:,.0f}' if y_mid > 10 else f'{y_mid:.2f}'),
            (y_min, f'${y_min:,.0f}' if y_min > 10 else f'{y_min:.2f}'),
        ]
        y_tick_svg = ''
        for val, label in y_ticks:
            yy = y(val)
            y_tick_svg += (
                f'<text x="{pad_l - 6}" y="{yy + 3:.1f}" '
                f'text-anchor="end" fill="{_TEXT_MUTED}" font-size="9">{label}</text>'
            )
            # Grid line
            y_tick_svg += (
                f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + plot_w}" y2="{yy:.1f}" '
                f'stroke="{_CARD_BORDER}" stroke-width="0.5"/>'
            )

        # X-axis labels
        x_tick_svg = (
            f'<text x="{pad_l}" y="{svg_h - 5}" fill="{_TEXT_MUTED}" font-size="9">Day 0</text>'
            f'<text x="{pad_l + plot_w}" y="{svg_h - 5}" text-anchor="end" '
            f'fill="{_TEXT_MUTED}" font-size="9">Day {horizon}</text>'
        )

        # Last price reference line
        lp_y = y(last_price)
        lp_svg = (
            f'<line x1="{pad_l}" y1="{lp_y:.1f}" x2="{pad_l + plot_w}" y2="{lp_y:.1f}" '
            f'stroke="{_TEXT_MUTED}" stroke-width="0.8" stroke-dasharray="4,3"/>'
            f'<text x="{pad_l + plot_w + 2}" y="{lp_y + 3:.1f}" fill="{_TEXT_MUTED}" '
            f'font-size="8">now</text>'
        )

        # Constraint bounds (red dashed lines)
        constraint_svg = ''
        for c in constraint_map.get(name, []):
            for bound_key, label in [('lower', 'min'), ('upper', 'max')]:
                bv = c.get(bound_key)
                if bv is not None:
                    by = y(float(bv))
                    if pad_t <= by <= pad_t + plot_h:
                        constraint_svg += (
                            f'<line x1="{pad_l}" y1="{by:.1f}" '
                            f'x2="{pad_l + plot_w}" y2="{by:.1f}" '
                            f'stroke="{_RED}" stroke-width="1" stroke-dasharray="5,3"/>'
                        )

        svg = f'''<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}"
             xmlns="http://www.w3.org/2000/svg" style="display:block">
          {y_tick_svg}
          {x_tick_svg}
          {lp_svg}
          <polygon points="{band_points(p5, p95)}"
                   fill="{_ACCENT}" fill-opacity="0.12" stroke="none"/>
          <polygon points="{band_points(p25, p75)}"
                   fill="{_ACCENT}" fill-opacity="0.25" stroke="none"/>
          <polyline points="{med_pts}"
                    fill="none" stroke="{_ACCENT}" stroke-width="2"/>
          {constraint_svg}
        </svg>'''

        title_html = (
            f'<div style="font-size:13px;font-weight:600;margin-bottom:4px">'
            f'{html.escape(name)} '
            f'<span style="color:{ret_color};font-weight:700">{ret_str}</span>'
            f'</div>'
        )

        cards_html.append(f'<div class="card" style="padding:12px">{title_html}{svg}</div>')

    # Legend
    legend = (
        f'<div style="display:flex;gap:16px;font-size:10px;color:{_TEXT_MUTED};margin-top:4px;margin-bottom:8px">'
        f'<span><span style="display:inline-block;width:20px;height:8px;'
        f'background:{_ACCENT};opacity:0.12;vertical-align:middle;margin-right:4px;border-radius:2px"></span>p5–p95</span>'
        f'<span><span style="display:inline-block;width:20px;height:8px;'
        f'background:{_ACCENT};opacity:0.35;vertical-align:middle;margin-right:4px;border-radius:2px"></span>p25–p75</span>'
        f'<span><span style="display:inline-block;width:20px;height:2px;'
        f'background:{_ACCENT};vertical-align:middle;margin-right:4px"></span>median</span>'
        f'<span><span style="display:inline-block;width:20px;height:1px;border-top:1px dashed {_TEXT_MUTED};'
        f'vertical-align:middle;margin-right:4px"></span>current</span>'
        f'</div>'
    )

    # Grid layout: 2 columns
    ncols = 2 if len(cards_html) > 1 else 1
    grid = (
        f'<div style="display:grid;grid-template-columns:repeat({ncols},1fr);gap:10px">'
        + ''.join(cards_html)
        + '</div>'
    )

    body = legend + grid
    width = 420 * ncols + 20
    return _wrap("Flow Path Distribution", body, width=width)
