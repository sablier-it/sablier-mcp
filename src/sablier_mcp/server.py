"""
Sablier MCP Server

Gives AI agents the ability to perform regime-dependent factor modeling,
qualitative analysis, portfolio risk testing, and return simulation
through the Sablier platform.

Supports both local (stdio) and remote (streamable-http) transport.
Remote mode uses OAuth 2.0 — Claude Desktop opens a browser for login.
"""

import asyncio
import json
import logging
import os
import re
import urllib.parse
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.types import (
    EmbeddedResource,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from pydantic import Field

from sablier_mcp.auth import SablierOAuthProvider, current_sablier_token, login_page, _pending_auth_redirect
from sablier_mcp.client import SablierClient, SablierAPIError
from sablier_mcp.widgets import (
    betas_heatmap,
    grain_score_card,
    portfolio_overview,
)


logger = logging.getLogger("sablier-mcp")

# ══════════════════════════════════════════════════
# Server setup
# ══════════════════════════════════════════════════

_transport = os.getenv("MCP_TRANSPORT", "streamable-http")
_oauth_provider: SablierOAuthProvider | None = None

if _transport != "stdio":
    # Remote mode: enable OAuth
    _oauth_provider = SablierOAuthProvider()
    _port = int(os.getenv("PORT", os.getenv("MCP_PORT", "8000")))
    _issuer_url = os.getenv("MCP_ISSUER_URL", f"http://localhost:{_port}")
    server = FastMCP(
        name="Sablier",
        host="0.0.0.0",
        port=_port,
        auth_server_provider=_oauth_provider,
        auth=AuthSettings(
            issuer_url=_issuer_url,
            resource_server_url=_issuer_url,
            client_registration_options=ClientRegistrationOptions(enabled=True),
            required_scopes=[],
        ),
    )
else:
    # Local/stdio mode: no auth (uses SABLIER_API_KEY from env)
    server = FastMCP(name="Sablier")


# ── Custom routes (remote mode only) ─────────────

if _oauth_provider is not None:
    from starlette.responses import Response

    @server.custom_route("/login", methods=["GET", "POST"])
    async def _login_handler(request) -> Response:
        return await login_page(request, _oauth_provider)


# ══════════════════════════════════════════════════
# Client management
# ══════════════════════════════════════════════════

_stdio_client: SablierClient | None = None
_token_clients: dict[str, SablierClient] = {}


def get_client() -> SablierClient:
    """Get a SablierClient for the current user.

    In remote (OAuth) mode: uses the API key from the authenticated session.
    In stdio mode: uses SABLIER_API_KEY from the environment.
    """
    global _stdio_client

    # Check for API key (set by load_access_token in auth middleware)
    token = current_sablier_token.get(None)
    if token:
        if token not in _token_clients:
            _token_clients[token] = SablierClient.from_token(token)
        return _token_clients[token]

    # Fallback: stdio mode with API key
    if _stdio_client is None:
        _stdio_client = SablierClient()
    return _stdio_client


# ══════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════


def _fmt(data) -> str:
    """Format API response as readable JSON."""
    return json.dumps(data, indent=2, default=str)


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _validate_uuid(value: str, label: str = "ID") -> str | None:
    """Return an error string if value is not a valid UUID, else None."""
    if not _UUID_RE.match(value):
        return f"Error: '{value}' is not a valid {label}. Expected a UUID like '8f3a1b2c-...'. Use the relevant list tool to find valid IDs."
    return None


def _api_error(e: SablierAPIError) -> str:
    """Convert an API error into a friendly message for the LLM."""
    if e.status_code == 404:
        return f"Error: Not found — {e.detail}"
    if e.status_code == 422:
        return f"Error: Invalid input — {e.detail}"
    return f"Error: API returned {e.status_code} — {e.detail}"


def _with_widget(text: str, html: str) -> list:
    """Return both a text summary (for the LLM) and an HTML widget (for visual clients)."""
    return [
        TextContent(type="text", text=text),
        EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri=f"data:text/html,{urllib.parse.quote(html)}",
                mimeType="text/html",
                text=text,
            ),
        ),
    ]


def _portfolio_tickers(portfolio: dict) -> list[str]:
    """Extract ticker symbols from a portfolio response.

    The API returns asset_names as {display_name: ticker} and weights as
    {display_name: weight}.  Fall back to an 'assets' list if present.
    """
    asset_names = portfolio.get("asset_names", {})
    if asset_names:
        return list(asset_names.values())
    # Fallback for list-style responses
    assets = portfolio.get("assets", [])
    return [a.get("ticker") for a in assets if a.get("ticker")]


def _flatten_betas(results: dict) -> dict:
    """Flatten the nested betas API response into a widget-friendly format.

    The API returns per_asset_results keyed by model UUID, with nested dicts
    for linear_betas ({display_name: {factor: float}}), alpha, residual_std.
    Factor stats (means, stds) are returned as parallel lists.

    This flattens everything so:
    - assets is keyed by display name (e.g. "Apple Inc.")
    - linear_betas is flat {factor: float}
    - alpha / residual_std are plain floats
    - factor_means / factor_stds are dicts {factor: float}
    """
    features = results.get("conditioning_features", [])
    factor_stats = results.get("factor_stats", {})
    factor_names = factor_stats.get("factor_names", features)

    # Flatten per-asset data
    per_asset = results.get("per_asset_results", {})
    assets = {}
    for _model_id, data in per_asset.items():
        lb = data.get("linear_betas", {})
        alpha_dict = data.get("alpha", {}) or {}
        resid_dict = data.get("residual_std", {}) or {}

        # Each model maps to one asset by display name
        for display_name, betas in lb.items():
            assets[display_name] = {
                "status": data.get("status"),
                "linear_betas": betas if isinstance(betas, dict) else {},
                "alpha": alpha_dict.get(display_name) if isinstance(alpha_dict, dict) else alpha_dict,
                "residual_std": resid_dict.get(display_name) if isinstance(resid_dict, dict) else resid_dict,
            }

    # Convert parallel lists to dicts
    means_list = factor_stats.get("factor_means", [])
    stds_list = factor_stats.get("factor_stds", [])
    means_raw_list = factor_stats.get("factor_means_raw", [])
    stds_raw_list = factor_stats.get("factor_stds_raw", [])

    factor_means = dict(zip(factor_names, means_list)) if means_list else {}
    factor_stds = dict(zip(factor_names, stds_list)) if stds_list else {}
    factor_means_raw = dict(zip(factor_names, means_raw_list)) if means_raw_list else {}
    factor_stds_raw = dict(zip(factor_names, stds_raw_list)) if stds_raw_list else {}

    return {
        "conditioning_features": features,
        "assets": assets,
        "factor_means_raw": factor_means_raw,
        "factor_stds_raw": factor_stds_raw,
        "factor_means": factor_means,
        "factor_stds": factor_stds,
    }


async def _ensure_portfolio(
    portfolio_id: str | None,
    tickers: list[str] | None,
    weights: list[float] | None,
) -> tuple[dict, str | None]:
    """Resolve or auto-create a portfolio.

    Returns (portfolio_dict, error_string).  If error_string is not None
    the caller should return it immediately.
    """
    client = get_client()

    if portfolio_id:
        if err := _validate_uuid(portfolio_id, "portfolio_id"):
            return {}, err
        portfolio = await client.get_portfolio(portfolio_id)
        if not _portfolio_tickers(portfolio):
            return {}, "Error: portfolio has no assets."
        return portfolio, None

    if not tickers:
        return {}, (
            "Error: provide either portfolio_id (from create_portfolio / list_portfolios) "
            "or tickers (e.g. ['AAPL', 'MSFT'])."
        )

    # Auto-assign weights: explicit > equal
    if not weights:
        weights = [1.0 / len(tickers)] * len(tickers)
    if len(weights) != len(tickers):
        return {}, "Error: tickers and weights must have the same length."

    # Auto-create portfolio
    name = ", ".join(tickers)
    assets = [{"ticker": t, "weight": w} for t, w in zip(tickers, weights)]
    portfolio = await client.create_portfolio(name, assets)
    return portfolio, None


_NOT_LOGGED_IN = (
    "Error: Not authenticated. "
    "Set the SABLIER_API_KEY environment variable to your Sablier API key. "
    "You can create one at https://app.sablier.co/settings/api-keys."
)


def _require_auth() -> str | None:
    """Return an error string if not authenticated, else None.

    In remote (OAuth) mode the SDK middleware handles auth, so this
    always passes.  In stdio mode it checks SABLIER_API_KEY.
    """
    if current_sablier_token.get(None):
        return None  # OAuth-authenticated
    client = get_client()
    if not client.is_authenticated:
        return _NOT_LOGGED_IN
    return None


# ══════════════════════════════════════════════════
# Search & Discovery
# ══════════════════════════════════════════════════


@server.tool(
    name="search_features",
    description="Search for tickers (stocks, ETFs) and market indicators (VIX, DXY, rates). Returns matching symbols with descriptions. Use this to validate tickers before creating a portfolio.",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def search_features(
    query: Annotated[str, Field(description="Search term (e.g. 'AAPL', 'technology', 'volatility', 'gold')")],
    is_asset: Annotated[bool | None, Field(description="If True, only assets. If False, only indicators.", default=None)] = None,
    limit: Annotated[int, Field(description="Max results (default 20)", default=20)] = 20,
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        results = await client.search_features(query, is_asset=is_asset, limit=limit)
        summary = []
        for f in results:
            entry = {
                "ticker": f.get("ticker"),
                "name": f.get("display_name"),
                "source": f.get("source"),
                "category": f.get("category"),
                "is_asset": f.get("is_asset"),
            }
            if f.get("description"):
                entry["description"] = f["description"]
            summary.append(entry)
        return _fmt(summary)
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Portfolios
# ══════════════════════════════════════════════════


@server.tool(
    name="list_portfolios",
    description="List the user's existing portfolios with names, IDs, asset compositions, and status.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_portfolios(
    limit: Annotated[int, Field(description="Max portfolios to return", default=50)] = 50,
) -> list:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        result = await client.list_portfolios(limit=limit)
        portfolios = result.get("portfolios", [])
        summary = []
        for p in portfolios:
            summary.append({
                "id": p["id"],
                "name": p["name"],
                "tickers": _portfolio_tickers(p),
                "weights": p.get("weights", {}),
                "capital": p.get("capital", 100_000),
                "description": p.get("description", ""),
                "created_at": p.get("created_at"),
            })
        data = {"total": result.get("total", len(summary)), "portfolios": summary}
        return _with_widget(_fmt(data), portfolio_overview(data))
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_portfolio",
    description="Get detailed information about a specific portfolio including assets, weights, and associated feature sets. Use the portfolio ID from list_portfolios.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_portfolio(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        result = await client.get_portfolio(portfolio_id)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="create_portfolio",
    description="Create a new portfolio from tickers and weights. Weights must sum to 1.0. For a single asset, use weight 1.0.",
)
async def create_portfolio(
    name: Annotated[str, Field(description="Portfolio name (e.g. 'Tech Portfolio')")],
    tickers: Annotated[list[str], Field(description="Ticker symbols (e.g. ['AAPL', 'MSFT', 'NVDA'])")],
    weights: Annotated[list[float], Field(description="Corresponding weights summing to 1.0 (e.g. [0.4, 0.3, 0.3])")],
    capital: Annotated[float, Field(description="Total capital allocation in USD (default $100,000)", default=100_000.0)] = 100_000.0,
    description: Annotated[str, Field(description="Optional description", default="")] = "",
) -> str:
    if err := _require_auth():
        return err
    if len(tickers) != len(weights):
        return "Error: tickers and weights must have the same length"

    try:
        assets = [{"ticker": t, "weight": w} for t, w in zip(tickers, weights)]
        client = get_client()
        result = await client.create_portfolio(name, assets, description=description or None, capital=capital)
        portfolio_id = result["id"]
        return _fmt({
            "portfolio_id": portfolio_id,
            "name": result["name"],
            "tickers": _portfolio_tickers(result),
            "weights": result.get("weights", {}),
            "capital": result.get("capital", capital),
            "target_set_id": result.get("target_set_id"),
            "message": (
                f"Portfolio created (ID: {portfolio_id}). "
                "Next: use this portfolio_id with analyze_quantitative "
                "and/or analyze_qualitative (thematic) for a full risk picture."
            ),
        })
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="update_portfolio",
    description=(
        "Update an existing portfolio. Can change name, description, weights, and/or capital. "
        "Only pass the fields you want to update — omitted fields stay unchanged. "
        "Weights must sum to 1.0 if provided."
    ),
)
async def update_portfolio(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
    name: Annotated[str | None, Field(description="New portfolio name", default=None)] = None,
    description: Annotated[str | None, Field(description="New description", default=None)] = None,
    weights: Annotated[dict[str, float] | None, Field(description="New weights by ticker (e.g. {'AAPL': 0.5, 'MSFT': 0.5}). Must sum to 1.0.", default=None)] = None,
    capital: Annotated[float | None, Field(description="New capital allocation in USD", default=None)] = None,
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    fields: dict = {}
    if name is not None:
        fields["name"] = name
    if description is not None:
        fields["description"] = description
    if weights is not None:
        fields["weights"] = weights
    if capital is not None:
        fields["capital"] = capital
    if not fields:
        return "Error: provide at least one field to update (name, description, weights, capital)."
    try:
        client = get_client()
        result = await client.update_portfolio(portfolio_id, **fields)
        return _fmt({
            "portfolio_id": result["id"],
            "name": result["name"],
            "weights": result.get("weights", {}),
            "capital": result.get("capital"),
            "message": "Portfolio updated successfully.",
        })
    except SablierAPIError as e:
        return _api_error(e)



@server.tool(
    name="get_portfolio_value",
    description="Get the current live value of a portfolio: total value, P&L, and per-position breakdown.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_portfolio_value(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        result = await client.get_portfolio_live_value(portfolio_id)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_portfolio_analytics",
    description="Get portfolio analytics: Sharpe ratio, volatility, expected return, max drawdown, and beta. Supports timeframes: 1W, 1M, 1Y, 2Y, 5Y.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_portfolio_analytics(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
    timeframe: Annotated[str, Field(description="Timeframe: 1W, 1M, 1Y, 2Y, or 5Y (default 1Y)", default="1Y")] = "1Y",
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        result = await client.get_portfolio_analytics(portfolio_id, timeframe=timeframe)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_asset_profiles",
    description="Get asset classification for a portfolio: sector, industry, country, exchange, and asset type per holding.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_asset_profiles(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        result = await client.get_asset_profiles(portfolio_id)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Qualitative Analysis (GRAIN)
# ══════════════════════════════════════════════════


@server.tool(
    name="analyze_qualitative",
    description=(
        "Run qualitative (GRAIN) analysis: scans SEC filings and earnings calls to score company exposure "
        "to themes (0-100). Supports predefined themes ('AI exposure') or custom themes. "
        "Pass either portfolio_id or tickers directly."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def analyze_qualitative(
    themes: Annotated[list[str], Field(description="Themes to score (e.g. ['AI exposure', 'Saudi Arabia risk', 'debt levels'])")],
    portfolio_id: Annotated[str | None, Field(description="UUID of an existing portfolio. If omitted, provide tickers instead.", default=None)] = None,
    tickers: Annotated[list[str] | None, Field(description="Tickers to analyze (e.g. ['AAPL', 'MSFT']). Auto-creates a portfolio if portfolio_id is not given.", default=None)] = None,
    source_types: Annotated[list[str] | None, Field(description="Document types: ['10-K', '10-Q', 'earnings_call']. Default: all.", default=None)] = None,
    min_year: Annotated[int | None, Field(description="Earliest filing year to include", default=None)] = None,
    max_year: Annotated[int | None, Field(description="Latest filing year to include", default=None)] = None,
) -> list | str:
    if err := _require_auth():
        return err
    try:
        portfolio, err = await _ensure_portfolio(portfolio_id, tickers, None)
        if err:
            return err
        resolved_tickers = _portfolio_tickers(portfolio)

        # Portfolio stores weights keyed by display name (e.g. "Apple Inc.": 0.2)
        # but GRAIN backend looks up by ticker (e.g. "AAPL"). Remap using asset_names.
        raw_weights = portfolio.get("weights", {})
        asset_names_map = portfolio.get("asset_names", {})  # {display_name: ticker}
        if raw_weights and asset_names_map:
            weights_dict = {
                asset_names_map.get(display, display): w
                for display, w in raw_weights.items()
            }
        else:
            weights_dict = raw_weights
        p_id = portfolio.get("id")
        p_name = portfolio.get("name")

        client = get_client()

        # Auto-generate keywords for each theme.  The backend returns
        # is_predefined=True for library themes (keywords already built-in)
        # and is_predefined=False for custom themes (keywords generated via LLM).
        # We only need to send custom_keywords for non-predefined themes.
        custom_keywords: dict[str, list[str]] = {}
        resolved_themes: list[str] = []
        for theme in themes:
            kw_result = await client.generate_grain_keywords(theme)
            resolved_themes.append(kw_result.get("theme", theme))
            if not kw_result.get("is_predefined", False):
                keywords = kw_result.get("keywords", [])[:10]
                if keywords:
                    custom_keywords[kw_result.get("theme", theme)] = keywords

        job = await client.start_grain_analysis(
            tickers=resolved_tickers,
            themes=resolved_themes,
            source_types=source_types,
            min_year=min_year,
            max_year=max_year,
            weights=weights_dict or None,
            portfolio_id=p_id,
            portfolio_name=p_name,
            custom_keywords=custom_keywords or None,
        )
        job_id = job.get("job_id")
        if not job_id:
            return "Error: GRAIN analysis did not return a job_id."

        # Poll until complete
        result = await client.poll_grain_job(job_id)

        if result.get("status") == "failed":
            return f"Analysis failed: {result.get('error_message', 'Unknown error')}"

        if result.get("status") != "completed":
            return _fmt({
                "status": result.get("status"),
                "job_id": job_id,
                "message": "Analysis is still running (this can take a few minutes). Please try again shortly.",
            })

        # Summarize results
        raw_results = result.get("results", {})
        themes_data = raw_results.get("results", []) or raw_results.get("themes", [])
        summary = {
            "status": "completed",
            "processing_time_seconds": result.get("processing_time_seconds"),
            "themes": [],
        }
        for theme in themes_data:
            theme_summary = {
                "theme": theme.get("theme"),
                "display_name": theme.get("display_name"),
                "portfolio_score": theme.get("portfolio_score") or theme.get("max_exposure", {}).get("score"),
                "portfolio_tier": theme.get("portfolio_tier"),
                "direction": theme.get("direction"),
                "exposure_at_risk": theme.get("exposure_at_risk"),
                "confidence": theme.get("confidence"),
                "top_contributors": theme.get("top_contributors"),
                "ticker_scores": [],
            }
            for ts in theme.get("ticker_scores", []):
                ticker_entry = {
                    "ticker": ts.get("ticker"),
                    "score": ts.get("score"),
                    "tier": ts.get("tier") or ts.get("tier_name"),
                    "direction": ts.get("direction"),
                    "confidence": ts.get("confidence"),
                    "top_evidence": [],
                }
                for ev in (ts.get("evidence") or [])[:3]:
                    ticker_entry["top_evidence"].append({
                        "passage": (ev.get("passage") or "")[:500],
                        "source": ev.get("source"),
                        "source_type": ev.get("source_type"),
                        "fiscal_period": ev.get("fiscal_period"),
                        "filing_date": ev.get("filing_date"),
                        "why_relevant": ev.get("why_relevant"),
                    })
                theme_summary["ticker_scores"].append(ticker_entry)
            summary["themes"].append(theme_summary)

        return _with_widget(_fmt(summary), grain_score_card(summary))
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_themes",
    description="Browse the GRAIN theme library. Returns predefined themes with names, descriptions, keywords, and categories.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_themes() -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        themes = await client.list_grain_themes()
        summary = []
        for t in themes:
            summary.append({
                "id": t.get("id") or t.get("name"),
                "display_name": t.get("display_name"),
                "description": t.get("description"),
                "category": t.get("category"),
                "keywords": t.get("keywords", [])[:5],
            })
        return _fmt(summary)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_grain_analyses",
    description="List past qualitative (GRAIN) analyses. Optionally filter by portfolio_id.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_grain_analyses(
    portfolio_id: Annotated[str | None, Field(description="Optional: filter by portfolio UUID", default=None)] = None,
) -> str:
    if err := _require_auth():
        return err
    if portfolio_id and (err := _validate_uuid(portfolio_id, "portfolio_id")):
        return err
    try:
        client = get_client()
        analyses = await client.list_grain_analyses(portfolio_id=portfolio_id)
        return _fmt(analyses)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_grain_analysis",
    description="Load a saved GRAIN analysis with full results: theme scores, per-ticker breakdown, and evidence passages.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_grain_analysis(
    analysis_id: Annotated[str, Field(description="The analysis UUID from list_grain_analyses")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(analysis_id, "analysis_id"):
        return err
    try:
        client = get_client()
        result = await client.get_grain_analysis(analysis_id)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)



# ══════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════


@server.tool(
    name="list_model_groups",
    description="List all existing analyses. Each analysis ties a portfolio to a set of market drivers and contains per-asset models. Use this to find previous work, check if training is done, or retrieve simulation results for risk testing and scenarios.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_model_groups() -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        groups = await client.list_model_groups()
        summary = []
        for g in groups:
            models = g.get("models", [])
            summary.append({
                "id": g["id"],
                "name": g.get("name"),
                "conditioning_set": g.get("conditioning_set_name"),
                "status": g.get("status"),
                "model_count": len(models),
                "models": [
                    {
                        "asset": m.get("asset_id"),
                        "status": m.get("status"),
                        "model_id": m.get("model_id"),
                    }
                    for m in models
                ],
                "latest_simulation": g.get("latest_simulation"),
            })
        return _fmt(summary)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_feature_set_templates",
    description="Browse pre-built sets of market drivers (e.g. interest rates, volatility, commodities). Returns template names, factors, and conditioning_set_id needed by analyze_quantitative.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_feature_set_templates() -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        templates = await client.list_feature_set_templates()
        return _fmt(templates)
    except SablierAPIError as e:
        return _api_error(e)






@server.tool(
    name="simulate_betas",
    description=(
        "Compute how each asset in a trained model group responds to the chosen market drivers. "
        "Requires model_group_id (status must be 'trained'). Synchronous — returns results directly. "
        "Returns per-asset factor exposures, simulation_batch_id, and current factor levels (factor_means)."
    ),
    annotations=ToolAnnotations(openWorldHint=True),
)
async def simulate_betas(
    model_group_id: Annotated[str, Field(description="UUID of the trained model group (from analyze_quantitative or list_model_groups)")],
    lookback_days: Annotated[int | None, Field(description="Historical lookback window in trading days", default=None)] = None,
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        results = await client.simulate_betas_batch(
            model_group_id=model_group_id,
            historical_lookback_days=lookback_days,
        )
        sim_batch_id = results.get("simulation_batch_id")
        if not sim_batch_id:
            return "Error: betas simulation did not return a simulation_batch_id."

        if not results.get("all_completed"):
            return _fmt({
                "simulation_batch_id": sim_batch_id,
                "status": results.get("status", "unknown"),
                "message": "Simulation did not complete successfully.",
            })

        flat = _flatten_betas(results)
        summary = {
            "simulation_batch_id": sim_batch_id,
            **flat,
        }

        return _with_widget(_fmt(summary), betas_heatmap(summary))
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="run_model_validation",
    description=(
        "Trigger validation for all models in a trained model group. "
        "Computes per-asset quality metrics: R², autocorrelation, regime sensitivity, pass rate, "
        "and quality badge (EXCELLENT/GOOD/ACCEPTABLE/POOR). "
        "Use this after analyze_quantitative to assess model reliability before running scenarios. "
        "For cached results from a previous run, use get_model_validation instead."
    ),
    annotations=ToolAnnotations(openWorldHint=True),
)
async def run_model_validation(
    model_group_id: Annotated[str, Field(description="UUID of the model group (from analyze_quantitative or list_model_groups)")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        batch = await client.trigger_batch_validation(model_group_id)

        validation_batch_id = batch.get("validation_batch_id")
        if not validation_batch_id:
            return "Error: validation did not return a validation_batch_id."

        # Moment validation is synchronous — fetch full results
        results = await client.get_batch_validation_results(validation_batch_id)

        return _format_validation_results(model_group_id, results)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_model_validation",
    description=(
        "Get cached validation metrics from the latest validation run for a model group. "
        "Returns per-asset model quality: R², autocorrelation, regime sensitivity, pass rate, "
        "and quality badge. To trigger a fresh validation, use run_model_validation instead."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_model_validation(
    model_group_id: Annotated[str, Field(description="UUID of the model group (from analyze_quantitative or list_model_groups)")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        result = await client.get_latest_group_validation(model_group_id)
        return _format_validation_results(model_group_id, result)
    except SablierAPIError as e:
        return _api_error(e)


def _format_validation_results(model_group_id: str, result: dict) -> str:
    per_asset = result.get("per_asset", [])
    formatted = []
    for asset in per_asset:
        entry = {
            "asset": asset.get("asset_id") or asset.get("model_name"),
            "quality": asset.get("quality"),
            "pass_rate": asset.get("pass_rate"),
            "n_passed": asset.get("n_passed"),
            "n_tests": asset.get("n_tests"),
            "r_squared": asset.get("r_squared"),
            "r_squared_train": asset.get("r_squared_train"),
            "autocorrelation": asset.get("autocorrelation"),
            "regime_sensitivity": asset.get("regime_sensitivity"),
        }
        formatted.append(entry)

    return _fmt({
        "model_group_id": model_group_id,
        "assets": formatted,
    })


# ══════════════════════════════════════════════════
# Return Simulation
# ══════════════════════════════════════════════════


@server.tool(
    name="simulate_returns",
    description=(
        "Simulate return distributions under a what-if scenario. "
        "Requires simulation_batch_id from analyze_quantitative or simulate_betas. "
        "IMPORTANT: factor values must be absolute price levels (use factor_means_raw as baseline). "
        "Example: to shock US Market -5% from 682.39, pass {'US Market': 648.27}. "
        "Returns per-asset expected returns and risk metrics (VaR, ES, volatility)."
    ),
    annotations=ToolAnnotations(openWorldHint=True),
)
async def simulate_returns(
    simulation_batch_id: Annotated[str, Field(description="From analyze_quantitative or simulate_betas results")],
    factors: Annotated[dict[str, float], Field(
        description=(
            "Absolute target price levels for each factor. "
            "Use factor_means_raw from betas output as current levels, "
            "then compute target (e.g. US Market -5%: 682.39 * 0.95 = 648.27). "
            "Keys must match conditioning_features exactly."
        )
    )],
    n_samples: Annotated[int, Field(description="Number of Monte Carlo samples", default=5000)] = 5000,
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(simulation_batch_id, "simulation_batch_id"):
        return err
    try:
        client = get_client()
        results = await client.simulate_returns_batch(
            simulation_batch_id=simulation_batch_id,
            factors=factors,
            n_samples=n_samples,
        )
        returns_batch_id = results.get("returns_batch_id")
        if not returns_batch_id:
            return "Error: returns simulation did not return a returns_batch_id."

        if not results.get("all_completed"):
            return _fmt({
                "returns_batch_id": returns_batch_id,
                "status": results.get("status", "unknown"),
                "message": "Returns simulation did not complete successfully.",
            })

        # Format per-asset risk metrics from the summary field
        per_asset = results.get("per_asset_results", {})
        formatted = {}
        for asset_id, data in per_asset.items():
            if data.get("status") != "completed":
                formatted[asset_id] = {"status": data.get("status")}
                continue
            summary = data.get("summary", [{}])
            s = summary[0] if summary else {}
            formatted[asset_id] = {
                "expected_return": s.get("expected"),
                "mean": s.get("mean"),
                "std": s.get("std"),
                "VaR_95": s.get("VaR_95"),
                "ES_95": s.get("ES_95"),
                "p05": s.get("p05"),
                "p25": s.get("p25"),
                "p50": s.get("p50"),
                "p75": s.get("p75"),
                "p95": s.get("p95"),
            }

        return _fmt({
            "returns_batch_id": returns_batch_id,
            "factors_used": next(iter(per_asset.values()), {}).get("factors_used", {}),
            "per_asset_risk": formatted,
        })
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Scenarios
# ══════════════════════════════════════════════════


@server.tool(
    name="create_scenario",
    description="Save a named what-if scenario for reuse. Factor spec types: 'fixed' (exact value), 'percentile' (historical), 'shock' (std dev shift).",
)
async def create_scenario(
    model_id: Annotated[str, Field(description="UUID of the model this scenario applies to")],
    name: Annotated[str, Field(description="Scenario name (e.g. 'Recession', 'Tech Bubble')")],
    factor_values: Annotated[dict[str, dict], Field(description="Factor specs (e.g. {'VIX': {'type': 'fixed', 'value': 35}})")],
    description: Annotated[str, Field(description="Optional description", default="")] = "",
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_id, "model_id"):
        return err
    try:
        client = get_client()
        result = await client.create_scenario(
            model_id=model_id,
            name=name,
            specs=factor_values,
            description=description or None,
        )
        return _fmt({
            "id": result["id"],
            "name": result["name"],
            "specs": result.get("specs"),
            "message": "Scenario created. Use simulate_returns with the factor values to sample returns.",
        })
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_scenarios",
    description="List saved scenarios. Optionally filter by model_id from list_model_groups.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_scenarios(
    model_id: Annotated[str | None, Field(description="Optional model UUID to filter by", default=None)] = None,
) -> str:
    if err := _require_auth():
        return err
    if model_id:
        if err := _validate_uuid(model_id, "model_id"):
            return err
    try:
        client = get_client()
        result = await client.list_scenarios(model_id=model_id)
        scenarios = result.get("scenarios", [])
        summary = []
        for s in scenarios:
            summary.append({
                "id": s["id"],
                "name": s.get("name"),
                "description": s.get("description"),
                "specs": s.get("specs"),
                "model_id": s.get("model_id"),
                "created_at": s.get("created_at"),
            })
        return _fmt({"total": result.get("total", len(summary)), "scenarios": summary})
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Full Analysis Orchestrator
# ══════════════════════════════════════════════════


@server.tool(
    name="analyze_quantitative",
    description=(
        "One-shot quantitative analysis: builds factor models, trains, and computes asset sensitivities to "
        "market drivers. Pass either portfolio_id or tickers directly (auto-creates portfolio with equal weights). "
        "Requires conditioning_set_id from list_feature_set_templates. "
        "Returns factor exposures, simulation_batch_id for use with simulate_returns."
    ),
)
async def analyze_quantitative(
    conditioning_set_id: Annotated[str, Field(description="UUID of the conditioning set (from list_feature_set_templates)")],
    portfolio_id: Annotated[str | None, Field(description="UUID of an existing portfolio. If omitted, provide tickers instead.", default=None)] = None,
    tickers: Annotated[list[str] | None, Field(description="Tickers to analyze (e.g. ['AAPL', 'MSFT']). Auto-creates a portfolio if portfolio_id is not given.", default=None)] = None,
    weights: Annotated[list[float] | None, Field(description="Optional weights for tickers (must sum to 1.0). Defaults to equal weights.", default=None)] = None,
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(conditioning_set_id, "conditioning_set_id"):
        return err

    try:
        # Step 0: Resolve or auto-create portfolio
        portfolio, err = await _ensure_portfolio(portfolio_id, tickers, weights)
        if err:
            return err
        portfolio_id = portfolio["id"]
        target_set_id = portfolio.get("target_set_id")
        asset_tickers = _portfolio_tickers(portfolio)
        portfolio_name = portfolio.get("name", "")

        client = get_client()

        # Step 1: Create models (linked to portfolio via parent_target_set_id)
        create_result = await _retry_api_call(lambda: client.batch_create_models(
            conditioning_set_id=conditioning_set_id,
            asset_tickers=asset_tickers,
            parent_target_set_id=target_set_id,
            group_name=portfolio_name or None,
        ))
        model_group_id = create_result.get("model_group_id")
        if not model_group_id:
            return "Error: model creation did not return a model_group_id."

        total_created = create_result.get("total_created", 0)
        total_failed = create_result.get("total_failed", 0)
        if total_created == 0:
            return _fmt({
                "error": "No models were created.",
                "total_failed": total_failed,
                "failed_assets": create_result.get("failed_assets", []),
            })

        # Step 2: Train (synchronous — returns completed results directly)
        train_result = await _retry_api_call(lambda: client.train_batch(
            model_group_id=model_group_id,
        ))
        if train_result.get("status") == "failed":
            return _fmt({
                "error": "Training failed.",
                "model_group_id": model_group_id,
                "details": train_result.get("error_message"),
            })
        if train_result.get("status") != "completed":
            return _fmt({
                "error": "Training did not complete.",
                "model_group_id": model_group_id,
                "status": train_result.get("status"),
            })

        # Step 3: Simulate betas (synchronous — returns full results directly)
        results = await _retry_api_call(lambda: client.simulate_betas_batch(
            model_group_id=model_group_id,
        ))
        sim_batch_id = results.get("simulation_batch_id")
        if not sim_batch_id:
            return "Error: betas simulation did not return a simulation_batch_id."

        if not results.get("all_completed"):
            return _fmt({
                "error": "Simulation did not complete.",
                "model_group_id": model_group_id,
                "simulation_batch_id": sim_batch_id,
            })

        # Build summary
        flat = _flatten_betas(results)
        summary = {
            "status": "completed",
            "model_group_id": model_group_id,
            "simulation_batch_id": sim_batch_id,
            "portfolio_id": portfolio_id,
            "models_created": total_created,
            **flat,
            "next_steps": (
                "Use simulate_returns with factor values (from factor_means_raw) to run scenarios "
                "and get per-asset risk metrics (VaR, ES, volatility). "
                "Use run_model_validation to validate model quality."
            ),
        }

        return _with_widget(_fmt(summary), betas_heatmap(summary))
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("analyze_quantitative failed: %s", e, exc_info=True)
        return "Analysis failed unexpectedly. Please try again."


# NOTE: sablier_full_analysis was removed — LLM can call analyze_quantitative + analyze_qualitative separately.




async def _retry_api_call(coro_fn, max_retries: int = 2, delay: float = 3.0):
    """Retry an async API call on transient server errors (5xx).

    coro_fn must be a zero-arg callable that returns a coroutine, e.g.:
        await _retry_api_call(lambda: client.train_batch(model_group_id=mgid))
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except SablierAPIError as e:
            last_exc = e
            if e.status_code < 500 or attempt == max_retries:
                raise
            logger.warning("Transient %s on attempt %d/%d — retrying in %.0fs", e, attempt + 1, max_retries + 1, delay)
            await asyncio.sleep(delay)
            delay *= 2  # exponential backoff
    raise last_exc  # unreachable, but keeps type checker happy


# ══════════════════════════════════════════════════
# ASGI middleware — captures redirect_uri from /authorize
# before the SDK's handler calls get_client().
# ══════════════════════════════════════════════════


class CaptureRedirectMiddleware:
    """Pure ASGI middleware that sets _pending_auth_redirect contextvar."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "") == "/authorize":
            qs = scope.get("query_string", b"").decode()
            params = urllib.parse.parse_qs(qs)
            redirect_uri = params.get("redirect_uri", [None])[0]
            if redirect_uri:
                _pending_auth_redirect.set(redirect_uri)
        await self.app(scope, receive, send)


# ══════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════


def main():
    if _transport == "stdio":
        server.run(transport="stdio")
    else:
        import uvicorn

        starlette_app = server.streamable_http_app()
        wrapped_app = CaptureRedirectMiddleware(starlette_app)

        config = uvicorn.Config(
            wrapped_app,
            host="0.0.0.0",
            port=_port,
            log_level="info",
        )
        uvicorn_server = uvicorn.Server(config)

        import asyncio
        asyncio.run(uvicorn_server.serve())


if __name__ == "__main__":
    main()
