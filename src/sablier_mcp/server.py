"""
Sablier MCP Server — tool definitions.

All 66 tools are registered here. Infrastructure (server setup, client
management, helpers, retry logic) lives in _core.py.
"""

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from sablier_mcp._core import (
    # Server instance
    server,
    logger,
    # Client
    get_client,
    SablierAPIError,
    # Helpers
    _fmt,
    _validate_uuid,
    _api_error,
    _with_widget,
    _require_auth,
    _portfolio_tickers,
    _build_per_asset_output,
    _flatten_betas,
    _ensure_portfolio,
    _validate_conditioning_data,
    # Flow helpers
    _train_flow_model_impl,
    _generate_flow_paths_impl,
    _acquire_gpu,
    _FlowGPUBusy,
    _retry_api_call,
    _retry_gpu_call,
    # Widgets
    betas_heatmap,
    flow_fan_chart,
    flow_risk_card,
    grain_score_card,
    portfolio_overview,
)


# ══════════════════════════════════════════════════
# Search & Discovery
# ══════════════════════════════════════════════════


@server.tool(
    name="search_features",
    description="Search for tickers (stocks, ETFs) and market indicators (VIX, DXY, rates). Returns matching symbols with descriptions. Use this to validate tickers before creating a portfolio.",
    annotations=ToolAnnotations(title="Search Features", readOnlyHint=True, openWorldHint=True),
)
async def search_features(
    query: Annotated[str, Field(description="Search term (e.g. 'AAPL', 'technology', 'volatility', 'gold')")],
    is_asset: Annotated[bool | None, Field(description="If True, only assets. If False, only indicators.", default=None)] = None,
    limit: Annotated[int, Field(description="Max results (default 50)", default=50)] = 50,
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        results = await client.search_features(query, is_asset=is_asset, limit=limit)  # source filter available via client
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
# User API Keys (third-party services)
# ══════════════════════════════════════════════════




# ══════════════════════════════════════════════════
# Feature Catalog Management
# ══════════════════════════════════════════════════


@server.tool(
    name="add_feature",
    description=(
        "Add a ticker to the feature catalog so it can be used in portfolios or conditioning sets. "
        "IMPORTANT: First use search_features to check if the ticker already exists — "
        "calling add_feature for an existing ticker returns a 409 error. "
        "Specify source ('yahoo' for stocks/ETFs/futures, 'fred' for rates/economic indicators). "
        "Validates the ticker exists on the source API and auto-populates metadata "
        "(display_name, category, units, etc.) from the API response. "
        "Set is_asset=true for assets that go into portfolios, false for conditioning factors. "
        "After adding, call refresh_feature_data to populate its historical data."
    ),
    annotations=ToolAnnotations(title="Add Feature", destructiveHint=True, openWorldHint=True),
)
async def add_feature(
    ticker: Annotated[str, Field(description="Ticker symbol (e.g. 'AAPL', 'DFF', 'CL=F')")],
    source: Annotated[str, Field(description="Data source: 'yahoo' (stocks, ETFs, futures) or 'fred' (rates, economic)")],
    display_name: Annotated[str | None, Field(description="Human-readable name (e.g. 'Apple Inc.'). Auto-detected if omitted.", default=None)] = None,
    description: Annotated[str | None, Field(description="Brief description", default=None)] = None,
    category: Annotated[str | None, Field(description="Category: equity, rates, fx, commodity, volatility, economic, etc. Auto-detected if omitted.", default=None)] = None,
    is_asset: Annotated[bool | None, Field(description="True for portfolio assets, False for conditioning factors. Auto-detected if omitted.", default=None)] = None,
    data_type: Annotated[str | None, Field(description="Data type: price, rate, index, level. Auto-detected if omitted.", default=None)] = None,
    units: Annotated[str | None, Field(description="Units (e.g. 'USD', 'percent', 'index'). Auto-detected if omitted.", default=None)] = None,
    skip_validation: Annotated[bool, Field(description="Skip ticker validation against source API", default=False)] = False,
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        result = await client.add_feature(
            ticker=ticker, source=source, display_name=display_name,
            description=description, category=category, is_asset=is_asset,
            data_type=data_type, units=units, skip_validation=skip_validation,
        )
        return _fmt({
            "id": result.get("id"),
            "ticker": result.get("ticker"),
            "display_name": result.get("display_name"),
            "category": result.get("category"),
            "is_asset": result.get("is_asset"),
            "data_type": result.get("data_type"),
            "units": result.get("units"),
            "source": result.get("source"),
            "message": f"Feature '{ticker}' added to catalog. Call refresh_feature_data to populate historical data.",
        })
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="refresh_feature_data",
    description=(
        "Fetch/update historical training data for specific tickers from Yahoo Finance or FRED. "
        "For new features: full fetch from 2000. For existing: incremental update to today. "
        "Use this after add_feature, or to force-update stale data."
    ),
    annotations=ToolAnnotations(title="Refresh Feature Data", destructiveHint=True, openWorldHint=True),
)
async def refresh_feature_data(
    tickers: Annotated[list[str], Field(description="Tickers to refresh (e.g. ['AAPL', 'CL=F', 'DFF'])")],
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        result = await client.refresh_feature_data(tickers)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)




# ══════════════════════════════════════════════════
# Portfolios
# ══════════════════════════════════════════════════


@server.tool(
    name="list_portfolios",
    description="List the user's existing portfolios with names, IDs, asset compositions, and status.",
    annotations=ToolAnnotations(title="List Portfolios", readOnlyHint=True),
)
async def list_portfolios(
    limit: Annotated[int, Field(description="Max portfolios to return", default=100)] = 100,
) -> list | str:
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
    annotations=ToolAnnotations(title="Get Portfolio", readOnlyHint=True),
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
    annotations=ToolAnnotations(title="Create Portfolio", destructiveHint=True),
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
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 0.01:
        return f"Error: weights must sum to 1.0 (got {weight_sum:.4f})"

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
        "Update an existing portfolio. Can change name, description, weights, capital, and/or options_positions. "
        "Only pass the fields you want to update — omitted fields stay unchanged. "
        "Weights must sum to 1.0 if provided. "
        "options_positions sets the options overlay for derivatives analysis (persisted on the portfolio)."
    ),
    annotations=ToolAnnotations(title="Update Portfolio", destructiveHint=True),
)
async def update_portfolio(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
    name: Annotated[str | None, Field(description="New portfolio name", default=None)] = None,
    description: Annotated[str | None, Field(description="New description", default=None)] = None,
    weights: Annotated[dict[str, float] | None, Field(description="New weights by ticker (e.g. {'AAPL': 0.5, 'MSFT': 0.5}). Must sum to 1.0.", default=None)] = None,
    capital: Annotated[float | None, Field(description="New capital allocation in USD", default=None)] = None,
    options_positions: Annotated[list[dict] | None, Field(
        description=(
            "Options overlay positions to persist on the portfolio. Each dict: "
            "underlying (display_name), option_type ('call'/'put'), strike (float), "
            "days_to_expiry (int), quantity (int, negative=short), implied_vol (float), "
            "entry_premium (float). These are used by analyze_derivatives when no positions are passed inline."
        ),
        default=None,
    )] = None,
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
    if options_positions is not None:
        fields["options_positions"] = options_positions
    if not fields:
        return "Error: provide at least one field to update (name, description, weights, capital, options_positions)."
    try:
        client = get_client()
        result = await client.update_portfolio(portfolio_id, **fields)
        output = {
            "portfolio_id": result["id"],
            "name": result["name"],
            "weights": result.get("weights", {}),
            "capital": result.get("capital"),
            "message": "Portfolio updated successfully.",
        }
        if result.get("options_positions"):
            output["options_positions"] = result["options_positions"]
        return _fmt(output)
    except SablierAPIError as e:
        return _api_error(e)



@server.tool(
    name="get_portfolio_value",
    description="Get the current live value of a portfolio: total value, P&L, and per-position breakdown.",
    annotations=ToolAnnotations(title="Get Portfolio Value", readOnlyHint=True),
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
    description=(
        "Get historical portfolio analytics: Sharpe ratio, volatility, expected return, max drawdown, "
        "and market beta (benchmarked vs SPY). Supports timeframes: 1W, 1M, 1Y, 2Y, 5Y. "
        "This is backward-looking — for forward-looking risk, use compute_returns or test_flow_risk. "
        "NEW: Pass model_group_id to also get factor return attribution — shows which factors "
        "(VIX, rates, oil, etc.) drove your portfolio returns over the period. "
        "Requires compute_betas to have been run first via analyze_quantitative."
    ),
    annotations=ToolAnnotations(title="Get Portfolio Analytics", readOnlyHint=True),
)
async def get_portfolio_analytics(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
    timeframe: Annotated[str, Field(description="Timeframe: 1W, 1M, 1Y, 2Y, or 5Y (default 1Y)", default="1Y")] = "1Y",
    model_group_id: Annotated[str | None, Field(description="Model group ID for factor return attribution. From analyze_quantitative or list_model_groups.", default=None)] = None,
    rollup: Annotated[str, Field(description="Attribution period: 'daily', 'weekly', 'monthly' (default 'daily')", default="daily")] = "daily",
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        params: dict = {"timeframe": timeframe}
        if model_group_id:
            params["model_group_id"] = model_group_id
            params["rollup"] = rollup
        result = await client.get_portfolio_analytics(portfolio_id, **params)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_asset_profiles",
    description="Get asset classification for a portfolio: sector, industry, country, exchange, and asset type per holding.",
    annotations=ToolAnnotations(title="Get Asset Profiles", readOnlyHint=True),
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


@server.tool(
    name="delete_portfolio",
    description="Delete a portfolio by ID. Permanent, cannot be undone.",
    annotations=ToolAnnotations(title="Delete Portfolio", destructiveHint=True),
)
async def delete_portfolio(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID to delete")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        await client.delete_portfolio(portfolio_id)
        return _fmt({"message": f"Portfolio {portfolio_id} deleted successfully."})
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="optimize_portfolio",
    description=(
        "Find optimal portfolio weights using per-asset factor exposures from compute_betas or analyze_quantitative. "
        "Requires simulation_batch_id (from their output). "
        "Objectives: 'max_sharpe' (maximize risk-adjusted return), 'min_variance' (minimize portfolio volatility), "
        "'max_return' (maximize expected return for given risk). "
        "Advanced objectives (pass the string directly): 'analytical_risk_parity' (equalize risk contributions), "
        "'mean_cvar' (minimize CVaR, requires simulation_ids not beta_simulation_ids), "
        "'expected_utility' (maximize CRRA utility), 'risk_parity' (CVaR-based equal risk), "
        "'exposure_target' (match target factor exposures — set target_exposures on the API). "
        "Default: 'max_sharpe'. Long-only constraint applied by default."
    ),
    annotations=ToolAnnotations(title="Optimize Portfolio", readOnlyHint=True, openWorldHint=True),
)
async def optimize_portfolio(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
    simulation_batch_id: Annotated[str, Field(description="From compute_betas or analyze_quantitative")],
    objective: Annotated[str, Field(description="Optimization objective: 'max_sharpe' (default), 'min_variance', 'max_return', 'analytical_risk_parity' (equal risk contributions), 'expected_utility' (CRRA), 'risk_parity' (CVaR-based), 'mean_cvar' (minimize CVaR)", default="max_sharpe")] = "max_sharpe",
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    if err := _validate_uuid(simulation_batch_id, "simulation_batch_id"):
        return err
    try:
        client = get_client()
        result = await client.optimize_portfolio(
            portfolio_id, simulation_batch_id, objective=objective,
        )
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_efficient_frontier",
    description=(
        "Calculate the mean-variance efficient frontier for portfolio assets using historical returns. "
        "Returns a curve of optimal risk-return tradeoffs with long-only constraints (no shorting). "
        "Each point includes optimal weights, expected return, and volatility. "
        "This is a historical analysis — for forward-looking optimization, use optimize_portfolio with simulation data."
    ),
    annotations=ToolAnnotations(title="Get Efficient Frontier", readOnlyHint=True),
)
async def get_efficient_frontier(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
    num_portfolios: Annotated[int, Field(description="Number of points on the frontier curve (default 50)", default=50)] = 50,
    timeframe: Annotated[str, Field(description="Historical lookback period: '1Y', '2Y', '5Y', etc. Default '1Y'.", default="1Y")] = "1Y",
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        result = await client.get_efficient_frontier(portfolio_id, num_portfolios=num_portfolios, timeframe=timeframe)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_optimization_history",
    description=(
        "Retrieve past portfolio optimization results. Each entry includes the objective, "
        "optimal weights, expected return, volatility, VaR, and Sharpe ratio. "
        "Optionally filter by simulation_batch_id to see results for a specific beta computation."
    ),
    annotations=ToolAnnotations(title="Get Optimization History", readOnlyHint=True),
)
async def get_optimization_history(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
    simulation_batch_id: Annotated[str | None, Field(description="Filter to a specific beta simulation run. Optional.", default=None)] = None,
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        result = await client.get_optimization_history(portfolio_id, simulation_batch_id=simulation_batch_id)
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
    annotations=ToolAnnotations(title="Analyze Qualitative", readOnlyHint=True, openWorldHint=True),
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
        raw_results = result.get("results") or {}
        themes_data = raw_results.get("results", []) or raw_results.get("themes", [])

        # Surface not-covered tickers clearly (ETFs, non-US equities with no SEC filings)
        # not_covered_tickers lives in each theme's coverage_summary (same list across themes)
        not_covered_tickers: list[str] = []
        if themes_data:
            first_coverage = (themes_data[0] or {}).get("coverage_summary") or {}
            not_covered_tickers = first_coverage.get("not_covered_tickers") or []
        covered_count = (themes_data[0] or {}).get("coverage_summary", {}).get("covered_count", 1) if themes_data else 0

        if not covered_count and not_covered_tickers:
            return _fmt({
                "status": "not_covered",
                "not_covered_tickers": not_covered_tickers,
                "message": (
                    f"{', '.join(not_covered_tickers)} {'have' if len(not_covered_tickers) > 1 else 'has'} "
                    "no SEC filings in GRAIN. GRAIN requires equities with 10-K/10-Q filings. "
                    "ETFs, non-US equities, and funds are not supported."
                ),
            })

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
                    "tier": ts.get("tier") if ts.get("tier") is not None else ts.get("tier_name"),
                    "direction": ts.get("direction"),
                    "confidence": ts.get("confidence"),
                    "top_evidence": [],
                }
                for ev in (ts.get("evidence") or [])[:3]:
                    ev_entry: dict[str, Any] = {
                        "passage": (ev.get("passage") or "")[:500],
                        "source": ev.get("source"),
                        "source_type": ev.get("source_type"),
                        "fiscal_period": ev.get("fiscal_period"),
                        "filing_date": ev.get("filing_date"),
                        "why_relevant": ev.get("why_relevant"),
                    }
                    if ev.get("filing_url"):
                        ev_entry["filing_url"] = ev["filing_url"]
                    ticker_entry["top_evidence"].append(ev_entry)
                theme_summary["ticker_scores"].append(ticker_entry)
            summary["themes"].append(theme_summary)

        return _with_widget(_fmt(summary), grain_score_card(summary))
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_themes",
    description="Browse the GRAIN theme library. Returns predefined themes with names, descriptions, keywords, and categories.",
    annotations=ToolAnnotations(title="List Themes", readOnlyHint=True),
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
    annotations=ToolAnnotations(title="List GRAIN Analyses", readOnlyHint=True),
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
    description="Retrieve full results of a completed GRAIN analysis: theme scores, per-ticker breakdown, and evidence passages. Use list_grain_analyses first to find the analysis_id.",
    annotations=ToolAnnotations(title="Get GRAIN Analysis", readOnlyHint=True),
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


@server.tool(
    name="delete_grain_analysis",
    description="Delete a saved GRAIN qualitative analysis by ID. Permanent, cannot be undone.",
    annotations=ToolAnnotations(title="Delete GRAIN Analysis", readOnlyHint=False, destructiveHint=True),
)
async def delete_grain_analysis(
    analysis_id: Annotated[str, Field(description="The analysis UUID to delete")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(analysis_id, "analysis_id"):
        return err
    try:
        client = get_client()
        result = await client.delete_grain_analysis(analysis_id)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════


@server.tool(
    name="list_model_groups",
    description="List all model groups (each created by analyze_quantitative). A model group ties a portfolio to a conditioning set and contains per-asset models. Check model_type: null/absent = Moment (linear), 'flow_generative' = Flow. Use this to find model_group_ids for compute_betas, compute_returns, (resume).",
    annotations=ToolAnnotations(title="List Model Groups", readOnlyHint=True),
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
                "model_type": g.get("model_type"),  # 'flow_generative' or None (Moment)
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
    annotations=ToolAnnotations(title="List Feature Set Templates", readOnlyHint=True),
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
    name="create_feature_set",
    description=(
        "Create a custom conditioning set (or target set) from features in the catalog. "
        "Use this to build arbitrary factor sets for analyze_quantitative instead of using pre-built templates. "
        "Each feature needs at minimum a 'ticker' and 'source' ('YAHOO' or 'FRED'). "
        "The display_name is auto-resolved from available_features if omitted. "
        "Returns the conditioning_set_id that can be passed to analyze_quantitative."
    ),
    annotations=ToolAnnotations(title="Create Feature Set", destructiveHint=True),
)
async def create_feature_set(
    name: Annotated[str, Field(description="Name for the set (e.g. 'Custom Macro Factors')")],
    features: Annotated[list[dict], Field(description="List of features. Each needs 'ticker' and 'source' (YAHOO/FRED). Optional: 'display_name'.")],
    description: Annotated[str, Field(description="Optional description", default="")] = "",
    set_type: Annotated[str, Field(description="'conditioning' (market drivers) or 'target' (assets to model)", default="conditioning")] = "conditioning",
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        result = await client.create_feature_set(
            name=name, features=features, description=description, set_type=set_type,
        )
        set_id = result.get("id")
        actual_type = result.get("set_type", set_type)
        id_key = "conditioning_set_id" if actual_type == "conditioning" else "target_set_id"
        return _fmt({
            id_key: set_id,
            "name": result.get("name"),
            "set_type": actual_type,
            "features": [
                {"display_name": f.get("display_name"), "ticker": f.get("ticker"), "source": f.get("source")}
                for f in (result.get("features") or [])
            ],
            "message": f"Feature set '{name}' created (ID: {set_id}). "
                       + (f"Use the {id_key} with analyze_quantitative." if actual_type == "conditioning"
                          else "Use the target_set_id when creating portfolios."),
        })
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_feature_sets",
    description=(
        "List all accessible feature sets: your custom sets plus shared templates. "
        "Filter by set_type ('conditioning' or 'target'). "
        "Use this to find conditioning_set_id values for analyze_quantitative."
    ),
    annotations=ToolAnnotations(title="List Feature Sets", readOnlyHint=True),
)
async def list_feature_sets(
    set_type: Annotated[str | None, Field(description="Filter: 'conditioning' or 'target'. Omit for all.", default=None)] = None,
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        result = await client.list_all_feature_sets(set_type=set_type)
        sets = result.get("feature_sets", []) if isinstance(result, dict) else result
        return _fmt([
            {
                "id": s.get("id"),
                "name": s.get("name"),
                "set_type": s.get("set_type"),
                "features_count": len(s.get("features") or []),
                "features": [f.get("display_name") or f.get("ticker") for f in (s.get("features") or [])],
                "is_template": bool(s.get("tag")),
            }
            for s in sets
        ])
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_feature_set",
    description="Get detailed information about a specific feature set including all features and their configuration.",
    annotations=ToolAnnotations(title="Get Feature Set", readOnlyHint=True),
)
async def get_feature_set(
    feature_set_id: Annotated[str, Field(description="UUID of the feature set")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(feature_set_id, "feature_set_id"):
        return err
    try:
        client = get_client()
        result = await client.get_feature_set(feature_set_id)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="delete_feature_set",
    description="Delete a custom feature set. Permanent, cannot be undone. Cannot delete shared templates.",
    annotations=ToolAnnotations(title="Delete Feature Set", destructiveHint=True),
)
async def delete_feature_set(
    feature_set_id: Annotated[str, Field(description="UUID of the feature set to delete")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(feature_set_id, "feature_set_id"):
        return err
    try:
        client = get_client()
        result = await client.delete_feature_set(feature_set_id)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="delete_model_group",
    description="Delete a model group and all its models, simulations, and associated data. Permanent, cannot be undone.",
    annotations=ToolAnnotations(title="Delete Model Group", destructiveHint=True),
)
async def delete_model_group(
    model_group_id: Annotated[str, Field(description="UUID of the model group to delete")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        await client.delete_model_group(model_group_id)
        return _fmt({"message": f"Model group {model_group_id} deleted successfully."})
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_residual_correlation",
    description=(
        "Get the cross-asset residual correlation matrix for a model group. "
        "Shows how much unexplained co-movement exists between assets after accounting for factor exposures. "
        "High residual correlations suggest missing common factors."
    ),
    annotations=ToolAnnotations(title="Get Residual Correlation", readOnlyHint=True),
)
async def get_residual_correlation(
    model_group_id: Annotated[str, Field(description="UUID of the model group")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        result = await client.get_residual_correlation(model_group_id)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_simulations",
    description="List past beta computation runs for a Moment model group. Each entry is a compute_betas invocation with its simulation_batch_id, date, and status. Use this to find older simulation_batch_ids for compute_returns. NOT for Flow models — use list_flow_scenarios for those.",
    annotations=ToolAnnotations(title="List Simulations", readOnlyHint=True),
)
async def list_simulations(
    model_group_id: Annotated[str, Field(description="UUID of the model group")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        result = await client.list_group_simulations(model_group_id)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)




@server.tool(
    name="compute_betas",
    description=(
        "Compute factor exposures (betas) for an already-trained model group. "
        "Use this when you already have a trained model_group_id (from analyze_quantitative or list_model_groups) "
        "and want to refresh betas with a different lookback window, or get a new simulation_batch_id. "
        "You do NOT need this if you just ran analyze_quantitative — it already includes this step. "
        "Returns per-asset factor exposures with R² (goodness-of-fit), rolling_window used, factor_last_date "
        "(effective beta date), data_truncated_by (stale factors), and simulation_batch_id (for compute_returns). "
        "Key use: call with different lookback_days (e.g. 63, 126, 252) to compare betas across time horizons — "
        "divergence signals regime changes. Check R² to gauge how well factors explain each asset."
    ),
    annotations=ToolAnnotations(title="Compute Betas", readOnlyHint=True, openWorldHint=True),
)
async def compute_betas(
    model_group_id: Annotated[str, Field(description="UUID of the trained model group (from analyze_quantitative or list_model_groups)")],
    lookback_days: Annotated[int | None, Field(description="Historical lookback window in trading days", default=None)] = None,
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        # Synchronous — Moment returns results directly
        results = await client.compute_betas_batch(
            model_group_id=model_group_id,
            historical_lookback_days=lookback_days,
        )
        sim_batch_id = results.get("simulation_batch_id")
        if not sim_batch_id:
            return "Error: betas computation did not return a simulation_batch_id."

        flat = _flatten_betas(results)
        summary = {
            "simulation_batch_id": sim_batch_id,
            **flat,
        }

        return _with_widget(_fmt(summary), betas_heatmap(summary))
    except SablierAPIError as e:
        return _api_error(e)




# ══════════════════════════════════════════════════
# Return Simulation
# ══════════════════════════════════════════════════


@server.tool(
    name="compute_returns",
    description=(
        "Run an ad-hoc what-if stress test on a Moment (linear) model — the PRIMARY tool for quick scenario analysis. "
        "Requires simulation_batch_id from analyze_quantitative or compute_betas. "
        "Pass ALL conditioning_features as absolute price levels in the factors dict. "
        "BEFORE calling: read factor_last_values_raw from the betas output — these are today's values. "
        "Always state current vs stressed levels explicitly (e.g. 'Oil is at $82 today, stressing to $120'). "
        "Also check data_freshness_warning in betas output — if present, betas may be stale. "
        "Use factor_last_values_raw as baseline, then compute targets "
        "(e.g. oil currently 82, stress to 120: pass {'Crude Oil': 120.0}). "
        "For saved/reusable scenarios, use create_scenario then compute_returns with the factor values. "
        "For Flow (generative) models, use simulate_flow_scenario instead."
    ),
    annotations=ToolAnnotations(title="Compute Returns", readOnlyHint=True, openWorldHint=True),
)
async def compute_returns(
    simulation_batch_id: Annotated[str, Field(description="From analyze_quantitative or compute_betas results")],
    factors: Annotated[dict[str, float], Field(
        description=(
            "Absolute target price levels for each factor. "
            "Use factor_last_values_raw from betas output as current levels, "
            "then compute target (e.g. US Market -5%: 682.39 * 0.95 = 648.27). "
            "Keys must match conditioning_features exactly."
        )
    )],
    n_samples: Annotated[int, Field(description="Number of Monte Carlo samples (default 1000)", default=1000)] = 1000,
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(simulation_batch_id, "simulation_batch_id"):
        return err
    try:
        client = get_client()
        # Synchronous — Moment returns results directly
        results = await client.compute_returns_batch(
            simulation_batch_id=simulation_batch_id,
            factors=factors,
            n_samples=n_samples,
        )
        returns_batch_id = results.get("returns_batch_id")
        if not returns_batch_id:
            return "Error: returns simulation did not return a returns_batch_id."

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
    description=(
        "Save a named Moment scenario template for later reuse. "
        "This does NOT run a simulation — use compute_returns with the factor values instead, or for ad-hoc tests. "
        "IMPORTANT: requires model_id — this is an individual per-asset model UUID from list_model_groups → models[].model_id, "
        "NOT the model_group_id. Each scenario is tied to one asset's model. "
        "Factor spec format: {'VIX': {'type': 'fixed', 'value': 35}}. "
        "Supported types: 'fixed' (exact value), 'percentile' (historical percentile), 'shock' (std dev shift)."
    ),
    annotations=ToolAnnotations(title="Create Scenario", destructiveHint=True),
)
async def create_scenario(
    model_id: Annotated[str, Field(description="UUID of the model this scenario applies to")],
    name: Annotated[str, Field(description="Scenario name (e.g. 'Recession', 'Tech Bubble')")],
    factor_values: Annotated[dict[str, dict], Field(description="Factor specs (e.g. {'VIX': {'type': 'fixed', 'value': 35}})")],
    model_group_id: Annotated[str | None, Field(description="Model group UUID (optional, for reference)", default=None)] = None,
    description: Annotated[str, Field(description="Optional description", default="")] = "",
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_id, "model_id"):
        return err
    if model_group_id:
        if err := _validate_uuid(model_group_id, "model_group_id"):
            return err
    try:
        client = get_client()
        result = await client.create_scenario(
            model_id=model_id,
            name=name,
            specs=factor_values,
            model_group_id=model_group_id,
            description=description or None,
        )
        return _fmt({
            "id": result["id"],
            "name": result["name"],
            "specs": result.get("specs"),
            "message": "Scenario created. Use compute_returns with the factor values to sample returns.",
        })
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_scenarios",
    description=(
        "List saved Moment scenario templates (created via create_scenario). "
        "These are stored factor specs tied to individual model_ids — "
        "to execute one, use compute_returns with the factor values. For ad-hoc tests, use compute_returns directly. "
        "NOT for Flow scenarios — use list_flow_scenarios for those."
    ),
    annotations=ToolAnnotations(title="List Scenarios", readOnlyHint=True),
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


@server.tool(
    name="get_scenario",
    description="Get detailed information about a saved scenario including its factor specs.",
    annotations=ToolAnnotations(title="Get Scenario", readOnlyHint=True),
)
async def get_scenario(
    scenario_id: Annotated[str, Field(description="The scenario UUID")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(scenario_id, "scenario_id"):
        return err
    try:
        client = get_client()
        result = await client.get_scenario(scenario_id)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="update_scenario",
    description=(
        "Update a saved scenario. Can change name, description, or factor specs. "
        "Only pass the fields you want to update."
    ),
    annotations=ToolAnnotations(title="Update Scenario", destructiveHint=True),
)
async def update_scenario(
    scenario_id: Annotated[str, Field(description="The scenario UUID")],
    name: Annotated[str | None, Field(description="New scenario name", default=None)] = None,
    description: Annotated[str | None, Field(description="New description", default=None)] = None,
    specs: Annotated[dict[str, dict] | None, Field(description="New factor specs", default=None)] = None,
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(scenario_id, "scenario_id"):
        return err
    fields: dict = {}
    if name is not None:
        fields["name"] = name
    if description is not None:
        fields["description"] = description
    if specs is not None:
        fields["specs"] = specs
    if not fields:
        return "Error: provide at least one field to update (name, description, specs)."
    try:
        client = get_client()
        result = await client.update_scenario(scenario_id, **fields)
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="delete_scenario",
    description="Delete a saved scenario. Permanent, cannot be undone.",
    annotations=ToolAnnotations(title="Delete Scenario", destructiveHint=True),
)
async def delete_scenario(
    scenario_id: Annotated[str, Field(description="The scenario UUID to delete")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(scenario_id, "scenario_id"):
        return err
    try:
        client = get_client()
        await client.delete_scenario(scenario_id)
        return _fmt({"message": f"Scenario {scenario_id} deleted successfully."})
    except SablierAPIError as e:
        return _api_error(e)




# ══════════════════════════════════════════════════
# Full Analysis Orchestrator
# ══════════════════════════════════════════════════


@server.tool(
    name="analyze_quantitative",
    description=(
        "Build and train linear factor models for a portfolio in one step (creates models → trains → computes betas). "
        "This is the starting point for Moment (linear) analysis. "
        "Requires conditioning_set_id (the market drivers — get one from list_feature_set_templates or create_feature_set). "
        "Uses a two-layer architecture: thematic factors (conditioning_set_id) + optional baseline factors "
        "(baseline_mode='us' absorbs market/value/growth variance via real-time ETF proxies before thematic factors). "
        "Pass either portfolio_id or tickers directly (auto-creates portfolio with equal weights). "
        "Returns: factor exposures (betas), per-asset R² (in-sample goodness-of-fit — tells you how linear the relationship "
        "actually is for the given window), rolling_window used, factor_last_date (effective beta date — may be truncated "
        "if a factor has stale data), and data_truncated_by (which factors caused truncation). "
        "Tip: vary rolling_window (e.g. 63 vs 126 vs 252) to detect regime changes — diverging betas across windows "
        "signal non-stationarity. Low R² suggests nonlinear dynamics or missing factors. "
        "Next step: call compute_returns with the simulation_batch_id to run what-if stress tests."
    ),
    annotations=ToolAnnotations(title="Analyze Quantitative", destructiveHint=True, openWorldHint=True),
)
async def analyze_quantitative(
    conditioning_set_id: Annotated[str, Field(description="UUID of the thematic conditioning set (from list_feature_set_templates or create_feature_set)")],
    portfolio_id: Annotated[str | None, Field(description="UUID of an existing portfolio. If omitted, provide tickers instead.", default=None)] = None,
    tickers: Annotated[list[str] | None, Field(description="Tickers to analyze (e.g. ['AAPL', 'MSFT']). Auto-creates a portfolio if portfolio_id is not given.", default=None)] = None,
    weights: Annotated[list[float] | None, Field(description="Optional weights for tickers (must sum to 1.0). Defaults to equal weights.", default=None)] = None,
    baseline_mode: Annotated[str | None, Field(description="Baseline factor orthogonalization region. ETF-based (real-time): 'us', 'global', 'developed_ex_us', 'europe', 'japan'. Legacy FF5 (~2mo lag): 'us_ff5', 'global_ff5'. Set 'none' or omit to skip baseline.", default=None)] = None,
    nonlinear: Annotated[bool, Field(description="Also fit nonlinear factor exposure model on top of linear betas, producing sensitivity curves. Requires Pro+ tier. Default True (runs if tier allows).", default=True)] = True,
    rolling_window: Annotated[int | None, Field(description="Rolling window size in trading days for beta estimation (default 252 = ~1 year). Smaller = more responsive to recent regime changes, larger = more stable.", default=None)] = None,
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

        # Pre-flight: ensure conditioning set has fetched data
        if data_err := await _validate_conditioning_data(conditioning_set_id):
            return data_err

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

        # Step 2: Train (synchronous — Moment returns results directly)
        train_result = await _retry_api_call(lambda: client.train_batch(
            model_group_id=model_group_id,
            nonlinear=nonlinear,
            baseline_mode=baseline_mode,
            rolling_huber_window=rolling_window,
        ))
        if train_result.get("status") == "failed" or train_result.get("failed", 0) == train_result.get("total", 0):
            return _fmt({
                "error": "Training failed.",
                "model_group_id": model_group_id,
                "details": train_result.get("results", []),
            })

        # Step 3: Compute betas (synchronous — Moment returns results directly)
        results = await _retry_api_call(lambda: client.compute_betas_batch(
            model_group_id=model_group_id,
        ))
        sim_batch_id = results.get("simulation_batch_id")
        if not sim_batch_id:
            return "Error: betas computation did not return a simulation_batch_id."

        if not results.get("all_completed"):
            return _fmt({
                "error": "Computation did not complete.",
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
                "To run a what-if scenario, call compute_returns with simulation_batch_id "
                f"and a factors dict containing ALL of these at your desired levels: "
                f"{', '.join(flat.get('conditioning_features', []))}. "
                "Use factor_last_values_raw above as current baseline values. "
                "To save a scenario template for later, use create_scenario."
            ),
        }

        return _with_widget(_fmt(summary), betas_heatmap(summary))
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("analyze_quantitative failed: %s", e, exc_info=True)
        return "Analysis failed unexpectedly. Please try again."


# NOTE: sablier_full_analysis was removed — LLM can call analyze_quantitative + analyze_qualitative separately.


# ══════════════════════════════════════════════════
# Flow (Generative Time Series)
# ══════════════════════════════════════════════════


# ── Flow internal helpers ──

async def _train_flow_model_impl(
    client, conditioning_set_id: str, portfolio_id: str | None,
    tickers: list[str] | None, weights: list[float] | None, horizon: int,
) -> tuple[dict | str, str | None, str | None, list]:
    """Train a Flow model. Returns (output_or_error, model_group_id, portfolio_id, feature_names)."""
    if err := _validate_uuid(conditioning_set_id, "conditioning_set_id"):
        return err, None, None, []

    portfolio, err = await _ensure_portfolio(portfolio_id, tickers, weights)
    if err:
        return err, None, None, []
    portfolio_id = portfolio["id"]
    target_set_id = portfolio.get("target_set_id")
    asset_tickers = _portfolio_tickers(portfolio)
    portfolio_name = portfolio.get("name", "")

    if data_err := await _validate_conditioning_data(conditioning_set_id):
        return data_err, None, None, []

    create_result = await _retry_api_call(lambda: client.batch_create_models(
        conditioning_set_id=conditioning_set_id,
        asset_tickers=asset_tickers,
        parent_target_set_id=target_set_id,
        group_name=portfolio_name or None,
    ))
    model_group_id = create_result.get("model_group_id")
    if not model_group_id:
        return "Error: model creation did not return a model_group_id.", None, None, []

    if create_result.get("total_created", 0) == 0:
        return _fmt({
            "error": "No models were created.",
            "total_failed": create_result.get("total_failed", 0),
            "failed_assets": create_result.get("failed_assets", []),
        }), None, None, []

    # Dispatch training job to GPU worker — returns immediately.
    # Training runs async on GPU (~5-15 min).
    train_job = await _retry_gpu_call(lambda: client.flow_train(
        model_group_id=model_group_id,
        horizon=horizon,
    ))
    train_job_id = train_job.get("job_id")
    if not train_job_id:
        return "Error: Flow training did not return a job_id.", None, None, []

    feature_names = train_job.get("feature_names", [])

    output = {
        "status": "training_started",
        "model_group_id": model_group_id,
        "portfolio_id": portfolio_id,
        "job_id": train_job_id,
        "feature_names": feature_names,
        "estimated_time": "5-15 minutes",
        "instructions": (
            f"GPU training job dispatched (job_id='{train_job_id}'). "
            f"Tell the user: 'Training started — it takes about 5-15 minutes. "
            f"You can keep chatting with me, and ask me to check progress anytime.' "
            f"When the user asks to check: check_flow_job(job_id='{train_job_id}'). "
            f"When training completes: generate_flow_paths(model_group_id='{model_group_id}')."
        ),
    }
    return output, model_group_id, portfolio_id, feature_names


async def _generate_flow_paths_impl(
    client, model_group_id: str, portfolio_id: str | None,
    horizon: int, n_paths: int, price_history_length: int | None = None,
) -> list | str:
    """Generate paths from a trained model. Returns formatted output."""
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err

    feature_names: list = []

    # Resolve portfolio_id from model group if not provided
    _mg_info = None
    try:
        groups = await client.list_model_groups()
        _mg_info = next((g for g in groups if g.get("id") == model_group_id), None)
    except Exception:
        logger.debug("Could not fetch model groups", exc_info=True)

    if not portfolio_id and _mg_info:
        try:
            ptsi = _mg_info.get("parent_target_set_id")
            if ptsi:
                plist = await client.list_portfolios()
                items = plist.get("portfolios", plist) if isinstance(plist, dict) else plist
                match = next((p for p in items if p.get("target_set_id") == ptsi), None)
                if match:
                    portfolio_id = match["id"]
        except Exception:
            logger.debug("Could not resolve portfolio_id from model group", exc_info=True)

    # Try existing results first — skip if caller explicitly set price_history_length
    # (they want fresh paths with specific indicator seeding, not stale cached results)
    if price_history_length is None:
        try:
            results = await client.flow_get_latest_results(model_group_id)
            summary = results.get("summary") or {}
            feature_names = results.get("feature_names", [])
            gen_job_id = results.get("job_id", "")

            if summary:
                per_asset = _build_per_asset_output(summary, include_paths=True)
                pid_str = f"'{portfolio_id}'" if portfolio_id else "'<find via list_portfolios>'"
                feat_hint = f" Features for constraints: {', '.join(feature_names[:10])}." if feature_names else ""
                output = {
                    "status": "completed (existing results)",
                    "model_group_id": model_group_id,
                    "portfolio_id": portfolio_id,
                    "flow_job_id": str(gen_job_id),
                    "horizon": results.get("horizon", horizon),
                    "n_paths": results.get("n_paths", n_paths),
                    "feature_names": feature_names,
                    "per_asset": per_asset,
                    "next_steps": (
                        f"Available actions: "
                        f"simulate_flow_scenario(model_group_id='{model_group_id}', portfolio_id={pid_str}, constraints=...) for what-if scenarios | "
                        f"test_flow_risk(portfolio_id={pid_str}, flow_job_id='{gen_job_id}') for risk metrics | "
                        f"flow_validate(model_group_id='{model_group_id}') for model quality check | "
                        f"list_flow_scenarios(model_group_id='{model_group_id}') to see past scenarios.{feat_hint}"
                    ),
                }
                try:
                    chart_html = flow_fan_chart(summary, results.get("horizon", horizon))
                    if chart_html:
                        return _with_widget(_fmt(output), chart_html)
                except Exception:
                    pass
                return _fmt(output)
        except SablierAPIError as e:
            if e.status_code != 404:
                raise

    # Check model status
    try:
        mg_status = (_mg_info.get("status") or "").lower() if _mg_info else ""
        if mg_status in ("failed", "error"):
            return _fmt({
                "error": f"Model group status is '{mg_status}'. Cannot generate paths from a failed model.",
                "model_group_id": model_group_id,
                "hint": "Create a new model with train_flow_model(conditioning_set_id=..., tickers=...).",
            })
        if mg_status not in ("trained", "completed", ""):
            training_job_id = (_mg_info or {}).get("active_training_job_id")
            return _fmt({
                "error": f"Model group status is '{mg_status}'. Training may still be in progress.",
                "model_group_id": model_group_id,
                "training_job_id": training_job_id,
                "hint": (
                    f"Use check_flow_job(job_id='{training_job_id}', job_type='train') to monitor progress."
                    if training_job_id else
                    "Wait for training to complete, or train a new model with train_flow_model."
                ),
            })
    except Exception:
        logger.debug("Could not check model group status", exc_info=True)

    # Generate new paths
    async with _acquire_gpu():
        gen_job = await _retry_gpu_call(lambda: client.flow_generate_paths(
            model_group_id=model_group_id,
            n_paths=n_paths,
            horizon=horizon,
            price_history_length=price_history_length,
        ))
        gen_job_id = gen_job.get("job_id")
        if not gen_job_id:
            return _fmt({"error": "Path generation did not return a job_id.", "model_group_id": model_group_id})

        gen_result = await client.poll_flow_job(gen_job_id, f"/flow/{gen_job_id}/results")

    if gen_result.get("status", "") == "failed":
        return _fmt({
            "error": "Path generation failed.",
            "model_group_id": model_group_id,
            "details": gen_result.get("error") or gen_result,
        })

    results = await client.flow_get_results(gen_job_id)
    summary = results.get("summary") or {}
    if results.get("feature_names"):
        feature_names = results["feature_names"]

    per_asset = _build_per_asset_output(summary, include_paths=True)
    feat_hint = f" Features for constraints: {', '.join(feature_names[:10])}." if feature_names else ""
    pid_str = f"'{portfolio_id}'" if portfolio_id else "'<find via list_portfolios>'"
    output = {
        "status": "completed",
        "model_group_id": model_group_id,
        "portfolio_id": portfolio_id,
        "flow_job_id": str(gen_job_id),
        "horizon": horizon,
        "n_paths": results.get("n_paths", n_paths),
        "feature_names": feature_names,
        "per_asset": per_asset,
        "next_steps": (
            f"Available actions: "
            f"simulate_flow_scenario(model_group_id='{model_group_id}', portfolio_id={pid_str}, constraints=...) for what-if scenarios | "
            f"test_flow_risk(portfolio_id={pid_str}, flow_job_id='{gen_job_id}') for risk metrics | "
            f"flow_validate(model_group_id='{model_group_id}') for model quality check | "
            f"list_flow_scenarios(model_group_id='{model_group_id}') to see past scenarios.{feat_hint}"
        ),
    }
    try:
        chart_html = flow_fan_chart(summary, horizon)
        if chart_html:
            return _with_widget(_fmt(output), chart_html)
    except Exception:
        logger.warning("Fan chart generation failed, returning text-only", exc_info=True)
    return _fmt(output)


# ── Flow MCP tools ──

@server.tool(
    name="train_flow_model",
    description=(
        "Train a generative Flow model on a portfolio and conditioning set. "
        "Returns immediately — training runs on a GPU and takes 5-15 minutes. "
        "After calling this, STOP and tell the user training has started. Let them keep chatting. "
        "The user will ask you to check progress — use check_flow_job(job_id=...) when they do. "
        "Do NOT automatically poll or call check_flow_job yourself. "
        "Requires conditioning_set_id (from list_feature_set_templates or create_feature_set) "
        "and tickers or portfolio_id."
    ),
    annotations=ToolAnnotations(title="Train Flow Model", destructiveHint=True, openWorldHint=True),
)
async def train_flow_model(
    conditioning_set_id: Annotated[str, Field(
        description="UUID of the conditioning set (from list_feature_set_templates or create_feature_set)."
    )],
    portfolio_id: Annotated[str | None, Field(
        description="UUID of an existing portfolio. If omitted, provide tickers instead.",
        default=None,
    )] = None,
    tickers: Annotated[list[str] | None, Field(
        description="Tickers to analyze (e.g. ['AAPL', 'MSFT']). Auto-creates a portfolio if portfolio_id is not given.",
        default=None,
    )] = None,
    weights: Annotated[list[float] | None, Field(
        description="Optional weights (must sum to 1.0). Defaults to equal weights.",
        default=None,
    )] = None,
    horizon: Annotated[int | None, Field(
        description="Forecast horizon in trading days. ~1 month = 20, ~1 quarter = 60, ~6 months = 120. Defaults to 60 if omitted.",
        default=None,
    )] = None,
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        result, mgid, pid, fnames = await _train_flow_model_impl(
            client, conditioning_set_id, portfolio_id, tickers, weights, horizon or 60,
        )
        if isinstance(result, str):
            return result
        return _fmt(result)
    except _FlowGPUBusy as e:
        return str(e)
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("train_flow_model failed: %s", e, exc_info=True)
        return "Flow training failed unexpectedly. Please try again."


@server.tool(
    name="check_flow_job",
    description=(
        "Check the status of an async Flow job (training, generation, or validation). "
        "Returns status ('running', 'completed', 'failed') and progress details. "
        "Does NOT return results — when completed, call get_flow_results(job_id) to fetch data. "
        "Call this ONCE, report status to the user, then STOP — do not poll in a loop. "
        "Typical times: training 5-15 min, generation 1-3 min, validation 3-5 min."
    ),
    annotations=ToolAnnotations(title="Check Flow Job", readOnlyHint=True),
)
async def check_flow_job(
    job_id: Annotated[str, Field(description="Job ID returned by train_flow_model or other flow tools")],
    job_type: Annotated[str, Field(
        description="Type of job: 'train' (default), 'generate', or 'validate'.",
        default="train",
    )] = "train",
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        if job_type == "train":
            result = await client.flow_train_status(job_id)
        elif job_type == "validate":
            result = await client.flow_validate_status(job_id)
        else:
            result = await client.flow_get_results(job_id)

        status = result.get("status", "unknown")
        output: dict = {"job_id": job_id, "status": status}

        if status == "completed":
            if job_type == "train":
                mgid = result.get("model_group_id", "")
                output["model_group_id"] = mgid
                output["feature_names"] = result.get("feature_names", [])
                output["next_steps"] = (
                    f"Training complete! Call generate_flow_paths(model_group_id='{mgid}') "
                    f"to produce simulated price trajectories."
                )
            elif job_type == "validate":
                output["next_steps"] = (
                    f"Validation complete! Call get_flow_results(job_id='{job_id}', job_type='validate') "
                    f"to see quality badge, pass rate, and per-feature metrics (Wasserstein, KS, coverage)."
                )
            else:
                mgid = result.get("model_group_id", "")
                output["next_steps"] = (
                    f"Generation complete! Use get_flow_results(job_id='{job_id}') to fetch full results. "
                    f"Use test_flow_risk(flow_job_id='{job_id}') for portfolio risk metrics."
                )
        elif status == "failed":
            output["error"] = result.get("error") or result.get("error_message") or "Unknown error"
        else:
            # Still running — surface progress info
            progress = result.get("progress") or result.get("result") or {}
            if isinstance(progress, dict):
                prog = progress.get("progress", {})
                if prog.get("phase"):
                    output["phase"] = prog["phase"]
                if prog.get("message"):
                    output["progress_message"] = prog["message"]
                if prog.get("step") is not None and prog.get("total_steps"):
                    output["progress"] = f"{prog['step']}/{prog['total_steps']}"
            current_epoch = result.get("current_epoch") or result.get("epoch")
            max_epochs = result.get("max_epochs")
            if current_epoch is not None:
                output["current_epoch"] = current_epoch
            if max_epochs is not None:
                output["max_epochs"] = max_epochs
            if current_epoch and max_epochs:
                output["progress"] = f"{current_epoch}/{max_epochs} epochs"
            output["hint"] = (
                "STOP. Job is still running. Report progress to the user and STOP calling tools. "
                "Do NOT call check_flow_job again — the user will ask you when they want an update. "
                "Suggest they can ask 'check my training job' in a few minutes."
            )

        return _fmt(output)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="generate_flow_paths",
    description=(
        "Generate simulated multi-step price trajectories from a trained Flow model. "
        "Returns per-asset percentile bands (p5/p25/p50/p75/p95 per timestep), "
        "sample paths per target asset, and scalar terminal statistics. "
        "Requires model_group_id from train_flow_model or list_model_groups. "
        "If paths already exist, returns cached results instantly. "
        "Path generation takes ~1-3 min on GPU. "
        "Defaults: horizon=60 (~1 quarter), n_paths=1000."
    ),
    annotations=ToolAnnotations(title="Generate Flow Paths", destructiveHint=True, openWorldHint=True),
)
async def generate_flow_paths(
    model_group_id: Annotated[str, Field(
        description="UUID of a trained Flow model group (from train_flow_model or list_model_groups)."
    )],
    portfolio_id: Annotated[str | None, Field(
        description="UUID of the portfolio. Resolved automatically if omitted.",
        default=None,
    )] = None,
    horizon: Annotated[int, Field(
        description="Forecast horizon in trading days. Default 60.",
        default=60,
    )] = 60,
    n_paths: Annotated[int, Field(
        description="Number of paths to generate. 1000 default.",
        default=1000,
    )] = 1000,
    price_history_length: Annotated[int | None, Field(
        description="Days of historical prices to include before the paths start. Defaults to horizon. Set higher (e.g. 120) to warm up indicators like MACD or z-score before forward_test_rules.",
        default=None,
    )] = None,
) -> list | str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        return await _generate_flow_paths_impl(client, model_group_id, portfolio_id, horizon, n_paths, price_history_length)
    except _FlowGPUBusy as e:
        return str(e)
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("generate_flow_paths failed: %s", e, exc_info=True)
        return "Flow path generation failed unexpectedly. Please try again."


# NOTE: `train_and_generate` was removed. The new-run mode dispatched
# training (async, 5-15 min) and then immediately invoked
# `_generate_flow_paths_impl` against an as-yet-untrained model — the
# generate call could only return a misleading "no results yet" or
# silently produce nothing useful. The resume mode was a pure rename of
# `generate_flow_paths`. Use the explicit chain instead:
#   train_flow_model -> check_flow_job -> flow_validate -> generate_flow_paths
# The agent already drives this chain cleanly (see stress-test run 4).


@server.tool(
    name="simulate_flow_scenario",
    description=(
        "Start constrained what-if scenario generation from a trained Flow model. "
        "Returns immediately with a job_id — use check_flow_job(job_id, job_type='generate') to poll for results. "
        "IMPORTANT: Always run generate_flow_paths FIRST on the same day before running any scenarios. "
        "This establishes a same-day baseline — when you retrieve results via get_flow_results, "
        "scenario_probability is automatically computed: the fraction of unconstrained baseline paths "
        "that naturally satisfy your constraints, telling you how likely the scenario is. "
        "≥5% = within normal range | 1-5% = rare | <1% = outside training distribution. "
        "scenario_probability is measured from the baseline n_paths (default 1000), giving a resolution floor of 0.1% — "
        "0% means below measurable threshold, not necessarily impossible. "
        "THREE WAYS TO GET 0% PROBABILITY — avoid all: "
        "(1) DURATION: mean-reverting features (VIX, spreads, rates) spike for days to weeks, not months. "
        "Always use t_start/t_end to window constraints (e.g. t_start=10, t_end=20), not the full horizon. "
        "(2) JUMP TOO ABRUPT: if today's value is far from the threshold, t_start must give enough time to get there. "
        "(3) MULTIPLE CONSTRAINTS: joint probability multiplies — if oil ≥155 is 2% and VIX ≥40 is 3%, the joint is ~0.06%, below the 0.1% resolution floor. "
        "With 2+ constraints, check scenario_probability first and consider testing each constraint individually before combining. "
        "Check last_price from generate_flow_paths first — if VIX is at 15 and you constrain it above 30 from day 5, "
        "that's a 2x move in 5 days (essentially never happens). Set t_start large enough for a realistic transition: "
        "the bigger the gap between current value and threshold, the later t_start should be. "
        "VIX all-time high ~89, never sustained above 30 for more than a few weeks. "
        "IMPORTANT: feature_name in constraints must be the DISPLAY NAME from feature_names "
        "(e.g. 'Apple Inc.', 'SPDR S&P 500 ETF Trust'), NOT ticker symbols. "
        "Constraint types: 'level' (absolute price bounds), 'return' (per-step return bounds). "
        "Pass portfolio_id through so test_flow_risk can be called directly on results. "
        "Run scenarios SEQUENTIALLY (one at a time), not in parallel, to avoid GPU queue contention."
    ),
    annotations=ToolAnnotations(title="Simulate Flow Scenario", readOnlyHint=True, openWorldHint=True),
)
async def simulate_flow_scenario(
    model_group_id: Annotated[str, Field(
        description="UUID of the model group with a trained Flow model (from train_flow_model)"
    )],
    constraints: Annotated[list[dict], Field(
        description=(
            "List of constraints. Each MUST have 'lower' and/or 'upper' (NOT 'threshold'). "
            "Required keys: feature_name, lower and/or upper. Optional: type, t_start, t_end. "
            "Example: {'feature_name': 'Equity Volatility (VIX)', 'lower': 30, 't_start': 10, 't_end': 25} "
            "(VIX > 30 from day 10 to 25). "
            "For upper bound: {'feature_name': 'Apple Inc.', 'upper': 200, 't_start': 5, 't_end': 40} "
            "(AAPL < 200). "
            "Types: 'level' (default, absolute price) or 'return'. "
            "feature_name must be DISPLAY NAME from feature_names (not ticker). "
            "ALWAYS set t_start/t_end to window the constraint realistically."
        )
    )],
    portfolio_id: Annotated[str | None, Field(
        description="UUID of the portfolio (from train_flow_model). Pass it through so test_flow_risk can be called directly on the results.",
        default=None,
    )] = None,
    n_paths: Annotated[int, Field(
        description="Number of paths to generate. More = better diversity. 1000 default.",
        default=1000,
    )] = 1000,
    horizon: Annotated[int | None, Field(
        description="Override horizon (defaults to training horizon).",
        default=None,
    )] = None,
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()

        # Auto-check: if no baseline exists for today, generate one first
        from datetime import date
        today = date.today().isoformat()
        try:
            baselines = await client.flow_list_baselines(model_group_id)
            has_today_baseline = any(
                b.get("created_at", "")[:10] == today
                for b in (baselines.get("baselines") or [])
            )
        except Exception:
            has_today_baseline = False

        baseline_job_id = None
        if not has_today_baseline:
            logger.info("No baseline for today — auto-generating before scenario")
            bl_job = await _retry_gpu_call(lambda: client.flow_generate_paths(
                model_group_id=model_group_id,
                n_paths=n_paths,
                horizon=horizon,
            ))
            baseline_job_id = bl_job.get("job_id")
            if baseline_job_id:
                # Poll until baseline completes (typically 1-2 min)
                bl_result = await client.poll_flow_job(baseline_job_id, f"/flow/{baseline_job_id}/results")
                if bl_result.get("status") == "failed":
                    logger.warning("Auto-baseline generation failed, proceeding with scenario anyway")

        # Dispatch constrained generation
        job = await _retry_gpu_call(lambda: client.flow_generate_constrained_paths(
            model_group_id=model_group_id,
            constraints=constraints,
            n_paths=n_paths,
            horizon=horizon,
        ))
        job_id = job.get("job_id")
        if not job_id:
            return "Error: Constrained generation did not return a job_id."

        pid_str = f"'{portfolio_id}'" if portfolio_id else "'<find via list_portfolios>'"
        output: dict = {
            "status": "generating",
            "model_group_id": model_group_id,
            "portfolio_id": portfolio_id,
            "flow_job_id": str(job_id),
            "constraints": constraints,
            "n_paths": n_paths,
            "next_steps": (
                f"Constrained generation started (job_id='{job_id}'). Takes 1-3 minutes. "
                f"STOP HERE — tell the user and wait. Do NOT poll in a loop. "
                f"The user can ask you to check with: check_flow_job(job_id='{job_id}', job_type='generate'). "
                f"Once completed: test_flow_risk(portfolio_id={pid_str}, flow_job_id='{job_id}') for futures-only risk, "
                f"or analyze_derivatives(flow_job_id='{job_id}', portfolio_id={pid_str}, options_positions=...) "
                f"if the user has options positions to overlay."
            ),
        }
        if baseline_job_id:
            output["auto_baseline_job_id"] = baseline_job_id
            output["note"] = "A baseline was auto-generated for today before running the scenario."

        return _fmt(output)
    except _FlowGPUBusy as e:
        return str(e)
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("simulate_flow_scenario failed: %s", e, exc_info=True)
        return "Constrained scenario generation failed unexpectedly. Please try again."


@server.tool(
    name="test_flow_risk",
    description=(
        "Run portfolio risk analytics on Flow-generated paths (FUTURES/EQUITIES ONLY — no options). "
        "Computes expected return, volatility, Sharpe ratio, Sortino ratio, Calmar ratio, "
        "VaR 95%, CVaR 95%, max drawdown, profitability rate, and return distribution percentiles. "
        "Requires portfolio_id and flow_job_id from generate_flow_paths, or simulate_flow_scenario. "
        "If the user has OPTIONS positions, use analyze_derivatives instead — it reprices options "
        "on every path using Black-76 and shows combined futures+options risk. "
        "TIP: Call this on multiple flow_job_ids (baseline + different scenarios) to build a "
        "side-by-side comparison of risk metrics across scenarios."
    ),
    annotations=ToolAnnotations(title="Test Flow Risk", readOnlyHint=True, openWorldHint=True),
)
async def test_flow_risk(
    portfolio_id: Annotated[str, Field(
        description="UUID of the portfolio (from generate_flow_paths, or list_portfolios)"
    )],
    flow_job_id: Annotated[str, Field(
        description="Flow generation job ID (from generate_flow_paths, or simulate_flow_scenario)"
    )],
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    if err := _validate_uuid(flow_job_id, "flow_job_id"):
        return err
    try:
        client = get_client()
        result = await client.flow_portfolio_test(portfolio_id, flow_job_id)

        agg = result.get("aggregated_results") or {}
        stats = result.get("summary_stats") or {}

        output = {
            "status": "completed",
            "portfolio_id": portfolio_id,
            "flow_job_id": flow_job_id,
            "n_days": result.get("n_days"),
            "risk_metrics": {
                "expected_return": agg.get("mean_return"),
                "volatility": agg.get("std_return"),
                "sharpe_ratio": agg.get("sharpe_ratio"),
                "sortino_ratio": agg.get("sortino_ratio"),
                "calmar_ratio": agg.get("calmar_ratio"),
                "var_95": agg.get("var_95"),
                "cvar_95": agg.get("cvar_95"),
                "max_drawdown": agg.get("max_drawdown"),
                "profitability_rate": agg.get("profitability_rate"),
            },
            "return_distribution": {
                "mean": stats.get("mean_return"),
                "median": stats.get("median_return"),
                "std": stats.get("mean_volatility") or agg.get("std_return"),
                "skewness": agg.get("skewness"),
                "kurtosis": agg.get("kurtosis"),
            },
        }

        try:
            card_html = flow_risk_card(result)
            if card_html:
                return _with_widget(_fmt(output), card_html)
        except Exception:
            logger.warning("Flow risk card generation failed, returning text-only", exc_info=True)

        return _fmt(output)
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("test_flow_risk failed: %s", e, exc_info=True)
        return "Flow portfolio risk test failed unexpectedly. Please try again."


@server.tool(
    name="list_flow_scenarios",
    description=(
        "List completed constrained scenarios for a Flow model group. "
        "Returns job IDs, constraints used, satisfaction rates, and timestamps. "
        "Use this to find previous scenario results without re-running them — "
        "pass any flow_job_id to test_flow_risk for risk metrics."
    ),
    annotations=ToolAnnotations(title="List Flow Scenarios", readOnlyHint=True),
)
async def list_flow_scenarios(
    model_group_id: Annotated[str, Field(
        description="UUID of the Flow model group (from train_flow_model, or list_model_groups)"
    )],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        result = await client.flow_list_scenarios(model_group_id)
        if isinstance(result, list):
            scenarios = result
        elif isinstance(result, dict):
            scenarios = result.get("scenarios", [result])
        else:
            scenarios = [result]
        return _fmt(scenarios)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_flow_baselines",
    description=(
        "List completed baseline (unconstrained) generation jobs for a Flow model group. "
        "Returns job IDs, path counts, horizons, and creation dates. "
        "Baselines are standalone unconstrained simulations used for comparison with scenarios. "
        "Use generate_flow_paths to create new baselines."
    ),
    annotations=ToolAnnotations(title="List Flow Baselines", readOnlyHint=True),
)
async def list_flow_baselines(
    model_group_id: Annotated[str, Field(
        description="UUID of the Flow model group"
    )],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        result = await client.flow_list_baselines(model_group_id)
        baselines = result.get("baselines", []) if isinstance(result, dict) else result
        return _fmt(baselines)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="download_flow_paths",
    description=(
        "Download all generated paths from a Flow generation job as CSV. "
        "Returns raw path data with columns: path_idx, day, then one column per feature. "
        "Works for both baseline and scenario generation jobs. "
        "Use the flow_job_id from generate_flow_paths, simulate_flow_scenario, or list_flow_baselines/list_flow_scenarios."
    ),
    annotations=ToolAnnotations(title="Download Flow Paths", readOnlyHint=True),
)
async def download_flow_paths(
    flow_job_id: Annotated[str, Field(
        description="Flow generation job ID (from generate_flow_paths or simulate_flow_scenario)"
    )],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(flow_job_id, "flow_job_id"):
        return err
    try:
        client = get_client()
        csv_bytes = await client.flow_download_paths(flow_job_id)
        # Return first 50 lines as preview + total line count
        lines = csv_bytes.decode('utf-8', errors='replace').split('\n')
        total_lines = len(lines)
        preview = '\n'.join(lines[:51])
        return _fmt({
            "total_rows": total_lines - 1,  # exclude header
            "preview_rows": min(50, total_lines - 1),
            "csv_preview": preview,
            "note": f"Showing first 50 of {total_lines - 1} rows. Full CSV has {total_lines - 1} data rows.",
        })
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="delete_flow_job",
    description="Delete a flow simulation job (baseline or constrained scenario). Permanent, cannot be undone.",
    annotations=ToolAnnotations(title="Delete Flow Job", destructiveHint=True),
)
async def delete_flow_job(
    job_id: Annotated[str, Field(description="The flow job UUID to delete (from list_flow_baselines or list_flow_scenarios)")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(job_id, "job_id"):
        return err
    try:
        client = get_client()
        result = await client.delete_flow_job(job_id)
        return _fmt({"message": f"Flow job {job_id} deleted.", **result})
    except SablierAPIError as e:
        return _api_error(e)


# ── Systematic Trading Rules ──────────────────────────────────────────────────

@server.tool(
    name="create_rule",
    description=(
        "Add a systematic trading rule to a portfolio. Rules are evaluated day-by-day on FLOW "
        "forward paths during forward_test_rules — not backtested on history.\n\n"
        "TWO RULE TYPES:\n"
        "  • Signal rules  (action.type='signal_weight') — continuous indicator → proportional position. For CTAs and trend-followers.\n"
        "  • Binary rules  (all other action types) — trigger fires → discrete weight change. For risk overlays, hard stops, regime gates.\n\n"
        "Use signal rules (priority 0) for the core strategy; binary rules (priority 1+) for risk overrides.\n\n"

        "── SIGNAL RULE ──\n"
        "trigger: {indicator, asset, params}  ← no operator/threshold\n"
        "action:  {type:'signal_weight', asset, normalizer, max_weight, min_weight}\n"
        "  normalizer = typical signal magnitude; clip(signal/normalizer, -1, 1) → position\n"
        "  weight = scaled*max_weight if scaled≥0 else scaled*|min_weight|\n\n"
        "  trigger={indicator:'macd_line', asset:'CL=F', params:{fast:12, slow:60}}\n"
        "  action={type:'signal_weight', asset:'CL=F', normalizer:2.0, max_weight:0.6, min_weight:-0.3}\n\n"
        "  trigger={indicator:'z_score', asset:'ZN=F', params:{window:60}}\n"
        "  action={type:'signal_weight', asset:'ZN=F', normalizer:2.0, max_weight:0.5, min_weight:-0.5}\n\n"

        "── BINARY RULE ──\n"
        "trigger: {indicator, asset, params, operator, threshold}  OR  {combinator:'all'|'any', conditions:[...]}\n"
        "  indicators: raw | moving_average | ema | rsi | bollinger_upper | bollinger_lower |\n"
        "              bollinger_width | macd_line | macd_signal | rolling_std | rolling_volatility | rate_of_change | z_score\n"
        "  asset: portfolio assets OR conditioning factors ('^VIX', 'DX-Y.NYB', 'T10Y2Y', 'ZN=F', ...)\n"
        "  operator: '>' | '<' | '>=' | '<=' | '==' | 'crosses_above' | 'crosses_below'\n"
        "action: exit | set_weight (exact value, negative=short) | scale_weight (multiplier) | reverse\n\n"
        "  trigger={indicator:'rsi', asset:'CL=F', params:{period:14}, operator:'>', threshold:70}\n"
        "  action={type:'exit', asset:'CL=F'}\n\n"
        "  trigger={combinator:'all', conditions:[\n"
        "    {indicator:'raw', asset:'^VIX', params:{}, operator:'>', threshold:30},\n"
        "    {indicator:'rsi', asset:'CL=F', params:{period:14}, operator:'>', threshold:65}]}\n"
        "  action={type:'scale_weight', asset:'CL=F', value:0.5}\n\n"
        "IMPORTANT: Trigger assets can be portfolio assets OR conditioning factors (VIX, DXY, etc.). "
        "For forward_test_rules, the FLOW model must include ALL referenced features — missing features cause rules to silently fail. "
        "For evaluate_rules (live data), any feature in training_data works with no model dependency."
    ),
    annotations=ToolAnnotations(title="Create Trading Rule"),
)
async def create_rule(
    portfolio_id: Annotated[str, Field(description="Portfolio UUID")],
    name: Annotated[str, Field(description="Short descriptive name for the rule")],
    trigger: Annotated[dict, Field(description="For BINARY rules: single condition {indicator,asset,params,operator,threshold} or multi-condition {combinator:'all'|'any', conditions:[...]}. For SIGNAL rules: just {indicator,asset,params} — no operator or threshold needed.")],
    action: Annotated[dict, Field(description="Signal: {type:'signal_weight', asset, normalizer, max_weight, min_weight} OR binary: {type:'exit'|'set_weight'|'scale_weight'|'reverse', asset, value?}")],
    description: Annotated[str | None, Field(description="Optional longer description")] = None,
    is_active: Annotated[bool, Field(description="Whether the rule is active (default false)")] = False,
    priority: Annotated[int, Field(description="Evaluation order when multiple rules fire (lower = first, default 0)")] = 0,
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        result = await client.create_trading_rule(
            portfolio_id=portfolio_id, name=name, trigger=trigger, action=action,
            description=description, is_active=is_active, priority=priority,
        )
        return _fmt({
            "rule_id": result.get("id"),
            "name": name,
            "is_active": is_active,
            "trigger": trigger,
            "action": action,
            "message": f"Rule '{name}' created. Use forward_test_rules to test on FLOW paths, or evaluate_rules to check against live market data.",
        })
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_rules",
    description="List all systematic trading rules attached to a portfolio, including their trigger/action definitions, active status, and priority order.",
    annotations=ToolAnnotations(title="List Trading Rules", readOnlyHint=True),
)
async def list_rules(
    portfolio_id: Annotated[str, Field(description="Portfolio UUID")],
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        result = await client.list_trading_rules(portfolio_id)
        rules = result.get("rules", [])
        return _fmt({
            "portfolio_id": portfolio_id,
            "count": len(rules),
            "rules": [{
                "rule_id": r["id"],
                "name": r["name"],
                "description": r.get("description"),
                "is_active": r["is_active"],
                "priority": r["priority"],
                "trigger": r["trigger"],
                "action": r["action"],
            } for r in rules],
        })
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="toggle_rule",
    description="Activate or deactivate a systematic trading rule. Only active rules are included in forward_test_rules by default.",
    annotations=ToolAnnotations(title="Toggle Trading Rule"),
)
async def toggle_rule(
    portfolio_id: Annotated[str, Field(description="Portfolio UUID")],
    rule_id: Annotated[str, Field(description="Rule UUID")],
    is_active: Annotated[bool, Field(description="True to activate, False to deactivate")],
) -> list | str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        await client.update_trading_rule(portfolio_id, rule_id, is_active=is_active)
        status = "activated" if is_active else "deactivated"
        return _fmt({"rule_id": rule_id, "is_active": is_active, "message": f"Rule {status}."})
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="delete_rule",
    description="Permanently delete a systematic trading rule from a portfolio.",
    annotations=ToolAnnotations(title="Delete Trading Rule"),
)
async def delete_rule(
    portfolio_id: Annotated[str, Field(description="Portfolio UUID")],
    rule_id: Annotated[str, Field(description="Rule UUID")],
) -> list | str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        await client.delete_trading_rule(portfolio_id, rule_id)
        return _fmt({"rule_id": rule_id, "deleted": True})
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="forward_test_rules",
    description=(
        "Forward-test systematic trading rules against FLOW-generated price paths. "
        "Returns TWO levels of output:\n"
        "  • combined_strategy — ALL rules applied together in priority order on every path. "
        "This is your actual strategy performance vs the base static portfolio.\n"
        "  • rule_attribution — each rule tested individually to show which rules help vs hurt.\n\n"
        "How it works:\n"
        "  1. Loads the FLOW price paths (same N paths for every evaluation — fair comparison)\n"
        "  2. Steps through each path day-by-day, applies rules in priority order, tracks P&L\n"
        "  3. Returns Sharpe, CVaR, max drawdown, return for combined strategy and each rule alone\n\n"
        "IMPORTANT: The FLOW model must include paths for ALL features referenced in rule triggers "
        "(both portfolio assets AND conditioning factors like VIX, DXY, etc.). Rules referencing features "
        "not in the FLOW model will silently fail — check warnings in the response.\n"
        "For checking rules against today's real market data (no FLOW dependency), use evaluate_rules instead.\n\n"
        "Prerequisites: (1) create rules with create_rule; "
        "(2) activate them with toggle_rule(is_active=True); "
        "(3) generate FLOW paths with generate_flow_paths.\n"
        "If rule_ids is omitted, tests all active rules."
    ),
    annotations=ToolAnnotations(title="Forward-Test Trading Rules"),
)
async def forward_test_rules(
    portfolio_id: Annotated[str, Field(description="Portfolio UUID")],
    flow_job_id: Annotated[str, Field(description="Completed FLOW job ID")],
    rule_ids: Annotated[list[str] | None, Field(description="Specific rule UUIDs to test. Omit to test all active rules.")] = None,
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    if err := _validate_uuid(flow_job_id, "flow_job_id"):
        return err
    try:
        client = get_client()
        result = await client.forward_test_rules(portfolio_id, flow_job_id, rule_ids)

        base = result.get("base_strategy", {})
        base_stats = base.get("summary_stats", {})

        def _stats_row(stats, base_stats):
            return {
                "mean_return": stats.get("mean_return"),
                "mean_sharpe": stats.get("mean_sharpe"),
                "mean_max_drawdown": stats.get("mean_max_drawdown"),
                "mean_volatility": stats.get("mean_volatility"),
                "cvar_95": stats.get("cvar_95"),
                "vs_base_return": round(stats.get("mean_return", 0) - base_stats.get("mean_return", 0), 4)
                    if stats.get("mean_return") is not None else None,
                "vs_base_sharpe": round(stats.get("mean_sharpe", 0) - base_stats.get("mean_sharpe", 0), 4)
                    if stats.get("mean_sharpe") is not None else None,
                "vs_base_drawdown": round(stats.get("mean_max_drawdown", 0) - base_stats.get("mean_max_drawdown", 0), 4)
                    if stats.get("mean_max_drawdown") is not None else None,
            }

        # Combined strategy (all rules together — the actual strategy)
        combined = result.get("combined_strategy")
        combined_out = None
        if combined:
            combined_out = {
                "rule_names": combined.get("rule_names"),
                **_stats_row(combined.get("summary_stats", {}), base_stats),
            }
        elif result.get("combined_strategy_error"):
            combined_out = {"error": result["combined_strategy_error"]}

        # Individual rule attribution (each rule alone vs base)
        attribution_rows = []
        for r in result.get("rule_attribution", []):
            if r.get("error"):
                attribution_rows.append({"rule": r["rule_name"], "priority": r.get("priority"), "error": r["error"]})
                continue
            attribution_rows.append({
                "rule": r["rule_name"],
                "rule_id": r["rule_id"],
                "priority": r.get("priority"),
                **_stats_row(r.get("summary_stats", {}), base_stats),
            })

        out = {
            "portfolio_id": portfolio_id,
            "flow_job_id": flow_job_id,
            "horizon_days": result.get("horizon_days"),
            "n_rules": result.get("n_rules_tested"),
            "base_strategy": _stats_row(base_stats, base_stats),
            "combined_strategy": combined_out,
            "rule_attribution": attribution_rows,
            "tip": (
                "combined_strategy shows performance with ALL rules applied together in priority order — "
                "this is your actual strategy. rule_attribution shows each rule tested alone to identify "
                "which rules help vs hurt. Negative vs_base_drawdown = less drawdown = better."
            ),
        }
        warnings = result.get("warnings", [])
        if warnings:
            out["warnings"] = warnings
        return _fmt(out)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="evaluate_rules",
    description=(
        "Check which portfolio trading rules trigger on TODAY's real market data. "
        "Unlike forward_test_rules (which tests against simulated FLOW paths), this evaluates "
        "rules against actual historical prices from training_data — no FLOW model needed.\n\n"
        "Returns per-rule: triggered (bool), action prescribed, current indicator values. "
        "Also returns recommended_weights (combined effect of all triggered rules) and weight_changes.\n\n"
        "Data source: training_data (refreshed daily at 21:00 UTC after US market close). "
        "On weekends/holidays, evaluates against the most recent trading day.\n\n"
        "Use this for daily 'any rules fired?' monitoring. "
        "For simulated forward-testing across 1000+ scenarios, use forward_test_rules instead."
    ),
    annotations=ToolAnnotations(title="Evaluate Rules (Live Data)", readOnlyHint=True),
)
async def evaluate_rules(
    portfolio_id: Annotated[str, Field(description="Portfolio UUID")],
    rule_ids: Annotated[list[str] | None, Field(
        description="Specific rule UUIDs to evaluate. Omit to evaluate all active rules.",
        default=None,
    )] = None,
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        result = await client.evaluate_rules(portfolio_id, rule_ids)

        rules = result.get("rules", [])
        triggered = [r for r in rules if r.get("triggered")]
        not_triggered = [r for r in rules if not r.get("triggered")]

        output = {
            "evaluation_date": result.get("evaluation_date"),
            "n_rules_evaluated": len(rules),
            "n_rules_triggered": len(triggered),
            "triggered_rules": [{
                "rule_name": r["rule_name"],
                "rule_id": r["rule_id"],
                "priority": r.get("priority"),
                "action": r.get("action_prescribed"),
                "indicator_values": r.get("indicator_values"),
            } for r in triggered],
            "not_triggered_rules": [{
                "rule_name": r["rule_name"],
                "rule_id": r["rule_id"],
                "indicator_values": r.get("indicator_values"),
            } for r in not_triggered],
            "base_weights": result.get("base_weights"),
            "recommended_weights": result.get("recommended_weights"),
            "weight_changes": result.get("weight_changes"),
        }
        warnings = result.get("warnings", [])
        if warnings:
            output["warnings"] = warnings
        return _fmt(output)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_flow_results",
    description=(
        "Get results of a completed Flow job (generation or validation). "
        "For generation jobs (default): returns per-asset terminal statistics, "
        "percentile bands (P5–P95 timeseries), sample paths, and price_history for indicator warmup. "
        "Set summary_only=true to keep stats + bands but drop sample paths (~60%% smaller). "
        "Use download_flow_paths to get full raw path data as CSV. "
        "For scenario jobs: also returns satisfaction_rate and constraint details. "
        "For validation jobs (job_type='validate'): returns quality badge, pass_rate, and per-feature metrics "
        "(Wasserstein distance, KS tests, coverage, marginal checks). "
        "Use check_flow_job first to verify the job is completed."
    ),
    annotations=ToolAnnotations(title="Get Flow Results", readOnlyHint=True),
)
async def get_flow_results(
    job_id: Annotated[str, Field(description="Job ID from generate_flow_paths, simulate_flow_scenario, or flow_validate")],
    job_type: Annotated[str, Field(
        description="Type of job: 'generate' (default) for path generation, or 'validate' for validation results.",
        default="generate",
    )] = "generate",
    summary_only: Annotated[bool, Field(
        description="If true, return terminal stats + percentile bands but skip individual sample paths. "
        "~60%% smaller output. Use download_flow_paths to get full raw path data separately.",
        default=False,
    )] = False,
    max_sample_paths: Annotated[int, Field(
        description="Max sample paths to return per asset (default 10, max 50). Ignored if summary_only=true.",
        default=10,
    )] = 10,
) -> list | str:
    if err := _require_auth():
        return err
    try:
        client = get_client()

        # Validation jobs use a different endpoint and return different data
        if job_type == "validate":
            results = await client.flow_validate_results(job_id)
            status = results.get("status", "unknown")
            if status != "completed":
                return _fmt({"error": f"Validation not completed (status: {status}). Use check_flow_job(job_id='{job_id}', job_type='validate') to monitor."})
            output = {
                "job_id": job_id,
                "model_group_id": results.get("model_group_id", ""),
                "status": "completed",
                "quality": results.get("quality"),
                "pass_rate": results.get("pass_rate"),
                "n_paths": results.get("n_paths"),
                "horizon": results.get("horizon"),
                "feature_names": results.get("feature_names"),
                "metrics": results.get("metrics"),
            }
            return _fmt(output)

        results = await client.flow_get_results(job_id)
        status = results.get("status", "unknown")
        if status != "completed":
            return _fmt({"error": f"Job not completed (status: {status}). Use check_flow_job to monitor progress."})

        summary = results.get("summary") or {}
        baseline_summary = results.get("baseline_summary") or {}
        satisfaction_rate = results.get("satisfaction_rate")
        result_horizon = results.get("horizon")
        mgid = results.get("model_group_id", "")

        # Build per-asset output with configurable sample path count
        max_paths = min(max(1, max_sample_paths), 50)
        per_asset = _build_per_asset_output(
            summary, include_paths=True, summary_only=summary_only,
            baseline_summary=baseline_summary,
        )
        # Adjust sample path count if requested beyond default 10
        if max_paths != 10 and not summary_only:
            MAX_TS_POINTS = 60
            for name, data in summary.items():
                if not isinstance(data, dict) or data.get("feature_type") != "target":
                    continue
                ts = data.get("timeseries") or {}
                sample = ts.get("sample_paths") or []
                if sample and name in per_asset:
                    step = max(1, len(sample) // max_paths)
                    picked = sample[::step][:max_paths]
                    def _ds(arr):
                        if not arr or len(arr) <= MAX_TS_POINTS:
                            return arr
                        s = max(1, len(arr) // MAX_TS_POINTS)
                        return arr[::s]
                    per_asset[name]["sample_paths"] = [_ds(p) for p in picked]

        output = {
            "job_id": job_id,
            "model_group_id": mgid,
            "horizon": result_horizon,
            "n_paths": results.get("n_paths"),
            "per_asset": per_asset,
        }
        if results.get("price_history"):
            output["price_history"] = results["price_history"]
        if satisfaction_rate is not None:
            output["satisfaction_rate"] = satisfaction_rate
            output["constraints"] = results.get("constraints", [])
        if results.get("scenario_probability") is not None:
            output["scenario_probability"] = results["scenario_probability"]
            output["scenario_probability_note"] = results.get("scenario_probability_note")
        elif results.get("scenario_probability_note"):
            # No baseline found — surface the warning
            output["scenario_probability_note"] = results["scenario_probability_note"]

        try:
            chart_html = flow_fan_chart(summary, result_horizon, results.get("constraints"))
            if chart_html:
                return _with_widget(_fmt(output), chart_html)
        except Exception:
            logger.warning("Fan chart generation failed, returning text-only", exc_info=True)

        return _fmt(output)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="flow_validate",
    description=(
        "Validate a trained Flow model against real data. "
        "Generates paths and compares them to historical distributions using "
        "Wasserstein distance, KS tests, coverage tests, and marginal distribution checks. "
        "Returns immediately with a job_id — validation runs asynchronously (~3-5 min). "
        "Use check_flow_job(job_id=..., job_type='validate') to monitor progress. "
        "Requires a trained Flow model (run train_flow_model first)."
    ),
    annotations=ToolAnnotations(title="Validate Flow Model", destructiveHint=True, openWorldHint=True),
)
async def flow_validate(
    model_group_id: Annotated[str, Field(description="UUID of the model group with a trained Flow model")],
    n_paths: Annotated[int, Field(description="Number of paths to generate for validation (default 500)", default=500)] = 500,
    horizon: Annotated[int | None, Field(description="Validation horizon (defaults to training horizon)", default=None)] = None,
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()

        job = await _retry_gpu_call(lambda: client.flow_validate(
            model_group_id=model_group_id,
            n_paths=n_paths,
            horizon=horizon,
        ))
        job_id = job.get("job_id")
        if not job_id:
            return "Error: Flow validation did not return a job_id."

        return _fmt({
            "status": "validating",
            "job_id": job_id,
            "model_group_id": model_group_id,
            "next_steps": (
                f"Validation started (job_id='{job_id}'). It typically takes 3-5 minutes. "
                f"Use check_flow_job(job_id='{job_id}', job_type='validate') to check progress. "
                f"Once completed, results include a quality badge and per-feature metrics."
            ),
        })
    except _FlowGPUBusy as e:
        return str(e)
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Derivatives Analysis
# ══════════════════════════════════════════════════


@server.tool(
    name="analyze_derivatives",
    description=(
        "Run options risk analysis on FLOW-generated paths for a mixed futures + options portfolio. "
        "Reprices each option at every timestep of every path using Black-76, then computes portfolio-level "
        "risk metrics (VaR, CVaR, Sharpe, Sortino, max drawdown) and per-position Greeks (delta, gamma, vega, theta, rho). "
        "Returns separate risk breakdowns for: combined portfolio, futures-only, and options-only components, "
        "plus P&L timeseries percentile bands. "
        "Requires a flow_job_id from generate_flow_paths or simulate_flow_scenario. "
        "For scenario analysis: run simulate_flow_scenario first (e.g., 'VIX > 30 and crude drops 20%'), "
        "then call this tool to see how your options hedge performs under that scenario."
    ),
    annotations=ToolAnnotations(title="Analyze Derivatives Risk", readOnlyHint=True, openWorldHint=True),
)
async def analyze_derivatives(
    flow_job_id: Annotated[str, Field(
        description="Flow generation job ID (from generate_flow_paths, or simulate_flow_scenario)"
    )],
    options_positions: Annotated[list[dict], Field(
        description=(
            "List of option positions. Each dict must have: "
            "underlying (display_name of the futures, e.g. 'E-mini S&P 500 Futures'), "
            "option_type ('call' or 'put'), strike (float), days_to_expiry (int), "
            "quantity (int, negative for short), implied_vol (float, annualized e.g. 0.20). "
            "Optional: entry_premium (float), multiplier (float, defaults to contract spec)."
        )
    )],
    portfolio_id: Annotated[str | None, Field(
        description="Portfolio UUID for underlying futures weights. Optional for standalone options analysis.",
        default=None,
    )] = None,
    risk_free_rate: Annotated[float, Field(
        description="Annualized risk-free rate (default 0.045 = 4.5%)",
        default=0.045,
    )] = 0.045,
    capital: Annotated[float | None, Field(
        description="Override portfolio capital. If None, uses portfolio's capital.",
        default=None,
    )] = None,
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(flow_job_id, "flow_job_id"):
        return err
    if portfolio_id:
        if err := _validate_uuid(portfolio_id, "portfolio_id"):
            return err

    try:
        client = get_client()
        result = await client.analyze_derivatives(
            flow_job_id=flow_job_id,
            options_positions=options_positions,
            portfolio_id=portfolio_id,
            risk_free_rate=risk_free_rate,
            capital=capital,
        )

        # Extract key metrics for compact output
        portfolio = result.get("portfolio", {})
        options_only = result.get("options_only", {})
        positions = result.get("positions", [])
        agg_greeks = result.get("aggregate_greeks", {})

        output = {
            "status": "completed",
            "flow_job_id": flow_job_id,
            "portfolio_id": portfolio_id,
            "n_paths": result.get("n_paths"),
            "horizon": result.get("horizon"),
            "combined_portfolio": {
                "expected_return": portfolio.get("expected_return"),
                "volatility_ann": portfolio.get("volatility_ann"),
                "sharpe_ratio": portfolio.get("sharpe_ratio"),
                "sortino_ratio": portfolio.get("sortino_ratio"),
                "var_95": portfolio.get("var_95"),
                "cvar_95": portfolio.get("cvar_95"),
                "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
                "profitability": portfolio.get("profitability"),
                "terminal_pnl": portfolio.get("terminal_pnl"),
            },
            "options_only": {
                "expected_return": options_only.get("expected_return"),
                "var_95": options_only.get("var_95"),
                "terminal_pnl": options_only.get("terminal_pnl"),
            },
            "aggregate_greeks": agg_greeks,
            "per_position": [
                {
                    "position": p.get("position"),
                    "greeks": p.get("greeks"),
                    "terminal_payoff": p.get("terminal_payoff"),
                }
                for p in positions
            ],
        }

        return _fmt(output)
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("analyze_derivatives failed: %s", e, exc_info=True)
        return "Derivatives analysis failed unexpectedly. Please try again."


@server.tool(
    name="price_option",
    description=(
        "Price a single option on a futures contract using the Black-76 model. "
        "Returns the option price, per-contract value (price × contract multiplier), "
        "and analytical Greeks (delta, gamma, vega, theta, rho). "
        "Supports all major futures: ES=F, NQ=F, CL=F, GC=F, SI=F, ZB=F, ZN=F, ZC=F, ZW=F, ZS=F, etc. "
        "If a flow_job_id is provided, also computes an Esscher fair-value estimate from FLOW paths "
        "(captures fat tails and vol clustering that Black-76 misses). "
        "Use this for quick pricing checks; use analyze_derivatives for full portfolio risk."
    ),
    annotations=ToolAnnotations(title="Price Option", readOnlyHint=True),
)
async def price_option_tool(
    underlying_ticker: Annotated[str, Field(
        description="Ticker of the underlying futures (e.g. 'ES=F', 'CL=F', 'GC=F')"
    )],
    strike: Annotated[float, Field(
        description="Option strike price"
    )],
    days_to_expiry: Annotated[int, Field(
        description="Trading days until option expiration"
    )],
    option_type: Annotated[str, Field(
        description="'call' or 'put'",
        default="call",
    )] = "call",
    implied_vol: Annotated[float | None, Field(
        description="Annualized implied volatility (e.g. 0.20 for 20%). If omitted, uses 20% default.",
        default=None,
    )] = None,
    risk_free_rate: Annotated[float, Field(
        description="Annualized risk-free rate (default 0.045 = 4.5%)",
        default=0.045,
    )] = 0.045,
    flow_job_id: Annotated[str | None, Field(
        description="Optional Flow job ID. If provided, also computes Esscher fair-value from FLOW paths.",
        default=None,
    )] = None,
) -> str:
    if err := _require_auth():
        return err
    if flow_job_id:
        if err := _validate_uuid(flow_job_id, "flow_job_id"):
            return err

    try:
        client = get_client()
        result = await client.price_option(
            underlying_ticker=underlying_ticker,
            strike=strike,
            days_to_expiry=days_to_expiry,
            option_type=option_type,
            implied_vol=implied_vol,
            risk_free_rate=risk_free_rate,
            flow_job_id=flow_job_id,
        )
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("price_option failed: %s", e, exc_info=True)
        return "Option pricing failed unexpectedly. Please try again."


# ══════════════════════════════════════════════════
# Historical Backtesting
# ══════════════════════════════════════════════════


@server.tool(
    name="backtest_rules",
    description=(
        "Run a historical backtest of trading rules on REAL market data — not simulated FLOW paths. "
        "Tests how rules would have performed over a historical period. "
        "Returns same structure as forward_test_rules (base vs combined vs per-rule attribution) "
        "PLUS monthly returns table, drawdown series, turnover stats, and transaction cost analysis. "
        "Prerequisites: create rules with create_rule (and activate them). "
        "Transaction costs: configurable (default 10bps per trade). "
        "Warmup period (default 252 days) pre-fills indicator state before the test period starts. "
        "For forward-looking testing on synthetic FLOW paths, use forward_test_rules instead."
    ),
    annotations=ToolAnnotations(title="Backtest Rules (Historical)", readOnlyHint=True, openWorldHint=True),
)
async def backtest_rules(
    portfolio_id: Annotated[str, Field(description="Portfolio UUID")],
    start_date: Annotated[str, Field(description="Backtest start date (YYYY-MM-DD). E.g. '2020-01-01' for a COVID-era test.")],
    end_date: Annotated[str | None, Field(description="End date (YYYY-MM-DD). Defaults to latest available data.")] = None,
    rule_ids: Annotated[list[str] | None, Field(description="Specific rule UUIDs. Omit to test all active rules.")] = None,
    warmup_days: Annotated[int, Field(description="Days of history before start_date for indicator seeding (default 252 = ~1 year)")] = 252,
    cost_bps: Annotated[float, Field(description="Transaction cost in basis points per trade (default 10.0)")] = 10.0,
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(portfolio_id, "portfolio_id"):
        return err
    try:
        client = get_client()
        result = await client.backtest_rules(
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
            rule_ids=rule_ids,
            warmup_days=warmup_days,
            cost_bps=cost_bps,
        )
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("backtest_rules failed: %s", e, exc_info=True)
        return "Historical backtest failed unexpectedly. Please try again."


# ══════════════════════════════════════════════════
# Universe Screening
# ══════════════════════════════════════════════════


@server.tool(
    name="screen_universe",
    description=(
        "Screen the asset universe by metadata (sector, region, asset type) and price-based metrics "
        "(momentum, volatility, percentile rank, z-score, RSI, MA distance). "
        "Only screens assets already in the Sablier catalog with training data. "
        "Use search_features + add_feature first to expand the catalog if needed. "
        "Metadata fields: sector, region, asset_type, category, source. "
        "Price fields: momentum_20d/60d/252d, volatility_20d/60d, percentile_1y, z_score_60d, "
        "ma_distance_50d/200d, rsi_14, current_price, change_1d_pct/1w_pct/1m_pct. "
        "Operators: eq, neq, in (metadata); gt, gte, lt, lte, between (price). "
        "Results include computed metrics per asset. Use top results to create a portfolio."
    ),
    annotations=ToolAnnotations(title="Screen Universe", readOnlyHint=True, openWorldHint=True),
)
async def screen_universe(
    criteria: Annotated[list[dict], Field(
        description=(
            "List of filter criteria. Each: {field, operator, value}. "
            "Examples: "
            "{field: 'sector', operator: 'in', value: ['Technology', 'Healthcare']}, "
            "{field: 'momentum_60d', operator: 'gt', value: 0.05}, "
            "{field: 'volatility_20d', operator: 'lt', value: 0.30}, "
            "{field: 'percentile_1y', operator: 'gt', value: 80}"
        )
    )],
    sort_by: Annotated[str | None, Field(description="Field to sort results by (default: first price criterion or momentum_60d)")] = None,
    sort_order: Annotated[str, Field(description="'desc' (default) or 'asc'")] = "desc",
    limit: Annotated[int, Field(description="Max results (default 20)")] = 20,
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        result = await client.screen_universe(
            criteria=criteria,
            sort_by=sort_by or "momentum_60d",
            sort_order=sort_order,
            limit=limit,
        )
        return _fmt(result)
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("screen_universe failed: %s", e, exc_info=True)
        return "Universe screening failed unexpectedly. Please try again."


# ══════════════════════════════════════════════════
# Market Radar
# ══════════════════════════════════════════════════


@server.tool(
    name="market_radar",
    description=(
        "Get a Bloomberg-terminal-grade market briefing with 50+ indicators and computed regime signals. "
        "Returns current levels, 1-day/1-week/1-month changes, z-scores, and percentiles for equities, "
        "rates, credit, FX, commodities, volatility, international markets, and crypto. "
        "Also computes cross-asset signals: Risk-On/Risk-Off score, yield curve regime, credit stress, "
        "volatility regime, sector rotation, copper/gold ratio, stock-bond correlation, and inflation momentum. "
        "Flags significant moves (|z-score| > 2) as content opportunities. "
        "Use this to understand the current market environment and decide what Sablier analyses to run."
    ),
    annotations=ToolAnnotations(title="Market Radar", readOnlyHint=True, openWorldHint=True),
)
async def market_radar() -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        data = await client.get_market_radar()

        # Build a concise summary for the LLM
        regime = data.get("regime_summary", {})
        sig_moves = data.get("significant_moves", [])
        sectors = data.get("sector_performance", {})

        summary_parts = []

        # Regime overview
        roro = regime.get("roro_score", {})
        summary_parts.append(f"REGIME: {roro.get('label', 'Unknown')} (RORO score: {roro.get('score', 'N/A')})")

        curve = regime.get("yield_curve", {})
        summary_parts.append(f"YIELD CURVE: {curve.get('label', 'Unknown')} (10Y-2Y: {curve.get('spread_10y2y', 'N/A')}%)")

        credit = regime.get("credit_stress", {})
        summary_parts.append(f"CREDIT: {credit.get('label', 'Unknown')} (HY OAS: {credit.get('hy_oas', 'N/A')}bp)")

        vol = regime.get("vol_regime", {})
        summary_parts.append(f"VOLATILITY: {vol.get('label', 'Unknown')} (VIX: {vol.get('vix', 'N/A')})")

        rotation = regime.get("sector_rotation", {})
        summary_parts.append(f"SECTORS: {rotation.get('label', 'Unknown')} — {rotation.get('breadth', 'N/A')}")

        cg = regime.get("copper_gold_trend", {})
        summary_parts.append(f"COPPER/GOLD: {cg.get('label', 'Unknown')}")

        sb = regime.get("stock_bond_correlation", {})
        summary_parts.append(f"STOCK-BOND CORR: {sb.get('label', 'Unknown')} ({sb.get('correlation', 'N/A')})")

        infl = regime.get("inflation_momentum", {})
        summary_parts.append(f"INFLATION EXPECTATIONS: {infl.get('label', 'Unknown')}")

        summary_parts.append("")

        # Significant moves
        if sig_moves:
            summary_parts.append(f"SIGNIFICANT MOVES ({len(sig_moves)}):")
            for m in sig_moves[:10]:
                direction = "↑" if m.get("change_1w_pct", 0) and m["change_1w_pct"] > 0 else "↓"
                summary_parts.append(
                    f"  {direction} {m['name']} ({m['ticker']}): "
                    f"z={m['z_score']:.1f}, 1w: {m.get('change_1w_pct', 'N/A')}%, "
                    f"current: {m['current']} [{m['level'].upper()}]"
                )
        else:
            summary_parts.append("NO SIGNIFICANT MOVES (quiet market)")

        summary_parts.append("")

        # Sector performance
        leaders = sectors.get("leaders", [])
        laggards = sectors.get("laggards", [])
        if leaders:
            summary_parts.append("SECTOR LEADERS (5d): " + ", ".join(
                f"{s['name']} ({s['return_5d']:+.1f}%)" for s in leaders
            ))
        if laggards:
            summary_parts.append("SECTOR LAGGARDS (5d): " + ", ".join(
                f"{s['name']} ({s['return_5d']:+.1f}%)" for s in laggards
            ))

        summary_text = "\n".join(summary_parts)

        # Also include the full data for detailed analysis
        full_output = {
            "summary": summary_text,
            "regime_summary": regime,
            "significant_moves": sig_moves,
            "sector_performance": sectors,
            "indicators_count": data.get("indicators_count", 0),
            "as_of_date": data.get("as_of_date"),
            "all_indicators": data.get("indicators", []),
        }

        return _fmt(full_output)
    except SablierAPIError as e:
        return _api_error(e)
    except Exception as e:
        logger.error("market_radar failed: %s", e, exc_info=True)
        return f"Market radar failed: {e}"


# ══════════════════════════════════════════════════
# Account
# ══════════════════════════════════════════════════

@server.tool(
    name="whoami",
    description=(
        "Quick account summary: name, email, tier, credit balance, and billing period. "
        "Use this first to orient yourself — single call covers identity and credit status."
    ),
    annotations=ToolAnnotations(title="Who Am I", readOnlyHint=True),
)
async def whoami() -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        user = await client.get_user_info()
        credits = await client.get_credits()
        output = {
            "name": user.get("name"),
            "email": user.get("email"),
            "company": user.get("company"),
            "tier": credits.get("tier", "free"),
            "credits_remaining": credits.get("credits_remaining", 0),
            "credits_used": credits.get("credits_used", 0),
            "purchased_credits": credits.get("purchased_credits", 0),
            "allow_overage": credits.get("allow_overage", False),
        }
        if credits.get("billing_period_end"):
            output["billing_period_end"] = credits["billing_period_end"]
        return _fmt(output)
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Billing
# ══════════════════════════════════════════════════

@server.tool(
    name="get_credits",
    description=(
        "Credit balance details: used, remaining, purchased packs, and overage status. "
        "Use whoami for a quick summary; use this when you need the full credit object."
    ),
    annotations=ToolAnnotations(title="Get Credits", readOnlyHint=True),
)
async def get_credits() -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        data = await client.get_credits()
        return _fmt(data)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_billing_info",
    description=(
        "Subscription plan details: tier limits, overage rates, and per-operation costs. "
        "Use this to understand pricing before running expensive operations (not for credit balance — use whoami or get_credits)."
    ),
    annotations=ToolAnnotations(title="Get Billing Info", readOnlyHint=True),
)
async def get_billing_info() -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        data = await client.get_billing_info()
        return _fmt(data)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_billing_usage",
    description=(
        "Get detailed usage breakdown for the current or a specific billing month. "
        "Shows per-operation counts, included limits, overage counts, and costs. "
        "Month format: YYYY-MM (e.g. '2026-03')."
    ),
    annotations=ToolAnnotations(title="Get Billing Usage", readOnlyHint=True),
)
async def get_billing_usage(
    month: Annotated[str | None, Field(
        description="Billing month in YYYY-MM format. Defaults to current month.",
        default=None,
    )] = None,
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        data = await client.get_billing_usage(month)
        return _fmt(data)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="subscribe",
    description=(
        "Subscribe to a Sablier plan (new subscription). Returns a Stripe Checkout URL to complete payment. "
        "Tiers: 'pro' (Pro Monthly €499/mo or Pro Annual €349/mo — 1,000 credits/month, overage at €0.50/credit monthly or €0.35/credit annual). "
        "Enterprise pricing is custom — contact team@sablier.it. "
        "To manage an existing subscription (upgrade, downgrade, cancel, update payment), "
        "use manage_subscription instead."
    ),
    annotations=ToolAnnotations(title="Subscribe", destructiveHint=True),
)
async def subscribe(
    tier: Annotated[str, Field(description="Subscription tier: 'pro' (€499/mo or €349/mo annual, 1000 credits)")],
) -> str:
    if err := _require_auth():
        return err
    if tier not in ('pro', 'pro_annual', 'enterprise'):
        return "Invalid tier. Choose 'pro' (€499/mo), 'pro_annual' (€349/mo billed annually), or contact team@sablier.it for Enterprise."
    try:
        client = get_client()
        data = await client.create_checkout_session(tier)
        return _fmt(data)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="manage_subscription",
    description=(
        "Open the Stripe Customer Portal to manage an EXISTING subscription: "
        "upgrade, downgrade, cancel, or update payment method. "
        "Returns a portal URL. For new subscriptions, use subscribe instead."
    ),
    annotations=ToolAnnotations(title="Manage Subscription", readOnlyHint=True),
)
async def manage_subscription() -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        data = await client.create_portal_session()
        return _fmt(data)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="list_credit_packs",
    description=(
        "List available credit packs for one-time purchase. "
        "Returns pack options with credits, price, and per-credit cost. "
        "Credit packs are available to all tiers and never expire. "
        "To purchase, use buy_credit_pack with the pack_id."
    ),
    annotations=ToolAnnotations(title="List Credit Packs", readOnlyHint=True),
)
async def list_credit_packs() -> str:
    try:
        client = get_client()
        data = await client.list_credit_packs()
        return _fmt(data)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="buy_credit_pack",
    description=(
        "Purchase a one-time credit pack. Returns a Stripe Checkout URL to complete payment. "
        "Available packs: 'pack_100' (100 credits, €69), 'pack_500' (500 credits, €299), "
        "'pack_1000' (1000 credits, €549). Credits are added instantly after payment and never expire. "
        "Use list_credit_packs to see current pricing. Use get_credits to check your balance first."
    ),
    annotations=ToolAnnotations(title="Buy Credit Pack", destructiveHint=True),
)
async def buy_credit_pack(
    pack_id: Annotated[str, Field(description="Pack to purchase: 'pack_100', 'pack_500', or 'pack_1000'")],
) -> str:
    if err := _require_auth():
        return err
    if pack_id not in ('pack_100', 'pack_500', 'pack_1000'):
        return "Invalid pack_id. Choose 'pack_100' (100 credits, €69), 'pack_500' (500 credits, €299), or 'pack_1000' (1000 credits, €549). Use list_credit_packs for details."
    try:
        client = get_client()
        data = await client.purchase_credit_pack(pack_id)
        return _fmt(data)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="toggle_overage",
    description=(
        "Enable or disable on-demand overage credits for Pro subscribers. "
        "When enabled, operations continue beyond the monthly credit allocation "
        "and are billed at the overage rate (€0.50/credit monthly, €0.35/credit annual). "
        "When disabled, operations are blocked once monthly credits run out. "
        "Only available for Pro tier — free users should buy credit packs or subscribe."
    ),
    annotations=ToolAnnotations(title="Toggle Overage", destructiveHint=True),
)
async def toggle_overage(
    enabled: Annotated[bool, Field(description="True to enable overage, False to disable")],
) -> str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        data = await client.toggle_overage(enabled)
        return _fmt(data)
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════


def main():
    from sablier_mcp._core import _transport, _port, CaptureRedirectMiddleware

    if _transport == "stdio":
        server.run(transport="stdio")
    else:
        import asyncio
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
        asyncio.run(uvicorn_server.serve())


if __name__ == "__main__":
    main()
