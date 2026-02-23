"""
Sablier MCP Server

Gives AI agents the ability to perform regime-dependent factor modeling,
qualitative analysis, portfolio risk testing, and return simulation
through the Sablier platform.

Supports both local (stdio) and remote (streamable-http) transport.
"""

import json
import os
import re
import urllib.parse
from typing import Annotated

from mcp.types import (
    EmbeddedResource,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from pydantic import Field
from mcp_use import MCPServer

from sablier_mcp.client import SablierClient, SablierAPIError
from sablier_mcp.widgets import (
    betas_heatmap,
    grain_score_card,
    portfolio_overview,
    risk_dashboard,
    training_progress,
)

server = MCPServer(name="Sablier", version="0.1.0")

_client: SablierClient | None = None


def get_client() -> SablierClient:
    global _client
    if _client is None:
        _client = SablierClient()
    return _client


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


_NOT_LOGGED_IN = (
    "Error: You need to log in first. "
    "If you already have a Sablier account, use the login tool with your email and password. "
    "If you're new, use create_account to sign up — it's free."
)


def _require_auth() -> str | None:
    """Return an error string if not authenticated, else None."""
    client = get_client()
    if not client.is_authenticated:
        return _NOT_LOGGED_IN
    return None


# ══════════════════════════════════════════════════
# Account — Sign up & Log in
# ══════════════════════════════════════════════════


@server.tool(
    name="create_account",
    description=(
        "Create a free Sablier account. This is the first step for new users. "
        "After signing up, the user will receive a verification email — "
        "they must click the link to activate their account. "
        "Once verified, they can use the login tool to start analyzing."
    ),
)
async def create_account(
    email: Annotated[str, Field(description="User's email address")],
    name: Annotated[str, Field(description="User's full name")],
    company: Annotated[str, Field(description="Company or organization name")],
    role: Annotated[str, Field(description="User's role (e.g. 'Portfolio Manager', 'Analyst', 'Student')")],
    password: Annotated[str, Field(description="Password (min 8 characters)")],
) -> str:
    try:
        client = get_client()
        result = await client.register(
            email=email, name=name, company=company, role=role, password=password
        )
        return _fmt({
            "status": "account_created",
            "message": (
                f"Account created for {email}! "
                "A verification email has been sent — the user needs to click the link in their inbox to activate the account. "
                "Once verified, use the login tool with this email and password to start using Sablier."
            ),
            "user_id": result.get("user_id"),
        })
    except SablierAPIError as e:
        if e.status_code == 409 or "already" in str(e.detail).lower():
            return (
                "This email is already registered. "
                "Use the login tool instead, or check your inbox for the verification email if you haven't activated yet."
            )
        return _api_error(e)


@server.tool(
    name="login",
    description=(
        "Log in to Sablier with email and password. This authenticates the session "
        "so you can access portfolios, run analyses, and manage scenarios. "
        "The user must have a verified account (created via create_account). "
        "IMPORTANT: Always start here if the user hasn't logged in yet. "
        "After login, the user can also access the Sablier web UI with the same credentials."
    ),
)
async def login(
    email: Annotated[str, Field(description="Email address")],
    password: Annotated[str, Field(description="Password")],
) -> str:
    try:
        client = get_client()
        result = await client.login(email=email, password=password)
        user = result.get("user", {})
        return _fmt({
            "status": "logged_in",
            "message": f"Welcome back, {user.get('name', 'there')}! You're now logged in and ready to go.",
            "user": {
                "name": user.get("name"),
                "email": user.get("email"),
                "company": user.get("company"),
            },
        })
    except SablierAPIError as e:
        if e.status_code == 401:
            return "Incorrect email or password. Please try again, or use create_account if you don't have an account yet."
        if e.status_code == 403 or "verif" in str(e.detail).lower():
            return "Your email hasn't been verified yet. Please check your inbox and click the verification link first."
        return _api_error(e)


# ══════════════════════════════════════════════════
# Search & Discovery
# ══════════════════════════════════════════════════


@server.tool(
    name="search_features",
    description="Search for tickers (stocks, ETFs) and market features (VIX, DXY, rates). Use this to find valid tickers before create_portfolio, or to discover macro factors. Start here if the user mentions a company or indicator you need to look up. WORKFLOW: search_features → create_portfolio → run_full_analysis (or step-by-step: create_models → train_models → simulate_betas).",
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
    description="List the user's existing portfolios with names, IDs, asset compositions, and status. Use this first when the user refers to a portfolio by name — you need the portfolio ID and its tickers for modeling. WORKFLOW: list_portfolios → list_feature_set_templates (pick factors) → run_full_analysis.",
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
                "assets": p.get("assets", []),
                "status": p.get("status"),
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
    description="Create a new portfolio from tickers and weights. Weights must sum to 1.0. NEXT STEP: ask the user 'What market drivers do you want to test this portfolio against?' and call list_feature_set_templates to show options (Macro, Market, Themes), then run_full_analysis.",
)
async def create_portfolio(
    name: Annotated[str, Field(description="Portfolio name (e.g. 'Tech Portfolio')")],
    tickers: Annotated[list[str], Field(description="Ticker symbols (e.g. ['AAPL', 'MSFT', 'NVDA'])")],
    weights: Annotated[list[float], Field(description="Corresponding weights (e.g. [0.4, 0.3, 0.3])")],
    description: Annotated[str, Field(description="Optional description", default="")] = "",
) -> str:
    if err := _require_auth():
        return err
    if len(tickers) != len(weights):
        return "Error: tickers and weights must have the same length"

    try:
        assets = [{"ticker": t, "weight": w} for t, w in zip(tickers, weights)]
        client = get_client()
        result = await client.create_portfolio(name, assets, description=description or None)
        return _fmt({
            "id": result["id"],
            "name": result["name"],
            "assets": result.get("assets", []),
            "target_set_id": result.get("target_set_id"),
            "status": result.get("status"),
            "message": "Portfolio created. Use list_model_groups to check for existing models, or create_models to create new ones.",
        })
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Qualitative Analysis (GRAIN)
# ══════════════════════════════════════════════════


@server.tool(
    name="analyze_qualitative",
    description="Run qualitative analysis on tickers for specific themes. Scans SEC filings (10-K, 10-Q) and earnings calls to score how exposed each company is to a theme (0-100 scale). Returns scores, supporting evidence passages, and sources. To analyze a portfolio, first use list_portfolios to get its tickers. Auto-polls up to 5 min; if it times out use get_analysis_status.",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def analyze_qualitative(
    tickers: Annotated[list[str], Field(description="Tickers to analyze (e.g. ['AAPL', 'MSFT'])")],
    themes: Annotated[list[str], Field(description="Themes to score (e.g. ['AI exposure', 'China risk', 'debt levels'])")],
    source_types: Annotated[list[str] | None, Field(description="Optional filter: ['10-K', '10-Q', 'earnings_call']", default=None)] = None,
    min_year: Annotated[int | None, Field(description="Earliest filing year to include", default=None)] = None,
    max_year: Annotated[int | None, Field(description="Latest filing year to include", default=None)] = None,
) -> list | str:
    if err := _require_auth():
        return err
    try:
        client = get_client()
        job = await client.start_grain_analysis(
            tickers=tickers,
            themes=themes,
            source_types=source_types,
            min_year=min_year,
            max_year=max_year,
        )
        job_id = job["job_id"]

        # Poll until complete
        result = await client.poll_grain_job(job_id)

        if result.get("status") == "failed":
            return f"Analysis failed: {result.get('error_message', 'Unknown error')}"

        if result.get("status") != "completed":
            return _fmt({
                "status": result.get("status"),
                "job_id": job_id,
                "message": "Analysis still running. Use get_analysis_status with this job_id to check later.",
            })

        # Summarize results — API nests as results.results (list of theme objects)
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
                "direction": theme.get("direction"),
                "confidence": theme.get("confidence"),
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
                        "passage": (ev.get("passage") or "")[:300],
                        "source": ev.get("source"),
                        "source_type": ev.get("source_type"),
                        "fiscal_period": ev.get("fiscal_period"),
                        "why_relevant": ev.get("why_relevant"),
                    })
                theme_summary["ticker_scores"].append(ticker_entry)
            summary["themes"].append(theme_summary)

        return _with_widget(_fmt(summary), grain_score_card(summary))
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_analysis_status",
    description="Check the status of a running GRAIN analysis job. Only needed if analyze_qualitative timed out — pass the job_id it returned.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_analysis_status(
    job_id: Annotated[str, Field(description="The GRAIN job ID from analyze_qualitative")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(job_id, "job_id"):
        return err
    try:
        client = get_client()
        result = await client.get_grain_job(job_id)
        return _fmt({
            "job_id": job_id,
            "status": result.get("status"),
            "progress": result.get("progress"),
            "error_message": result.get("error_message"),
        })
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
    description="Browse the Factors Library — pre-built sets of market drivers organized by category (Macro, Market, Themes). Each template defines what forces the portfolio is tested against (e.g. interest rates, volatility, commodities, credit spreads). Returns conditioning_set_id needed by create_models or run_full_analysis. IMPORTANT: Present these as 'What do you want to test your portfolio against?' — avoid jargon like 'conditioning set'. Show the template names and their factors so the user can pick.",
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
    name="create_models",
    description="Set up an analysis for a portfolio against chosen market drivers. Requires: conditioning_set_id from list_feature_set_templates. Returns model_group_id. NEXT: train_models → simulate_betas. TIP: prefer run_full_analysis which does everything in one call.",
)
async def create_models(
    conditioning_set_id: Annotated[str, Field(description="UUID of the conditioning set (market factors)")],
    asset_tickers: Annotated[list[str], Field(description="Tickers to create models for (e.g. ['AAPL', 'MSFT'])")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(conditioning_set_id, "conditioning_set_id"):
        return err
    try:
        client = get_client()
        result = await client.batch_create_models(
            conditioning_set_id=conditioning_set_id,
            asset_tickers=asset_tickers,
        )
        return _fmt({
            "model_group_id": result.get("model_group_id"),
            "conditioning_set_id": result.get("conditioning_set_id"),
            "total_created": result.get("total_created"),
            "total_failed": result.get("total_failed"),
            "models": result.get("created_models", []),
            "message": "Models created. Next step: use train_models with this model_group_id to start training.",
        })
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════


@server.tool(
    name="train_models",
    description="Start training for all models in a model group. Requires: model_group_id from create_models or list_model_groups. Returns job_id. Use get_training_progress to poll until 'completed', then simulate_betas. TIP: prefer run_full_analysis which handles the full chain automatically.",
)
async def train_models(
    model_group_id: Annotated[str, Field(description="UUID of the model group to train")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        result = await client.train_batch(
            model_group_id=model_group_id,
            training_mode="single_shot_linear",
        )
        return _fmt({
            "batch_id": result.get("batch_id"),
            "job_id": result.get("job_id"),
            "status": result.get("status"),
            "total_models": result.get("total_models"),
            "message": "Training started. Use get_training_progress with the job_id to monitor.",
        })
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_training_progress",
    description="Check how the analysis training is progressing. Pass the job_id from train_models. Once status is 'completed', use simulate_betas. Tell the user which asset is currently being processed and overall progress percentage.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_training_progress(
    job_id: Annotated[str, Field(description="The training job ID from train_models")],
) -> list:
    if err := _require_auth():
        return err
    if err := _validate_uuid(job_id, "job_id"):
        return err
    try:
        client = get_client()
        result = await client.get_batch_training_progress(job_id)
        data = {
            "job_id": result.get("job_id"),
            "status": result.get("status"),
            "current_asset": result.get("current_asset"),
            "completed_models": result.get("completed_models"),
            "total_models": result.get("total_models"),
            "progress_percent": result.get("progress_percent"),
            "current_epoch": result.get("current_epoch"),
            "max_epochs": result.get("max_epochs"),
            "train_loss": result.get("train_loss"),
            "val_loss": result.get("val_loss"),
            "error_message": result.get("error_message"),
        }
        return _with_widget(_fmt(data), training_progress(data))
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Factor Analysis (Betas Simulation)
# ══════════════════════════════════════════════════


@server.tool(
    name="simulate_betas",
    description="Compute how each asset in a trained model group responds to the chosen market drivers. Requires: model_group_id (status must be 'trained'). Auto-polls up to 5 min. Returns per-asset factor exposures + simulation_batch_id. NEXT: use test_portfolio_risk for risk metrics or simulate_returns for what-if scenarios. The results include factor_means and factor_stds — use these to show the user current market levels when they want to run scenarios.",
    annotations=ToolAnnotations(openWorldHint=True),
)
async def simulate_betas(
    model_group_id: Annotated[str, Field(description="UUID of the trained model group")],
    horizon: Annotated[int, Field(description="Forecast horizon in trading days", default=20)] = 20,
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(model_group_id, "model_group_id"):
        return err
    try:
        client = get_client()
        batch = await client.simulate_betas_batch(
            model_group_id=model_group_id,
            simulation_mode="single_shot_linear",
            horizon=horizon,
        )
        sim_batch_id = batch.get("simulation_batch_id")

        # Poll until complete
        results = await client.poll_betas_batch(sim_batch_id)

        if not results or not results.get("all_completed"):
            return _fmt({
                "simulation_batch_id": sim_batch_id,
                "status": "running",
                "completed": results.get("completed_count", 0) if results else 0,
                "total": results.get("total_count", 0) if results else 0,
                "message": "Simulation still running. Use get_betas_results to check later.",
            })

        per_asset = results.get("per_asset_results", {})
        summary = {
            "simulation_batch_id": sim_batch_id,
            "conditioning_features": results.get("conditioning_features", []),
            "completed": results.get("completed_count"),
            "total": results.get("total_count"),
            "assets": {},
        }
        for asset, asset_data in per_asset.items():
            summary["assets"][asset] = {
                "status": asset_data.get("status"),
                "linear_betas": asset_data.get("linear_betas"),
                "alpha": asset_data.get("alpha"),
                "residual_std": asset_data.get("residual_std"),
            }

        factor_stats = results.get("factor_stats", {})
        if factor_stats:
            summary["factor_means"] = factor_stats.get("means")
            summary["factor_stds"] = factor_stats.get("stds")

        return _with_widget(_fmt(summary), betas_heatmap(summary))
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_betas_results",
    description="Get full beta results from a completed simulation. Only needed if simulate_betas timed out or you want raw details. Pass the simulation_batch_id from simulate_betas.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_betas_results(
    simulation_batch_id: Annotated[str, Field(description="The simulation batch ID from simulate_betas")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(simulation_batch_id, "simulation_batch_id"):
        return err
    try:
        client = get_client()
        results = await client.get_betas_batch_results(simulation_batch_id)
        return _fmt(results)
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Return Simulation
# ══════════════════════════════════════════════════


@server.tool(
    name="simulate_returns",
    description="Simulate return distributions under a what-if scenario. Requires: simulation_batch_id from simulate_betas or run_full_analysis. IMPORTANT: Before calling this, show the user the full list of market drivers from the betas results (conditioning_features) along with their current values (factor_means from the betas output). Ask: 'Which of these would you like to change, and to what value?' Then pass the modified values as the factors dict. Factor keys must match the conditioning_features exactly. Auto-polls until ready.",
    annotations=ToolAnnotations(openWorldHint=True),
)
async def simulate_returns(
    simulation_batch_id: Annotated[str, Field(description="From simulate_betas")],
    factors: Annotated[dict[str, float], Field(description="Factor scenario values (e.g. {'VIX': 35, 'DXY': 110})")],
    n_samples: Annotated[int, Field(description="Number of Monte Carlo samples", default=5000)] = 5000,
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(simulation_batch_id, "simulation_batch_id"):
        return err
    try:
        client = get_client()
        batch = await client.simulate_returns_batch(
            simulation_batch_id=simulation_batch_id,
            factors=factors,
            n_samples=n_samples,
        )
        returns_batch_id = batch.get("returns_batch_id")

        results = await client.poll_returns_batch(returns_batch_id)

        if not results or not results.get("all_completed"):
            return _fmt({
                "returns_batch_id": returns_batch_id,
                "status": "running",
                "message": "Returns simulation still running. Use get_returns_results to check later.",
            })

        return _fmt(results)
    except SablierAPIError as e:
        return _api_error(e)


@server.tool(
    name="get_returns_results",
    description="Get results from a completed returns simulation. Only needed if simulate_returns timed out. Pass the returns_batch_id from simulate_returns.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_returns_results(
    returns_batch_id: Annotated[str, Field(description="The returns batch ID from simulate_returns")],
) -> str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(returns_batch_id, "returns_batch_id"):
        return err
    try:
        client = get_client()
        results = await client.get_returns_batch_results(returns_batch_id)
        return _fmt(results)
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Portfolio Risk Testing
# ══════════════════════════════════════════════════


@server.tool(
    name="test_portfolio_risk",
    description="Run a portfolio risk test. Requires: simulation_batch_id from simulate_betas or run_full_analysis. Returns expected return, VaR 95%, CVaR 95%, risk contribution per factor, and marginal contribution per asset. Weights must sum to 1.0 and tickers must match the simulated assets.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def test_portfolio_risk(
    simulation_batch_id: Annotated[str, Field(description="From simulate_betas")],
    weights: Annotated[dict[str, float], Field(description="Portfolio weights by ticker (e.g. {'AAPL': 0.4, 'MSFT': 0.3})")],
) -> list:
    if err := _require_auth():
        return err
    if err := _validate_uuid(simulation_batch_id, "simulation_batch_id"):
        return err
    try:
        client = get_client()
        result = await client.portfolio_test(simulation_batch_id, weights)
        data = {
            "portfolio_betas": result.get("portfolio_betas"),
            "weighted_beta": result.get("weighted_beta"),
            "expected_return": result.get("expected_return"),
            "portfolio_alpha": result.get("portfolio_alpha"),
            "var_95": result.get("var_95"),
            "cvar_95": result.get("cvar_95"),
            "diversification_ratio": result.get("diversification_ratio"),
            "risk_contribution": result.get("risk_contribution"),
            "marginal_ctr": result.get("marginal_ctr"),
            "n_assets": result.get("n_assets"),
        }
        return _with_widget(_fmt(data), risk_dashboard(data))
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Scenarios
# ══════════════════════════════════════════════════


@server.tool(
    name="create_scenario",
    description="Save a named what-if scenario for reuse. Requires a model_id from list_model_groups. Factor spec format: {'type': 'fixed', 'value': 35} for exact value, {'type': 'percentile', 'value': 95} for historical percentile, {'type': 'shock', 'value': 2.0} for std dev shift. Use simulate_returns to run the scenario.",
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
    name="run_full_analysis",
    description=(
        "One-shot analysis: builds factor models, trains them, and computes how each asset responds to market drivers — "
        "all in one call. This replicates the full Moment workflow from the Sablier UI. "
        "Requires: conditioning_set_id (from list_feature_set_templates — ask the user 'what do you want to test your portfolio against?') "
        "and asset_tickers (from the portfolio). Returns per-asset factor exposures (betas) + simulation_batch_id. "
        "After this completes, use test_portfolio_risk for risk metrics or simulate_returns for what-if scenarios."
    ),
)
async def run_full_analysis(
    conditioning_set_id: Annotated[str, Field(description="UUID of the conditioning set (from list_feature_set_templates)")],
    asset_tickers: Annotated[list[str], Field(description="Tickers to model (e.g. ['AAPL', 'MSFT', 'NVDA'])")],
    horizon: Annotated[int, Field(description="Forecast horizon in trading days", default=20)] = 20,
) -> list | str:
    if err := _require_auth():
        return err
    if err := _validate_uuid(conditioning_set_id, "conditioning_set_id"):
        return err

    client = get_client()

    try:
        # Step 1: Create models
        create_result = await client.batch_create_models(
            conditioning_set_id=conditioning_set_id,
            asset_tickers=asset_tickers,
        )
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

        # Step 2: Train
        train_result = await client.train_batch(
            model_group_id=model_group_id,
            training_mode="single_shot_linear",
        )
        job_id = train_result.get("job_id")
        if not job_id:
            return "Error: training did not return a job_id."

        # Step 3: Poll training until complete
        train_status = await client.poll_training(job_id)
        if train_status.get("status") == "failed":
            return _fmt({
                "error": "Training failed.",
                "model_group_id": model_group_id,
                "details": train_status.get("error_message"),
            })
        if train_status.get("status") != "completed":
            return _fmt({
                "status": "training_timeout",
                "model_group_id": model_group_id,
                "job_id": job_id,
                "message": "Training is still running. Use get_training_progress with job_id to monitor, then simulate_betas once complete.",
            })

        # Step 4: Simulate betas
        batch = await client.simulate_betas_batch(
            model_group_id=model_group_id,
            simulation_mode="single_shot_linear",
            horizon=horizon,
        )
        sim_batch_id = batch.get("simulation_batch_id")

        # Step 5: Poll betas until complete
        results = await client.poll_betas_batch(sim_batch_id)

        if not results or not results.get("all_completed"):
            return _fmt({
                "status": "simulation_timeout",
                "model_group_id": model_group_id,
                "simulation_batch_id": sim_batch_id,
                "message": "Betas simulation still running. Use get_betas_results with simulation_batch_id to check later.",
            })

        # Build summary
        per_asset = results.get("per_asset_results", {})
        summary = {
            "status": "completed",
            "model_group_id": model_group_id,
            "simulation_batch_id": sim_batch_id,
            "models_created": total_created,
            "conditioning_features": results.get("conditioning_features", []),
            "assets": {},
        }
        for asset, asset_data in per_asset.items():
            summary["assets"][asset] = {
                "status": asset_data.get("status"),
                "linear_betas": asset_data.get("linear_betas"),
                "alpha": asset_data.get("alpha"),
                "residual_std": asset_data.get("residual_std"),
            }

        factor_stats = results.get("factor_stats", {})
        if factor_stats:
            summary["factor_means"] = factor_stats.get("means")
            summary["factor_stds"] = factor_stats.get("stds")

        summary["next_steps"] = (
            "Use test_portfolio_risk with simulation_batch_id and portfolio weights to get VaR/CVaR, "
            "or simulate_returns with factor values to generate return distributions under a scenario."
        )

        return _with_widget(_fmt(summary), betas_heatmap(summary))
    except SablierAPIError as e:
        return _api_error(e)


# ══════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════


def main():
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    port = int(os.getenv("MCP_PORT", "8000"))
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="streamable-http", port=port)


if __name__ == "__main__":
    main()
