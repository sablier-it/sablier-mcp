"""
Sablier MCP Server

Gives AI agents the ability to perform regime-dependent factor modeling,
qualitative analysis, portfolio risk testing, and return simulation
through the Sablier platform.

Supports both local (stdio) and remote (streamable-http) transport.
"""

import json
import os
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


# ══════════════════════════════════════════════════
# Search & Discovery
# ══════════════════════════════════════════════════


@server.tool(
    name="search_features",
    description="Search for tickers (stocks, ETFs) and market features (VIX, DXY, rates). Use this to find available tickers before creating a portfolio, or to discover macro factors for conditioning sets.",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def search_features(
    query: Annotated[str, Field(description="Search term (e.g. 'AAPL', 'technology', 'volatility', 'gold')")],
    is_asset: Annotated[bool | None, Field(description="If True, only assets. If False, only indicators.", default=None)] = None,
    limit: Annotated[int, Field(description="Max results (default 20)", default=20)] = 20,
) -> str:
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


@server.tool(
    name="get_portfolio",
    description="Get detailed information about a specific portfolio including assets and weights.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_portfolio(
    portfolio_id: Annotated[str, Field(description="The portfolio UUID")],
) -> str:
    client = get_client()
    result = await client.get_portfolio(portfolio_id)
    return _fmt(result)


@server.tool(
    name="create_portfolio",
    description="Create a new portfolio from tickers and weights. Weights should sum to 1.0. This automatically creates the underlying feature sets needed for modeling.",
)
async def create_portfolio(
    name: Annotated[str, Field(description="Portfolio name (e.g. 'Tech Portfolio')")],
    tickers: Annotated[list[str], Field(description="Ticker symbols (e.g. ['AAPL', 'MSFT', 'NVDA'])")],
    weights: Annotated[list[float], Field(description="Corresponding weights (e.g. [0.4, 0.3, 0.3])")],
    description: Annotated[str, Field(description="Optional description", default="")] = "",
) -> str:
    if len(tickers) != len(weights):
        return "Error: tickers and weights must have the same length"

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


# ══════════════════════════════════════════════════
# Qualitative Analysis (GRAIN)
# ══════════════════════════════════════════════════


@server.tool(
    name="analyze_qualitative",
    description="Run GRAIN qualitative analysis on tickers for specific themes. Analyzes SEC filings (10-K, 10-Q) and earnings call transcripts to score thematic exposures. Returns scores, evidence passages, and source breakdowns. Auto-polls until results are ready (up to 5 minutes).",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def analyze_qualitative(
    tickers: Annotated[list[str], Field(description="Tickers to analyze (e.g. ['AAPL', 'MSFT'])")],
    themes: Annotated[list[str], Field(description="Themes to score (e.g. ['AI exposure', 'China risk', 'debt levels'])")],
    source_types: Annotated[list[str] | None, Field(description="Optional filter: ['10-K', '10-Q', 'earnings_transcript']", default=None)] = None,
    min_year: Annotated[int | None, Field(description="Earliest filing year to include", default=None)] = None,
    max_year: Annotated[int | None, Field(description="Latest filing year to include", default=None)] = None,
) -> list | str:
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


@server.tool(
    name="get_analysis_status",
    description="Check the status of a running GRAIN qualitative analysis. Use this if analyze_qualitative timed out.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_analysis_status(
    job_id: Annotated[str, Field(description="The GRAIN job ID from analyze_qualitative")],
) -> str:
    client = get_client()
    result = await client.get_grain_job(job_id)
    return _fmt({
        "job_id": job_id,
        "status": result.get("status"),
        "progress": result.get("progress"),
        "error_message": result.get("error_message"),
    })


# ══════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════


@server.tool(
    name="list_model_groups",
    description="List all model groups with their training and simulation status. A model group contains per-asset models sharing the same conditioning set (market factors).",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_model_groups() -> str:
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


@server.tool(
    name="list_feature_set_templates",
    description="List available conditioning set templates (predefined market factor sets). Use these when creating models — a conditioning set defines what factors the model learns exposures to.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_feature_set_templates() -> str:
    client = get_client()
    templates = await client.list_feature_set_templates()
    return _fmt(templates)


@server.tool(
    name="create_models",
    description="Batch create per-asset factor models. Each ticker gets its own model that learns regime-dependent exposures to the factors in the conditioning set.",
)
async def create_models(
    conditioning_set_id: Annotated[str, Field(description="UUID of the conditioning set (market factors)")],
    asset_tickers: Annotated[list[str], Field(description="Tickers to create models for (e.g. ['AAPL', 'MSFT'])")],
) -> str:
    client = get_client()
    result = await client.batch_create_models(
        conditioning_set_id=conditioning_set_id,
        asset_tickers=asset_tickers,
    )
    return _fmt({
        "total_created": result.get("total_created"),
        "total_failed": result.get("total_failed"),
        "models": result.get("models", []),
        "message": "Models created. Use train_models with the model_group_id to start training.",
    })


# ══════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════


@server.tool(
    name="train_models",
    description="Start batch training for all models in a model group. Training runs on GPU. Use get_training_progress to monitor. Modes: single_shot_linear (fast, ~1 min), single_shot_nonlinear (~5 min), rollout_nonlinear (~15 min).",
)
async def train_models(
    model_group_id: Annotated[str, Field(description="UUID of the model group to train")],
    training_mode: Annotated[str, Field(description="Training mode", default="single_shot_linear")] = "single_shot_linear",
    max_epochs: Annotated[int, Field(description="Maximum training epochs", default=100)] = 100,
) -> str:
    client = get_client()
    result = await client.train_batch(
        model_group_id=model_group_id,
        training_mode=training_mode,
        max_epochs=max_epochs,
    )
    return _fmt({
        "batch_id": result.get("batch_id"),
        "job_id": result.get("job_id"),
        "status": result.get("status"),
        "total_models": result.get("total_models"),
        "message": "Training started. Use get_training_progress with the job_id to monitor.",
    })


@server.tool(
    name="get_training_progress",
    description="Check the progress of a batch training job including current epoch, loss, and completion status.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_training_progress(
    job_id: Annotated[str, Field(description="The training job ID from train_models")],
) -> list:
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


# ══════════════════════════════════════════════════
# Factor Analysis (Betas Simulation)
# ══════════════════════════════════════════════════


@server.tool(
    name="simulate_betas",
    description="Compute regime-dependent factor exposures (betas) for all assets in a model group. Unlike static betas, these change based on current market conditions. Runs on GPU, auto-polls until ready (up to 5 min).",
    annotations=ToolAnnotations(openWorldHint=True),
)
async def simulate_betas(
    model_group_id: Annotated[str, Field(description="UUID of the trained model group")],
    simulation_mode: Annotated[str, Field(description="Must match training mode", default="single_shot_linear")] = "single_shot_linear",
    horizon: Annotated[int, Field(description="Forecast horizon in trading days", default=20)] = 20,
) -> list | str:
    client = get_client()
    batch = await client.simulate_betas_batch(
        model_group_id=model_group_id,
        simulation_mode=simulation_mode,
        horizon=horizon,
    )
    sim_batch_id = batch.get("simulation_batch_id")

    # Poll until complete
    results = await client.poll_betas_batch(sim_batch_id)

    if not results:
        return _fmt({
            "simulation_batch_id": sim_batch_id,
            "status": "running",
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


@server.tool(
    name="get_betas_results",
    description="Get detailed factor beta results from a completed simulation batch. Includes per-asset betas, sensitivity curves, factor statistics, and residual correlations.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_betas_results(
    simulation_batch_id: Annotated[str, Field(description="The simulation batch ID from simulate_betas")],
) -> str:
    client = get_client()
    results = await client.get_betas_batch_results(simulation_batch_id)
    return _fmt(results)


# ══════════════════════════════════════════════════
# Return Simulation
# ══════════════════════════════════════════════════


@server.tool(
    name="simulate_returns",
    description="Simulate return distributions under a specific factor scenario. Given factor values (e.g. VIX=35 for stress), samples thousands of possible return outcomes. Auto-polls until ready.",
    annotations=ToolAnnotations(openWorldHint=True),
)
async def simulate_returns(
    simulation_batch_id: Annotated[str, Field(description="From simulate_betas")],
    factors: Annotated[dict[str, float], Field(description="Factor scenario values (e.g. {'VIX': 35, 'DXY': 110})")],
    n_samples: Annotated[int, Field(description="Number of Monte Carlo samples", default=5000)] = 5000,
) -> str:
    client = get_client()
    batch = await client.simulate_returns_batch(
        simulation_batch_id=simulation_batch_id,
        factors=factors,
        n_samples=n_samples,
    )
    returns_batch_id = batch.get("returns_batch_id")

    results = await client.poll_returns_batch(returns_batch_id)

    if not results:
        return _fmt({
            "returns_batch_id": returns_batch_id,
            "status": "running",
            "message": "Returns simulation still running.",
        })

    return _fmt(results)


@server.tool(
    name="get_returns_results",
    description="Get results from a completed returns simulation.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_returns_results(
    returns_batch_id: Annotated[str, Field(description="The returns batch ID from simulate_returns")],
) -> str:
    client = get_client()
    results = await client.get_returns_batch_results(returns_batch_id)
    return _fmt(results)


# ══════════════════════════════════════════════════
# Portfolio Risk Testing
# ══════════════════════════════════════════════════


@server.tool(
    name="test_portfolio_risk",
    description="Run a portfolio risk test using simulation results. Returns portfolio-level factor betas, expected return, VaR, CVaR, risk contribution per factor, and marginal contribution per asset.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def test_portfolio_risk(
    simulation_batch_id: Annotated[str, Field(description="From simulate_betas")],
    weights: Annotated[dict[str, float], Field(description="Portfolio weights by ticker (e.g. {'AAPL': 0.4, 'MSFT': 0.3})")],
) -> list:
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


# ══════════════════════════════════════════════════
# Scenarios
# ══════════════════════════════════════════════════


@server.tool(
    name="create_scenario",
    description="Create a named what-if scenario. Each factor uses: {'type': 'fixed', 'value': 35} for exact value, {'type': 'percentile', 'value': 95} for historical percentile, or {'type': 'shock', 'value': 2.0} for std dev shift.",
)
async def create_scenario(
    model_id: Annotated[str, Field(description="UUID of the model this scenario applies to")],
    name: Annotated[str, Field(description="Scenario name (e.g. 'Recession', 'Tech Bubble')")],
    factor_values: Annotated[dict[str, dict], Field(description="Factor specs (e.g. {'VIX': {'type': 'fixed', 'value': 35}})")],
    description: Annotated[str, Field(description="Optional description", default="")] = "",
) -> str:
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


@server.tool(
    name="list_scenarios",
    description="List existing scenarios, optionally filtered by model.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_scenarios(
    model_id: Annotated[str | None, Field(description="Optional model UUID to filter by", default=None)] = None,
) -> str:
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
