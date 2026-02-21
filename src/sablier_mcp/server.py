"""
Sablier MCP Server

Gives AI agents the ability to perform regime-dependent factor modeling,
qualitative analysis, portfolio risk testing, and return simulation
through the Sablier platform.
"""

import json
from mcp.server.fastmcp import FastMCP

from sablier_mcp.client import SablierClient, SablierAPIError

mcp = FastMCP(
    "sablier",
    instructions="""You have access to Sablier, an AI-powered portfolio analytics platform
that provides regime-dependent factor modeling, qualitative analysis from SEC filings,
and Monte Carlo return simulation.

Typical workflow:
1. Create a portfolio from tickers and weights (or list existing ones)
2. Run qualitative analysis (GRAIN) to understand thematic exposures from filings
3. Use existing trained models to simulate factor betas and returns
4. Test portfolio risk under different market scenarios

For new portfolios that need models trained:
1. create_portfolio → get portfolio with assets
2. list_feature_set_templates → pick a conditioning set of market factors
3. create_models → batch create per-asset models
4. train_models → start training (takes 5-30 min per asset)
5. simulate_betas → compute regime-dependent factor exposures
6. simulate_returns → sample return distributions under scenarios
7. test_portfolio_risk → get VaR, CVaR, Sharpe, risk decomposition
""",
)

_client: SablierClient | None = None


def get_client() -> SablierClient:
    global _client
    if _client is None:
        _client = SablierClient()
    return _client


def _fmt(data) -> str:
    """Format API response as readable JSON."""
    return json.dumps(data, indent=2, default=str)


# ══════════════════════════════════════════════════
# Search & Discovery
# ══════════════════════════════════════════════════


@mcp.tool()
async def search_features(
    query: str,
    is_asset: bool | None = None,
    limit: int = 20,
) -> str:
    """Search for tickers (stocks, ETFs) and market features (VIX, DXY, rates).

    Use this to find available tickers before creating a portfolio,
    or to discover macro factors for conditioning sets.

    Args:
        query: Search term (e.g. "AAPL", "technology", "volatility", "gold")
        is_asset: If True, only return assets (stocks/ETFs). If False, only indicators.
        limit: Max results to return (default 20)
    """
    client = get_client()
    results = await client.search_features(query, is_asset=is_asset, limit=limit)
    # Summarize for readability
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


@mcp.tool()
async def list_portfolios(limit: int = 50) -> str:
    """List the user's existing portfolios.

    Returns portfolio names, IDs, asset compositions, and status.

    Args:
        limit: Max portfolios to return (default 50)
    """
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
    return _fmt({"total": result.get("total", len(summary)), "portfolios": summary})


@mcp.tool()
async def get_portfolio(portfolio_id: str) -> str:
    """Get detailed information about a specific portfolio.

    Args:
        portfolio_id: The portfolio UUID
    """
    client = get_client()
    result = await client.get_portfolio(portfolio_id)
    return _fmt(result)


@mcp.tool()
async def create_portfolio(
    name: str,
    tickers: list[str],
    weights: list[float],
    description: str = "",
) -> str:
    """Create a new portfolio from tickers and weights.

    This automatically creates the underlying feature sets needed for modeling.
    Weights should sum to 1.0.

    Args:
        name: Portfolio name (e.g. "Tech Portfolio")
        tickers: List of ticker symbols (e.g. ["AAPL", "MSFT", "NVDA"])
        weights: Corresponding weights (e.g. [0.4, 0.3, 0.3])
        description: Optional description
    """
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
        "message": "Portfolio created. To run factor analysis, you need trained models. "
                   "Use list_model_groups to check for existing models, or create_models "
                   "to create new ones.",
    })


# ══════════════════════════════════════════════════
# Qualitative Analysis (GRAIN)
# ══════════════════════════════════════════════════


@mcp.tool()
async def analyze_qualitative(
    tickers: list[str],
    themes: list[str],
    source_types: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
) -> str:
    """Run GRAIN qualitative analysis on tickers for specific themes.

    Analyzes SEC filings (10-K, 10-Q) and earnings call transcripts to score
    how exposed each ticker is to the given themes. Returns scores, evidence
    passages, and source breakdowns.

    This is an async operation — the tool will poll until results are ready
    (up to 5 minutes).

    Args:
        tickers: Tickers to analyze (e.g. ["AAPL", "MSFT"])
        themes: Themes to score (e.g. ["AI exposure", "China risk", "debt levels"])
        source_types: Optional filter: ["10-K", "10-Q", "earnings_transcript"]
        min_year: Earliest filing year to include
        max_year: Latest filing year to include
    """
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
            # Include top 3 evidence passages
            for ev in (ts.get("evidence") or [])[:3]:
                ticker_entry["top_evidence"].append({
                    "passage": ev.get("passage", "")[:300],
                    "source": ev.get("source"),
                    "source_type": ev.get("source_type"),
                    "fiscal_period": ev.get("fiscal_period"),
                    "why_relevant": ev.get("why_relevant"),
                })
            theme_summary["ticker_scores"].append(ticker_entry)
        summary["themes"].append(theme_summary)

    return _fmt(summary)


@mcp.tool()
async def get_analysis_status(job_id: str) -> str:
    """Check the status of a running GRAIN qualitative analysis.

    Use this if analyze_qualitative timed out and you need to check progress.

    Args:
        job_id: The GRAIN job ID returned from analyze_qualitative
    """
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


@mcp.tool()
async def list_model_groups() -> str:
    """List all model groups with their training and simulation status.

    A model group contains per-asset models that share the same conditioning
    set (market factors). This is the main way to see what's available for
    simulation.
    """
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


@mcp.tool()
async def list_feature_set_templates() -> str:
    """List available conditioning set templates (predefined market factor sets).

    Use these when creating models — a conditioning set defines what market
    factors (VIX, rates, dollar index, etc.) the model learns exposures to.
    """
    client = get_client()
    templates = await client.list_feature_set_templates()
    return _fmt(templates)


@mcp.tool()
async def create_models(
    conditioning_set_id: str,
    asset_tickers: list[str],
) -> str:
    """Batch create per-asset factor models for a set of tickers.

    Each ticker gets its own model that learns regime-dependent exposures
    to the factors in the conditioning set.

    Args:
        conditioning_set_id: UUID of the conditioning set (market factors)
        asset_tickers: Tickers to create models for (e.g. ["AAPL", "MSFT"])
    """
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


@mcp.tool()
async def train_models(
    model_group_id: str,
    training_mode: str = "single_shot_linear",
    max_epochs: int = 100,
) -> str:
    """Start batch training for all models in a model group.

    Training runs on GPU and takes 5-30 minutes per asset depending on
    data size and training mode. Use get_training_progress to monitor.

    Training modes (progressive complexity):
    - single_shot_linear: Fast linear factor model (~5 min)
    - single_shot_nonlinear: Nonlinear regime model (~15 min)
    - rollout_nonlinear: Full generative rollout model (~30 min)

    Args:
        model_group_id: UUID of the model group to train
        training_mode: Training mode (default: single_shot_linear)
        max_epochs: Maximum training epochs (default: 100)
    """
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


@mcp.tool()
async def get_training_progress(job_id: str) -> str:
    """Check the progress of a batch training job.

    Args:
        job_id: The training job ID from train_models
    """
    client = get_client()
    result = await client.get_batch_training_progress(job_id)
    return _fmt({
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
    })


# ══════════════════════════════════════════════════
# Factor Analysis (Betas Simulation)
# ══════════════════════════════════════════════════


@mcp.tool()
async def simulate_betas(
    model_group_id: str,
    simulation_mode: str = "single_shot_linear",
    horizon: int = 20,
) -> str:
    """Compute regime-dependent factor exposures (betas) for all assets in a model group.

    This reveals how each asset's returns respond to market factors like VIX,
    interest rates, dollar strength etc. Unlike static betas, these are
    regime-dependent — they change based on current market conditions.

    Runs on GPU. Auto-polls until results are ready (up to 5 min).

    Args:
        model_group_id: UUID of the trained model group
        simulation_mode: Must match training mode (default: single_shot_linear)
        horizon: Forecast horizon in trading days (default: 20)
    """
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

    # Summarize
    per_asset = results.get("per_asset_results", {})
    summary = {
        "simulation_batch_id": sim_batch_id,
        "conditioning_features": results.get("conditioning_features", []),
        "completed": results.get("completed_count"),
        "total": results.get("total_count"),
        "assets": {},
    }
    for asset, data in per_asset.items():
        summary["assets"][asset] = {
            "status": data.get("status"),
            "linear_betas": data.get("linear_betas"),
            "alpha": data.get("alpha"),
            "residual_std": data.get("residual_std"),
        }

    factor_stats = results.get("factor_stats", {})
    if factor_stats:
        summary["factor_means"] = factor_stats.get("means")
        summary["factor_stds"] = factor_stats.get("stds")

    return _fmt(summary)


@mcp.tool()
async def get_betas_results(simulation_batch_id: str) -> str:
    """Get detailed factor beta results from a completed simulation batch.

    Includes per-asset betas, sensitivity curves, factor statistics,
    and residual correlations.

    Args:
        simulation_batch_id: The simulation batch ID from simulate_betas
    """
    client = get_client()
    results = await client.get_betas_batch_results(simulation_batch_id)
    return _fmt(results)


# ══════════════════════════════════════════════════
# Return Simulation
# ══════════════════════════════════════════════════


@mcp.tool()
async def simulate_returns(
    simulation_batch_id: str,
    factors: dict[str, float],
    n_samples: int = 5000,
) -> str:
    """Simulate return distributions under a specific factor scenario.

    Given factor values (e.g. VIX=35 for stress, DXY=110 for strong dollar),
    samples thousands of possible return outcomes using the trained model.

    Auto-polls until results are ready.

    Args:
        simulation_batch_id: From simulate_betas
        factors: Factor scenario values (e.g. {"VIX": 35, "DXY": 110, "TLT": 4.5})
        n_samples: Number of Monte Carlo samples (default: 5000)
    """
    client = get_client()
    batch = await client.simulate_returns_batch(
        simulation_batch_id=simulation_batch_id,
        factors=factors,
        n_samples=n_samples,
    )
    returns_batch_id = batch.get("returns_batch_id")

    # Poll until complete
    results = await client.poll_returns_batch(returns_batch_id)

    if not results:
        return _fmt({
            "returns_batch_id": returns_batch_id,
            "status": "running",
            "message": "Returns simulation still running.",
        })

    return _fmt(results)


@mcp.tool()
async def get_returns_results(returns_batch_id: str) -> str:
    """Get results from a completed returns simulation.

    Args:
        returns_batch_id: The returns batch ID from simulate_returns
    """
    client = get_client()
    results = await client.get_returns_batch_results(returns_batch_id)
    return _fmt(results)


# ══════════════════════════════════════════════════
# Portfolio Risk Testing
# ══════════════════════════════════════════════════


@mcp.tool()
async def test_portfolio_risk(
    simulation_batch_id: str,
    weights: dict[str, float],
) -> str:
    """Run a portfolio risk test using simulation results.

    Computes portfolio-level factor betas, expected return, VaR, CVaR,
    risk contribution per factor, and marginal contribution to risk per asset.

    Args:
        simulation_batch_id: From simulate_betas
        weights: Portfolio weights by ticker (e.g. {"AAPL": 0.4, "MSFT": 0.3, "NVDA": 0.3})
    """
    client = get_client()
    result = await client.portfolio_test(simulation_batch_id, weights)
    return _fmt({
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
    })


# ══════════════════════════════════════════════════
# Scenarios
# ══════════════════════════════════════════════════


@mcp.tool()
async def create_scenario(
    model_id: str,
    name: str,
    factor_values: dict[str, dict],
    description: str = "",
) -> str:
    """Create a named what-if scenario for factor simulation.

    Each factor can be set using different spec types:
    - {"type": "fixed", "value": 35} — set to exact value
    - {"type": "percentile", "value": 95} — set to historical percentile
    - {"type": "shock", "value": 2.0} — shift by N standard deviations

    Args:
        model_id: UUID of the model this scenario applies to
        name: Scenario name (e.g. "Recession", "Tech Bubble")
        factor_values: Factor specifications (e.g. {"VIX": {"type": "fixed", "value": 35}})
        description: Optional description of the scenario
    """
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


@mcp.tool()
async def list_scenarios(model_id: str | None = None) -> str:
    """List existing scenarios, optionally filtered by model.

    Args:
        model_id: Optional model UUID to filter by
    """
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
    mcp.run()


if __name__ == "__main__":
    main()
