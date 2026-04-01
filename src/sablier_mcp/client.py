"""Async HTTP client for the Sablier API."""

import asyncio
import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://sablier-api-215397666394.us-central1.run.app/api/v1"
DEFAULT_TIMEOUT = 60.0
LONG_TIMEOUT = 300.0  # 5 minutes — for synchronous training/simulation
POLL_INTERVAL = 3.0
MAX_POLL_TIME = 300.0  # 5 minutes — for generation jobs
TRAIN_POLL_TIME = 1200.0  # 20 minutes — training takes longer


class SablierAPIError(Exception):
    """Raised when the Sablier API returns an error."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class SablierClient:
    """Async HTTP client for the Sablier REST API."""

    def __init__(self):
        self.base_url = os.getenv("SABLIER_API_URL", DEFAULT_BASE_URL).rstrip("/")
        self._auth_token: str | None = os.getenv("SABLIER_API_KEY") or None
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        if self._auth_token:
            self._client.headers["Authorization"] = f"Bearer {self._auth_token}"

    @classmethod
    def from_token(cls, jwt_token: str) -> "SablierClient":
        """Create a client pre-authenticated with a specific JWT (for OAuth flow)."""
        instance = cls.__new__(cls)
        instance.base_url = os.getenv("SABLIER_API_URL", DEFAULT_BASE_URL).rstrip("/")
        instance._auth_token = jwt_token
        instance._client = httpx.AsyncClient(
            base_url=instance.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {jwt_token}",
            },
            timeout=DEFAULT_TIMEOUT,
        )
        return instance

    @property
    def is_authenticated(self) -> bool:
        return self._auth_token is not None

    def set_auth_token(self, token: str) -> None:
        """Set the auth token (API key or JWT) for this session."""
        self._auth_token = token
        self._client.headers["Authorization"] = f"Bearer {token}"

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        response = await self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise SablierAPIError(response.status_code, str(detail))
        if response.status_code == 204:
            return None
        return response.json()

    async def _get(self, path: str, **kwargs) -> Any:
        return await self._request("GET", path, **kwargs)

    async def _post(self, path: str, **kwargs) -> Any:
        return await self._request("POST", path, **kwargs)

    async def _post_long(self, path: str, **kwargs) -> Any:
        """POST with extended timeout for synchronous training/simulation."""
        kwargs.setdefault("timeout", LONG_TIMEOUT)
        return await self._request("POST", path, **kwargs)

    async def _delete(self, path: str, **kwargs) -> Any:
        return await self._request("DELETE", path, **kwargs)

    # ──────────────────────────────────────────────
    # Features / Search
    # ──────────────────────────────────────────────

    async def search_features(
        self,
        query: str,
        is_asset: bool | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        params: dict[str, Any] = {"q": query, "limit": limit}
        if is_asset is not None:
            params["is_asset"] = is_asset
        if source:
            params["source"] = source
        return await self._get("/features/search", params=params)

    async def add_feature(
        self,
        ticker: str,
        source: str,
        display_name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        is_asset: bool | None = None,
        data_type: str | None = None,
        units: str | None = None,
        frequency: str = "daily",
        metadata: dict | None = None,
        skip_validation: bool = False,
    ) -> dict:
        """Add a feature to the available_features catalog with validation and auto-enrichment."""
        body: dict[str, Any] = {"ticker": ticker, "source": source, "frequency": frequency}
        if display_name:
            body["display_name"] = display_name
        if description:
            body["description"] = description
        if category:
            body["category"] = category
        if is_asset is not None:
            body["is_asset"] = is_asset
        if data_type:
            body["data_type"] = data_type
        if units:
            body["units"] = units
        if metadata:
            body["metadata"] = metadata
        if skip_validation:
            body["skip_validation"] = True
        return await self._post("/features/available", json=body)

    async def refresh_feature_data(self, tickers: list[str]) -> dict:
        """Refresh training data for specific tickers."""
        return await self._post_long("/features/available/refresh", json={"tickers": tickers})

    async def list_transformations(self) -> list[dict]:
        """List available transformation types for derived features."""
        return await self._get("/features/transformations")

    async def create_derived_feature(
        self,
        name: str,
        base_feature: str,
        transformation: str,
        parameters: dict,
        display_name: str | None = None,
        description: str | None = None,
    ) -> dict:
        """Create a derived feature."""
        body: dict[str, Any] = {
            "name": name,
            "base_feature": base_feature,
            "transformation": transformation,
            "parameters": parameters,
        }
        if display_name:
            body["display_name"] = display_name
        if description:
            body["description"] = description
        return await self._post("/features/derived", json=body)

    # ──────────────────────────────────────────────
    # Portfolios
    # ──────────────────────────────────────────────

    async def list_portfolios(self, limit: int = 100, offset: int = 0) -> dict:
        return await self._get("/portfolios", params={"limit": limit, "offset": offset})

    async def get_portfolio(self, portfolio_id: str) -> dict:
        return await self._get(f"/portfolios/{portfolio_id}")

    async def create_portfolio(
        self, name: str, assets: list[dict], description: str | None = None,
        capital: float = 100_000.0,
    ) -> dict:
        body: dict[str, Any] = {"name": name, "assets": assets, "capital": capital}
        if description:
            body["description"] = description
        return await self._post("/portfolios/from-assets", json=body)

    async def update_portfolio(
        self, portfolio_id: str, **fields: Any
    ) -> dict:
        """Update a portfolio. Pass only the fields to change (name, description, weights, capital)."""
        return await self._request("PATCH", f"/portfolios/{portfolio_id}", json=fields)

    async def delete_portfolio(self, portfolio_id: str) -> dict:
        return await self._delete(f"/portfolios/{portfolio_id}")

    async def get_portfolio_live_value(self, portfolio_id: str) -> dict:
        return await self._get(f"/portfolios/{portfolio_id}/live-value")

    async def get_portfolio_analytics(
        self, portfolio_id: str, timeframe: str = "1Y"
    ) -> dict:
        return await self._get(
            f"/portfolios/{portfolio_id}/analytics", params={"timeframe": timeframe}
        )

    async def get_asset_profiles(self, portfolio_id: str) -> dict:
        return await self._get(f"/portfolios/{portfolio_id}/asset-profiles")

    async def optimize_portfolio(
        self, portfolio_id: str, simulation_batch_id: str,
        objective: str = "max_sharpe",
        risk_aversion: float | None = None,
        risk_free_rate: float | None = None,
        max_position: float | None = None,
        long_only: bool | None = None,
        max_drawdown_limit: float | None = None,
        target_exposures: dict | None = None,
    ) -> dict:
        """Find optimal portfolio weights.

        Resolves simulation_batch_id to per-asset beta_simulation_ids,
        then calls the analytical beta-based optimization endpoint.
        """
        # Resolve batch → per-asset simulation IDs
        batch = await self.get_betas_batch_results(simulation_batch_id)
        per_asset = batch.get("per_asset_results", {})
        beta_sim_ids = {}
        for asset_name, asset_data in per_asset.items():
            sid = asset_data.get("simulation_id")
            if sid:
                beta_sim_ids[asset_name] = sid

        if not beta_sim_ids:
            raise ValueError(
                f"No per-asset simulation IDs found in batch {simulation_batch_id}"
            )

        # Map objective names to analytical variants
        obj_map = {
            "max_sharpe": "analytical_max_sharpe",
            "min_variance": "analytical_min_variance",
            "max_return": "analytical_mean_variance",
        }
        api_objective = obj_map.get(objective, objective)

        body: dict[str, Any] = {
            "beta_simulation_ids": beta_sim_ids,
            "objective": api_objective,
        }
        if risk_aversion is not None:
            body["risk_aversion"] = risk_aversion
        if risk_free_rate is not None:
            body["risk_free_rate"] = risk_free_rate
        if max_position is not None:
            body["max_position"] = max_position
        if long_only is not None:
            body["long_only"] = long_only
        if max_drawdown_limit is not None:
            body["max_drawdown_limit"] = max_drawdown_limit
        if target_exposures is not None:
            body["target_exposures"] = target_exposures

        return await self._post(
            f"/portfolios/{portfolio_id}/optimize",
            json=body,
        )

    async def get_efficient_frontier(
        self, portfolio_id: str, num_portfolios: int = 50,
        timeframe: str = "1Y", risk_free_rate: float = 0.02,
    ) -> dict:
        """Calculate efficient frontier for portfolio assets."""
        return await self._get(
            f"/portfolios/{portfolio_id}/efficient-frontier",
            params={
                "num_portfolios": num_portfolios,
                "timeframe": timeframe,
                "risk_free_rate": risk_free_rate,
            },
        )

    async def get_optimization_history(
        self, portfolio_id: str, simulation_batch_id: str | None = None,
    ) -> dict:
        """Retrieve past optimization results for a portfolio."""
        params: dict[str, Any] = {}
        if simulation_batch_id:
            params["simulation_batch_id"] = simulation_batch_id
        return await self._get(
            f"/portfolios/{portfolio_id}/optimization-history",
            params=params or None,
        )

    # ──────────────────────────────────────────────
    # GRAIN (Qualitative Analysis)
    # ──────────────────────────────────────────────

    async def start_grain_analysis(
        self,
        tickers: list[str],
        themes: list[str],
        source_types: list[str] | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        use_transcripts: bool = True,
        use_cache: bool = True,
        force_refresh: bool = False,
        weights: dict[str, float] | None = None,
        portfolio_id: str | None = None,
        portfolio_name: str | None = None,
        custom_keywords: dict[str, list[str]] | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "tickers": tickers,
            "themes": themes,
            "use_transcripts": use_transcripts,
            "use_cache": use_cache,
            "force_refresh": force_refresh,
        }
        if source_types:
            body["source_types"] = source_types
        if min_year:
            body["min_year"] = min_year
        if max_year:
            body["max_year"] = max_year
        if weights:
            body["weights"] = weights
        if portfolio_id:
            body["portfolio_id"] = portfolio_id
        if portfolio_name:
            body["portfolio_name"] = portfolio_name
        if custom_keywords:
            body["custom_keywords"] = custom_keywords
        return await self._post("/grain/analyze", json=body)

    async def get_grain_job(self, job_id: str) -> dict:
        return await self._get(f"/grain/jobs/{job_id}")

    async def poll_grain_job(
        self, job_id: str, timeout: float = MAX_POLL_TIME
    ) -> dict:
        """Poll a GRAIN job until completion or timeout."""
        elapsed = 0.0
        while elapsed < timeout:
            result = await self.get_grain_job(job_id)
            status = result.get("status", "")
            if status in ("completed", "failed"):
                return result
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
        return result  # Return last status even if timed out

    async def generate_grain_keywords(self, theme: str) -> dict:
        return await self._post("/grain/generate-keywords", json={"theme": theme})

    async def list_grain_themes(self) -> list[dict]:
        return await self._get("/grain/themes")

    async def list_grain_analyses(self, portfolio_id: str | None = None) -> list[dict]:
        params: dict[str, Any] = {}
        if portfolio_id:
            params["portfolio_id"] = portfolio_id
        return await self._get("/grain/analyses", params=params)

    async def get_grain_analysis(self, analysis_id: str) -> dict:
        return await self._get(f"/grain/analyses/{analysis_id}")

    async def delete_grain_analysis(self, analysis_id: str) -> dict:
        return await self._delete(f"/grain/analyses/{analysis_id}")

    # ──────────────────────────────────────────────
    # Models
    # ──────────────────────────────────────────────

    async def list_model_groups(self) -> list[dict]:
        return await self._get("/models/groups")

    async def delete_model_group(self, group_id: str) -> dict:
        return await self._delete(f"/models/groups/{group_id}")

    async def list_group_simulations(self, group_id: str) -> dict:
        return await self._get(f"/moment/model-groups/{group_id}/simulations")

    async def get_residual_correlation(self, group_id: str) -> dict:
        return await self._get(f"/moment/model-groups/{group_id}/residual-correlation")

    async def batch_create_models(
        self,
        conditioning_set_id: str,
        asset_tickers: list[str],
        name_template: str = "{ticker}_model",
        parent_target_set_id: str | None = None,
        group_name: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "conditioning_set_id": conditioning_set_id,
            "asset_tickers": asset_tickers,
            "name_template": name_template,
        }
        if parent_target_set_id:
            body["parent_target_set_id"] = parent_target_set_id
        if group_name:
            body["group_name"] = group_name
        return await self._post("/models/batch", json=body)

    # ──────────────────────────────────────────────
    # Feature Sets
    # ──────────────────────────────────────────────

    async def list_feature_set_templates(self) -> list[dict]:
        return await self._get("/feature-sets/templates")

    async def list_all_feature_sets(self, set_type: str | None = None) -> dict:
        """List all accessible feature sets (user's own + shared templates)."""
        params: dict[str, Any] = {}
        if set_type:
            params["set_type"] = set_type
        return await self._get("/feature-sets/all", params=params)

    async def create_feature_set(
        self,
        name: str,
        features: list[dict],
        description: str = "",
        set_type: str = "conditioning",
    ) -> dict:
        """Create a custom feature set (conditioning or target)."""
        return await self._post("/feature-sets", json={
            "name": name,
            "set_type": set_type,
            "features": features,
            "description": description,
        })

    async def get_feature_set(self, feature_set_id: str) -> dict:
        """Get details of a specific feature set."""
        return await self._get(f"/feature-sets/{feature_set_id}")

    async def delete_feature_set(self, feature_set_id: str) -> dict:
        """Delete a feature set."""
        return await self._delete(f"/feature-sets/{feature_set_id}")

    # ──────────────────────────────────────────────
    # Training (Moment — synchronous, returns results directly)
    # ──────────────────────────────────────────────

    async def train_batch(
        self,
        model_group_id: str,
        nonlinear: bool = True,
        baseline_mode: str | None = None,
        use_baseline: bool = True,
        baseline_set_id: str | None = None,
        use_factor_selection: bool = False,
        rolling_huber_window: int | None = None,
        rolling_huber_epsilon: float | None = None,
    ) -> dict:
        """Batch train all models in a group. Synchronous — returns results directly.

        baseline_mode: ETF-based factor orthogonalization region.
            'us', 'global', 'developed_ex_us', 'europe', 'japan' (real-time ETF proxies)
            'us_ff5', 'global_ff5', etc. (legacy academic FF5, ~2 month lag)
            'none' or None to skip baseline.
        use_baseline: Whether to include baseline factors (default True).
        baseline_set_id: UUID of a custom baseline feature set (overrides baseline_mode).
        use_factor_selection: Enable LASSO-based factor selection (default False).
        rolling_huber_window: Window size for rolling Huber regression (default 252).
        rolling_huber_epsilon: Huber epsilon for outlier robustness (default 1.345).
        """
        body: dict = {
            "model_group_id": model_group_id,
            "nonlinear": nonlinear,
            "use_baseline": use_baseline,
        }
        if baseline_mode is not None:
            body["baseline_mode"] = baseline_mode
        if baseline_set_id is not None:
            body["baseline_set_id"] = baseline_set_id
        if use_factor_selection:
            body["use_factor_selection"] = use_factor_selection
        if rolling_huber_window is not None:
            body["rolling_huber_window"] = rolling_huber_window
        if rolling_huber_epsilon is not None:
            body["rolling_huber_epsilon"] = rolling_huber_epsilon
        return await self._post_long("/moment/train/batch", json=body)

    # ──────────────────────────────────────────────
    # Simulation — Betas (Moment — synchronous)
    # ──────────────────────────────────────────────

    async def simulate_betas_batch(
        self,
        model_group_id: str,
        historical_lookback_days: int | None = None,
    ) -> dict:
        """Compute factor exposures for all models in a group. Synchronous."""
        body: dict[str, Any] = {
            "model_group_id": model_group_id,
        }
        if historical_lookback_days is not None:
            body["historical_lookback_days"] = historical_lookback_days
        return await self._post_long("/moment/simulate-betas/batch", json=body)

    async def get_betas_batch_results(self, simulation_batch_id: str) -> dict:
        return await self._get(
            f"/moment/simulate-betas/batch/{simulation_batch_id}/results"
        )

    async def portfolio_test(
        self, simulation_batch_id: str, weights: dict[str, float]
    ) -> dict:
        return await self._post(
            f"/moment/simulate-betas/batch/{simulation_batch_id}/portfolio-test",
            json={"weights": weights},
        )

    # ──────────────────────────────────────────────
    # Simulation — Returns (Moment — synchronous)
    # ──────────────────────────────────────────────

    async def simulate_returns_batch(
        self,
        simulation_batch_id: str,
        factors: dict[str, float],
        n_samples: int = 1000,
    ) -> dict:
        """Sample returns under stressed factor values. Synchronous."""
        return await self._post_long(
            "/moment/simulate-returns/batch",
            json={
                "simulation_batch_id": simulation_batch_id,
                "factors": factors,
                "n_samples": n_samples,
                "use_raw_values": True,
            },
        )

    async def get_returns_batch_results(self, returns_batch_id: str) -> dict:
        return await self._get(
            f"/moment/simulate-returns/batch/{returns_batch_id}/results"
        )

    # ──────────────────────────────────────────────
    # Scenarios
    # ──────────────────────────────────────────────

    async def create_scenario(
        self,
        model_id: str,
        name: str,
        specs: dict[str, dict],
        model_group_id: str | None = None,
        description: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "model_id": model_id,
            "name": name,
            "factor_values": specs,
        }
        if model_group_id:
            body["model_group_id"] = model_group_id
        if description:
            body["description"] = description
        return await self._post("/scenarios", json=body)

    async def list_scenarios(
        self, model_id: str | None = None, limit: int = 100
    ) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if model_id:
            params["model_id"] = model_id
        return await self._get("/scenarios", params=params)

    async def get_scenario(self, scenario_id: str) -> dict:
        return await self._get(f"/scenarios/{scenario_id}")

    async def update_scenario(self, scenario_id: str, **fields: Any) -> dict:
        """Update a scenario. Pass only the fields to change."""
        # Remap MCP 'specs' to backend 'factor_values'
        if "specs" in fields:
            fields["factor_values"] = fields.pop("specs")
        return await self._request("PATCH", f"/scenarios/{scenario_id}", json=fields)

    async def delete_scenario(self, scenario_id: str) -> dict:
        return await self._delete(f"/scenarios/{scenario_id}")

    async def clone_scenario(self, scenario_id: str) -> dict:
        return await self._post(f"/scenarios/{scenario_id}/clone")

    async def run_scenario(self, scenario_id: str) -> dict:
        """Run a scenario through its pipeline (Moment sync, Flow async)."""
        return await self._post_long(f"/scenarios/{scenario_id}/run")

    # ──────────────────────────────────────────────
    # Validation (Moment — synchronous)
    # ──────────────────────────────────────────────

    async def get_latest_group_validation(self, model_group_id: str) -> dict:
        return await self._get(f"/moment/validation/group/{model_group_id}/latest")

    async def validate_single_model(self, model_id: str) -> dict:
        """Validate a single trained model. Synchronous — returns on completion."""
        return await self._post_long(
            "/moment/validation/validate",
            json={"model_id": model_id},
        )

    async def trigger_batch_validation(
        self,
        model_group_id: str,
        n_samples: int = 200,
        max_starting_points: int = 100,
        timeout: float | None = None,
    ) -> dict:
        """Run validation for all models in a group. Synchronous."""
        kwargs: dict[str, Any] = {
            "json": {
                "model_group_id": model_group_id,
                "n_samples": n_samples,
                "max_starting_points": max_starting_points,
            },
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await self._post_long(
            "/moment/validation/batch",
            **kwargs,
        )

    async def get_batch_validation_results(self, validation_batch_id: str) -> dict:
        return await self._get(
            f"/moment/validation/batch/{validation_batch_id}/results"
        )

    # ──────────────────────────────────────────────
    # Market Radar
    # ──────────────────────────────────────────────

    async def get_market_radar(self) -> dict:
        """Get comprehensive market radar snapshot with 50+ indicators and regime signals."""
        return await self._get("/market/radar")

    # ──────────────────────────────────────────────
    # Flow (Generative Time Series)
    # ──────────────────────────────────────────────

    async def flow_train(
        self,
        model_group_id: str,
        horizon: int | None = None,
        obs_length: int | None = None,
        max_epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        patience: int | None = None,
        context_dim: int | None = None,
        encoder_d_model: int | None = None,
        encoder_n_layers: int | None = None,
        denoiser_d_model: int | None = None,
        denoiser_n_layers: int | None = None,
    ) -> dict:
        """Start async OT-CFM flow model training job.

        All hyperparameters are optional — the backend applies sensible defaults
        (horizon=120, obs_length=200, max_epochs=500, lr=1e-4, batch_size=64,
        patience=80, context_dim=96, encoder_d_model=192, encoder_n_layers=2,
        denoiser_d_model=192, denoiser_n_layers=6).
        """
        body: dict[str, Any] = {"model_group_id": model_group_id}
        for key, val in [
            ("horizon", horizon), ("obs_length", obs_length),
            ("max_epochs", max_epochs), ("lr", lr),
            ("batch_size", batch_size), ("patience", patience),
            ("context_dim", context_dim), ("encoder_d_model", encoder_d_model),
            ("encoder_n_layers", encoder_n_layers),
            ("denoiser_d_model", denoiser_d_model),
            ("denoiser_n_layers", denoiser_n_layers),
        ]:
            if val is not None:
                body[key] = val
        return await self._post("/flow/train", json=body)

    async def flow_train_status(self, job_id: str) -> dict:
        return await self._get(f"/flow/train/{job_id}/status")

    async def poll_flow_train(
        self, job_id: str, timeout: float = TRAIN_POLL_TIME
    ) -> dict:
        """Poll a flow training job until completion or timeout."""
        elapsed = 0.0
        while elapsed < timeout:
            result = await self.flow_train_status(job_id)
            status = result.get("status", "")
            if status in ("completed", "failed"):
                return result
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
        return result

    async def flow_generate_paths(
        self,
        model_group_id: str,
        n_paths: int = 1000,
        horizon: int | None = None,
    ) -> dict:
        """Start async path generation job."""
        body: dict[str, Any] = {
            "model_group_id": model_group_id,
            "n_paths": n_paths,
        }
        if horizon is not None:
            body["horizon"] = horizon
        return await self._post("/flow/generate-paths", json=body)

    async def flow_generate_constrained_paths(
        self,
        model_group_id: str,
        constraints: list[dict],
        n_paths: int = 1000,
        horizon: int | None = None,
        name: str | None = None,
        skip_baseline: bool = True,
    ) -> dict:
        """Start async constrained path generation job (latent optimization)."""
        body: dict[str, Any] = {
            "model_group_id": model_group_id,
            "constraints": constraints,
            "n_paths": n_paths,
            "skip_baseline": skip_baseline,
        }
        if horizon is not None:
            body["horizon"] = horizon
        if name is not None:
            body["name"] = name
        return await self._post("/flow/generate-constrained-paths", json=body)

    async def flow_get_results(self, job_id: str) -> dict:
        return await self._get(f"/flow/{job_id}/results")

    async def flow_validate(
        self,
        model_group_id: str,
        n_paths: int = 500,
        horizon: int | None = None,
    ) -> dict:
        body: dict = {
            "model_group_id": model_group_id,
            "n_paths": n_paths,
        }
        if horizon is not None:
            body["horizon"] = horizon
        return await self._post("/flow/validate", json=body)

    async def flow_validate_status(self, job_id: str) -> dict:
        return await self._get(f"/flow/validate/{job_id}/status")

    async def flow_validate_results(self, job_id: str) -> dict:
        return await self._get(f"/flow/validate/{job_id}/results")

    async def poll_flow_job(
        self, job_id: str, status_path: str, timeout: float = MAX_POLL_TIME
    ) -> dict:
        """Poll any flow job (train or generate) until completion."""
        elapsed = 0.0
        while elapsed < timeout:
            result = await self._get(status_path)
            status = result.get("status", "")
            if status in ("completed", "failed"):
                return result
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
        return result

    async def flow_get_latest_results(self, model_group_id: str) -> dict:
        """Get latest completed baseline generation results for a flow model group."""
        return await self._get(f"/flow/model-group/{model_group_id}/latest-results")

    async def flow_list_scenarios(self, model_group_id: str) -> dict:
        """List completed constrained scenarios for a flow model group."""
        return await self._get(f"/flow/model-group/{model_group_id}/scenarios")

    async def flow_list_baselines(self, model_group_id: str) -> dict:
        """List completed baseline (unconstrained) generation jobs for a flow model group."""
        return await self._get(f"/flow/model-group/{model_group_id}/baselines")

    async def flow_download_paths(self, job_id: str) -> bytes:
        """Download all generated paths as CSV for a flow generation job."""
        response = await self._client.request("GET", f"/flow/{job_id}/download-paths")
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            from sablier_mcp.client import SablierAPIError
            raise SablierAPIError(response.status_code, str(detail))
        return response.content

    async def delete_flow_job(self, job_id: str) -> dict:
        """Delete a flow simulation job (baseline or scenario)."""
        return await self._delete(f"/flow/{job_id}")

    async def flow_portfolio_test(self, portfolio_id: str, flow_job_id: str) -> dict:
        """Run portfolio risk analytics on flow-generated paths."""
        return await self._post(
            "/flow/portfolio-test",
            json={"portfolio_id": portfolio_id, "flow_job_id": flow_job_id},
        )

    # ──────────────────────────────────────────────
    # Account
    # ──────────────────────────────────────────────

    async def get_user_info(self) -> dict:
        """Get current user's account info (name, email, tier, etc.)."""
        return await self._get("/account/user")

    # ──────────────────────────────────────────────
    # User API Keys (third-party: FRED, Finnhub)
    # ──────────────────────────────────────────────

    async def set_user_api_key(self, provider: str, api_key: str, name: str | None = None) -> dict:
        """Create or update a user's third-party API key."""
        body: dict = {"provider": provider, "api_key": api_key}
        if name:
            body["name"] = name
        return await self._post("/user-api-keys", json=body)

    async def list_user_api_keys(self) -> dict:
        """List the user's stored third-party API keys (no secrets returned)."""
        return await self._get("/user-api-keys")

    async def delete_user_api_key(self, provider: str) -> dict:
        """Delete a user's third-party API key by provider."""
        return await self._delete(f"/user-api-keys/{provider}")

    # ──────────────────────────────────────────────
    # Billing
    # ──────────────────────────────────────────────

    async def create_checkout_session(self, tier: str, success_url: str | None = None, cancel_url: str | None = None) -> dict:
        """Create a Stripe Checkout Session for subscribing to a tier."""
        body: dict = {"tier": tier}
        if success_url:
            body["success_url"] = success_url
        if cancel_url:
            body["cancel_url"] = cancel_url
        return await self._post("/billing/checkout", json=body)

    async def create_portal_session(self, return_url: str | None = None) -> dict:
        """Create a Stripe Customer Portal session to manage subscription."""
        body: dict = {}
        if return_url:
            body["return_url"] = return_url
        return await self._post("/billing/portal", json=body)

    async def get_billing_info(self) -> dict:
        """Get billing info: tier, usage, limits, overage rates."""
        return await self._get("/billing/info")

    async def get_billing_usage(self, month: str | None = None) -> dict:
        """Get usage breakdown for current or specified month."""
        params = {}
        if month:
            params["month"] = month
        return await self._get("/billing/usage", params=params)

    async def get_credits(self) -> dict:
        """Get current credit balance for this month."""
        return await self._get("/billing/credits")

    async def list_credit_packs(self) -> list[dict]:
        """List available credit packages with pricing. No auth required."""
        return await self._get("/billing/credit-packs")

    async def purchase_credit_pack(
        self,
        pack_id: str,
        success_url: str | None = None,
        cancel_url: str | None = None,
    ) -> dict:
        """Create a Stripe Checkout Session to purchase a credit pack.

        pack_id: 'pack_100', 'pack_500', or 'pack_1000'.
        """
        body: dict[str, Any] = {"pack_id": pack_id}
        if success_url:
            body["success_url"] = success_url
        if cancel_url:
            body["cancel_url"] = cancel_url
        return await self._post("/billing/credit-packs/checkout", json=body)

    async def toggle_overage(self, enabled: bool) -> dict:
        """Toggle on-demand overage credits for Pro subscribers."""
        return await self._post("/billing/overage", json={"enabled": enabled})

    # ──────────────────────────────────────────────
    # Portfolio Aggregation
    # ──────────────────────────────────────────────

    async def aggregate_portfolio_simulations(
        self,
        portfolio_id: str,
        simulation_ids: dict[str, str],
        mode: str = "single_shot_linear",
    ) -> dict:
        """Aggregate per-asset simulation results into portfolio-level analytics.

        Args:
            portfolio_id: Portfolio defining assets and weights.
            simulation_ids: {asset_ticker: returns_simulation_id} from simulate_returns_batch.
            mode: 'single_shot_linear', 'single_shot_nonlinear', or 'rollout_nonlinear'.

        Returns:
            Portfolio-level analytics including VaR, ES, expected return,
            portfolio betas, and (for nonlinear modes) distribution stats.
        """
        return await self._post_long(
            f"/portfolios/{portfolio_id}/aggregate",
            json={
                "simulation_ids": simulation_ids,
                "mode": mode,
            },
        )

    # ──────────────────────────────────────────────
    # Tests
    # ──────────────────────────────────────────────

