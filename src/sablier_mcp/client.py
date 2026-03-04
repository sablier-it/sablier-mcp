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

    async def close(self):
        await self._client.aclose()

    # ──────────────────────────────────────────────
    # Auth (unauthenticated endpoints)
    # ──────────────────────────────────────────────

    async def register(
        self, email: str, name: str, company: str, role: str, password: str
    ) -> dict:
        """Register a new Sablier account. No auth required."""
        return await self._post(
            "/auth/register",
            json={
                "email": email,
                "name": name,
                "company": company,
                "role": role,
                "password": password,
            },
        )

    async def login(self, email: str, password: str) -> dict:
        """Login and get JWT tokens. No auth required."""
        result = await self._post(
            "/auth/login",
            json={"email": email, "password": password},
        )
        # Auto-set the access token for this session
        access_token = result.get("access_token")
        if access_token:
            self.set_auth_token(access_token)
        return result

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
    ) -> dict:
        """Find optimal portfolio weights."""
        return await self._post(
            f"/portfolios/{portfolio_id}/optimize",
            json={"simulation_batch_id": simulation_batch_id, "objective": objective},
        )

    async def get_efficient_frontier(
        self, portfolio_id: str, n_points: int = 20,
    ) -> dict:
        """Calculate efficient frontier for portfolio assets."""
        return await self._get(
            f"/portfolios/{portfolio_id}/efficient-frontier",
            params={"n_points": n_points},
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
        weights: dict[str, float] | None = None,
        portfolio_id: str | None = None,
        portfolio_name: str | None = None,
        custom_keywords: dict[str, list[str]] | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "tickers": tickers,
            "themes": themes,
            "use_transcripts": use_transcripts,
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

    async def list_models(self, limit: int = 100, offset: int = 0) -> dict:
        return await self._get("/models", params={"limit": limit, "offset": offset})

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

    # ──────────────────────────────────────────────
    # Training (Moment — synchronous, returns results directly)
    # ──────────────────────────────────────────────

    async def train_batch(
        self,
        model_group_id: str,
    ) -> dict:
        """Batch train all models in a group. Synchronous — returns results directly."""
        return await self._post_long(
            "/moment/train/batch",
            json={
                "model_group_id": model_group_id,
            },
        )

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
        n_samples: int = 5000,
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
        description: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "model_id": model_id,
            "name": name,
            "specs": specs,
        }
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
        return await self._request("PATCH", f"/scenarios/{scenario_id}", json=fields)

    async def delete_scenario(self, scenario_id: str) -> dict:
        return await self._delete(f"/scenarios/{scenario_id}")

    async def clone_scenario(self, scenario_id: str) -> dict:
        return await self._post(f"/scenarios/{scenario_id}/clone")

    # ──────────────────────────────────────────────
    # Validation (Moment — synchronous)
    # ──────────────────────────────────────────────

    async def get_latest_group_validation(self, model_group_id: str) -> dict:
        return await self._get(f"/moment/validation/group/{model_group_id}/latest")

    async def trigger_batch_validation(
        self,
        model_group_id: str,
        n_samples: int = 200,
        max_starting_points: int = 100,
    ) -> dict:
        """Run validation for all models in a group. Synchronous."""
        return await self._post_long(
            "/moment/validation/batch",
            json={
                "model_group_id": model_group_id,
                "n_samples": n_samples,
                "max_starting_points": max_starting_points,
            },
        )

    async def get_batch_validation_results(self, validation_batch_id: str) -> dict:
        return await self._get(
            f"/moment/validation/batch/{validation_batch_id}/results"
        )

    async def list_simulation_history(
        self, simulation_batch_id: str
    ) -> dict:
        """List past scenario runs for a given betas simulation batch."""
        return await self._get(
            f"/moment/simulate-returns/batch/history/{simulation_batch_id}"
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
        horizon: int = 20,
        obs_length: int = 60,
        max_epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 50,
    ) -> dict:
        """Start async OT-CFM flow model training job."""
        body: dict[str, Any] = {
            "model_group_id": model_group_id,
            "horizon": horizon,
            "obs_length": obs_length,
            "max_epochs": max_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "patience": patience,
        }
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
        n_paths: int = 100,
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
        n_paths: int = 100,
        horizon: int | None = None,
    ) -> dict:
        """Start async constrained path generation job (SMC filtering)."""
        body: dict[str, Any] = {
            "model_group_id": model_group_id,
            "constraints": constraints,
            "n_paths": n_paths,
        }
        if horizon is not None:
            body["horizon"] = horizon
        return await self._post("/flow/generate-constrained-paths", json=body)

    async def flow_get_results(self, job_id: str) -> dict:
        return await self._get(f"/flow/{job_id}/results")

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

    # ──────────────────────────────────────────────
    # Tests
    # ──────────────────────────────────────────────

