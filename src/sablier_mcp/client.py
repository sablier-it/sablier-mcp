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
POLL_INTERVAL = 3.0
MAX_POLL_TIME = 300.0  # 5 minutes


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

    # ──────────────────────────────────────────────
    # Portfolios
    # ──────────────────────────────────────────────

    async def list_portfolios(self, limit: int = 100, offset: int = 0) -> dict:
        return await self._get("/portfolios", params={"limit": limit, "offset": offset})

    async def get_portfolio(self, portfolio_id: str) -> dict:
        return await self._get(f"/portfolios/{portfolio_id}")

    async def create_portfolio(
        self, name: str, assets: list[dict], description: str | None = None
    ) -> dict:
        body: dict[str, Any] = {"name": name, "assets": assets}
        if description:
            body["description"] = description
        return await self._post("/portfolios/from-assets", json=body)

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

    # ──────────────────────────────────────────────
    # Models
    # ──────────────────────────────────────────────

    async def list_models(self, limit: int = 100, offset: int = 0) -> dict:
        return await self._get("/models", params={"limit": limit, "offset": offset})

    async def list_model_groups(self) -> list[dict]:
        return await self._get("/models/groups")

    async def batch_create_models(
        self,
        conditioning_set_id: str,
        asset_tickers: list[str],
        name_template: str = "{ticker}_model",
    ) -> dict:
        return await self._post(
            "/models/batch",
            json={
                "conditioning_set_id": conditioning_set_id,
                "asset_tickers": asset_tickers,
                "name_template": name_template,
            },
        )

    # ──────────────────────────────────────────────
    # Feature Sets
    # ──────────────────────────────────────────────

    async def list_feature_set_templates(self) -> list[dict]:
        return await self._get("/feature-sets/templates")

    # ──────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────

    async def train_batch(
        self,
        model_group_id: str,
        training_mode: str = "single_shot_linear",
        max_epochs: int = 100,
        learning_rate: float = 0.001,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ) -> dict:
        return await self._post(
            "/ml/train/batch",
            json={
                "model_group_id": model_group_id,
                "training_mode": training_mode,
                "max_epochs": max_epochs,
                "learning_rate": learning_rate,
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
            },
        )

    async def get_batch_training_progress(self, job_id: str) -> dict:
        return await self._get(f"/ml/train/batch/{job_id}/progress")

    async def poll_training(
        self, job_id: str, timeout: float = MAX_POLL_TIME
    ) -> dict:
        """Poll a training job until completion or timeout."""
        elapsed = 0.0
        result = {}
        while elapsed < timeout:
            result = await self.get_batch_training_progress(job_id)
            status = result.get("status", "")
            if status in ("completed", "failed"):
                return result
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
        return result

    # ──────────────────────────────────────────────
    # Simulation — Betas
    # ──────────────────────────────────────────────

    async def simulate_betas_batch(
        self,
        model_group_id: str,
        simulation_mode: str = "single_shot_linear",
        horizon: int = 20,
    ) -> dict:
        return await self._post(
            "/ml/simulate-betas/batch",
            json={
                "model_group_id": model_group_id,
                "simulation_mode": simulation_mode,
                "horizon": horizon,
            },
        )

    async def get_betas_batch_status(self, simulation_batch_id: str) -> dict:
        return await self._get(
            f"/ml/simulate-betas/batch/{simulation_batch_id}/status"
        )

    async def get_betas_batch_results(self, simulation_batch_id: str) -> dict:
        return await self._get(
            f"/ml/simulate-betas/batch/{simulation_batch_id}/results"
        )

    async def poll_betas_batch(
        self, simulation_batch_id: str, timeout: float = MAX_POLL_TIME
    ) -> dict:
        """Poll batch betas simulation until all complete or timeout."""
        elapsed = 0.0
        while elapsed < timeout:
            result = await self.get_betas_batch_status(simulation_batch_id)
            if result.get("all_completed") or result.get("status") in (
                "completed",
                "failed",
            ):
                return await self.get_betas_batch_results(simulation_batch_id)
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
        return await self.get_betas_batch_results(simulation_batch_id)

    async def portfolio_test(
        self, simulation_batch_id: str, weights: dict[str, float]
    ) -> dict:
        return await self._post(
            f"/ml/simulate-betas/batch/{simulation_batch_id}/portfolio-test",
            json={"weights": weights},
        )

    # ──────────────────────────────────────────────
    # Simulation — Returns
    # ──────────────────────────────────────────────

    async def simulate_returns_batch(
        self,
        simulation_batch_id: str,
        factors: dict[str, float],
        n_samples: int = 5000,
    ) -> dict:
        return await self._post(
            "/ml/simulate-returns/batch",
            json={
                "simulation_batch_id": simulation_batch_id,
                "factors": factors,
                "n_samples": n_samples,
            },
        )

    async def get_returns_batch_status(self, returns_batch_id: str) -> dict:
        return await self._get(
            f"/ml/simulate-returns/batch/{returns_batch_id}/status"
        )

    async def get_returns_batch_results(self, returns_batch_id: str) -> dict:
        return await self._get(
            f"/ml/simulate-returns/batch/{returns_batch_id}/results"
        )

    async def poll_returns_batch(
        self, returns_batch_id: str, timeout: float = MAX_POLL_TIME
    ) -> dict:
        """Poll batch returns simulation until complete or timeout."""
        elapsed = 0.0
        while elapsed < timeout:
            result = await self.get_returns_batch_status(returns_batch_id)
            if result.get("all_completed") or result.get("status") in (
                "completed",
                "failed",
            ):
                return await self.get_returns_batch_results(returns_batch_id)
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
        return await self.get_returns_batch_results(returns_batch_id)

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

    # ──────────────────────────────────────────────
    # Tests
    # ──────────────────────────────────────────────

