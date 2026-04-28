"""
Sablier MCP — shared infrastructure.

Server setup, client management, helpers, and retry logic.
Extracted from server.py to keep tool definitions separate from plumbing.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import date as _date
import json
import logging
import os
import re
import urllib.parse

from mcp.server.fastmcp import FastMCP
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.types import (
    EmbeddedResource,
    Icon,
    TextContent,
    TextResourceContents,
)

from sablier_mcp.auth import SablierOAuthProvider, current_sablier_token, login_page, _pending_auth_redirect
from sablier_mcp.client import SablierClient, SablierAPIError
from sablier_mcp.widgets import (
    betas_heatmap,
    flow_fan_chart,
    flow_risk_card,
    grain_score_card,
    portfolio_overview,
)


logger = logging.getLogger("sablier-mcp")

# ══════════════════════════════════════════════════
# Server setup
# ══════════════════════════════════════════════════

_transport = os.getenv("MCP_TRANSPORT", "stdio" if os.getenv("SABLIER_API_KEY") else "streamable-http")
_oauth_provider: SablierOAuthProvider | None = None

_ICONS = [
    Icon(src="https://sablier.ai/logo-mcp.svg", mimeType="image/svg+xml"),
]

if _transport != "stdio":
    _oauth_provider = SablierOAuthProvider()
    _port = int(os.getenv("PORT", os.getenv("MCP_PORT", "8000")))
    _issuer_url = os.getenv("MCP_ISSUER_URL", f"http://localhost:{_port}")
    server = FastMCP(
        name="Sablier",
        icons=_ICONS,
        host="0.0.0.0",
        port=_port,
        stateless_http=True,
        auth_server_provider=_oauth_provider,
        auth=AuthSettings(
            issuer_url=_issuer_url,
            resource_server_url=_issuer_url,
            client_registration_options=ClientRegistrationOptions(enabled=True),
            required_scopes=[],
        ),
    )
else:
    _port = 8000
    server = FastMCP(name="Sablier", icons=_ICONS)


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

_flow_gpu_lock = asyncio.Lock()


class _FlowGPUBusy(Exception):
    """Raised when the GPU lock cannot be acquired (another Flow job is running)."""
    pass


@asynccontextmanager
async def _acquire_gpu(timeout: float = 30.0):
    """Acquire the Flow GPU lock with a timeout."""
    try:
        await asyncio.wait_for(_flow_gpu_lock.acquire(), timeout=timeout)
    except asyncio.TimeoutError:
        raise _FlowGPUBusy(
            "Another Flow GPU job is already running. "
            "Please wait for it to finish before starting a new one."
        )
    try:
        yield
    finally:
        _flow_gpu_lock.release()


def get_client() -> SablierClient:
    """Get a SablierClient for the current user."""
    global _stdio_client

    token = current_sablier_token.get(None)
    if token:
        if token not in _token_clients:
            _token_clients[token] = SablierClient.from_token(token)
        return _token_clients[token]

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
    if e.status_code == 409:
        return f"Already exists — {e.detail}"
    if e.status_code == 422:
        return f"Error: Invalid input — {e.detail}"
    if e.status_code == 429:
        return (
            f"Credit limit reached — {e.detail}\n\n"
            f"To check your balance, use the get_credits tool. "
            f"To upgrade, use the subscribe tool or visit https://sablier.ai/app/billing"
        )
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
                text="[interactive chart]",
            ),
        ),
    ]


def _portfolio_tickers(portfolio: dict) -> list[str]:
    """Extract ticker symbols from a portfolio response."""
    asset_names = portfolio.get("asset_names", {})
    if asset_names:
        return list(asset_names.values())
    assets = portfolio.get("assets", [])
    return [a.get("ticker") for a in assets if a.get("ticker")]


def _build_per_asset_output(
    summary: dict,
    include_paths: bool = True,
    summary_only: bool = False,
    baseline_summary: dict | None = None,
) -> dict:
    """Build per-asset output from backend summary, optionally including paths."""
    MAX_SAMPLE_PATHS = 10
    MAX_TS_POINTS = 60

    per_asset: dict = {}
    for name, data in summary.items():
        if not isinstance(data, dict):
            continue
        entry: dict = {
            "last_price": data.get("last_price"),
            "mean_return": data.get("mean_return"),
            "median_terminal": data.get("median_terminal"),
            "p5": data.get("p5"),
            "p95": data.get("p95"),
            "feature_type": data.get("feature_type"),
        }

        if include_paths:
            ts = data.get("timeseries") or {}
            is_target = data.get("feature_type") == "target"
            if ts and is_target:
                def _maybe_downsample(arr: list) -> list:
                    if not arr or len(arr) <= MAX_TS_POINTS:
                        return arr
                    step = max(1, len(arr) // MAX_TS_POINTS)
                    return arr[::step]

                entry["percentile_bands"] = {
                    k: _maybe_downsample(ts[k])
                    for k in ("p5", "p25", "p50", "p75", "p95")
                    if k in ts
                }
                if not summary_only:
                    sample = ts.get("sample_paths") or []
                    if sample:
                        step = max(1, len(sample) // MAX_SAMPLE_PATHS)
                        picked = sample[::step][:MAX_SAMPLE_PATHS]
                        entry["sample_paths"] = [_maybe_downsample(p) for p in picked]
            elif ts and not is_target:
                p50 = ts.get("p50")
                if p50:
                    entry["median_path"] = p50

        if baseline_summary:
            bl = baseline_summary.get(name) or {}
            if bl:
                entry["baseline_mean_return"] = bl.get("mean_return")
                entry["baseline_median_terminal"] = bl.get("median_terminal")

        per_asset[name] = entry
    return per_asset


def _flatten_betas(results: dict) -> dict:
    """Flatten the nested betas API response into a widget-friendly format."""
    features = results.get("conditioning_features", [])
    factor_stats = results.get("factor_stats") or {}
    factor_names = factor_stats.get("factor_names", features)

    per_asset = results.get("per_asset_results", {})
    assets = {}
    for _model_id, data in per_asset.items():
        lb = data.get("linear_betas", {})
        alpha_dict = data.get("alpha", {}) or {}
        resid_dict = data.get("residual_std", {}) or {}
        r2_dict = data.get("per_asset_r2", {}) or {}

        for display_name, betas in lb.items():
            asset_entry = {
                "status": data.get("status"),
                "linear_betas": betas if isinstance(betas, dict) else {},
                "alpha": alpha_dict.get(display_name) if isinstance(alpha_dict, dict) else alpha_dict,
                "residual_std": resid_dict.get(display_name) if isinstance(resid_dict, dict) else resid_dict,
            }
            for target_key, r2_vals in r2_dict.items():
                asset_entry["r2"] = r2_vals.get("r2")
                if r2_vals.get("gam_r2") is not None:
                    asset_entry["gam_r2"] = r2_vals["gam_r2"]
                break
            assets[display_name] = asset_entry

    means_list = factor_stats.get("factor_means", [])
    stds_list = factor_stats.get("factor_stds", [])
    means_raw_list = factor_stats.get("factor_means_raw", [])
    stds_raw_list = factor_stats.get("factor_stds_raw", [])
    last_values_list = factor_stats.get("factor_last_values_raw", [])

    factor_means = dict(zip(factor_names, means_list)) if means_list else {}
    factor_stds = dict(zip(factor_names, stds_list)) if stds_list else {}
    factor_means_raw = dict(zip(factor_names, means_raw_list)) if means_raw_list else {}
    factor_stds_raw = dict(zip(factor_names, stds_raw_list)) if stds_raw_list else {}
    factor_last_values_raw = dict(zip(factor_names, last_values_list)) if last_values_list else {}

    factor_last_date_str = factor_stats.get("factor_last_date")
    out = {
        "conditioning_features": features,
        "assets": assets,
        "factor_last_values_raw": factor_last_values_raw,
        "factor_last_date": factor_last_date_str,
        "factor_means_raw": factor_means_raw,
        "factor_stds_raw": factor_stds_raw,
        "factor_means": factor_means,
        "factor_stds": factor_stds,
    }

    if results.get("rolling_window"):
        out["rolling_window"] = results["rolling_window"]
    if results.get("data_truncated_by"):
        out["data_truncated_by"] = results["data_truncated_by"]

    if factor_last_date_str:
        try:
            last_date = _date.fromisoformat(factor_last_date_str)
            days_old = (_date.today() - last_date).days
            business_days_old = round(days_old * 5 / 7)
            if business_days_old > 5:
                out["data_freshness_warning"] = (
                    f"Factor data ends {factor_last_date_str} ({business_days_old} business days ago). "
                    f"At least one conditioning feature has stale data — the beta estimation window "
                    f"was truncated to this date for ALL features. Betas and factor_last_values_raw "
                    f"may not reflect current market conditions. "
                    f"Run refresh_feature_data on your conditioning set tickers to update."
                )
        except (ValueError, TypeError):
            pass
    if results.get("collinear_groups"):
        out["collinear_groups"] = results["collinear_groups"]
    return out


async def _ensure_portfolio(
    portfolio_id: str | None,
    tickers: list[str] | None,
    weights: list[float] | None,
) -> tuple[dict, str | None]:
    """Resolve or auto-create a portfolio."""
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

    if not weights:
        weights = [1.0 / len(tickers)] * len(tickers)
    if len(weights) != len(tickers):
        return {}, "Error: tickers and weights must have the same length."

    name = ", ".join(tickers)
    assets = [{"ticker": t, "weight": w} for t, w in zip(tickers, weights)]
    try:
        portfolio = await client.create_portfolio(name, assets)
    except SablierAPIError as e:
        if e.status_code == 409 or (e.status_code == 500 and "unique" in str(e).lower()):
            all_portfolios = await client.list_portfolios()
            items = all_portfolios.get("portfolios", all_portfolios) if isinstance(all_portfolios, dict) else all_portfolios
            match = next((p for p in items if p.get("name") == name), None)
            if match:
                portfolio = await client.get_portfolio(match["id"])
                return portfolio, None
        raise
    return portfolio, None


async def _validate_conditioning_data(conditioning_set_id: str) -> str | None:
    """Ensure features in a conditioning set have training data."""
    client = get_client()
    feature_set = await client.get_feature_set(conditioning_set_id)

    if feature_set.get("fetched_data_available") is not False:
        return None

    features = feature_set.get("features", [])
    tickers = [
        f.get("ticker")
        for f in features
        if isinstance(f, dict) and f.get("type") != "indicator" and f.get("ticker")
    ]

    if not tickers:
        return None

    try:
        logger.info("Auto-refreshing %d features for conditioning set %s", len(tickers), conditioning_set_id)
        await client.refresh_feature_data(tickers)
    except Exception:
        logger.warning("Auto-refresh failed for conditioning set %s — proceeding anyway", conditioning_set_id, exc_info=True)

    return None


_NOT_LOGGED_IN = (
    "Error: Not authenticated. "
    "Set the SABLIER_API_KEY environment variable to your Sablier API key. "
    "You can create one at https://sablier-ai.com."
)


def _require_auth() -> str | None:
    """Return an error string if not authenticated, else None."""
    if current_sablier_token.get(None):
        return None
    client = get_client()
    if not client.is_authenticated:
        return _NOT_LOGGED_IN
    return None


# ══════════════════════════════════════════════════
# Flow internal helpers
# ══════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════
# Retry helpers
# ══════════════════════════════════════════════════


async def _retry_api_call(coro_fn, max_retries: int = 2, delay: float = 3.0):
    """Retry an async API call on transient server errors (5xx) and rate limits (429)."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except SablierAPIError as e:
            last_exc = e
            retryable = e.status_code >= 500 or e.status_code == 429
            if not retryable or attempt == max_retries:
                raise
            logger.warning("Transient %s on attempt %d/%d — retrying in %.0fs", e, attempt + 1, max_retries + 1, delay)
            await asyncio.sleep(delay)
            delay *= 2
    raise last_exc


async def _retry_gpu_call(coro_fn, max_wait: float = 300.0, initial_delay: float = 15.0):
    """Retry a GPU worker call that may fail with 'Worker busy'."""
    last_exc = None
    delay = initial_delay
    elapsed = 0.0
    attempt = 0
    while elapsed < max_wait:
        try:
            return await coro_fn()
        except SablierAPIError as e:
            last_exc = e
            is_busy = e.status_code in (503, 429) or (e.status_code >= 500 and "busy" in str(e).lower())
            if not is_busy:
                raise
            attempt += 1
            if elapsed + delay >= max_wait:
                raise
            logger.warning("GPU worker busy (attempt %d) — retrying in %.0fs", attempt, delay)
            await asyncio.sleep(delay)
            elapsed += delay
            delay = min(delay * 1.5, 60.0)
    raise last_exc


# ══════════════════════════════════════════════════
# ASGI middleware
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
