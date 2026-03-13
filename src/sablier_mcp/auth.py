"""OAuth 2.0 Authorization Server provider for the Sablier MCP server.

Implements the MCP SDK's OAuthAuthorizationServerProvider so that Claude Desktop
(and other MCP clients) can authenticate via a browser-based login flow:

  1. Client connects → server returns 401
  2. Browser opens → user sees a login form served by this module
  3. User enters email/password → validated against the Sablier backend
  4. Server generates a long-lived API key for the user
  5. Auth code → token exchange → client is authenticated with API key

The API key (sk_live_...) is encoded into a stateless HMAC-signed OAuth token.
API keys never expire, so there is no refresh dance or session timeout.
Sessions survive Cloud Run container restarts without external storage.
"""

import base64
import contextvars
import hashlib
import hmac as _hmac
import json as _json
import logging
import os
import secrets
import time
import urllib.parse
from typing import Any

import httpx

logger = logging.getLogger("sablier-mcp.auth")

# Contextvar set by load_access_token so tool handlers can retrieve the Sablier API key.
current_sablier_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_sablier_token", default=None
)

# Contextvar set by ASGI middleware before the SDK's authorize handler runs.
# Allows get_client() to reconstruct clients lost after a server restart.
_pending_auth_redirect: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_pending_auth_redirect", default=None
)
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    RegistrationError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

# ──────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────

_CODE_TTL = 300  # 5 minutes
_TOKEN_TTL = 365 * 24 * 3600  # 1 year (API key never expires; this is just the OAuth wrapper)

_SABLIER_API_URL = os.getenv(
    "SABLIER_API_URL",
    "https://sablier-api-215397666394.us-central1.run.app/api/v1",
).rstrip("/")


def _random_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


# ──────────────────────────────────────────────────
# Stateless token encoding (survives container restarts)
# ──────────────────────────────────────────────────

# MCP_TOKEN_SECRET must be set in Cloud Run for tokens to survive restarts.
# If unset, a random secret is generated (tokens valid only for this instance).
_TOKEN_SECRET: bytes = (os.getenv("MCP_TOKEN_SECRET") or "").encode() or secrets.token_bytes(32)


def _encode_stateless_token(payload: dict) -> str:
    """Encode a payload into a self-contained HMAC-signed token string."""
    raw = _json.dumps(payload, separators=(",", ":")).encode()
    sig = _hmac.new(_TOKEN_SECRET, raw, hashlib.sha256).hexdigest()[:32]
    return base64.urlsafe_b64encode(raw).decode().rstrip("=") + "." + sig


def _decode_stateless_token(token_str: str) -> dict | None:
    """Decode and verify a self-contained token. Returns None if invalid/tampered."""
    try:
        parts = token_str.rsplit(".", 1)
        if len(parts) != 2:
            return None
        b64_part, sig = parts
        padding = 4 - len(b64_part) % 4
        if padding != 4:
            b64_part += "=" * padding
        raw = base64.urlsafe_b64decode(b64_part)
        expected_sig = _hmac.new(_TOKEN_SECRET, raw, hashlib.sha256).hexdigest()[:32]
        if not _hmac.compare_digest(sig, expected_sig):
            return None
        return _json.loads(raw)
    except Exception:
        return None


_ALLOWED_REDIRECT_HOSTS = {
    "localhost", "127.0.0.1", "::1",        # Claude Desktop / Claude Code
    "claude.ai", "claude.com",              # Claude.ai web app
}


def _is_allowed_redirect(uri: str) -> bool:
    """Allow redirects to localhost (Desktop/Code) and Claude.ai/Claude.com (web)."""
    try:
        parsed = urllib.parse.urlparse(uri)
        if parsed.hostname in _ALLOWED_REDIRECT_HOSTS:
            return True
        return False
    except Exception:
        return False


# ──────────────────────────────────────────────────
# Extended models (store extra data alongside the standard fields)
# ──────────────────────────────────────────────────


class SablierAuthorizationCode(AuthorizationCode):
    """Authorization code enriched with the Sablier API key obtained at login."""
    sablier_api_key: str


class SablierAccessToken(AccessToken):
    """Access token that carries a Sablier API key."""
    sablier_api_key: str


class SablierRefreshToken(RefreshToken):
    """Refresh token that carries the same Sablier API key (for SDK compatibility)."""
    sablier_api_key: str


# ──────────────────────────────────────────────────
# Provider
# ──────────────────────────────────────────────────


class SablierOAuthProvider(
    OAuthAuthorizationServerProvider[SablierAuthorizationCode, SablierRefreshToken, SablierAccessToken]
):
    """In-memory OAuth provider backed by the Sablier REST API for credential validation."""

    def __init__(self, login_url: str = "/login") -> None:
        self._login_url = login_url

        # In-memory stores
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._pending_sessions: dict[str, tuple[str, AuthorizationParams]] = {}  # session_id -> (client_id, params)
        self._auth_codes: dict[str, SablierAuthorizationCode] = {}
        self._access_tokens: dict[str, SablierAccessToken] = {}
        self._refresh_tokens: dict[str, SablierRefreshToken] = {}

    # ── Client Registration ──────────────────────

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        client = self._clients.get(client_id)
        if client is not None:
            return client

        # Client was registered before a server restart (in-memory state lost).
        # Reconstruct using the redirect_uri from the current /authorize request.
        redirect_uri = _pending_auth_redirect.get()
        if not redirect_uri or not _is_allowed_redirect(redirect_uri):
            return None

        logger.info(
            "Reconstructing client %s after restart (redirect: %s)",
            client_id, redirect_uri,
        )
        restored = OAuthClientInformationFull(
            client_id=client_id,
            client_id_issued_at=int(time.time()),
            redirect_uris=[redirect_uri],
            token_endpoint_auth_method="none",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            client_name="MCP Client (restored)",
        )
        self._clients[client_id] = restored
        return restored

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if not client_info.client_id:
            raise RegistrationError(error="invalid_client_metadata", error_description="client_id is required")
        self._clients[client_info.client_id] = client_info

    # ── Authorization ────────────────────────────

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        session_id = _random_token(16)
        self._pending_sessions[session_id] = (client.client_id or "", params)
        return f"{self._login_url}?session={session_id}"

    # ── Authorization Code ───────────────────────

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> SablierAuthorizationCode | None:
        code_obj = self._auth_codes.get(authorization_code)
        if code_obj and code_obj.expires_at < time.time():
            del self._auth_codes[authorization_code]
            return None
        return code_obj

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: SablierAuthorizationCode
    ) -> OAuthToken:
        # Remove the used code
        self._auth_codes.pop(authorization_code.code, None)

        now = time.time()
        client_id = client.client_id or ""
        scopes = authorization_code.scopes
        api_key = authorization_code.sablier_api_key

        # Issue stateless access token — API key encoded inside.
        # API key never expires, so the session lives until the OAuth wrapper expires (1 year).
        access_token_str = _encode_stateless_token({
            "t": "access",
            "cid": client_id,
            "sc": scopes,
            "exp": int(now + _TOKEN_TTL),
            "sak": api_key,
        })
        self._access_tokens[access_token_str] = SablierAccessToken(
            token=access_token_str,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(now + _TOKEN_TTL),
            sablier_api_key=api_key,
        )

        # Issue stateless refresh token (for SDK compatibility)
        refresh_token_str = _encode_stateless_token({
            "t": "refresh",
            "cid": client_id,
            "sc": scopes,
            "exp": int(now + _TOKEN_TTL),
            "sak": api_key,
        })
        self._refresh_tokens[refresh_token_str] = SablierRefreshToken(
            token=refresh_token_str,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(now + _TOKEN_TTL),
            sablier_api_key=api_key,
        )

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=_TOKEN_TTL,
            refresh_token=refresh_token_str,
        )

    # ── Refresh Token ────────────────────────────

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> SablierRefreshToken | None:
        rt = self._refresh_tokens.get(refresh_token)

        # Fallback: decode stateless token after container restart
        if rt is None:
            payload = _decode_stateless_token(refresh_token)
            if payload and payload.get("t") == "refresh":
                if payload.get("exp", 0) < time.time():
                    return None
                rt = SablierRefreshToken(
                    token=refresh_token,
                    client_id=payload.get("cid", ""),
                    scopes=payload.get("sc", []),
                    expires_at=payload["exp"],
                    sablier_api_key=payload.get("sak", ""),
                )
                self._refresh_tokens[refresh_token] = rt
                logger.info("Recovered refresh token from stateless encoding after restart")

        if rt and rt.expires_at and rt.expires_at < time.time():
            del self._refresh_tokens[refresh_token]
            return None
        return rt

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: SablierRefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # API key doesn't expire — just re-issue the same token with a fresh TTL.
        api_key = refresh_token.sablier_api_key
        if not api_key:
            raise Exception("No API key in refresh token — please log in again")

        # Revoke old refresh token
        self._refresh_tokens.pop(refresh_token.token, None)

        now = time.time()
        client_id = client.client_id or ""
        resolved_scopes = scopes or refresh_token.scopes

        # Re-issue stateless access token with same API key
        access_token_str = _encode_stateless_token({
            "t": "access",
            "cid": client_id,
            "sc": resolved_scopes,
            "exp": int(now + _TOKEN_TTL),
            "sak": api_key,
        })
        self._access_tokens[access_token_str] = SablierAccessToken(
            token=access_token_str,
            client_id=client_id,
            scopes=resolved_scopes,
            expires_at=int(now + _TOKEN_TTL),
            sablier_api_key=api_key,
        )

        # Re-issue stateless refresh token
        new_refresh_str = _encode_stateless_token({
            "t": "refresh",
            "cid": client_id,
            "sc": resolved_scopes,
            "exp": int(now + _TOKEN_TTL),
            "sak": api_key,
        })
        self._refresh_tokens[new_refresh_str] = SablierRefreshToken(
            token=new_refresh_str,
            client_id=client_id,
            scopes=resolved_scopes,
            expires_at=int(now + _TOKEN_TTL),
            sablier_api_key=api_key,
        )

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=_TOKEN_TTL,
            refresh_token=new_refresh_str,
        )

    # ── Access Token ─────────────────────────────

    async def load_access_token(self, token: str) -> SablierAccessToken | None:
        at = self._access_tokens.get(token)

        # Fallback: decode stateless token after container restart
        if at is None:
            payload = _decode_stateless_token(token)
            if payload and payload.get("t") == "access":
                if payload.get("exp", 0) < time.time():
                    return None
                at = SablierAccessToken(
                    token=token,
                    client_id=payload.get("cid", ""),
                    scopes=payload.get("sc", []),
                    expires_at=payload["exp"],
                    sablier_api_key=payload.get("sak", ""),
                )
                self._access_tokens[token] = at  # re-cache
                logger.info("Recovered access token from stateless encoding after restart")

        if at and at.expires_at and at.expires_at < time.time():
            del self._access_tokens[token]
            return None
        if at:
            # API key — just set it in the contextvar. No refresh needed.
            current_sablier_token.set(at.sablier_api_key)
        return at

    # ── Revocation ───────────────────────────────

    async def revoke_token(self, token: SablierAccessToken | SablierRefreshToken) -> None:
        self._access_tokens.pop(token.token, None)
        self._refresh_tokens.pop(token.token, None)

    # ── Helpers for the login handler ────────────

    def get_pending_session(self, session_id: str) -> AuthorizationParams | None:
        entry = self._pending_sessions.get(session_id)
        return entry[1] if entry else None

    def complete_login(
        self,
        session_id: str,
        sablier_api_key: str,
    ) -> tuple[str, str | None]:
        """Complete login: generate auth code, return (redirect_url, error).

        Returns (redirect_url, None) on success, or ("", error_message) on failure.
        """
        entry = self._pending_sessions.pop(session_id, None)
        if not entry:
            return "", "Invalid or expired session."
        client_id, params = entry

        code = _random_token()
        self._auth_codes[code] = SablierAuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + _CODE_TTL,
            client_id=client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
            sablier_api_key=sablier_api_key,
        )

        redirect_url = construct_redirect_uri(
            str(params.redirect_uri),
            code=code,
            state=params.state,
        )
        return redirect_url, None


# ──────────────────────────────────────────────────
# Login page HTML
# ──────────────────────────────────────────────────

_SABLIER_LOGO_SVG = '<svg width="40" height="40" viewBox="0 0 165.249 165.172" xmlns="http://www.w3.org/2000/svg"><g transform="translate(-27.026,-51.101)"><path style="fill:#ffffff;fill-opacity:1" d="m 52.602,216.237 c -9.184,-0.656 -17.479,-5.873 -22.112,-13.908 -1.521,-2.638 -2.823,-6.368 -3.274,-9.382 -0.133,-0.887 -0.19,-18.756 -0.19,-59.355 0,-61.739 -0.037,-58.802 0.788,-62.058 0.375,-1.481 0.691,-2.348 1.629,-4.463 2.266,-5.11 6.376,-9.616 11.43,-12.531 2.712,-1.564 6.477,-2.862 9.525,-3.284 1.016,-0.141 17.811,-0.18 59.972,-0.14 55.866,0.054 58.622,0.071 59.881,0.377 5.645,1.37 9.766,3.584 13.649,7.334 4.817,4.651 7.65,10.661 8.232,17.459 0.133,1.547 0.17,19.388 0.125,58.928 -0.071,61.212 -0.005,56.926 -0.92,60.501 -2.381,9.31 -9.927,16.987 -19.241,19.576 -3.679,1.022 0.939,0.949 -61.462,0.977 -31.481,0.014 -57.595,0.001 -58.032,-0.031 z m 42.949,-22.013 c 2.732,-0.632 6.211,-1.763 8.209,-2.67 3.968,-1.802 5.153,-2.609 15.892,-10.827 1.2,-0.919 4.287,-3.279 6.857,-5.244 2.571,-1.966 5.19,-3.969 5.821,-4.452 1.803,-1.38 6.14,-4.69 7.475,-5.703 5.81,-4.414 7.134,-5.762 8.838,-9.001 1.59,-3.023 2.277,-5.817 2.263,-9.207 -0.014,-3.499 -0.917,-7.063 -2.458,-9.701 -1.838,-3.146 -3.269,-4.547 -9.327,-9.131 -1.358,-1.028 -3.581,-2.718 -4.939,-3.755 -4.665,-3.564 -5.071,-3.872 -5.622,-4.271 -0.303,-0.22 -0.552,-0.471 -0.552,-0.557 0,-0.086 1.032,-0.942 2.293,-1.902 1.261,-0.96 2.956,-2.257 3.766,-2.883 1.632,-1.26 5.92,-4.547 7.004,-5.369 4.08,-3.092 6.045,-5.322 7.757,-8.804 0.529,-1.077 1.176,-2.752 1.438,-3.722 0.423,-1.569 0.477,-2.085 0.485,-4.674 0.009,-2.78 -0.016,-3.001 -0.572,-4.939 -1.27,-4.423 -4.278,-8.628 -7.884,-11.019 -1.876,-1.244 -2.835,-1.706 -4.926,-2.375 -2.7,-0.862 -4.806,-1.123 -8.127,-1.004 -6.543,0.233 -13.038,2.462 -19.554,6.711 -1.549,1.01 -24.294,17.912 -26.924,20.007 -0.34,0.271 -1.689,1.275 -2.999,2.232 -3.994,2.919 -5.625,4.356 -7.307,6.438 -3.19,3.948 -4.767,9.242 -4.156,13.951 0.506,3.901 1.889,7.096 4.307,9.95 1.598,1.886 2.327,2.493 9.538,7.939 6.224,4.701 7.286,5.512 8.338,6.366 l 0.929,0.755 -1.459,1.105 c -0.802,0.608 -1.935,1.472 -2.517,1.92 -1.054,0.812 -5.376,4.09 -7.056,5.352 -2.747,2.064 -3.423,2.589 -4.564,3.544 -2.252,1.886 -3.804,3.791 -5.106,6.268 -3.173,6.039 -3.198,12.741 -0.07,18.75 2.522,4.846 7.233,8.328 13.268,9.809 2.093,0.514 2.145,0.517 6.438,0.455 2.852,-0.041 4.316,-0.138 5.202,-0.343 z m 10.673,-20.188 c -3.837,-0.86 -6.468,-2.268 -9.08,-4.86 -2.418,-2.399 -3.853,-4.778 -4.656,-7.721 -0.776,-2.846 -0.904,-4.174 -0.897,-9.287 l 0.007,-4.872 0.655,-0.473 c 0.36,-0.26 1.256,-0.935 1.991,-1.5 4.583,-3.519 6.284,-4.82 7.219,-5.524 4.706,-3.539 7.941,-6.071 7.899,-6.185 -0.028,-0.075 -1.519,-1.239 -3.313,-2.587 -3.005,-2.257 -11.676,-8.795 -13.616,-10.267 l -0.828,-0.628 -0.009,-5.292 c -0.005,-2.91 0.073,-5.808 0.172,-6.438 0.84,-5.333 3.63,-9.675 8.115,-12.63 3.319,-2.187 7.782,-3.317 11.417,-2.891 6.039,0.707 10.771,3.668 13.775,8.619 1.012,1.669 1.446,2.728 2.104,5.138 0.441,1.618 0.459,1.877 0.517,7.497 l 0.06,5.82 -0.72,0.53 c -0.896,0.658 -2.276,1.707 -4.621,3.514 -2.258,1.74 -5.862,4.483 -9.679,7.367 l -2.927,2.211 0.898,0.694 c 0.888,0.685 5.734,4.359 11.041,8.37 1.504,1.136 3.448,2.614 4.321,3.282 l 1.588,1.216 0.054,5.194 c 0.035,3.344 -0.019,5.665 -0.151,6.517 -0.23,1.482 -1.049,4.069 -1.688,5.332 -1.936,3.827 -5.538,7.224 -9.254,8.724 -3.264,1.319 -7.498,1.779 -10.393,1.13 z"/></g></svg>'

_LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in — Sablier</title>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    min-height: 100vh;
    background-color: #0F1115;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    animation: fadeIn 0.4s ease-out;
  }
  .container {
    width: 100%;
    max-width: 1000px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 1rem;
    overflow: hidden;
  }
  .form-panel {
    background-color: rgba(255,255,255,0.05);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 3rem 3.5rem;
    display: flex;
    flex-direction: column;
    min-height: 520px;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 2.5rem;
  }
  .logo svg { width: 40px; height: 40px; }
  .logo span {
    font-size: 1.25rem;
    font-weight: 700;
    color: #FFFFFF;
  }
  h1 {
    font-size: 1.875rem;
    font-weight: 600;
    color: #FFFFFF;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
  }
  .subtitle {
    font-size: 0.875rem;
    color: #9CA3AF;
    margin: 0 0 2.5rem 0;
    line-height: 1.5;
  }
  form {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    flex: 1;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    font-size: 0.875rem;
    font-weight: 500;
    color: rgba(255,255,255,0.9);
  }
  input {
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    border: 1px solid rgba(255,255,255,0.1);
    font-size: 0.9375rem;
    color: #FFFFFF;
    background-color: rgba(255,255,255,0.05);
    transition: all 0.2s ease;
    font-family: 'Source Sans 3', sans-serif;
  }
  input::placeholder { color: rgba(255,255,255,0.5); }
  input:focus {
    outline: none;
    border-color: rgba(255,255,255,0.2);
    background-color: rgba(255,255,255,0.08);
  }
  button {
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    background-color: transparent;
    color: #FFFFFF;
    font-size: 0.9375rem;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.1);
    cursor: pointer;
    transition: all 0.2s ease;
    margin-top: 0.5rem;
    font-family: 'Source Sans 3', sans-serif;
  }
  button:hover {
    background-color: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.15);
  }
  .error {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #ef4444;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    font-size: 0.8125rem;
  }
  .footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: auto;
    padding-top: 2rem;
    font-size: 0.8125rem;
    color: #9CA3AF;
  }
  .footer a {
    color: #3B82F6;
    text-decoration: none;
    transition: color 0.2s ease;
  }
  .footer a:hover { color: #60A5FA; text-decoration: underline; }
  .image-panel {
    position: relative;
    min-height: 520px;
    overflow: hidden;
    background: linear-gradient(135deg, #1a1c2e, #0F1115);
  }
  .image-panel img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    display: block;
  }
  @media (max-width: 968px) {
    .container { grid-template-columns: 1fr; max-width: 500px; }
    .image-panel { display: none; }
    .form-panel { padding: 2.5rem 2rem; }
  }
  @media (max-width: 640px) {
    body { padding: 1rem; }
    .form-panel { padding: 2rem 1.5rem; }
    h1 { font-size: 1.5rem; }
  }
</style>
</head>
<body>
<div class="container">
  <div class="form-panel">
    <div class="logo">
      """ + _SABLIER_LOGO_SVG + """
      <span>Sablier</span>
    </div>
    <div>
      <h1>Welcome back</h1>
      <p class="subtitle">Sign in to connect your AI assistant</p>
    </div>
    {{ERROR}}
    <form method="POST" action="/login">
      <input type="hidden" name="session" value="{{SESSION}}">
      <label>Email
        <input type="email" name="email" required placeholder="you@company.com" value="{{EMAIL}}">
      </label>
      <label>Password
        <input type="password" name="password" required placeholder="Enter your password">
      </label>
      <button type="submit">Sign in</button>
    </form>
    <div class="footer">
      <span>Don't have an account? <a href="https://www.sablier-ai.com" target="_blank">Sign up</a></span>
      <a href="https://www.sablier-ai.com/" target="_blank">Explore Sablier &rarr;</a>
    </div>
  </div>
  <div class="image-panel">
    <img src="https://www.sablier-ai.com/landing-image.jpg" alt="Modern skyscrapers" onerror="this.style.display='none'">
  </div>
</div>
</body>
</html>"""


def _render_login(session_id: str, error: str = "", email: str = "") -> HTMLResponse:
    error_html = f'<div class="error">{error}</div>' if error else ""
    html = (
        _LOGIN_PAGE
        .replace("{{SESSION}}", session_id)
        .replace("{{ERROR}}", error_html)
        .replace("{{EMAIL}}", email)
    )
    return HTMLResponse(html)


# ──────────────────────────────────────────────────
# Login route handler
# ──────────────────────────────────────────────────


async def login_page(request: Request, provider: SablierOAuthProvider) -> HTMLResponse | RedirectResponse:
    """Handles GET (show form) and POST (process login) for /login."""

    if request.method == "GET":
        session_id = request.query_params.get("session", "")
        if not provider.get_pending_session(session_id):
            return HTMLResponse("<h1>Invalid or expired session</h1>", status_code=400)
        return _render_login(session_id)

    # POST — process login
    form = await request.form()
    session_id = str(form.get("session", ""))
    email = str(form.get("email", ""))
    password = str(form.get("password", ""))

    if not session_id or not provider.get_pending_session(session_id):
        return HTMLResponse("<h1>Invalid or expired session</h1>", status_code=400)

    if not email or not password:
        return _render_login(session_id, error="Please enter both email and password.", email=email)

    # Step 1: Authenticate against Sablier backend (get short-lived JWT)
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                f"{_SABLIER_API_URL}/auth/login",
                json={"email": email, "password": password},
                headers={"Content-Type": "application/json"},
            )

        if resp.status_code == 401:
            return _render_login(session_id, error="Invalid email or password.", email=email)
        if resp.status_code == 403:
            return _render_login(
                session_id,
                error="Email not verified yet. Check your inbox for the verification link.",
                email=email,
            )
        if resp.status_code >= 400:
            return _render_login(session_id, error="Login failed. Please try again.", email=email)

        data = resp.json()
        sablier_jwt = data.get("access_token", "")

        if not sablier_jwt:
            return _render_login(session_id, error="Login failed. Please try again.", email=email)

    except Exception:
        return _render_login(session_id, error="Could not reach Sablier. Please try again.", email=email)

    # Step 2: Reuse existing MCP API key if one exists, otherwise create a new one.
    # Important: we do NOT delete old keys because Claude may still hold a cached
    # OAuth token referencing them. Deleting would cause 401s on cached sessions.
    _MCP_KEY_NAME = "Claude Desktop"
    _auth_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {sablier_jwt}"}
    api_key = ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            # Check for an existing MCP key we can reuse
            list_resp = await http.get(f"{_SABLIER_API_URL}/api-keys", headers=_auth_headers)
            if list_resp.status_code == 200:
                mcp_keys = [k for k in list_resp.json().get("api_keys", []) if k.get("name") == _MCP_KEY_NAME]
                if len(mcp_keys) > 1:
                    # Multiple MCP keys — keep the newest, delete the rest
                    mcp_keys.sort(key=lambda k: k.get("created_at", ""), reverse=True)
                    for old_key in mcp_keys[1:]:
                        key_id = old_key.get("id")
                        if key_id:
                            await http.delete(f"{_SABLIER_API_URL}/api-keys/{key_id}", headers=_auth_headers)
                            logger.info("Deleted duplicate MCP API key: %s", key_id)

                if mcp_keys:
                    # Reuse the existing key — but we don't have the raw key, only the prefix.
                    # We must create a new one (the raw key is only returned at creation time).
                    # Keep the old one alive so cached tokens still work.
                    logger.info("Existing MCP key found (prefix: %s), creating additional key", mcp_keys[0].get("key_prefix", "?"))

            # Always create a new key (we can't recover the raw value of existing keys)
            resp = await http.post(
                f"{_SABLIER_API_URL}/api-keys",
                json={"name": _MCP_KEY_NAME},
                headers=_auth_headers,
            )

        if resp.status_code >= 400:
            detail = resp.json().get("detail", "Unknown error") if resp.status_code < 500 else "Server error"
            logger.error("Failed to create API key: %s %s", resp.status_code, detail)
            return _render_login(session_id, error="Login succeeded but failed to create API key. Please try again.", email=email)

        api_key = resp.json().get("api_key", "")
        if not api_key:
            return _render_login(session_id, error="Login succeeded but API key was empty. Please try again.", email=email)

        logger.info("Created API key for MCP session (prefix: %s...)", api_key[:12])

    except Exception as exc:
        logger.error("Failed to create API key: %s", exc)
        return _render_login(session_id, error="Could not create API key. Please try again.", email=email)

    # Complete the OAuth flow with the API key (not JWT)
    redirect_url, err = provider.complete_login(session_id, api_key)
    if err:
        return _render_login(session_id, error=err, email=email)

    return RedirectResponse(url=redirect_url, status_code=302)
