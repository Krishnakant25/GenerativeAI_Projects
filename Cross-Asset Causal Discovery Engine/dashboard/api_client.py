"""Thin HTTP client over the Cross-Asset Causal Discovery Engine API.

The dashboard talks to the FastAPI service **only** over HTTP — it never
imports ``causal/pipeline.py`` or re-runs the analysis. This keeps the two
deployables decoupled and the API as the single source of truth.

Every method returns plain parsed JSON (``dict`` / ``list``); the dashboard
deliberately does *not* depend on the server's Pydantic models, so a schema
change on the server can't break the client at import time.

Transport notes:
  * The build spec asks for ``httpx``. This machine ships ``httpx2`` (an
    httpx-compatible fork the test suite already relies on — Starlette's
    TestClient needs it here), so we import ``httpx`` and transparently fall
    back to ``httpx2``. The API surface used below is identical in both.
  * Failures (API down, timeout, non-2xx) are translated into a single
    ``APIError`` carrying a human-readable message — the dashboard renders that
    as a clean error state, never a stack trace.
"""

from __future__ import annotations

from typing import Any

try:  # the project ships httpx2 (an httpx-compatible fork); prefer real httpx
    import httpx
except ModuleNotFoundError:  # pragma: no cover - depends on the local env
    import httpx2 as httpx  # type: ignore[no-redef]

import config


class APIError(RuntimeError):
    """Raised when the API is unreachable or returns a non-2xx response.

    The message is safe to show directly in the UI.
    """


class CausalAPIClient:
    """Stateless client wrapping the engine's HTTP endpoints.

    Construct it with the API base URL (defaults to ``config.API_BASE_URL``).
    GET reads use a short timeout; ``analyze`` gets a long one because a full
    multi-year pipeline run is synchronous and can take minutes.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        get_timeout: float | None = None,
        analyze_timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or config.API_BASE_URL).rstrip("/")
        self.get_timeout = get_timeout or config.API_GET_TIMEOUT_SECONDS
        self.analyze_timeout = analyze_timeout or config.API_ANALYZE_TIMEOUT_SECONDS

    # -- internal ---------------------------------------------------------

    def _request(self, method: str, path: str, *, timeout: float, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise APIError(
                f"Cannot reach the API at {self.base_url}. "
                "Is it running?  Start it with:  uvicorn api.main:app"
            ) from exc
        except httpx.RequestError as exc:
            # Timeouts, read errors, DNS, etc. — anything below the HTTP layer.
            raise APIError(f"Request to {url} failed: {exc}") from exc

        if resp.status_code >= 400:
            raise APIError(f"API returned HTTP {resp.status_code}: {self._detail(resp)}")

        try:
            return resp.json()
        except ValueError as exc:
            raise APIError(f"API returned a non-JSON response from {url}.") from exc

    @staticmethod
    def _detail(resp: Any) -> str:
        """Pull FastAPI's ``{"detail": ...}`` message out of an error response."""
        try:
            body = resp.json()
            if isinstance(body, dict) and "detail" in body:
                return str(body["detail"])
        except ValueError:
            pass
        text = getattr(resp, "text", "") or ""
        return (text or "unknown error")[:300]

    # -- endpoints --------------------------------------------------------

    def health(self) -> dict:
        """``GET /health`` — liveness + DB reachability."""
        return self._request("GET", "/health", timeout=self.get_timeout)

    def analyze(self, body: dict) -> dict:
        """``POST /analyze`` — run the full pipeline and persist it. Slow."""
        return self._request(
            "POST", "/analyze", timeout=self.analyze_timeout, json=body
        )

    def list_runs(self) -> list:
        """``GET /runs`` — all persisted runs, most recent first."""
        return self._request("GET", "/runs", timeout=self.get_timeout)

    def get_run(self, run_id: str) -> dict:
        """``GET /runs/{run_id}`` — run metadata (window, universe, alpha)."""
        return self._request("GET", f"/runs/{run_id}", timeout=self.get_timeout)

    def get_candidates(self, run_id: str, significant_only: bool = False) -> list:
        """``GET /runs/{run_id}/candidates`` — directional candidates + stats."""
        return self._request(
            "GET",
            f"/runs/{run_id}/candidates",
            timeout=self.get_timeout,
            params={"significant_only": significant_only},
        )

    def get_graph(self, run_id: str) -> dict:
        """``GET /runs/{run_id}/graph`` — discovered causal graph (node-link)."""
        return self._request(
            "GET", f"/runs/{run_id}/graph", timeout=self.get_timeout
        )

    def get_regimes(self, run_id: str) -> list:
        """``GET /runs/{run_id}/regimes`` — time-bound regime windows per pair."""
        return self._request(
            "GET", f"/runs/{run_id}/regimes", timeout=self.get_timeout
        )

    # -- Layer 2 (LLM plausibility / explanation) -------------------------

    def llm_health(self) -> dict:
        """``GET /llm/health`` — is the local Ollama model reachable?"""
        return self._request("GET", "/llm/health", timeout=self.get_timeout)

    def get_cards(self, run_id: str) -> list:
        """``GET /runs/{run_id}/cards`` — Layer-2 hypothesis cards, most
        confident first. Each card embeds its underlying statistic."""
        return self._request(
            "GET", f"/runs/{run_id}/cards", timeout=self.get_timeout
        )

    def validate(self, run_id: str, limit: int | None = None) -> dict:
        """``POST /runs/{run_id}/validate`` — generate hypothesis cards for the
        run's significant candidates. SLOW (a local 8B model, ~1 min/card), so
        it uses the long analyze timeout."""
        params = {"limit": limit} if limit is not None else None
        return self._request(
            "POST",
            f"/runs/{run_id}/validate",
            timeout=self.analyze_timeout,
            params=params,
        )

    # -- Phase 3 (scheduled regime-flip monitoring) -----------------------

    def get_flips(
        self,
        *,
        status: str | None = None,
        asset_a: str | None = None,
        asset_b: str | None = None,
    ) -> list:
        """``GET /flips`` — detected regime-status flips, most recent first.

        Each event carries the corrected p-value of its new-run candidate (hard
        rule: no flip without its statistic) and a ``confirmed`` flag. Optionally
        filter by lifecycle ``status`` ('pending'|'confirmed'|'reverted') or pair.
        """
        params = {
            k: v
            for k, v in (
                ("status", status),
                ("asset_a", asset_a),
                ("asset_b", asset_b),
            )
            if v is not None
        }
        return self._request(
            "GET", "/flips", timeout=self.get_timeout, params=params or None
        )
