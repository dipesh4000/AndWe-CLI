"""
Model registry for KuroCode.

Fetches free models from OpenRouter, caches them in memory and to a JSON
sidecar on disk.  When the network is unavailable, falls back to the
on-disk cache and emits a visible warning.

Resolution order
----------------
1. In-memory cache          (fastest — skipped when ``force_refresh=True``)
2. Live network fetch        → saved to disk sidecar on success
3. Disk sidecar (offline)   → ``UserWarning`` emitted, stale data returned

Usage::

    from kurocode.model_registry import ModelRegistry

    registry = ModelRegistry()
    models = await registry.fetch()                    # cached after 1st call
    models = await registry.fetch(force_refresh=True)  # bypass caches
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

_DEFAULT_CACHE_PATH = (
    Path.home() / ".local" / "share" / "kurocode" / "models_cache.json"
)
_MODELS_URL = "https://openrouter.ai/api/v1/models"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """A single model entry returned by the OpenRouter ``/models`` endpoint."""

    id: str
    name: str
    context_length: int
    pricing: dict[str, str] = field(default_factory=dict)
    description: str = ""

    @property
    def is_free(self) -> bool:
        """Return *True* when both prompt **and** completion pricing are zero."""
        if self.id.endswith(":free") or self.id.endswith("/free") or self.id == "openrouter/free":
            return True
            
        p = self.pricing
        try:
            return float(p.get("prompt", "1")) == 0.0 and float(p.get("completion", "1")) == 0.0
        except (ValueError, TypeError):
            return False


# ---------------------------------------------------------------------------
# Pure serialisation helpers (easily unit-tested with no I/O)
# ---------------------------------------------------------------------------


def _parse_model(raw: dict[str, Any]) -> ModelInfo:
    """Convert a raw OpenRouter model dict into a :class:`ModelInfo`."""
    pricing: dict[str, str] = {
        k: str(v) for k, v in raw.get("pricing", {}).items()
    }
    return ModelInfo(
        id=raw.get("id", ""),
        name=raw.get("name", raw.get("id", "")),
        context_length=int(raw.get("context_length", 0)),
        pricing=pricing,
        description=raw.get("description", ""),
    )


def _model_to_dict(model: ModelInfo) -> dict[str, Any]:
    """Serialise *model* to a plain dict suitable for JSON encoding."""
    return {
        "id": model.id,
        "name": model.name,
        "context_length": model.context_length,
        "pricing": model.pricing,
        "description": model.description,
    }


def _model_from_dict(d: dict[str, Any]) -> ModelInfo:
    """Deserialise a dict (as produced by :func:`_model_to_dict`) back to a :class:`ModelInfo`."""
    return ModelInfo(
        id=d["id"],
        name=d["name"],
        context_length=int(d.get("context_length", 0)),
        pricing=d.get("pricing", {}),
        description=d.get("description", ""),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Async model registry with two-level cache (memory → disk) and offline
    fallback.

    Parameters
    ----------
    cache_path:
        Path to the JSON sidecar cache file.
    http_client:
        Optional pre-built ``httpx.AsyncClient`` injected for testing.
        When *None* a short-lived client is created per network call.
    """

    def __init__(
        self,
        cache_path: Path = _DEFAULT_CACHE_PATH,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cache_path = cache_path
        self._http = http_client
        self._memory: list[ModelInfo] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self, *, force_refresh: bool = False) -> list[ModelInfo]:
        """
        Return the list of free OpenRouter models.

        Parameters
        ----------
        force_refresh:
            When *True* skip the in-memory cache and re-fetch from the
            network (disk fallback still applies on failure).

        Raises
        ------
        httpx.HTTPError
            When the network call fails **and** no disk cache is available.
        """
        if not force_refresh and self._memory is not None:
            return self._memory

        try:
            models = await self._fetch_from_network()
            self._memory = models
            self._save_disk_cache(models)
            return models
        except Exception as exc:
            cached = self._load_disk_cache()
            if cached is not None:
                warnings.warn(
                    f"Could not reach OpenRouter ({exc!r}); "
                    "using stale model cache — some models may be unavailable.",
                    stacklevel=2,
                    category=UserWarning,
                )
                self._memory = cached
                return cached
            raise

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_from_network(self) -> list[ModelInfo]:
        """Hit ``/models``, filter to free-only, and return parsed results."""
        if self._http is not None:
            response = await self._http.get(_MODELS_URL)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        else:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(_MODELS_URL)
                response.raise_for_status()
                data = response.json()

        raw_models: list[dict[str, Any]] = data.get("data", [])
        return [m for m in (_parse_model(r) for r in raw_models) if m.is_free]

    def _load_disk_cache(self) -> list[ModelInfo] | None:
        """Return the cached list from disk, or *None* if absent/corrupt."""
        if not self._cache_path.exists():
            return None
        try:
            payload: list[dict[str, Any]] = json.loads(
                self._cache_path.read_text(encoding="utf-8")
            )
            return [_model_from_dict(item) for item in payload]
        except Exception:
            return None

    def _save_disk_cache(self, models: list[ModelInfo]) -> None:
        """Persist *models* to the JSON sidecar (best-effort; never raises)."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [_model_to_dict(m) for m in models]
            self._cache_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception:
            pass  # caching is non-critical — silently degrade
