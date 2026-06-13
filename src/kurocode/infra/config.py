"""
Configuration layer for KuroCode.

Resolution order (highest → lowest priority):
  1. Environment variables  (``KUROCODE_*``)
  2. ``.env`` file
  3. Active TOML profile    (``[profiles.<name>]``, merged on top of ``[default]``)
  4. TOML ``[default]`` section
  5. Pydantic field defaults

Usage::

    from kurocode.infra.config import load_config

    cfg = load_config()               # reads env + TOML [default]
    cfg = load_config(profile="work") # adds [profiles.work] on top
    print(cfg.api_key.get_secret_value())
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from kurocode.exceptions import ConfigError

# ---------------------------------------------------------------------------
# XDG-compatible default paths
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "kurocode" / "config.toml"


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Validated runtime configuration for KuroCode."""

    model_config = SettingsConfigDict(
        env_prefix="KUROCODE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_key: SecretStr = Field(..., description="OpenRouter API key.")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL.",
    )
    timeout: float = Field(
        default=60.0,
        gt=0,
        description="HTTP timeout in seconds.",
    )
    site_url: str = Field(
        default="",
        description="HTTP-Referer header sent with every request.",
    )
    app_name: str = Field(
        default="kurocode",
        description="X-Title header shown in the OpenRouter dashboard.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts on transient 429 / 5xx errors.",
    )
    db_path: Path = Field(
        default_factory=lambda: (
            Path.home() / ".local" / "share" / "kurocode" / "history.db"
        ),
        description="Filesystem path to the SQLite conversation history database.",
    )

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # env vars and .env beat TOML values (passed as init kwargs), which
        # beat pydantic field defaults.
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)


# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict[str, Any]:
    """Parse *path* as TOML; return ``{}`` when the file does not exist."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"Failed to parse config file: {path}",
            hint=str(exc),
        ) from exc


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge *override* on top of *base* (override wins on conflicts)."""
    return {**base, **override}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_config(profile: str | None = None) -> Settings:
    """
    Build and return a validated :class:`Settings` instance.

    Parameters
    ----------
    profile:
        Name of the ``[profiles.<name>]`` TOML section to activate.
        When *None* only the ``[default]`` section is applied (if present).

    Raises
    ------
    ConfigError
        If the TOML file is malformed, the requested profile does not exist,
        or pydantic validation fails (e.g. ``api_key`` is missing).
    """
    # Resolve TOML config path: env var override → XDG default.
    config_env = os.environ.get("KUROCODE_CONFIG")
    toml_path = Path(config_env) if config_env else _DEFAULT_CONFIG_PATH

    raw = _read_toml(toml_path)

    # Base: [default] section (may be absent).
    merged: dict[str, Any] = dict(raw.get("default", {}))

    # Overlay the requested profile.
    if profile is not None:
        profiles: dict[str, Any] = raw.get("profiles", {})
        if profile not in profiles:
            raise ConfigError(
                f"Profile '{profile}' not found in {toml_path}.",
                hint=(
                    f"Available profiles: {list(profiles.keys())}"
                    if profiles
                    else "No profiles defined in config."
                ),
            )
        merged = _merge(merged, profiles[profile])

    # Pass TOML values as init kwargs.  Because settings_customise_sources
    # puts env_settings before init_settings, environment variables still win.
    try:
        return Settings(**merged)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(
            "Configuration validation failed.",
            hint=str(exc),
        ) from exc
