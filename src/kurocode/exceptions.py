"""
KuroCode exception hierarchy.

    KurocodeError
    ├── ConfigError
    ├── StoreError
    └── APIError
         ├── AuthError
         └── RateLimitError
"""

from __future__ import annotations


class KurocodeError(Exception):
    """Base exception for all KuroCode errors."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(message)

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class ConfigError(KurocodeError):
    """Raised for configuration problems (bad TOML, missing required values, etc.)."""


class StoreError(KurocodeError):
    """Raised when the SQLite conversation store encounters an error."""


class APIError(KurocodeError):
    """Raised for HTTP-level failures from the OpenRouter API."""

    def __init__(
        self,
        message: str,
        status_code: int,
        response_body: str = "",
        hint: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message, hint)

    def __str__(self) -> str:
        base = f"[HTTP {self.status_code}] {self.message}"
        if self.response_body:
            base += f"\nResponse: {self.response_body[:300]}"
        if self.hint:
            base += f"\nHint: {self.hint}"
        return base


class AuthError(APIError):
    """Raised on 401 / 403 responses — bad or missing API key."""

    def __init__(
        self,
        message: str = "Authentication failed.",
        status_code: int = 401,
        response_body: str = "",
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            response_body=response_body,
            hint="Check that KUROCODE_API_KEY is set and valid.",
        )


class RateLimitError(APIError):
    """Raised on 429 responses. *retry_after* is in seconds when the header is present."""

    def __init__(
        self,
        message: str = "Rate limit exceeded.",
        response_body: str = "",
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        hint = (
            f"Retry after {retry_after:.0f}s." if retry_after is not None else None
        )
        super().__init__(
            message,
            status_code=429,
            response_body=response_body,
            hint=hint,
        )
