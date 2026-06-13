"""
Async HTTP client for the OpenRouter API.

Features
--------
- ``httpx.AsyncClient`` with typed auth / identity headers
- SSE streaming via ``aiter_lines()`` → :class:`StreamChunk`
- ``tenacity`` retry on 429 and 5xx for non-streaming calls
- Typed Pydantic response models

Usage::

    from kurocode.infra.config import load_config
    from kurocode.infra.openrouter_client import OpenRouterClient

    cfg = load_config()
    async with OpenRouterClient(cfg) as client:
        # Non-streaming
        resp = await client.chat(
            messages=[{"role": "user", "content": "Hello!"}],
            model="openai/gpt-4o-mini",
        )
        print(resp.choices[0].message.content)

        # Streaming
        async for chunk in client.stream_chat(
            messages=[{"role": "user", "content": "Tell me a joke."}],
            model="openai/gpt-4o-mini",
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any, Literal

import httpx
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from kurocode.exceptions import APIError, AuthError, RateLimitError
from kurocode.infra.config import Settings

# ---------------------------------------------------------------------------
# Typed response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatChoice(BaseModel):
    """One completion choice in a non-streaming response."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    """Full non-streaming chat completion response from OpenRouter."""

    id: str
    model: str
    choices: list[ChatChoice]
    usage: dict[str, int] | None = None


class StreamDelta(BaseModel):
    """Incremental token delta within a streaming chunk."""

    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    """One choice within a streaming SSE chunk."""

    index: int
    delta: StreamDelta
    finish_reason: str | None = None


class StreamChunk(BaseModel):
    """A single SSE ``data:`` event parsed from an OpenRouter stream."""

    id: str
    model: str
    choices: list[StreamChoice]


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _raise_for_status(response: httpx.Response) -> None:
    """Map HTTP error responses to typed :mod:`kurocode.exceptions`."""
    if response.is_success:
        return

    body = response.text
    status = response.status_code

    if status in (401, 403):
        raise AuthError(
            message=f"Authentication failed (HTTP {status}).",
            status_code=status,
            response_body=body,
        )

    if status == 429:
        retry_after: float | None = None
        raw_ra = response.headers.get("Retry-After")
        if raw_ra is not None:
            try:
                retry_after = float(raw_ra)
            except ValueError:
                pass
        raise RateLimitError(response_body=body, retry_after=retry_after)

    raise APIError(
        message=f"OpenRouter API error (HTTP {status}).",
        status_code=status,
        response_body=body,
    )


def _is_retryable(exc: BaseException) -> bool:
    """Return ``True`` for transient errors worth retrying."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIError) and exc.status_code >= 500:
        return True
    return False


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenRouterClient:
    """
    Async context manager wrapping ``httpx.AsyncClient``.

    Parameters
    ----------
    settings:
        A validated :class:`~kurocode.infra.config.Settings` instance.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def _build_client(self) -> httpx.AsyncClient:
        s = self._settings
        headers: dict[str, str] = {
            "Authorization": f"Bearer {s.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        if s.site_url:
            headers["HTTP-Referer"] = s.site_url
        if s.app_name:
            headers["X-Title"] = s.app_name
        return httpx.AsyncClient(
            base_url=s.base_url,
            headers=headers,
            timeout=httpx.Timeout(s.timeout),
        )

    async def __aenter__(self) -> "OpenRouterClient":
        self._client = self._build_client()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "OpenRouterClient is not open.  Use 'async with OpenRouterClient(...) as client:'."
            )
        return self._client

    # ------------------------------------------------------------------
    # Non-streaming chat  (with tenacity retry)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Send a non-streaming chat completion request.

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts.
        model:
            OpenRouter model identifier, e.g. ``"openai/gpt-4o-mini"``.
        **kwargs:
            Extra fields forwarded verbatim to the request body
            (``temperature``, ``max_tokens``, etc.).

        Returns
        -------
        ChatResponse
            Fully parsed response including all choices and usage stats.

        Raises
        ------
        AuthError
            On 401 / 403.
        RateLimitError
            On 429 (retried up to ``settings.max_retries`` times).
        APIError
            On any other non-2xx response (5xx are retried).
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs,
        }

        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(max(1, self._settings.max_retries)),
            reraise=True,
        ):
            with attempt:
                response = await self._http.post(
                    "/chat/completions",
                    content=json.dumps(payload),
                )
                _raise_for_status(response)
                return ChatResponse.model_validate(response.json())

        # Never reached — AsyncRetrying with reraise=True will raise before this.
        raise RuntimeError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream a chat completion, yielding :class:`StreamChunk` objects.

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts.
        model:
            OpenRouter model identifier.
        **kwargs:
            Extra fields forwarded to the request body.

        Yields
        ------
        StreamChunk
            One object per SSE ``data:`` event (``[DONE]`` terminates the loop).

        Example
        -------
        ::

            async for chunk in client.stream_chat(messages, model):
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        async with self._http.stream(
            "POST",
            "/chat/completions",
            content=json.dumps(payload),
        ) as response:
            _raise_for_status(response)
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = StreamChunk.model_validate_json(data)
                except Exception:
                    # Silently skip malformed / unexpected SSE payloads.
                    continue
                yield chunk
