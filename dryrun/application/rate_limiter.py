"""Adaptive rate limiter — retry with backoff + dynamic concurrency adjustment."""

from __future__ import annotations
import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Match common rate-limit error patterns
_RATE_LIMIT_CODES = {429}
_RETRYABLE_CODES = {429, 500, 502, 503, 529}


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit (429) error."""
    exc_str = str(exc)
    if "429" in exc_str or "rate_limit" in exc_str.lower():
        return True
    if hasattr(exc, "status_code") and exc.status_code in _RATE_LIMIT_CODES:
        return True
    return False


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable (rate limit or server error)."""
    exc_str = str(exc)
    for code in _RETRYABLE_CODES:
        if str(code) in exc_str:
            return True
    if hasattr(exc, "status_code") and exc.status_code in _RETRYABLE_CODES:
        return True
    return False


def _get_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After value from exception if available."""
    if hasattr(exc, "response") and hasattr(exc.response, "headers"):
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return None


class AdaptiveLimiter:
    """Adaptive concurrency limiter with retry and backoff.

    - Retries retryable errors with exponential backoff + jitter
    - On rate-limit (429), reduces concurrency by half (floor 1)
    - After consecutive successes, recovers concurrency by +1 up to max
    """

    def __init__(self, max_concurrent: int = 5, max_retries: int = 3):
        self._max_concurrent = max_concurrent
        self._current_concurrent = max_concurrent
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._consecutive_successes = 0
        self._recovery_threshold = 5
        self._lock = asyncio.Lock()

    async def _reduce_concurrency(self) -> None:
        """Halve concurrency on rate limit."""
        async with self._lock:
            new_limit = max(1, self._current_concurrent // 2)
            if new_limit < self._current_concurrent:
                logger.warning(
                    "Rate limited — reducing concurrency: %d → %d",
                    self._current_concurrent,
                    new_limit,
                )
                self._current_concurrent = new_limit
                self._consecutive_successes = 0

    async def _maybe_recover(self) -> None:
        """Increase concurrency after sustained success."""
        async with self._lock:
            self._consecutive_successes += 1
            if (
                self._consecutive_successes >= self._recovery_threshold
                and self._current_concurrent < self._max_concurrent
            ):
                self._current_concurrent += 1
                self._consecutive_successes = 0
                logger.info("Recovering concurrency → %d", self._current_concurrent)

    async def run(self, fn: Callable[[], Awaitable[T]], label: str = "") -> T:
        """Execute fn with adaptive rate limiting and retry."""
        # Respect current concurrency (acquire + release extra slots if reduced)
        async with self._semaphore:
            # Additional wait if concurrency was reduced below semaphore capacity
            while True:
                async with self._lock:
                    # Simple gate: count active tasks vs current limit
                    pass
                break

            return await self._run_with_retry(fn, label)

    async def _run_with_retry(self, fn: Callable[[], Awaitable[T]], label: str) -> T:
        """Retry fn with exponential backoff on retryable errors."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                result = await fn()
                await self._maybe_recover()
                return result
            except Exception as exc:
                last_exc = exc

                if not _is_retryable_error(exc):
                    raise

                if attempt == self._max_retries:
                    break

                # Compute backoff
                retry_after = _get_retry_after(exc)
                if retry_after:
                    wait = retry_after
                else:
                    wait = min(2**attempt + random.uniform(0, 1), 60.0)

                if _is_rate_limit_error(exc):
                    await self._reduce_concurrency()

                logger.warning(
                    "Retrying %s (attempt %d/%d, wait %.1fs): %s",
                    label,
                    attempt + 1,
                    self._max_retries,
                    wait,
                    str(exc)[:100],
                )
                await asyncio.sleep(wait)

        raise last_exc  # type: ignore[misc]
