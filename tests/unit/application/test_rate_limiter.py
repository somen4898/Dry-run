"""Tests for adaptive rate limiter."""

import pytest
from dryrun.application.rate_limiter import (
    AdaptiveLimiter,
    _is_rate_limit_error,
    _is_retryable_error,
)


class FakeRateLimitError(Exception):
    status_code = 429


class FakeServerError(Exception):
    status_code = 500


class TestErrorDetection:
    def test_rate_limit_by_status_code(self):
        assert _is_rate_limit_error(FakeRateLimitError("too many")) is True

    def test_rate_limit_by_message(self):
        assert _is_rate_limit_error(Exception("Error code: 429 - rate_limit")) is True

    def test_not_rate_limit(self):
        assert _is_rate_limit_error(Exception("something else")) is False

    def test_retryable_429(self):
        assert _is_retryable_error(FakeRateLimitError("")) is True

    def test_retryable_500(self):
        assert _is_retryable_error(FakeServerError("")) is True

    def test_not_retryable(self):
        assert _is_retryable_error(ValueError("bad input")) is False


class TestAdaptiveLimiter:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        limiter = AdaptiveLimiter(max_concurrent=3)
        result = await limiter.run(self._success_fn, label="test")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        call_count = 0

        async def _failing_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeRateLimitError("Error code: 429 - rate limit")
            return "recovered"

        limiter = AdaptiveLimiter(max_concurrent=3, max_retries=3)
        result = await limiter.run(_failing_then_ok, label="test")
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        async def _always_fails():
            raise FakeRateLimitError("Error code: 429 - rate limit")

        limiter = AdaptiveLimiter(max_concurrent=3, max_retries=2)
        with pytest.raises(FakeRateLimitError):
            await limiter.run(_always_fails, label="test")

    @pytest.mark.asyncio
    async def test_does_not_retry_non_retryable(self):
        call_count = 0

        async def _bad_input():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        limiter = AdaptiveLimiter(max_concurrent=3, max_retries=3)
        with pytest.raises(ValueError):
            await limiter.run(_bad_input, label="test")
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_reduces_concurrency_on_rate_limit(self):
        call_count = 0

        async def _rate_limit_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FakeRateLimitError("Error code: 429")
            return "ok"

        limiter = AdaptiveLimiter(max_concurrent=4, max_retries=3)
        await limiter.run(_rate_limit_once, label="test")
        assert limiter._current_concurrent == 2  # halved from 4

    async def _success_fn(self):
        return "ok"
