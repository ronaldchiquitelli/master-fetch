"""Robots.txt tests: is_allowed with mocked parser, cache behavior,
unreachable robots.txt defaults to allow.

The robots.txt fetch is mocked (no network), but the is_allowed/cache
logic is real.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from master_fetch.robots import is_allowed, clear_robots_cache, _extract_netloc


class TestIsAllowed:

    @pytest.mark.asyncio
    async def test_unreachable_robots_allows(self, monkeypatch):
        # When robots.txt can't be fetched, default is allow
        async def mock_fetch(domain):
            return None
        monkeypatch.setattr("master_fetch.robots._fetch_robots_txt", mock_fetch)
        await clear_robots_cache()
        assert await is_allowed("https://example.com/page") is True

    @pytest.mark.asyncio
    async def test_disallowed_url_blocked(self, monkeypatch):
        from urllib.robotparser import RobotFileParser
        async def mock_fetch(domain):
            return "User-agent: *\nDisallow: /private/"
        monkeypatch.setattr("master_fetch.robots._fetch_robots_txt", mock_fetch)
        await clear_robots_cache()
        assert await is_allowed("https://example.com/private/secret") is False

    @pytest.mark.asyncio
    async def test_allowed_url_passes(self, monkeypatch):
        async def mock_fetch(domain):
            return "User-agent: *\nDisallow: /private/"
        monkeypatch.setattr("master_fetch.robots._fetch_robots_txt", mock_fetch)
        await clear_robots_cache()
        assert await is_allowed("https://example.com/public/page") is True

    @pytest.mark.asyncio
    async def test_malformed_url_allowed(self, monkeypatch):
        await clear_robots_cache()
        assert await is_allowed("not a url") is True

    @pytest.mark.asyncio
    async def test_empty_url_allowed(self):
        assert await is_allowed("") is True

    @pytest.mark.asyncio
    async def test_cache_prevents_refetch(self, monkeypatch):
        fetch_count = 0
        async def mock_fetch(domain):
            nonlocal fetch_count
            fetch_count += 1
            return "User-agent: *\nAllow: /"
        monkeypatch.setattr("master_fetch.robots._fetch_robots_txt", mock_fetch)
        await clear_robots_cache()

        await is_allowed("https://example.com/page1")
        await is_allowed("https://example.com/page2")
        assert fetch_count == 1  # second call used cache

    @pytest.mark.asyncio
    async def test_unreachable_robots_is_cached(self, monkeypatch):
        """A domain with no robots.txt must not be re-fetched on every URL."""
        fetch_count = 0
        async def mock_fetch(domain):
            nonlocal fetch_count
            fetch_count += 1
            return None
        monkeypatch.setattr("master_fetch.robots._fetch_robots_txt", mock_fetch)
        await clear_robots_cache()

        for i in range(5):
            assert await is_allowed(f"https://example.com/page{i}") is True
        assert fetch_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_checks_share_one_fetch(self, monkeypatch):
        """A crawl hitting one domain concurrently issues a single robots.txt fetch."""
        fetch_count = 0
        async def mock_fetch(domain):
            nonlocal fetch_count
            fetch_count += 1
            await asyncio.sleep(0.01)  # keep the fetch in flight for the others
            return "User-agent: *\nAllow: /"
        monkeypatch.setattr("master_fetch.robots._fetch_robots_txt", mock_fetch)
        await clear_robots_cache()

        results = await asyncio.gather(
            *[is_allowed(f"https://example.com/page{i}") for i in range(20)]
        )
        assert all(results)
        assert fetch_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_checks_on_distinct_domains_are_not_serialized(
        self, monkeypatch,
    ):
        """Single-flight is per domain — different domains still fetch in parallel."""
        import time as _time
        async def mock_fetch(domain):
            await asyncio.sleep(0.05)
            return "User-agent: *\nAllow: /"
        monkeypatch.setattr("master_fetch.robots._fetch_robots_txt", mock_fetch)
        await clear_robots_cache()

        started = _time.monotonic()
        await asyncio.gather(
            *[is_allowed(f"https://site{i}.com/page") for i in range(5)]
        )
        # 5 domains x 50ms serialized would be 250ms; in parallel it is ~50ms.
        assert _time.monotonic() - started < 0.2

    @pytest.mark.asyncio
    async def test_cached_miss_expires(self, monkeypatch):
        """A cached miss uses the shorter miss TTL, so it is retried later."""
        fetch_count = 0
        async def mock_fetch(domain):
            nonlocal fetch_count
            fetch_count += 1
            return None
        monkeypatch.setattr("master_fetch.robots._fetch_robots_txt", mock_fetch)
        monkeypatch.setattr("master_fetch.robots._ROBOTS_MISS_TTL", 0)
        await clear_robots_cache()

        await is_allowed("https://example.com/page1")
        await is_allowed("https://example.com/page2")
        assert fetch_count == 2


class TestExtractNetloc:

    def test_extracts_domain(self):
        assert _extract_netloc("https://example.com/path") == "example.com"

    def test_extracts_domain_with_port(self):
        assert _extract_netloc("https://example.com:8080/path") == "example.com:8080"

    def test_lowercases(self):
        assert _extract_netloc("https://EXAMPLE.COM/Path") == "example.com"

    def test_invalid_url_returns_empty(self):
        assert _extract_netloc("not a url") == ""

    def test_empty_url_returns_empty(self):
        assert _extract_netloc("") == ""
