"""Tests for smart_crawl: sitemap fallback honesty + URL normalization."""
import pytest
from unittest.mock import AsyncMock, patch

from master_fetch.crawl import smart_crawl, normalize_url, score_link


@pytest.mark.asyncio
async def test_sitemap_on_with_no_sitemap_returns_honest_empty():
    """sitemap=true on a site without a sitemap must return an honest
    empty response, not crash (UnboundLocalError on `root`)."""
    srv = AsyncMock()
    with patch("master_fetch.crawl._sitemap_map", new=AsyncMock(return_value=None)):
        r = await smart_crawl(srv, "https://example.com", sitemap=True)
    assert r.error == ""
    assert r.pages == []
    assert r.discover_only is True
    assert "no sitemap.xml found" in r.summary
    assert "example.com" in r.summary


@pytest.mark.asyncio
async def test_sitemap_auto_falls_through_to_bfs():
    """sitemap='auto' with no sitemap falls back to BFS discovery."""
    from master_fetch.server import ResponseModel

    async def fake_fetch(**kwargs):
        return ResponseModel(
            url=kwargs["url"], status=200,
            content=["<html><title>t</title><body>"
                     + "real prose content here " * 30 + "</body></html>"],
            fetcher_used="http", content_ok=True,
        )

    srv = AsyncMock()
    srv.smart_fetch = AsyncMock(side_effect=fake_fetch)
    with patch("master_fetch.crawl._sitemap_map", new=AsyncMock(return_value=None)):
        r = await smart_crawl(srv, "https://example.com", sitemap="auto")
    assert r.pages_crawled == 1
    assert r.sitemap_used is False


class TestNormalizeUrl:

    def test_strips_tracking_params_and_fragment(self):
        assert normalize_url(
            "https://Ex.com/docs/?utm_source=x&page=2#frag"
        ) == "https://ex.com/docs?page=2"

    def test_trailing_slash_collapses(self):
        assert normalize_url("https://ex.com/docs/") == normalize_url("https://ex.com/docs")

    def test_default_ports_stripped(self):
        assert normalize_url("http://ex.com:80/a") == "http://ex.com/a"
        assert normalize_url("https://ex.com:443/a") == "https://ex.com/a"


class TestScoreLink:

    def test_junk_paths_penalized(self):
        assert score_link("https://ex.com/login", "", "") < 0
        assert score_link("https://ex.com/docs/api", "", "") > 0

    def test_focus_relevance_beats_content_boost(self):
        focused = score_link("https://ex.com/x/rate-limits", "rate limits guide", "rate limit")
        unfocused = score_link("https://ex.com/docs/other", "other", "rate limit")
        assert focused > unfocused
