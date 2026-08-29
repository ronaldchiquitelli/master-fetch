"""Fetcher tests: Response class properties, CSS selector queries,
cross-platform HTML parsing, follow_redirects coercion.

Tests the real Response class against real HTML. No mocks of the class itself.
HTTPSession tests use minimal mocking only for the primp client.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from master_fetch.fetcher import Response, HTTPSession, http_get, _extract_encoding


# ─── Response class ───────────────────────────────────────────────

class TestResponseClass:

    def test_properties_return_init_values(self):
        r = Response(
            url="https://example.com/page",
            body=b"<html><body>Hello</body></html>",
            status=200,
            headers={"content-type": "text/html"},
            encoding="utf-8",
            reason="OK",
            cookies={"session": "abc"},
        )
        assert r.status == 200
        assert r.url == "https://example.com/page"
        assert r.headers["content-type"] == "text/html"
        assert r.body == b"<html><body>Hello</body></html>"
        assert r.encoding == "utf-8"
        assert r.reason == "OK"
        assert r.cookies["session"] == "abc"

    def test_content_decodes_body(self):
        r = Response(url="https://x.com", body=b"Hello world", status=200)
        assert r.content == "Hello world"

    def test_content_caches_result(self):
        r = Response(url="https://x.com", body=b"Hello", status=200)
        first = r.content
        second = r.content
        assert first is second  # cached, not re-decoded

    def test_content_handles_non_utf8(self):
        r = Response(url="https://x.com", body="Hello".encode("latin-1"), status=200, encoding="latin-1")
        assert r.content == "Hello"

    def test_html_content_alias(self):
        r = Response(url="https://x.com", body=b"<p>Hi</p>", status=200)
        assert r.html_content == r.content

    def test_empty_body_returns_empty_content(self):
        r = Response(url="https://x.com", body=b"", status=200)
        assert r.content == ""

    def test_non_bytes_body_coerced_to_empty(self):
        r = Response(url="https://x.com", body=None, status=200)
        assert r.body == b""

    def test_default_headers_empty_dict(self):
        r = Response(url="https://x.com", body=b"x", status=200)
        assert r.headers == {}

    def test_default_cookies_empty_dict(self):
        r = Response(url="https://x.com", body=b"x", status=200)
        assert r.cookies == {}


# ─── CSS selector queries ──────────────────────────────────────────

class TestCSSSelectors:

    def test_css_finds_elements(self):
        html = b'<html><body><div class="main">Text</div></body></html>'
        r = Response(url="https://x.com", body=html, status=200)
        results = r.css(".main")
        assert len(results) == 1
        assert results[0].text_content() == "Text"

    def test_css_no_matches_returns_empty_list(self):
        html = b'<html><body><div>Text</div></body></html>'
        r = Response(url="https://x.com", body=html, status=200)
        results = r.css(".nonexistent")
        assert results == []

    def test_css_finds_multiple_elements(self):
        html = b'<html><body><p>A</p><p>B</p><p>C</p></body></html>'
        r = Response(url="https://x.com", body=html, status=200)
        results = r.css("p")
        assert len(results) == 3

    def test_css_selector_on_subtree(self):
        html = b'<html><body><div class="container"><p>Inner</p></div><p>Outer</p></body></html>'
        r = Response(url="https://x.com", body=html, status=200)
        container = r.css(".container")
        assert len(container) == 1
        inner = container[0].css("p")
        assert len(inner) == 1
        assert inner[0].text_content() == "Inner"

    def test_invalid_css_selector_raises_error(self):
        # PR #11.1.5: errors propagate, not silently swallowed
        html = b'<html><body><div>Text</div></body></html>'
        r = Response(url="https://x.com", body=html, status=200)
        from lxml.cssselect import SelectorSyntaxError
        with pytest.raises(SelectorSyntaxError):
            r.css("div > >")

    def test_get_all_text_extracts_text(self):
        html = b'<html><body><p>Hello</p><p>World</p></body></html>'
        r = Response(url="https://x.com", body=html, status=200)
        text = r.get_all_text()
        assert "Hello" in text and "World" in text

    def test_get_all_text_with_ignore_tags(self):
        html = b'<html><body><script>evil()</script><p>visible</p></body></html>'
        r = Response(url="https://x.com", body=html, status=200)
        text = r.get_all_text(ignore_tags={"script"})
        assert "evil" not in text
        assert "visible" in text

    def test_element_wrapper_url_propagated(self):
        html = b'<html><body><div class="x">Y</div></body></html>'
        r = Response(url="https://example.com/page", body=html, status=200)
        el = r.css(".x")[0]
        assert el.url == "https://example.com/page"


# ─── Cross-platform HTML parsing ──────────────────────────────────

class TestHTMLParsing:

    def test_full_html_document_parsed_correctly(self):
        html = b"""<!DOCTYPE html><html><head><title>Test</title></head>
        <body><div class="content">Hello</div></body></html>"""
        r = Response(url="https://x.com", body=html, status=200)
        results = r.css(".content")
        assert len(results) == 1
        assert results[0].text_content() == "Hello"

    def test_html_without_doctype_parsed(self):
        html = b'<html><body><p>No doctype</p></body></html>'
        r = Response(url="https://x.com", body=html, status=200)
        assert len(r.css("p")) == 1

    def test_fragment_html_parsed(self):
        html = b'<div><p>Fragment</p></div>'
        r = Response(url="https://x.com", body=html, status=200)
        assert len(r.css("p")) == 1

    def test_empty_body_does_not_crash(self):
        r = Response(url="https://x.com", body=b"", status=200)
        assert r.css("div") == []


# ─── Encoding extraction ──────────────────────────────────────────

class TestExtractEncoding:

    def test_extracts_utf8(self):
        assert _extract_encoding("text/html; charset=utf-8") == "utf-8"

    def test_extracts_latin1(self):
        assert _extract_encoding("text/html; charset=iso-8859-1") == "iso-8859-1"

    def test_returns_utf8_for_empty(self):
        assert _extract_encoding("") == "utf-8"

    def test_returns_utf8_for_no_charset(self):
        assert _extract_encoding("application/json") == "utf-8"

    def test_handles_quoted_charset(self):
        assert _extract_encoding('text/html; charset="utf-8"') == "utf-8"


# ─── HTTPSession follow_redirects coercion (v11.0.2 fix) ──────────

class TestFollowRedirectsCoercion:

    @pytest.mark.asyncio
    async def test_string_safe_coerced_to_true(self):
        # scrapling-style "safe" -> True (manual redirect following enabled)
        # primp always gets follow_redirects=False; the get() method follows
        # redirects manually with SSRF re-validation per hop.
        session = HTTPSession()
        session._client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.headers = {}
        mock_resp.status_code = 200
        mock_resp.content = b"ok"
        mock_resp.url = "https://example.com"
        mock_resp.reason = "OK"
        mock_resp.cookies = []
        session._client.get = MagicMock(return_value=mock_resp)

        await session.get("https://example.com", follow_redirects="safe")
        # primp always gets follow_redirects=False (we follow manually)
        call_kwargs = session._client.get.call_args
        assert call_kwargs.kwargs.get("follow_redirects") is False

    @pytest.mark.asyncio
    async def test_string_never_coerced_to_false(self):
        session = HTTPSession()
        session._client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.headers = {}
        mock_resp.status_code = 200
        mock_resp.content = b"ok"
        mock_resp.url = "https://example.com"
        mock_resp.reason = "OK"
        mock_resp.cookies = []
        session._client.get = MagicMock(return_value=mock_resp)

        await session.get("https://example.com", follow_redirects="never")
        call_kwargs = session._client.get.call_args
        assert call_kwargs.kwargs.get("follow_redirects") is False

    @pytest.mark.asyncio
    async def test_bool_passed_through(self):
        session = HTTPSession()
        session._client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.headers = {}
        mock_resp.status_code = 200
        mock_resp.content = b"ok"
        mock_resp.url = "https://example.com"
        mock_resp.reason = "OK"
        mock_resp.cookies = []
        session._client.get = MagicMock(return_value=mock_resp)

        await session.get("https://example.com", follow_redirects=False)
        call_kwargs = session._client.get.call_args
        assert call_kwargs.kwargs.get("follow_redirects") is False


# ─── SSRF: redirect re-validation regression tests ────────────────────────

class TestSSRFRedirectRevalidation:
    """The fetcher must re-validate every redirect hop through validate_url.
    Before the fix, primp followed redirects internally without ever calling
    validate_url on the target, so a public URL that 302-redirects to
    127.0.0.1 bypassed the SSRF guard entirely."""

    @pytest.mark.asyncio
    async def test_redirect_to_loopback_is_blocked(self):
        """A 302 from a public URL to http://127.0.0.1/ must raise SecurityError."""
        from master_fetch.security import SecurityError

        session = HTTPSession()
        session._client = MagicMock()

        # First response: 302 redirect from public URL to 127.0.0.1
        redirect_resp = MagicMock()
        redirect_resp.status_code = 302
        redirect_resp.headers = {"location": "http://127.0.0.1:9941/"}
        redirect_resp.content = b""
        redirect_resp.url = "https://example.com/"
        redirect_resp.reason = "Found"
        redirect_resp.cookies = []

        session._client.get = MagicMock(return_value=redirect_resp)

        with pytest.raises(SecurityError, match="internal/private IP"):
            await session.get("https://example.com/", follow_redirects=True)

    @pytest.mark.asyncio
    async def test_redirect_to_public_url_is_followed(self):
        """A 302 from a public URL to another public URL must be followed."""
        session = HTTPSession()
        session._client = MagicMock()

        redirect_resp = MagicMock()
        redirect_resp.status_code = 302
        redirect_resp.headers = {"location": "https://example.org/page"}
        redirect_resp.content = b""
        redirect_resp.url = "https://example.com/"
        redirect_resp.reason = "Found"
        redirect_resp.cookies = []

        final_resp = MagicMock()
        final_resp.status_code = 200
        final_resp.headers = {"content-type": "text/html"}
        final_resp.content = b"<html>content</html>"
        final_resp.url = "https://example.org/page"
        final_resp.reason = "OK"
        final_resp.cookies = []

        session._client.get = MagicMock(side_effect=[redirect_resp, final_resp])

        result = await session.get("https://example.com/", follow_redirects=True)
        assert result.status == 200
        assert "content" in result.content

    @pytest.mark.asyncio
    async def test_primp_always_gets_follow_redirects_false(self):
        """primp must never follow redirects internally — we handle it manually."""
        session = HTTPSession()
        session._client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.content = b"ok"
        mock_resp.url = "https://example.com/"
        mock_resp.reason = "OK"
        mock_resp.cookies = []

        session._client.get = MagicMock(return_value=mock_resp)
        await session.get("https://example.com/", follow_redirects=True)

        call_kwargs = session._client.get.call_args
        assert call_kwargs.kwargs.get("follow_redirects") is False

    @pytest.mark.asyncio
    async def test_never_redirects_does_not_follow(self):
        """When follow_redirects='never', a 302 response is returned as-is."""
        session = HTTPSession()
        session._client = MagicMock()

        redirect_resp = MagicMock()
        redirect_resp.status_code = 302
        redirect_resp.headers = {"location": "http://127.0.0.1/"}
        redirect_resp.content = b""
        redirect_resp.url = "https://example.com/"
        redirect_resp.reason = "Found"
        redirect_resp.cookies = []

        session._client.get = MagicMock(return_value=redirect_resp)

        result = await session.get("https://example.com/", follow_redirects="never")
        assert result.status == 302

    @pytest.mark.asyncio
    async def test_redirect_chain_each_hop_validated(self):
        """A redirect chain A -> B -> 127.0.0.1 must be caught at the third hop."""
        from master_fetch.security import SecurityError

        session = HTTPSession()
        session._client = MagicMock()

        resp_a = MagicMock()
        resp_a.status_code = 302
        resp_a.headers = {"location": "https://example.org/step2"}
        resp_a.content = b""
        resp_a.url = "https://example.com/"
        resp_a.reason = "Found"
        resp_a.cookies = []

        resp_b = MagicMock()
        resp_b.status_code = 302
        resp_b.headers = {"location": "http://127.0.0.1/"}
        resp_b.content = b""
        resp_b.url = "https://example.org/step2"
        resp_b.reason = "Found"
        resp_b.cookies = []

        session._client.get = MagicMock(side_effect=[resp_a, resp_b])

        with pytest.raises(SecurityError, match="internal/private IP"):
            await session.get("https://example.com/", follow_redirects=True)

    @pytest.mark.asyncio
    async def test_max_redirects_cap_enforced(self):
        """Redirect loop must stop at max_redirects, not loop forever."""
        session = HTTPSession()
        session._client = MagicMock()

        loop_resp = MagicMock()
        loop_resp.status_code = 302
        loop_resp.headers = {"location": "https://example.org/loop"}
        loop_resp.content = b""
        loop_resp.url = "https://example.com/"
        loop_resp.reason = "Found"
        loop_resp.cookies = []

        session._client.get = MagicMock(return_value=loop_resp)
        result = await session.get("https://example.com/", follow_redirects=True, max_redirects=3)
        # Should stop after 3 redirects (still 302, didn't loop forever)
        assert session._client.get.call_count == 4  # initial + 3 hops
        assert result.status == 302
