"""Server-side wiring for XHR capture: verdicts, hints, and browser plumbing.

Complements test_network_capture.py (which covers the capture heuristics in
isolation) by checking how a capture result reaches the agent: what lands in
content, what content_ok says, and what next_action tells it to do next.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from master_fetch.server import (
    ResponseModel,
    _agent_hints,
    _apply_capture_verdict,
    _translate_response,
    _AJAX_SHELL_ERROR,
)
from master_fetch.fetcher import Response


def _model(**kw):
    defaults = dict(
        status=200, content=["Boilerplate help text with no figures."],
        url="https://weather.test/city", fetcher_used="stealthy",
        content_type="text/html", total_size_bytes=260_000,
        extracted_type="markdown",
    )
    defaults.update(kw)
    return ResponseModel(**defaults)


def _network(primary_text="02:00pm 29°C 03:00pm 34°C", url="https://weather.test/ajax/hours"):
    return {
        "fragments": [{
            "url": url, "status": 200, "content_type": "text/html",
            "size_bytes": 25_228, "text": primary_text,
            "is_primary": True, "text_truncated": False,
        }],
        "primary_url": url,
        "captured_count": 1,
    }


class TestCaptureVerdict:
    def test_shell_with_captured_data_is_marked_not_ok(self):
        result = _model(network=_network())
        _apply_capture_verdict(result, fold_captured=False, shell_detected=True)
        assert result.error.startswith("ajax_shell_detected")
        assert "network.fragments" in result.error
        _, _, content_ok = _agent_hints(result)
        assert content_ok is False

    def test_data_stays_out_of_content_by_default(self):
        result = _model(network=_network())
        _apply_capture_verdict(result, fold_captured=False, shell_detected=True)
        assert "29°C" not in " ".join(result.content)

    def test_fold_captured_merges_the_primary_fragment_into_content(self):
        result = _model(network=_network())
        _apply_capture_verdict(result, fold_captured=True, shell_detected=True)
        joined = " ".join(result.content)
        assert "29°C" in joined and "34°C" in joined

    def test_folding_leaves_the_result_usable(self):
        # Folding is the "just give me the data in content" mode, so the result
        # must not also be flagged unusable.
        result = _model(network=_network())
        _apply_capture_verdict(result, fold_captured=True, shell_detected=True)
        assert not result.error
        _, _, content_ok = _agent_hints(result)
        assert content_ok is True

    def test_shell_with_nothing_captured_says_so(self):
        result = _model(network={})
        _apply_capture_verdict(result, fold_captured=False, shell_detected=True)
        assert "no data-bearing XHR" in result.error

    def test_capture_without_a_shell_verdict_stays_silent(self):
        result = _model(network=_network())
        _apply_capture_verdict(result, fold_captured=False, shell_detected=False)
        assert result.error == ""

    def test_an_existing_error_is_not_overwritten(self):
        result = _model(network=_network(), error="http_error_500: server returned error status")
        _apply_capture_verdict(result, fold_captured=False, shell_detected=True)
        assert result.error.startswith("http_error_500")


class TestAgentHints:
    def test_next_action_points_at_the_primary_fragment(self):
        result = _model(network=_network(), error=_AJAX_SHELL_ERROR + "; the data is in network.fragments")
        _, next_action, _ = _agent_hints(result)
        assert "network.fragments" in next_action
        assert "https://weather.test/ajax/hours" in next_action

    def test_next_action_suggests_widening_when_nothing_was_captured(self):
        result = _model(network={}, error=_AJAX_SHELL_ERROR + "; no data-bearing XHR was captured")
        _, next_action, _ = _agent_hints(result)
        assert "capture_pattern" in next_action


class TestTranslateResponseAttachesNetwork:
    def _page(self, captured):
        page = Response(
            url="https://weather.test/city",
            body=b"<html><body><p>Boilerplate.</p></body></html>",
            status=200, headers={"content-type": "text/html"},
            captured=captured,
        )
        return page

    def test_captured_bodies_become_network_fragments(self):
        page = self._page([{
            "url": "https://weather.test/ajax/hours", "status": 200,
            "content_type": "text/html", "size_bytes": 60,
            "_body": '<div class="fc-hours">02:00pm</div><div>29°C</div>' * 8,
        }])
        model = _translate_response(page, "markdown", None, True, False, "stealthy", 10)
        assert model.network["captured_count"] == 1
        assert "29°C" in model.network["fragments"][0]["text"]

    def test_no_capture_leaves_network_empty(self):
        model = _translate_response(self._page([]), "markdown", None, True, False, "stealthy", 10)
        assert model.network == {}

    def test_raw_bodies_are_never_leaked_to_the_caller(self):
        page = self._page([{
            "url": "https://weather.test/ajax/hours", "status": 200,
            "content_type": "text/html", "size_bytes": 60,
            "_body": "<div>29°C data here for the fragment body</div>" * 8,
        }])
        model = _translate_response(page, "markdown", None, True, False, "stealthy", 10)
        assert "_raw" not in model.network["fragments"][0]
        assert "_body" not in model.network["fragments"][0]


class TestBrowserCapture:
    @pytest.mark.asyncio
    async def test_capture_buffers_a_data_response(self):
        from master_fetch.browser import _capture_response

        resp = MagicMock()
        resp.url = "https://weather.test/ajax_pub/weathernexthoursdays?city_id=1"
        resp.status = 200
        resp.request.resource_type = "xhr"
        resp.header_values = AsyncMock(return_value=["text/html"])
        resp.body = AsyncMock(return_value=b"<div>29" + b"x" * 300 + b"</div>")

        captured, total = [], [0]
        await _capture_response(resp, captured, total)
        assert len(captured) == 1
        assert captured[0]["status"] == 200
        assert total[0] > 0

    @pytest.mark.asyncio
    async def test_error_responses_are_skipped(self):
        from master_fetch.browser import _capture_response

        resp = MagicMock()
        resp.url = "https://weather.test/ajax_pub/x"
        resp.status = 503
        resp.request.resource_type = "xhr"
        resp.header_values = AsyncMock(return_value=["text/html"])
        resp.body = AsyncMock(return_value=b"nope")

        captured, total = [], [0]
        await _capture_response(resp, captured, total)
        assert captured == []

    @pytest.mark.asyncio
    async def test_a_body_read_failure_never_propagates(self):
        # Bodies vanish when a page navigates away mid-flight; that must not
        # turn into a failed fetch.
        from master_fetch.browser import _capture_response

        resp = MagicMock()
        resp.url = "https://weather.test/ajax_pub/x"
        resp.status = 200
        resp.request.resource_type = "xhr"
        resp.header_values = AsyncMock(return_value=["text/html"])
        resp.body = AsyncMock(side_effect=Exception("Response body is unavailable"))

        captured, total = [], [0]
        await _capture_response(resp, captured, total)  # must not raise
        assert captured == []

    @pytest.mark.asyncio
    async def test_oversized_bodies_are_dropped(self):
        from master_fetch.browser import _capture_response
        from master_fetch.network_capture import MAX_FRAGMENT_BYTES

        resp = MagicMock()
        resp.url = "https://weather.test/ajax_pub/huge"
        resp.status = 200
        resp.request.resource_type = "xhr"
        resp.header_values = AsyncMock(return_value=["text/html"])
        resp.body = AsyncMock(return_value=b"x" * (MAX_FRAGMENT_BYTES + 1))

        captured, total = [], [0]
        await _capture_response(resp, captured, total)
        assert captured == []

    @pytest.mark.asyncio
    async def test_capture_stops_once_the_total_budget_is_spent(self):
        from master_fetch.browser import _capture_response
        from master_fetch.network_capture import MAX_TOTAL_BYTES

        resp = MagicMock()
        resp.url = "https://weather.test/ajax_pub/x"
        resp.status = 200
        resp.request.resource_type = "xhr"
        resp.header_values = AsyncMock(return_value=["text/html"])
        resp.body = AsyncMock(return_value=b"<div>data</div>" * 40)

        captured, total = [], [MAX_TOTAL_BYTES]
        await _capture_response(resp, captured, total)
        assert captured == []


class TestSmartFetchPlumbing:
    @pytest.mark.asyncio
    async def test_capture_xhr_rejects_the_http_tier(self):
        from master_fetch.server import MasterFetchServer

        server = MasterFetchServer()
        with pytest.raises(ValueError, match="capture_xhr requires the browser tier"):
            await server.smart_fetch(
                "https://weather.test/city", force_fetcher="http", capture_xhr=True,
            )

    @pytest.mark.asyncio
    async def test_capture_xhr_pins_the_browser_tier_and_skips_cache(self):
        from master_fetch.server import MasterFetchServer

        server = MasterFetchServer()
        server._force_fetch = AsyncMock(return_value=_model())
        await server.smart_fetch("https://weather.test/city", capture_xhr=True, cache_ttl=3600)

        args, kwargs = server._force_fetch.call_args
        assert args[1] == "stealthy"
        assert kwargs["capture_xhr"] is True
        # cache_ttl is positional in _force_fetch's signature (7th arg).
        assert args[7] == 0

    @pytest.mark.asyncio
    async def test_an_invalid_capture_pattern_is_rejected(self):
        from master_fetch.browser import StealthyBrowser

        session = StealthyBrowser()
        session._is_alive = True
        with pytest.raises(ValueError, match="invalid capture_pattern"):
            await session.fetch("https://weather.test/city",
                                capture_xhr=True, capture_pattern="[unclosed")


class TestActionsCaptureComposition:
    """capture_xhr must compose with actions so XHRs fired by a click/scroll/
    tab-switch are captured — data that only loads on interaction."""

    @pytest.mark.asyncio
    async def test_actions_and_capture_are_forwarded_together(self):
        from master_fetch.server import MasterFetchServer

        server = MasterFetchServer()
        server._force_fetch = AsyncMock(return_value=_model())
        await server.smart_fetch(
            "https://shop.test/list", actions=[{"scroll": 3}], capture_xhr=True,
        )
        _, kwargs = server._force_fetch.call_args
        assert kwargs["page_action"] is not None
        assert kwargs["capture_xhr"] is True

    @pytest.mark.asyncio
    async def test_actions_without_capture_do_not_capture(self):
        from master_fetch.server import MasterFetchServer

        server = MasterFetchServer()
        server._force_fetch = AsyncMock(return_value=_model())
        await server.smart_fetch("https://shop.test/list", actions=[{"click": ".more"}])
        _, kwargs = server._force_fetch.call_args
        assert kwargs["page_action"] is not None
        assert kwargs["capture_xhr"] is False

    @pytest.mark.asyncio
    async def test_interactions_keep_stylesheets_for_layout(self):
        # scroll thresholds and element visibility need real layout, so the
        # interactive path must not block stylesheets.
        from master_fetch.server import MasterFetchServer

        server = MasterFetchServer()
        server._ensure_auto_session = AsyncMock(return_value="sid")
        server.stealthy_fetch = AsyncMock(return_value=_model())
        server._finalize_result = AsyncMock(side_effect=lambda r, *a, **k: r)
        await server.smart_fetch(
            "https://shop.test/list", actions=[{"scroll": 3}],
            capture_xhr=True, cache_ttl=0,
        )
        _, kwargs = server.stealthy_fetch.call_args
        assert kwargs["disable_resources"] is False

    @pytest.mark.asyncio
    async def test_non_interactive_capture_still_blocks_resources(self):
        from master_fetch.server import MasterFetchServer

        server = MasterFetchServer()
        server._ensure_auto_session = AsyncMock(return_value="sid")
        server.stealthy_fetch = AsyncMock(return_value=_model())
        server._finalize_result = AsyncMock(side_effect=lambda r, *a, **k: r)
        await server.smart_fetch("https://shop.test/data", capture_xhr=True, cache_ttl=0)
        _, kwargs = server.stealthy_fetch.call_args
        assert kwargs["disable_resources"] is True


class TestScrollAction:
    @pytest.mark.asyncio
    async def test_scroll_jumps_to_the_current_bottom_each_step(self):
        # A fixed viewport delta stops re-reaching the bottom once infinite
        # scroll grows the page; scrollTo(scrollHeight) re-triggers every step.
        from master_fetch.actions import build_page_action

        page_action = build_page_action([{"scroll": 3}])
        evaluated = []
        page = MagicMock()
        page.evaluate = AsyncMock(side_effect=lambda expr: evaluated.append(expr))
        page.wait_for_timeout = AsyncMock()

        await page_action(page)
        assert len(evaluated) == 3
        assert all("scrollHeight" in expr for expr in evaluated)
