"""Regression tests for HTTP→stealthy escalation (Dondai's critical bug).

The HTTP tier reports transport-level failures (TLS fingerprint block,
connection reset, timeout) as status=0. The auto-escalation accepted
`status < 400` as success, so 0<400 finalized a network error WITHOUT
ever trying the stealthy browser. These tests pin the fixed behavior:
status=0 escalates (except deterministic DNS/refused failures), and only
real 2xx/3xx statuses are accepted.
"""
import pytest
from unittest.mock import AsyncMock, patch

from master_fetch.server import MasterFetchServer, ResponseModel, _is_deterministic_net_error


def _http_error(msg="4 attempts failed. Last: Connection reset by peer"):
    return ResponseModel(
        url="https://example.com/", status=0,
        content=[f"[Network error: {msg}]"],
        fetcher_used="none", error=f"Network error: {msg}",
    )


def _stealth_ok(url="https://example.com/"):
    return ResponseModel(
        url=url, status=200, content=["real page content here"],
        fetcher_used="stealthy", total_extracted_chars=23,
    )


@pytest.mark.asyncio
async def test_status_zero_transport_error_escalates_to_stealthy():
    srv = MasterFetchServer()
    with patch.object(srv, "_http_with_retry", new=AsyncMock(return_value=_http_error())), \
         patch.object(srv, "_ensure_auto_session", new=AsyncMock(return_value="sid")), \
         patch.object(srv, "stealthy_fetch", new=AsyncMock(return_value=_stealth_ok())) as stealth:
        result = await srv._auto_escalate(
            "https://example.com/", "markdown", None, True, True,
            0, 0, True, False, 0, None, 30000, False,
            False, False, False, None, None, None,
        )
    assert stealth.await_count == 1
    assert result.status == 200
    assert result.escalation_path == "http→stealthy"


@pytest.mark.asyncio
async def test_dns_failure_does_not_escalate():
    srv = MasterFetchServer()
    dns_error = _http_error("4 attempts failed. Last: [Errno -2] Name or service not known")
    with patch.object(srv, "_http_with_retry", new=AsyncMock(return_value=dns_error)), \
         patch.object(srv, "stealthy_fetch", new=AsyncMock()) as stealth:
        result = await srv._auto_escalate(
            "https://no-such-host.invalid/", "markdown", None, True, True,
            0, 0, True, False, 0, None, 30000, False,
            False, False, False, None, None, None,
        )
    assert stealth.await_count == 0
    assert result.status == 0


@pytest.mark.asyncio
async def test_connection_refused_does_not_escalate():
    srv = MasterFetchServer()
    refused = _http_error("4 attempts failed. Last: [Errno 111] Connection refused")
    with patch.object(srv, "_http_with_retry", new=AsyncMock(return_value=refused)), \
         patch.object(srv, "stealthy_fetch", new=AsyncMock()) as stealth:
        result = await srv._auto_escalate(
            "https://example.com:9/", "markdown", None, True, True,
            0, 0, True, False, 0, None, 30000, False,
            False, False, False, None, None, None,
        )
    assert stealth.await_count == 0


@pytest.mark.asyncio
async def test_real_200_is_accepted_without_escalation():
    srv = MasterFetchServer()
    ok = ResponseModel(
        url="https://example.com/", status=200,
        content=["real content " * 100], fetcher_used="http",
    )
    with patch.object(srv, "_http_with_retry", new=AsyncMock(return_value=ok)), \
         patch.object(srv, "stealthy_fetch", new=AsyncMock()) as stealth:
        result = await srv._auto_escalate(
            "https://example.com/", "markdown", None, True, True,
            0, 0, True, False, 0, None, 30000, False,
            False, False, False, None, None, None,
        )
    assert stealth.await_count == 0
    assert result.escalation_path == "direct:http"


@pytest.mark.asyncio
async def test_stealthy_status_zero_is_not_accepted_as_success():
    """If stealthy ALSO fails at transport level, the result must be the
    all-tiers-failed envelope, not a silently accepted status=0 'success'."""
    srv = MasterFetchServer()
    stealth_fail = _http_error("browser navigation failed")
    with patch.object(srv, "_http_with_retry", new=AsyncMock(return_value=_http_error())), \
         patch.object(srv, "_ensure_auto_session", new=AsyncMock(return_value="sid")), \
         patch.object(srv, "stealthy_fetch", new=AsyncMock(return_value=stealth_fail)):
        result = await srv._auto_escalate(
            "https://example.com/", "markdown", None, True, True,
            0, 0, True, False, 0, None, 30000, False,
            False, False, False, None, None, None,
        )
    assert "all_failed" in result.escalation_path
    assert result.content[0].startswith("[All fetch tiers failed")


class TestDeterministicNetError:

    @pytest.mark.parametrize("msg", [
        "Name or service not known",
        "nodename nor servname provided",
        "Temporary failure in name resolution",
        "getaddrinfo failed",
        "Connection refused",
        "connect ECONNREFUSED 1.2.3.4:443",
    ])
    def test_deterministic_signals(self, msg):
        assert _is_deterministic_net_error(_http_error(msg)) is True

    @pytest.mark.parametrize("msg", [
        "Connection reset by peer",
        "TLS handshake failed",
        "EOF occurred in violation of protocol",
        "timed out",
    ])
    def test_escalatable_signals(self, msg):
        assert _is_deterministic_net_error(_http_error(msg)) is False

    def test_nonzero_status_never_deterministic(self):
        r = ResponseModel(url="x", status=403, content=[""], error="whatever")
        assert _is_deterministic_net_error(r) is False


@pytest.mark.asyncio
async def test_stealthy_exception_becomes_failure_envelope():
    """A browser crash must not escape as a raw exception - the agent
    needs the structured all-tiers-failed envelope with HTTP context."""
    srv = MasterFetchServer()
    with patch.object(srv, "_http_with_retry", new=AsyncMock(return_value=_http_error())), \
         patch.object(srv, "_ensure_auto_session", new=AsyncMock(side_effect=RuntimeError("browser launch failed"))):
        result = await srv._auto_escalate(
            "https://example.com/", "markdown", None, True, True,
            0, 0, True, False, 0, None, 30000, False,
            False, False, False, None, None, None,
        )
    assert "all_failed" in result.escalation_path
    assert result.content[0].startswith("[All fetch tiers failed")


@pytest.mark.asyncio
async def test_http_retry_bails_early_on_dns_error():
    """DNS failures must not burn 4 attempts with backoff sleeps."""
    srv = MasterFetchServer()
    calls = []

    async def fake_get(url, **kwargs):
        calls.append(url)
        raise OSError("[Errno -2] Name or service not known")

    with patch.object(srv, "get", side_effect=fake_get):
        result = await srv._http_with_retry("https://nope.invalid/", retries=0)
    assert len(calls) == 1
    assert result.status == 0
    assert result.retry_count == 1


@pytest.mark.asyncio
async def test_http_retry_passes_retries_zero_to_inner_get():
    """The outer loop is the retry policy; inner get must not also retry."""
    srv = MasterFetchServer()
    seen = {}

    async def fake_get(url, **kwargs):
        seen.update(kwargs)
        raise OSError("Connection reset by peer")

    with patch.object(srv, "get", side_effect=fake_get), \
         patch("master_fetch.server.asyncio_sleep", new=AsyncMock()):
        await srv._http_with_retry("https://example.com/")
    assert seen.get("retries") == 0
