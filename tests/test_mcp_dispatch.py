"""Regression tests for the MCP binding layer (issue #31).

Omitting an optional enum arg (e.g. `mode`) through tools/call must behave
exactly like its declared default, whether the client sends the arg flat,
inside `options`, as explicit null, or not at all. Exercises the real mcp
2.x lowlevel Server in-process via the official Client.
"""
import pytest

from master_fetch.server import MasterFetchServer


class _FakeResponse:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def model_dump_json(self):
        return "{}"

    def model_dump(self):
        return {}


@pytest.fixture
def spy_search(monkeypatch):
    calls = []

    async def fake_smart_search(self, query, **kwargs):
        calls.append({"query": query, **kwargs})
        return _FakeResponse()

    monkeypatch.setattr(MasterFetchServer, "smart_search", fake_smart_search)
    return calls


@pytest.mark.asyncio
async def test_omitted_mode_uses_declared_default(spy_search):
    srv = MasterFetchServer()
    await srv._dispatch("mcp_smart_search", {"query": "python tutorial", "max_results": 3})
    assert spy_search[0]["query"] == "python tutorial"
    assert spy_search[0]["max_results"] == 3
    # mode must NOT leak through as None - the method default ("auto") applies
    assert "mode" not in spy_search[0] or spy_search[0]["mode"] is not None


@pytest.mark.asyncio
async def test_null_mode_is_dropped(spy_search):
    srv = MasterFetchServer()
    await srv._dispatch("mcp_smart_search", {"query": "x", "options": {"mode": None}})
    assert "mode" not in spy_search[0]


@pytest.mark.asyncio
async def test_flat_and_options_args_both_accepted(spy_search):
    srv = MasterFetchServer()
    await srv._dispatch("mcp_smart_search", {"query": "x", "mode": "neural"})
    assert spy_search[0]["mode"] == "neural"
    await srv._dispatch("mcp_smart_search", {"query": "x", "options": {"mode": "find_similar", "url": "https://a.b/c"}})
    assert spy_search[1]["mode"] == "find_similar"
    assert spy_search[1]["url"] == "https://a.b/c"


@pytest.mark.asyncio
async def test_top_level_arg_wins_over_options(spy_search):
    srv = MasterFetchServer()
    await srv._dispatch("mcp_smart_search",
                        {"query": "x", "max_results": 9, "options": {"max_results": 3}})
    assert spy_search[0]["max_results"] == 9


@pytest.mark.asyncio
async def test_missing_query_raises_clean_error():
    srv = MasterFetchServer()
    with pytest.raises(ValueError, match="query"):
        await srv._dispatch("mcp_smart_search", {"max_results": 3})


@pytest.mark.asyncio
async def test_non_dict_arguments_raise_clean_error():
    srv = MasterFetchServer()
    with pytest.raises(ValueError, match="JSON object"):
        await srv._dispatch("mcp_smart_search", "not-a-dict")


@pytest.mark.asyncio
async def test_unknown_tool_raises_clean_error():
    srv = MasterFetchServer()
    with pytest.raises(ValueError, match="Unknown tool"):
        await srv._dispatch("mcp_nope", {})


# ── End-to-end through the real mcp 2.x Server ──────────────────────────────

@pytest.mark.asyncio
async def test_tools_call_omitted_mode_over_mcp_protocol(spy_search):
    """The reporter's exact failure: tools/call without `mode` must return a
    result, not an Internal Server Error + hang."""
    from mcp import Client

    srv = MasterFetchServer()
    async with Client(srv.build_mcp_server()) as client:
        result = await client.call_tool(
            "mcp_smart_search", {"query": "python tutorial", "max_results": 3})
    assert not result.is_error
    assert spy_search[0]["query"] == "python tutorial"
    assert "mode" not in spy_search[0] or spy_search[0]["mode"] is not None


@pytest.mark.asyncio
async def test_tools_call_null_mode_over_mcp_protocol(spy_search):
    from mcp import Client

    srv = MasterFetchServer()
    async with Client(srv.build_mcp_server()) as client:
        result = await client.call_tool(
            "mcp_smart_search", {"query": "x", "options": {"mode": None}})
    assert not result.is_error


@pytest.mark.asyncio
async def test_tools_call_error_becomes_is_error_result_not_protocol_error():
    """A tool failure must surface as is_error=True content (LLM-visible),
    not a JSON-RPC protocol error that clients raise (mcp 2.x default for
    uncaught handler exceptions)."""
    from mcp import Client

    srv = MasterFetchServer()
    async with Client(srv.build_mcp_server()) as client:
        result = await client.call_tool("mcp_smart_search", {"max_results": 3})
    assert result.is_error
    assert "query" in result.content[0].text


# ─── MCP 2026-07-28 stateless spec conformance regression tests ──────────

class TestStatelessConformance:
    """The 2026-07-28 spec is stateless: no initialize handshake.
    server/discover MUST be callable without initialize, and version-less
    requests must be served on the default version."""

    @pytest.mark.asyncio
    async def test_server_discover_without_initialize(self):
        """server/discover must respond without prior initialize (2026-07-28 spec)."""
        from master_fetch.server import MasterFetchServer
        from mcp import Client
        from mcp.types import DiscoverResult

        srv = MasterFetchServer()
        server = srv.build_mcp_server()
        async with Client(server) as client:
            result = await client.session.send_discover("2026-07-28")
            assert "supportedVersions" in result or "supported_versions" in result, \
                f"expected supportedVersions in result: {result}"

    @pytest.mark.asyncio
    async def test_tools_list_without_initialize_via_stdio(self):
        """tools/list must respond without prior initialize (stateless mode).
        Sends a raw JSON-RPC tools/list without initialize through stdio."""
        import json, subprocess, sys, threading, time

        proc = subprocess.Popen(
            [sys.executable, "-m", "master_fetch.server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        lines = []
        threading.Thread(target=lambda: [lines.append(l) for l in proc.stdout], daemon=True).start()

        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}) + "\n")
        proc.stdin.flush()
        time.sleep(3)
        proc.kill()

        found = False
        for line in lines:
            try:
                msg = json.loads(line)
                if msg.get("id") == 1 and "result" in msg:
                    tools = msg["result"].get("tools", [])
                    assert len(tools) > 0, "expected tools in result"
                    found = True
            except (json.JSONDecodeError, KeyError):
                pass
        assert found, "tools/list without initialize returned no result"

    @pytest.mark.asyncio
    async def test_server_discover_handler_registered(self):
        """build_mcp_server must register the server/discover handler."""
        from master_fetch.server import MasterFetchServer

        srv = MasterFetchServer()
        server = srv.build_mcp_server()
        handler = server.get_request_handler("server/discover")
        assert handler is not None, "server/discover handler not registered"

    @pytest.mark.asyncio
    async def test_init_exempt_includes_all_spec_methods(self):
        """_INIT_EXEMPT must include all spec methods for stateless mode."""
        from mcp.server import runner
        from mcp_types.methods import SPEC_CLIENT_METHODS

        # build_mcp_server must have been called to apply the patch
        from master_fetch.server import MasterFetchServer
        srv = MasterFetchServer()
        srv.build_mcp_server()

        assert "server/discover" in runner._INIT_EXEMPT
        assert "tools/list" in runner._INIT_EXEMPT
        assert runner._INIT_EXEMPT == frozenset(SPEC_CLIENT_METHODS)

    @pytest.mark.asyncio
    async def test_validate_client_request_falls_back_to_latest(self):
        """validate_client_request must fall back to LATEST_PROTOCOL_VERSION
        for methods that only exist in the 2026-07-28 surface."""
        import mcp_types.methods as mt
        from mcp.server import runner
        from master_fetch.server import MasterFetchServer

        srv = MasterFetchServer()
        srv.build_mcp_server()  # applies the patch

        # server/discover doesn't exist for 2025-11-25, should fall back
        try:
            mt.validate_client_request("server/discover", "2025-11-25", {})
        except Exception:
            pytest.fail("validate_client_request should fall back to 2026-07-28")

    @pytest.mark.asyncio
    async def test_initialize_still_works_backward_compat(self):
        """Legacy clients that send initialize must still work."""
        import json, subprocess, sys, threading, time

        proc = subprocess.Popen(
            [sys.executable, "-m", "master_fetch.server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        lines = []
        threading.Thread(target=lambda: [lines.append(l) for l in proc.stdout], daemon=True).start()

        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }) + "\n")
        proc.stdin.flush()
        time.sleep(3)
        proc.kill()

        found = False
        for line in lines:
            try:
                msg = json.loads(line)
                if msg.get("id") == 1 and "result" in msg:
                    pv = msg["result"].get("protocolVersion", "")
                    assert pv in ("2025-11-25", "2025-06-18", "2025-03-26"), \
                        f"unexpected protocolVersion: {pv}"
                    found = True
            except (json.JSONDecodeError, KeyError):
                pass
        assert found, "initialize returned no result"
