"""Tests for XHR capture on AJAX-shell pages.

The reference case is a weather page (meteologix.com) that returns 260KB of
HTML, extracts to 5.6KB of help text, and carries not one temperature: the
forecast arrives over /ajax_pub/weather* XHRs after render. Before capture,
hound reported that page as content_ok=True with no data in it.
"""

import pytest

from master_fetch.network_capture import (
    AJAX_SHELL_MIN_PLACEHOLDERS,
    MAX_SECONDARY_TEXT_CHARS,
    MAX_TEXT_CHARS,
    analyze_fragment,
    build_network_field,
    count_placeholder_containers,
    detect_ajax_shell,
    fragment_text,
    is_animation_payload,
    is_consent_payload,
    rank_fragments,
    score_fragment,
    should_capture,
)


# ─── Fixtures mirroring the real page structures ─────────────────────────────

# An unfilled panel: spinner image plus a short "loading" label, exactly the
# shape meteologix ships (the label is why "container must be empty" failed).
PANEL = (
    '<div id="weather-overview-{n}" class="">'
    '<div class="llajax ll-nhours">'
    '<img src="/images/layout/v2/loader_ajax.svg" alt="loading">'
    'Loading forecast'
    '</div></div>'
)

# Prose with no figures in it — what extraction recovers from such a page.
SHELL_PROSE = (
    "Here you can see a detailed look at the forecast for the coming hours. "
    "Note that the base for this is our Meteogram product, which shows a good "
    "average forecast for this location. However, you can also look at our "
    "compact prediction based on any other model that forecasts for your "
    "chosen location. The following models are available for this town. "
) * 6


def _shell_html(panels: int = 3, filler: int = 30_000) -> str:
    return (
        "<html><body>"
        + "".join(PANEL.format(n=i) for i in range(panels))
        + f"<p>{SHELL_PROSE}</p>"
        + "<!--" + "x" * filler + "-->"
        + "</body></html>"
    )


class TestShouldCapture:
    def test_captures_a_data_endpoint(self):
        assert should_capture(
            "https://meteologix.com/es/ajax_pub/weathernexthoursdays?city_id=1",
            "xhr", "text/html; charset=UTF-8",
        )

    def test_ignores_non_xhr_resources(self):
        assert not should_capture("https://x.test/data.json", "document", "application/json")
        assert not should_capture("https://x.test/logo.png", "image", "image/png")

    def test_drops_analytics_and_ad_hosts(self):
        for url in (
            "https://www.google-analytics.com/g/collect?v=2",
            "https://sentry.io/api/1/store/",
            "https://securepubads.g.doubleclick.net/gampad/ads",
        ):
            assert not should_capture(url, "xhr", "application/json"), url

    def test_drops_telemetry_paths_on_first_party_hosts(self):
        assert not should_capture("https://shop.test/api/analytics", "xhr", "application/json")
        assert not should_capture("https://shop.test/v1/events", "xhr", "application/json")

    def test_drops_sourcepoint_cmp_endpoints(self):
        # Served from a first-party CNAME, so host matching alone misses it.
        assert not should_capture(
            "https://data-c032428347.meteologix.com/wrapper/v2/messages?env=prod",
            "xhr", "application/json",
        )

    def test_drops_static_asset_hosts(self):
        # CoinGecko serves Lottie animation JSON from its static bucket; it is
        # JSON, first-party and useless.
        assert not should_capture(
            "https://static.coingecko.com/s/rocket-emoji-bd92.json", "xhr", "application/json",
        )

    def test_drops_bot_challenge_infrastructure(self):
        for url in (
            "https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/fo/1",
            "https://3.stg.html-load.cc/session/cae/www.example.com/q",
            "https://www.example.com/cdn-cgi/rum",
        ):
            assert not should_capture(url, "xhr", "application/json"), url

    def test_drops_asset_extensions(self):
        assert not should_capture("https://x.test/app.js", "xhr", "")
        assert not should_capture("https://x.test/style.css", "xhr", "")

    def test_drops_unhelpful_content_types(self):
        assert not should_capture("https://x.test/data", "xhr", "application/octet-stream")

    def test_keeps_missing_content_type(self):
        assert should_capture("https://x.test/api/rows", "xhr", "")

    def test_explicit_pattern_overrides_heuristics(self):
        import re
        pattern = re.compile(r"analytics")
        # The caller asked for it by name, so the noise filter steps aside.
        assert should_capture("https://x.test/api/analytics", "xhr", "application/json", pattern)
        assert not should_capture("https://x.test/api/rows", "xhr", "application/json", pattern)


class TestConsentDetection:
    def test_flags_a_cmp_payload(self):
        body = '{"propertyId": 17364, "campaigns": [{"type": "GDPR", "consent": {}}]}'
        assert is_consent_payload(body)

    def test_leaves_ordinary_data_alone(self):
        assert not is_consent_payload("02:00pm 29°C 0% 03:00pm 34°C 0%")

    def test_single_incidental_mention_is_not_enough(self):
        assert not is_consent_payload("Our consent form is available on request.")

    def test_flags_a_lottie_animation_payload(self):
        body = '{"v":"5.7.1","ip":0,"op":60,"ddd":0,"assets":[],"layers":[{"ty":4}]}'
        assert is_animation_payload(body)

    def test_ordinary_json_is_not_an_animation(self):
        assert not is_animation_payload('{"temp":29,"hours":[{"t":"02:00pm"}]}')


class TestFragmentText:
    def test_flattens_html_rows_to_lines(self):
        html = '<div class="h"><div class="fc-hours">02:00pm</div><div>29°C</div></div>'
        text = fragment_text(html, "text/html")
        assert "02:00pm" in text and "29°C" in text
        assert "<div" not in text

    def test_unescapes_entities(self):
        assert "AT&T" in fragment_text("<p>AT&amp;T</p>", "text/html")

    def test_drops_script_and_style_bodies_from_prose(self):
        html = "<div>real<style>.a{color:red}</style><script>var x=1</script></div>"
        assert "color:red" not in fragment_text(html, "text/html")

    def test_pretty_prints_json(self):
        text = fragment_text('{"temp":29,"unit":"C"}', "application/json")
        assert '"temp"' in text and "\n" in text

    def test_detects_json_without_a_content_type(self):
        assert '"temp"' in fragment_text('{"temp":29}', "")

    def test_invalid_json_falls_back_to_raw(self):
        assert fragment_text("{not json", "application/json") == "{not json"

    def test_recovers_data_from_inline_script_literals(self):
        # Thin markup, data in a chart config — the weather14days shape.
        html = (
            "<div></div><script>var hc_data = [{name:'Temperature',"
            + "data:[" + ",".join(f"[{i},{20 + i}]" for i in range(40)) + "]}];</script>"
        )
        result = analyze_fragment(html, "text/html")
        assert result["from_script"] is True
        assert "Temperature" in result["text"]

    def test_does_not_mine_scripts_when_markup_has_content(self):
        html = "<div>" + ("real rendered content. " * 40) + "</div><script>var a={b:1}</script>"
        result = analyze_fragment(html, "text/html")
        assert result["from_script"] is False

    def test_braces_inside_strings_do_not_truncate_a_literal(self):
        html = '<div></div><script>var d = {"note":"a } brace", "rows":[1,2,3,4,5,6,7,8,9,10],' \
               '"pad":"' + "x" * 150 + '"};</script>'
        text = analyze_fragment(html, "text/html")["text"]
        assert '"rows"' in text

    def test_empty_body_yields_empty_text(self):
        assert fragment_text("", "text/html") == ""


class TestRanking:
    def _frag(self, url, text, **kw):
        frag = {"url": url, "status": 200, "content_type": "text/html",
                "size_bytes": len(text), "text": text, "_raw": text}
        frag.update(kw)
        return frag

    def test_number_dense_panel_beats_a_bigger_prose_blob(self):
        data = self._frag("https://x.test/hours", "02:00pm 29°C 0% " * 50)
        prose = self._frag("https://x.test/about", "some ordinary sentence about things " * 200)
        assert score_fragment(data) > score_fragment(prose)

    def test_tiny_acknowledgements_never_win(self):
        ack = self._frag("https://x.test/ok", '{"ok":true}')
        data = self._frag("https://x.test/hours", "02:00pm 29°C 0% " * 50)
        ranked = rank_fragments([ack, data])
        assert ranked[0]["url"].endswith("/hours")
        assert ranked[0]["is_primary"] is True

    def test_rendered_panel_is_preferred_over_script_recovered_data(self):
        # The chart config is far bigger, but epoch pairs are not an answer.
        script = self._frag("https://x.test/chart", "[1785758400000, 29.3]," * 900,
                            from_script=True)
        rendered = self._frag("https://x.test/hours", "02:00pm 29°C 0% " * 50)
        ranked = rank_fragments([script, rendered])
        assert ranked[0]["url"].endswith("/hours")
        assert ranked[0]["is_primary"] is True

    def test_script_data_still_wins_when_nothing_was_rendered(self):
        script = self._frag("https://x.test/chart", "[1785758400000, 29.3]," * 900,
                            from_script=True)
        stub = self._frag("https://x.test/ping", "ok")
        ranked = rank_fragments([script, stub])
        assert ranked[0]["url"].endswith("/chart")

    def test_only_one_fragment_is_primary(self):
        frags = [self._frag(f"https://x.test/{i}", f"{i} rows of data " * 40) for i in range(5)]
        ranked = rank_fragments(frags)
        assert sum(1 for f in ranked if f["is_primary"]) == 1

    def test_primary_keeps_a_bigger_text_budget_than_the_rest(self):
        big = self._frag("https://x.test/a", "1 2 3 4 5 6 7 8 9 " * 5000)
        other = self._frag("https://x.test/b", "9 8 7 6 5 4 3 2 1 " * 5000)
        ranked = rank_fragments([big, other])
        assert len(ranked[0]["text"]) <= MAX_TEXT_CHARS
        assert len(ranked[1]["text"]) <= MAX_SECONDARY_TEXT_CHARS
        assert ranked[1]["text_truncated"] is True

    def test_internals_are_stripped_from_the_output(self):
        ranked = rank_fragments([self._frag("https://x.test/a", "data 1 2 3 " * 40)])
        assert "_raw" not in ranked[0] and "_score" not in ranked[0]

    def test_empty_input_is_handled(self):
        assert rank_fragments([]) == []


class TestBuildNetworkField:
    def _frag(self, url, text):
        return {"url": url, "status": 200, "content_type": "text/html",
                "size_bytes": len(text), "text": text, "_raw": text}

    def test_reports_the_primary_url(self):
        net = build_network_field([self._frag("https://x.test/hours", "02:00pm 29°C " * 40)])
        assert net["primary_url"] == "https://x.test/hours"
        assert net["captured_count"] == 1

    def test_discards_consent_blobs(self):
        consent = self._frag("https://x.test/msg",
                             '{"propertyId":1,"campaigns":[{"type":"GDPR","consent":1}]}' * 40)
        data = self._frag("https://x.test/hours", "02:00pm 29°C " * 40)
        net = build_network_field([consent, data])
        assert net["captured_count"] == 1
        assert net["primary_url"] == "https://x.test/hours"

    def test_discards_fragments_with_no_text(self):
        net = build_network_field([self._frag("https://x.test/empty", "   ")])
        assert net == {}

    def test_no_fragments_yields_an_empty_envelope(self):
        assert build_network_field([]) == {}


class TestPlaceholderCounting:
    def test_counts_panels_holding_only_a_spinner_and_a_label(self):
        assert count_placeholder_containers(_shell_html(panels=3)) >= 3

    def test_ignores_containers_that_already_hold_content(self):
        html = "<html><body>" + "".join(
            f"<div data-url='/api/{i}'>Real content already rendered here, "
            f"well beyond the short-label threshold for a waiting panel.</div>"
            for i in range(5)
        ) + "</body></html>"
        assert count_placeholder_containers(html) == 0

    def test_malformed_html_does_not_raise(self):
        assert count_placeholder_containers("<<<not html") == 0


class TestDetectAjaxShell:
    def test_detects_a_data_free_page_with_unfilled_panels(self):
        assert detect_ajax_shell(_shell_html(), SHELL_PROSE)

    def test_page_whose_text_carries_figures_is_not_a_shell(self):
        # The discriminator: content with numbers in it is content that arrived.
        text = SHELL_PROSE + " 29 34 15 21 0 42 13 32 38 16 2 24 " * 20
        assert not detect_ajax_shell(_shell_html(), text)

    def test_deferred_panels_alone_are_not_enough(self):
        # Measured on real pages: GitHub shows ~79 such panels with its content
        # fully present. Panels must coincide with data-free text.
        text = "Release 2.32.5 has 3 fixes and 12 commits since 2026. " * 60
        assert not detect_ajax_shell(_shell_html(panels=20), text)

    def test_no_panels_means_not_a_shell(self):
        html = "<html><body><p>" + SHELL_PROSE + "</p><!--" + "x" * 40_000 + "--></body></html>"
        assert not detect_ajax_shell(html, SHELL_PROSE)

    def test_small_documents_are_ignored(self):
        assert not detect_ajax_shell(_shell_html(filler=0)[:5000], SHELL_PROSE)

    def test_empty_extraction_is_left_to_the_js_shell_path(self):
        assert not detect_ajax_shell(_shell_html(), "")

    def test_requires_the_configured_number_of_panels(self):
        one = _shell_html(panels=1)
        assert count_placeholder_containers(one) < AJAX_SHELL_MIN_PLACEHOLDERS
        assert not detect_ajax_shell(one, SHELL_PROSE)
