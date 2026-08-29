"""Robots.txt compliance for Hound.

Respects robots.txt Disallow rules with caching per domain.
Uses Scrapling's HTTP fetcher (curl_cffi) instead of stdlib urllib
for browser-impersonated requests that bypass basic bot blocking.
Non-blocking — all I/O is async.
"""

import asyncio
import logging
from urllib.robotparser import RobotFileParser
from time import time

logger = logging.getLogger("master-fetch.robots")

# Cache robots.txt parsers per domain: {domain: (RobotFileParser | None, fetch_time)}
# A None parser is a cached miss (unreachable or unparseable robots.txt), kept so
# a domain without robots.txt is not re-fetched on every single URL check.
_robots_cache: dict[str, tuple[RobotFileParser | None, float]] = {}
# In-flight fetches per domain, so N concurrent checks on one domain issue one
# request instead of N. Populated and cleared under _robots_lock.
_robots_inflight: dict[str, "asyncio.Task[RobotFileParser | None]"] = {}
_robots_lock: asyncio.Lock | None = None
_ROBOTS_CACHE_TTL = 3600  # 1 hour
# Misses expire sooner than hits: an unreachable robots.txt is often transient,
# so retry within the hour without paying _FETCH_TIMEOUT on every URL.
_ROBOTS_MISS_TTL = 300  # 5 minutes
_FETCH_TIMEOUT = 10  # seconds


def _get_robots_lock() -> asyncio.Lock:
    """Lazy-init the robots cache lock (needs running event loop)."""
    global _robots_lock
    if _robots_lock is None:
        _robots_lock = asyncio.Lock()
    return _robots_lock

DEFAULT_USER_AGENT = (
    "Hound/2.7 (web research for AI agents; https://github.com/dondai44423/master-fetch)"
)


def _extract_netloc(url: str) -> str:
    """Extract netloc from URL. Returns '' for invalid URLs."""
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


async def _fetch_robots_txt(domain: str) -> str | None:
    """Fetch robots.txt for a domain using primp (async, impersonated).

    Returns the raw text content or None if unreachable.
    """
    try:
        from master_fetch.fetcher import HTTPSession
        async with HTTPSession(stealthy_headers=False, retries=1) as sess:
            response = await sess.get(
                f"https://{domain}/robots.txt", timeout=_FETCH_TIMEOUT,
            )
            body = getattr(response, 'body', None)
            if body:
                return body.decode(
                    getattr(response, 'encoding', None) or 'utf-8', errors='replace',
                )
    except ImportError:
        logger.debug(f"fetcher not available for robots.txt fetch of {domain}, using fallback")
    except Exception as e:
        logger.debug(f"Robots.txt fetch failed for {domain}: {e}")

    # Fallback: urllib in thread (no impersonation, but works for basic sites)
    try:
        from urllib.request import Request, urlopen
        from urllib.error import URLError

        def _sync_fetch():
            req = Request(
                f"https://{domain}/robots.txt",
                headers={"User-Agent": DEFAULT_USER_AGENT},
            )
            with urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")

        return await asyncio.to_thread(_sync_fetch)
    except (URLError, OSError) as e:
        logger.debug(f"robots.txt unreachable for {domain}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error fetching robots.txt for {domain}: {e}")
        return None


async def _load_robots_parser(domain: str) -> RobotFileParser | None:
    """Fetch + parse robots.txt for one domain and record the outcome.

    Runs as a single shared task per domain. Both outcomes are cached — a None
    parser is a miss, so an unreachable robots.txt is not re-fetched per URL.
    """
    try:
        raw = await _fetch_robots_txt(domain)
        parser: RobotFileParser | None = None
        if raw is None:
            logger.debug(f"robots.txt unreachable for {domain}")
        else:
            try:
                parser = RobotFileParser()
                parser.parse(raw.splitlines())
                logger.debug(f"Fetched and parsed robots.txt for {domain}")
            except Exception as e:
                parser = None
                logger.debug(f"Failed to parse robots.txt for {domain}: {e}")

        # Timestamp after the fetch, so a slow fetch does not shorten its own TTL.
        async with _get_robots_lock():
            _robots_cache[domain] = (parser, time())
        return parser
    finally:
        async with _get_robots_lock():
            _robots_inflight.pop(domain, None)


async def _get_robots_parser(domain: str, user_agent: str = "*") -> RobotFileParser | None:
    """Fetch and parse robots.txt for a domain. Caches result.

    Returns None if robots.txt is unreachable (allow by default).
    Returns RobotFileParser if successfully fetched.
    """
    lock = _get_robots_lock()

    async with lock:
        # Check cache. Misses are cached too, under a shorter TTL.
        if domain in _robots_cache:
            parser, fetched_at = _robots_cache[domain]
            ttl = _ROBOTS_CACHE_TTL if parser is not None else _ROBOTS_MISS_TTL
            if time() - fetched_at < ttl:
                return parser
            del _robots_cache[domain]

        # Join the in-flight fetch for this domain rather than starting another.
        task = _robots_inflight.get(domain)
        if task is None:
            task = asyncio.ensure_future(_load_robots_parser(domain))
            _robots_inflight[domain] = task

    # shield: one caller being cancelled must not cancel the fetch the others
    # are waiting on.
    try:
        return await asyncio.shield(task)
    except Exception as e:
        logger.debug(f"robots.txt lookup failed for {domain}: {e}")
        return None


async def is_allowed(url: str, user_agent: str = "*") -> bool:
    """Check if a URL is allowed per robots.txt.

    Returns True if:
    - robots.txt is unreachable (allow by default)
    - robots.txt allows this URL
    - URL is invalid (malformed)

    Returns False only if robots.txt explicitly disallows this URL.
    """
    domain = _extract_netloc(url)
    if not domain:
        return True  # Malformed URL: allow

    parser = await _get_robots_parser(domain, user_agent)
    if parser is None:
        return True  # Can't reach robots.txt: allow

    try:
        return parser.can_fetch(user_agent, url)
    except Exception:
        return True  # Parse error: allow


async def clear_robots_cache() -> None:
    """Clear the robots.txt cache (both hits and cached misses)."""
    lock = _get_robots_lock()
    async with lock:
        _robots_cache.clear()
    logger.info("Robots.txt cache cleared")
