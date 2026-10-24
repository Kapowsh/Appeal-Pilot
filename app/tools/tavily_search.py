"""Optional public-web lookup. Results are display-only "external references"."""
import logging

from tavily import TavilyClient

log = logging.getLogger(__name__)


def search(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    try:
        resp = TavilyClient(api_key=api_key).search(query, max_results=max_results)
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": (r.get("content") or "")[:300]}
                for r in resp.get("results", [])]
    except Exception as e:
        log.warning("Tavily search failed: %s", e)
        return []
