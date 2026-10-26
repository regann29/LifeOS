import os
from typing import Any

from tavily import TavilyClient


def search_web(query: str) -> list[dict[str, str]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Web search needs TAVILY_API_KEY in the local environment.")
    response: dict[str, Any] = TavilyClient(api_key=api_key).search(
        query=query,
        max_results=5,
        search_depth="basic",
    )
    results: list[dict[str, str]] = []
    for item in response.get("results", []):
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "content": str(item.get("content", ""))[:1200],
            }
        )
    return results
