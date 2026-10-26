import os
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient


def search_web(query: str) -> list[dict[str, str]]:
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [
            {
                "title": "Web search is not set up",
                "url": "",
                "summary": "Add TAVILY_API_KEY to your local .env file to enable web search.",
            }
        ]
    response: dict[str, Any] = TavilyClient(api_key=api_key).search(
        query=query,
        max_results=5,
        search_depth="basic",
    )
    results: list[dict[str, str]] = []
    for item in response.get("results", [])[:5]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "summary": str(item.get("content", ""))[:300],
            }
        )
    return results
