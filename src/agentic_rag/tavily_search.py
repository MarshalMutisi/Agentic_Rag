from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web with Tavily

    Results have the same shape as VectorStore.search() results, so the rest of
    the pipeline can grade and cite both kinds of evidence the same way.

    Args:
        query: The search text
        max_results: Maximum number of results

    Returns:
        List of {"text", "score", "source_type": "web", "metadata": {"title", "url"}} dicts.
        Empty if the search fails, so a web outage doesn't crash the agent.
    """
    try:
        response = TavilySearch(max_results=max_results).invoke({"query": query})
    except Exception as e:
        print(f"Web search failed: {e}")
        return []

    # On API errors the tool returns an error message instead of a dict.
    if not isinstance(response, dict):
        print(f"Web search failed: {response}")
        return []

    return [
        {
            "text": result["content"],
            "score": result.get("score"),
            "source_type": "web",
            "metadata": {"title": result.get("title", ""), "url": result["url"]},
        }
        for result in response.get("results", [])
        if result.get("content")
    ]


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval augmented generation?"
    for result in web_search(query):
        print(f"- {result['metadata']['title']} ({result['metadata']['url']})\n  {result['text'][:150]!r}")
