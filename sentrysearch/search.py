"""Query and retrieval logic."""

from .embedder import embed_query
from .store import get_collection, query_collection


def search_clips(query: str, n_results: int = 5) -> list[dict]:
    """Search indexed video clips with a natural language query.

    Args:
        query: Natural language search string.
        n_results: Number of results to return.

    Returns:
        List of result dicts sorted by relevance score (descending).
    """
    query_embedding = embed_query(query)
    collection = get_collection()
    return query_collection(collection, query_embedding, n_results=n_results)
