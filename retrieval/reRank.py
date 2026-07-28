import os
from functools import lru_cache
import cohere
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_cohere():
    return cohere.ClientV2(
        api_key=os.getenv("COHERENCE_api")
    )


def ranker(query, docs, top_n=5):
    if not docs:
        return []
    co = get_cohere()
    results = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=[doc.page_content for doc in docs],
        top_n=min(top_n, len(docs))
    ).results
    return [docs[r.index] for r in results]