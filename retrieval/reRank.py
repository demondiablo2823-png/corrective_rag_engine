import os
import cohere
from dotenv import load_dotenv

load_dotenv()

def ranker(query, docs, top_n=5):
    if not docs:
        return []
    api_key = (os.getenv("COHERENCE_api") or os.getenv("COHERE_API_KEY") or "").strip()
    if not api_key:
        return docs[:top_n]
    try:
        co = cohere.ClientV2(api_key=api_key)
        results = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[doc.page_content for doc in docs],
            top_n=min(top_n, len(docs))
        ).results
        return [docs[r.index] for r in results]
    except Exception:
        return docs[:top_n]
