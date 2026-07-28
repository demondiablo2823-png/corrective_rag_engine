import asyncio

from langchain_core.documents import Document

from config.vectordb import get_vectorstore
from retrieval.multiquery import multi_query
from retrieval.crag import get_crag
from retrieval.reRank import ranker


async def custom_retriever(query: str) -> list[Document]:
    queries = await multi_query(query)

    vs = get_vectorstore()
    tasks = [vs.asimilarity_search(q, k=10) for q in queries]
    results = await asyncio.gather(*tasks)

    all_docs = []
    for batch in results:
        for doc in batch:
            all_docs.append(doc)

    seen = {}
    for doc in all_docs:
        key = (doc.page_content, doc.metadata.get("source", ""))
        if key not in seen:
            seen[key] = doc
    unique_docs = list(seen.values())

    graded_docs = await get_crag(query, unique_docs)

    if not graded_docs:
        return unique_docs[:5]

    return graded_docs
