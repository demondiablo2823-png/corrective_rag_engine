from langchain_core.documents import Document
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv

load_dotenv()

async def search(query):
    api_key = (os.getenv("TAVILY_key") or os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return []
    try:
        web_search = TavilySearch(max_results=3, tavily_api_key=api_key)
        response = await web_search.ainvoke(query)
        doc = []
        for r in response.get("results", []):
            doc.append(
                Document(
                    page_content=r["content"],
                    metadata={
                        "title": r.get("title", "Web Source"),
                        "source": r.get("url", "")
                    }
                )
            )
        return doc
    except Exception:
        return []
