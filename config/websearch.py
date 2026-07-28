from langchain_core.documents import Document
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv
load_dotenv()

web_search = TavilySearch(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_key")
)

async def search(query):
    response= await web_search.ainvoke(query)
    doc=[]
    for r in response["results"]:
        doc.append(
            Document(
                    page_content=r["content"],
                    metadata={
                        "title":r["title"],
                        "source":r["url"]
                    }                
            )
        )
    return doc