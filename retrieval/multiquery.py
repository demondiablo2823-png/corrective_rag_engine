from config.prompt import MULTI_QUERY_PROMPT
from config.llm import get_llm, output_parser

async def multi_query(query):
    try:
        chain = MULTI_QUERY_PROMPT | get_llm() | output_parser()
        response = await chain.ainvoke({"question": query})
        queries = [q.strip() for q in response.strip().split("\n") if q.strip()][:3]
        return [query] + queries if queries else [query]
    except Exception:
        return [query]
