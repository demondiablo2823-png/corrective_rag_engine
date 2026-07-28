from config.prompt import MULTI_QUERY_PROMPT
from config.llm import get_llm, output_parser

_query_chain = MULTI_QUERY_PROMPT | get_llm() | output_parser()

async def multi_query(query):
    response = await _query_chain.ainvoke({"question": query})
    queries = [q.strip() for q in response.strip().split("\n") if q.strip()][:3]
    return [query] + queries if queries else [query]