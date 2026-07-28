from config.prompt import MULTI_QUERY_PROMPT
from config.llm import get_llm,output_parser


async def multi_query(query):
    prompt=MULTI_QUERY_PROMPT
    llm=get_llm()
    op=output_parser()
    response = await (prompt | llm | op).ainvoke({"question": query})
    queries = [q.strip() for q in response.strip().split("\n") if q.strip()][:3]
    return [query] + queries if queries else [query]