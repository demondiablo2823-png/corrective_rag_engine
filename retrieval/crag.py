from config.prompt import CRAG_PROMPT
from config.llm import get_llm, output_parser
from config.websearch import search
from retrieval.reRank import ranker
import asyncio

_grade_chain = CRAG_PROMPT | get_llm() | output_parser()

async def grade_document(query, doc):
    response = await _grade_chain.ainvoke({"question": query, "context": doc})
    return 1 if response.strip().lower().startswith('y') else 0

async def grade_all(query, document):
    if not document:
        return 'INCORRECT', []
    task = []
    for d in document:
        task.append(grade_document(query, d.page_content))
    grade = await asyncio.gather(*task)
    score = sum(grade) / len(grade)
    if score >= 0.7:
        accept = 'CORRECT'
    elif score <= 0.2:
        accept = 'INCORRECT'
    else:
        accept = 'AMBIGUOUS'
    return accept, grade


async def get_crag(query, document):
    accept, grade = await grade_all(query, document)

    final = []
    if accept in ('CORRECT', 'AMBIGUOUS'):
        final += [d for d, g in zip(document, grade) if g == 1]
    if accept == 'INCORRECT':
        final += await search(query)
    final = ranker(query, final)
    return final
