from langchain_core.messages import trim_messages
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from config.llm import get_llm
from config.history import get_history
from config.prompt import CONTEXTUALIZE_PROMPT, QA_PROMPT
from retrieval.context import custom_retriever

llm = get_llm()

retriever               = RunnableLambda(custom_retriever)
qa_chain                = create_stuff_documents_chain(prompt=QA_PROMPT, llm=llm)
history_aware_retriever = create_history_aware_retriever(llm, retriever, CONTEXTUALIZE_PROMPT)
rag_chain               = create_retrieval_chain(history_aware_retriever, qa_chain)

def trim(x):
    return trim_messages(
        messages=x["chat_history"],
        include_system=True,
        max_tokens=400,
        end_on="human",
        strategy="last",
        token_counter=llm
    )

trimmed_chain = RunnablePassthrough.assign(
    chat_history=RunnableLambda(trim)
) | rag_chain

final_chain = RunnableWithMessageHistory(
    trimmed_chain,
    get_session_history=get_history,
    output_messages_key="answer",
    history_messages_key="chat_history"
)

async def ask(question: str, session_id: str = "default"):
    response = await final_chain.ainvoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in response["context"]
    })
    return response["answer"], sources