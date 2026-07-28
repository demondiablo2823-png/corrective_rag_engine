from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and latest user question, \
reformulate the question to be standalone and clear without the chat history. \
Do NOT answer the question, just reformulate it if needed, otherwise return as is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Generate exactly 3 alternative versions of the given question. \
Output only the 3 questions, one per line. \
No numbering, no bullets, no extra text, no blank lines."""),
    ("human", "{question}"),
])

CRAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing relevance of retrieved documents to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant.
Only respond with 'yes' or 'no', nothing else."""),
    ("human", """Retrieved document:
{context}

User question:
{question}"""),
])




QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and knowledgeable assistant. \
Answer the user's question using only the provided context. \
If the context does not contain enough information, say "I don't have enough information to answer this."

Rules:
- Be concise and precise
- Do not make up information
- If unsure, say so

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])