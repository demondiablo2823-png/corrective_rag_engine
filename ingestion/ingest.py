from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.vectordb import get_db

async def ingest(path: str):
    docs = PyMuPDFLoader(path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    ).split_documents(docs)
    await get_db(chunks)