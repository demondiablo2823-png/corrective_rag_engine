from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from config.models import *
import os
from dotenv import load_dotenv
load_dotenv()


def get_dense_embedd():
    return HuggingFaceEmbeddings(model_name=dense_embedding_model)

def get_sparse_embedd():
    return FastEmbedSparse(model_name="Qdrant/bm25")


def get_client():
    url = (os.getenv("QDRANT_endpoint") or "").strip()
    if url and not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"
    if url and ".qdrant.io" in url and not url.endswith(":6333") and not url.endswith(":6334"):
        url = f"{url}:6333"
    return QdrantClient(
        url=url,
        api_key=(os.getenv("QDRANT_key") or "").strip(),
    )

def get_vectorstore() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_client(),
        collection_name="rag",
        embedding=get_dense_embedd(),
        sparse_embedding=get_sparse_embedd(),
        retrieval_mode=RetrievalMode.HYBRID,
    )

async def get_db(docs, k=5):
    vectorstore = await QdrantVectorStore.afrom_documents(
        docs,
        embedding=get_dense_embedd(),
        sparse_embedding=get_sparse_embedd(),
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name="rag",
        client=get_client(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})