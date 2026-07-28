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
    return QdrantClient(
        url=os.getenv("QDRANT_endpoint"),
        api_key=os.getenv("QDRANT_key"),
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
        url=os.getenv("QDRANT_endpoint"),
        api_key=os.getenv("QDRANT_key"),
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})