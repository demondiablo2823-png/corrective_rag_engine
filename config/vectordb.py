from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from config.models import *
import os
from dotenv import load_dotenv

load_dotenv()

dense_embedd  = HuggingFaceEmbeddings(model_name=dense_embedding_model)
sparse_embedd = FastEmbedSparse(model_name="Qdrant/bm25")

sync_client = QdrantClient(
    url=os.getenv("QDRANT_endpoint").strip(),
    api_key=os.getenv("QDRANT_key").strip(),
)

def get_vectorstore() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=sync_client,
        collection_name="rag",
        embedding=dense_embedd,
        sparse_embedding=sparse_embedd,
        retrieval_mode=RetrievalMode.HYBRID,
    )

def delete_collection():
    if sync_client.collection_exists("rag"):
        sync_client.delete_collection("rag")


async def get_db(docs, k=5):
    vectorstore = await QdrantVectorStore.afrom_documents(
        docs,
        embedding=dense_embedd,
        sparse_embedding=sparse_embedd,
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name="rag",
        url=os.getenv("QDRANT_endpoint").strip(),
        api_key=os.getenv("QDRANT_key").strip(),
        force_recreate=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})