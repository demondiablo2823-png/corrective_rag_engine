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
    url = os.getenv("QDRANT_endpoint")
    if url and not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"
    return QdrantClient(
        url=url,
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

def reset_collection(collection_name="rag"):
    try:
        client = get_client()
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
    except Exception as e:
        print(f"Warning: Could not reset collection '{collection_name}': {e}")

async def get_db(docs, k=5):
    reset_collection("rag")
    client = get_client()
    vectorstore = await QdrantVectorStore.afrom_documents(
        docs,
        embedding=get_dense_embedd(),
        sparse_embedding=get_sparse_embedd(),
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name="rag",
        client=client,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})