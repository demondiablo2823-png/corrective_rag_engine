from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from config.models import *

dense_embedd  = HuggingFaceEmbeddings(model_name=dense_embedding_model)
sparse_embedd = FastEmbedSparse(model_name="Qdrant/bm25")

sync_client = QdrantClient(path="./local_qdrant")

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
    delete_collection()
    vs = get_vectorstore()
    await vs.aadd_documents(docs)
    return vs.as_retriever(search_kwargs={"k": k})
