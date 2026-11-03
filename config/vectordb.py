from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from config.models import *

dense_embedd  = HuggingFaceEmbeddings(model_name=dense_embedding_model)
sparse_embedd = FastEmbedSparse(model_name="Qdrant/bm25")

sync_client = QdrantClient(path="./local_qdrant")

_dim = None
def get_embedding_dim() -> int:
    global _dim
    if _dim is None:
        _dim = len(dense_embedd.embed_query("dim_check"))
    return _dim

def ensure_collection():
    if not sync_client.collection_exists("rag"):
        sync_client.create_collection(
            collection_name="rag",
            vectors_config={"": models.VectorParams(size=get_embedding_dim(), distance=models.Distance.COSINE)},
            sparse_vectors_config={"langchain-sparse": models.SparseVectorParams()},
        )

def get_vectorstore() -> QdrantVectorStore:
    ensure_collection()
    return QdrantVectorStore(
        client=sync_client,
        collection_name="rag",
        embedding=dense_embedd,
        sparse_embedding=sparse_embedd,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="",
        sparse_vector_name="langchain-sparse",
    )

def delete_collection():
    if sync_client.collection_exists("rag"):
        sync_client.delete_collection("rag")


async def get_db(docs, k=5):
    delete_collection()
    vs = get_vectorstore()
    await vs.aadd_documents(docs)
    return vs.as_retriever(search_kwargs={"k": k})

