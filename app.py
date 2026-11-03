import streamlit as st
import asyncio
import os
import tempfile
from chain.final_chain import ask
from ingestion.ingest import ingest
from config.vectordb import delete_collection

# Auto-load secrets from Streamlit Cloud if present
try:
    for k in ["GROQ_API_KEY", "COHERE_API_KEY", "COHERENCE_api", "TAVILY_API_KEY", "TAVILY_key"]:
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k]).strip()
except Exception:
    pass

st.set_page_config(page_title="Corrective RAG Engine", page_icon="🚀", layout="wide")
st.title("🚀 Corrective RAG Chatbot")

with st.sidebar:
    st.header("🔑 API Keys")
    groq_api_key = st.text_input(
        "Groq API Key *",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Required for response generation. Free key at https://console.groq.com/keys"
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key.strip()

    with st.expander("Additional Keys (Optional)"):
        cohere_key = st.text_input(
            "Cohere API Key (Re-Ranking)",
            value=os.getenv("COHERENCE_api", "") or os.getenv("COHERE_API_KEY", ""),
            type="password",
            help="Optional for cross-encoder reranking."
        )
        if cohere_key:
            os.environ["COHERENCE_api"] = cohere_key.strip()
            os.environ["COHERE_API_KEY"] = cohere_key.strip()

        tavily_key = st.text_input(
            "Tavily API Key (Web Search Fallback)",
            value=os.getenv("TAVILY_key", "") or os.getenv("TAVILY_API_KEY", ""),
            type="password",
            help="Optional for web search fallback when context is insufficient."
        )
        if tavily_key:
            os.environ["TAVILY_key"] = tavily_key.strip()
            os.environ["TAVILY_API_KEY"] = tavily_key.strip()

    st.markdown("---")
    st.markdown("💡 *Get a free Groq API key at [console.groq.com](https://console.groq.com/keys)*")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

MAX_FILE_SIZE = 50 * 1024 * 1024

if uploaded_file and "ingested" not in st.session_state:

    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Ingesting document..."):
        asyncio.run(ingest(tmp_path))

    os.unlink(tmp_path)
    st.session_state.ingested = True
    st.session_state.messages = []
    st.success("Document ready!")

if "ingested" in st.session_state:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask a question...")

    if question:
        if not os.getenv("GROQ_API_KEY"):
            st.error("⚠️ Please enter a valid Groq API Key in the left sidebar to chat.")
            st.stop()

        st.chat_message("user").write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            try:
                answer, sources = asyncio.run(ask(question))
            except Exception as e:
                err_str = str(e).lower()
                if "authentication" in err_str or "invalid api key" in err_str or "401" in err_str:
                    answer = "❌ **Groq Authentication Error**: The Groq API key provided is invalid or expired. Please check your key in the left sidebar."
                    sources = []
                else:
                    raise e

        st.chat_message("assistant").write(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(source)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()
    if st.button("End Session & Clear Data"):
        delete_collection()
        st.session_state.clear()
        st.rerun()

else:
    st.info("Please upload a PDF to start chatting.")
