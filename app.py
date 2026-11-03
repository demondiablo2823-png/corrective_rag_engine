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

# --- SIDEBAR: BRING YOUR OWN KEY (BYOK) ---
with st.sidebar:
    st.header("🔑 Bring Your Own Key")
    st.caption("This app uses a Bring-Your-Own-Key model. Your keys are kept in your browser session only and are never saved.")

    groq_api_key = st.text_input(
        "Groq API Key (Required)",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
        help="Required for generating answers. Instant & free at console.groq.com/keys"
    )

    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key.strip()
        st.success("✅ Groq Key Active")
    else:
        st.warning("⚠️ Groq Key Required")
        st.markdown("[👉 Get a free Groq key in 30 seconds](https://console.groq.com/keys)")

    with st.expander("⚙️ Optional API Keys"):
        cohere_key = st.text_input(
            "Cohere API Key (Re-Ranking)",
            value=os.getenv("COHERENCE_api", "") or os.getenv("COHERE_API_KEY", ""),
            type="password",
            placeholder="Optional",
            help="For cross-encoder reranking. If blank, original retrieval order is used."
        )
        if cohere_key:
            os.environ["COHERENCE_api"] = cohere_key.strip()
            os.environ["COHERE_API_KEY"] = cohere_key.strip()
            st.caption("✅ Cohere Active")
        else:
            st.markdown("[Get free Cohere key](https://dashboard.cohere.com/api-keys)")

        tavily_key = st.text_input(
            "Tavily API Key (Web Search Fallback)",
            value=os.getenv("TAVILY_key", "") or os.getenv("TAVILY_API_KEY", ""),
            type="password",
            placeholder="Optional",
            help="For live web fallback when document context is weak."
        )
        if tavily_key:
            os.environ["TAVILY_key"] = tavily_key.strip()
            os.environ["TAVILY_API_KEY"] = tavily_key.strip()
            st.caption("✅ Tavily Active")
        else:
            st.markdown("[Get free Tavily key](https://app.tavily.com/) recommendations")

    st.markdown("---")
    st.markdown("### 🛠 Architecture")
    st.caption("• Local Hybrid Vector DB (Qdrant)\n• Dense BGE + Sparse BM25\n• Multi-Query Expansion\n• CRAG Relevance Evaluation\n• Cross-Encoder Re-Ranking")

# Main page key banner if missing
has_groq = bool(os.getenv("GROQ_API_KEY", "").strip())
if not has_groq:
    st.info("👋 **Welcome!** To get started, please paste your free **[Groq API Key](https://console.groq.com/keys)** in the sidebar on the left.")

uploaded_file = st.file_uploader("Upload a PDF to chat with", type="pdf")

MAX_FILE_SIZE = 50 * 1024 * 1024

if uploaded_file and "ingested" not in st.session_state:

    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Ingesting document and indexing vectors locally..."):
        asyncio.run(ingest(tmp_path))

    os.unlink(tmp_path)
    st.session_state.ingested = True
    st.session_state.messages = []
    st.success("Document ready! You can now ask questions below.")

if "ingested" in st.session_state:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask a question about your document...")

    if question:
        current_key = os.getenv("GROQ_API_KEY", "").strip()
        if not current_key:
            st.error("⚠️ **Groq API Key required!** Please paste your key in the sidebar on the left to generate answers.")
            st.stop()

        st.chat_message("user").write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking & generating source-grounded response..."):
            try:
                answer, sources = asyncio.run(ask(question))
            except Exception as e:
                err_str = str(e).lower()
                if "authentication" in err_str or "invalid api key" in err_str or "401" in err_str:
                    answer = "❌ **Groq Authentication Error**: The Groq API key entered in the sidebar is invalid or expired. Please update your key in the sidebar."
                    sources = []
                else:
                    raise e

        st.chat_message("assistant").write(answer)
        if sources:
            with st.expander("Sources Attribution"):
                for source in sources:
                    st.write(source)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()
    if st.button("End Session & Clear Data"):
        delete_collection()
        st.session_state.clear()
        st.rerun()

elif not uploaded_file:
    st.info("📄 Please upload a PDF document above to begin.")
