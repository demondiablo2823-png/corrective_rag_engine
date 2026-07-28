import streamlit as st
import asyncio
import os
import tempfile
from chain.final_chain import ask
from ingestion.ingest import ingest

st.title("RAG Chatbot")

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
        st.chat_message("user").write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            answer, sources = asyncio.run(ask(question))

        st.chat_message("assistant").write(answer)
        with st.expander("Sources"):
            for source in sources:
                st.write(source)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Please upload a PDF to start chatting.")