import asyncio
import sys

# Patch: Ensure event loop exists for async gRPC clients
if sys.version_info >= (3, 10):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rag_chain import load_docs, create_vectorstore, get_rag_chain

st.set_page_config(page_title="📄 Gemini RAG Assistant", layout="wide")
st.title("📄 Ask Your Document (Google Gemini RAG)")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader("Upload a document (PDF/TXT/CSV)", type=["pdf", "txt", "csv", "docx"], accept_multiple_files=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if uploaded_files:
    all_docs = []
    total_size_bytes = 0
    combined_filename = "_".join([file.name for file in uploaded_files])
    vectorstore_path = os.path.join("vectorstores", combined_filename)

    os.makedirs("docs", exist_ok=True)
    os.makedirs("vectorstores", exist_ok=True)

    with st.spinner("Processing documents..."):
        # Calculate total size for all uploaded files
        for file in uploaded_files:
            file_path = os.path.join("docs", file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            total_size_bytes += os.path.getsize(file_path)
            all_docs.extend(load_docs(file_path))

        size_kb = total_size_bytes / 1024
        size_mb = size_kb / 1024

        if os.path.exists(vectorstore_path):
            vectordb = FAISS.load_local(
                vectorstore_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            vectordb = create_vectorstore(all_docs, size_kb, size_mb)
            vectordb.save_local(vectorstore_path)

        st.markdown("#### First 3 lines of the first document:")
        st.write(all_docs[0].page_content[:500])

        st.session_state.qa_chain = get_rag_chain(vectordb, size_kb, size_mb)

    st.success("All documents processed. You can now ask questions!")

query = st.chat_input("Ask a question based on the document:")

if query and st.session_state.qa_chain:
    with st.spinner("Getting answer..."):
        response_container = st.empty()
        answer_chunks = []
        retrieved_docs = None
        for output in st.session_state.qa_chain.stream(query):
            answer_chunks.append(output["answer"])
            retrieved_docs = output.get("retrieved_docs", None)
        full_answer = ''.join(chunk.content for chunk in answer_chunks)
        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({
            "role": "assistant", 
            "content": full_answer,
            "retrieved_docs": retrieved_docs})

# Display chat history in a chat-like format
for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Optionally show retrieved context for the last answer
if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
    retrieved_docs = st.session_state.history[-1].get("retrieved_docs", None)
    if retrieved_docs:
        with st.expander("Show retrieved context"):
            for doc in retrieved_docs:
                st.code(doc.page_content[:300])

