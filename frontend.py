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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rag_chain import load_docs, create_vectorstore, get_rag_chain

st.set_page_config(page_title="📄 Gemini RAG Assistant", layout="wide")
st.title("📄 Ask Your Document (Google Gemini RAG)")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader("Upload a document (DOCX/PDF/TXT/CSV)", type=["pdf", "txt", "csv", "docx"], accept_multiple_files=True)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if uploaded_files:
    all_docs = []
    combined_filename = "_".join([file.name for file in uploaded_files])
    vectorstore_path = os.path.join("vectorstores", combined_filename)

    os.makedirs("docs", exist_ok=True)
    os.makedirs("vectorstores", exist_ok=True)

    with st.spinner("Processing documents..."):
        # If cached vectorstore exists
        if os.path.exists(vectorstore_path):
            vectordb = FAISS.load_local(
                vectorstore_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # Load and merge all docs
            for file in uploaded_files:
                file_path = os.path.join("docs", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                all_docs.extend(load_docs(file_path))

            vectordb = create_vectorstore(all_docs)
            vectordb.save_local(vectorstore_path)

            # Optional preview
            st.markdown("#### First 3 lines of the first document:")
            st.write(all_docs[0].page_content[:500])

        st.session_state.qa_chain = get_rag_chain(vectordb)

    st.success("All documents processed. You can now ask questions!")

query = st.chat_input("Ask a question based on the document:")

if query and st.session_state.qa_chain:
    with st.spinner("Getting answer..."):
        result = st.session_state.qa_chain.invoke(query)
        st.markdown("### 📬 Answer:")
        #st.write(type(result["answer"]))
        st.write(result["answer"].content)
        with st.expander("Show retrieved context"):
            for doc in result['retrieved_docs']:
                st.code(doc.page_content[:300])

        st.session_state.history.append((query, result))
