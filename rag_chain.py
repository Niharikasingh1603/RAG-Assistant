import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ( ChatGoogleGenerativeAI )
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
#from dotenv import load_dotenv
#load_dotenv()

import streamlit as st
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


def load_docs(file_path: str):
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.lower().endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

def create_vectorstore(docs, size_kb, size_mb):
    if size_kb <= 200:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    elif size_kb > 200 and size_mb <= 2:
        splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=150)
    elif size_mb > 2 and size_mb <= 10:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    else: # size_mb > 10
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=250)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

def get_rag_chain(vectordb, size_kb, size_mb):
    if size_kb <= 200:
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":6})
    elif size_kb > 200 and size_mb <= 2:
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k":10, "fetch_k":20, "lambda_mult":0.5})
    elif size_mb > 2 and size_mb <= 10:
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k":13, "fetch_k":25, "lambda_mult":0.5})
    else: # size_mb > 10
        retriever = vectordb.as_retriever(search_type="mmr+rerank", search_kwargs={"k":16, "fetch_k":30, "lambda_mult":0.3, "rerank_top_k":6})
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template = '''
        You are an AI assistant helping users understand and evaluate profiles and documents.
        Use only the context below to answer the question as accurately as possible. You may summarize, interpret, or infer **based on what is present in the context**, but do not add unrelated or external information.
        If the question cannot be answered based on the context, say:
        "Sorry, I don't have enough information to answer that."

        Context:
        {context}

        Question: {question}
        '''
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    parser = StrOutputParser()

    def retrieve_context(docs):
        joined_text = "\n\n".join(doc.page_content for doc in docs)
        return {"context_text": joined_text, "docs": docs}

    parallel_chain = RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough()
    ) | RunnableLambda(lambda inputs: {
        "context": retrieve_context(inputs["docs"]),
        "question": inputs["question"]
    })

    def process_output(inputs):
        context_text = inputs["context"]["context_text"]
        docs = inputs["context"]["docs"]
        formatted_prompt = prompt.format(context=context_text, question=inputs["question"])
        for chunk in llm.stream(formatted_prompt):
            yield {"answer": chunk, "retrieved_docs": docs}

    return parallel_chain | RunnableLambda(process_output)
