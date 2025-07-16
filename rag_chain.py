import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ( GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, )
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
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def get_rag_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":4})
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

    '''parallel_chain = RunnableParallel(
        context = retriever | RunnableLambda(lambda retrieved_docs: "\n\n".join(doc.page_content for doc in retrieved_docs)),
        question = RunnablePassthrough()
    )'''

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
        answer = llm.invoke(formatted_prompt)
        return {"answer": answer, "retrieved_docs": docs}

    return parallel_chain | RunnableLambda(process_output)
