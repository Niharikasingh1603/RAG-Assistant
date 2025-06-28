# 📄 Gemini RAG Assistant

A Retrieval-Augmented Generation (RAG) document assistant built with LangChain, Google Gemini, and Streamlit.

## 🔍 Features

- 📁 Upload multiple PDF/TXT/CSV documents
- 🧠 Generates accurate answers using Gemini Pro (via LangChain)
- 🧾 Uses FAISS for fast semantic document search
- 🔒 Local vectorstore caching for speed
- 💬 Real-time question-answering interface with Streamlit
- 🧵 Context-aware answers + source chunk viewer

## 🚀 Demo

> 🔗 [Live App on Streamlit Cloud](https://your-app-link.streamlit.app)  
(Replace this with your actual URL after deploying)

## 🛠️ Tech Stack

- 🧠 Google Gemini API (`gemini-2.0-flash`, `embedding-001`)
- 🧪 LangChain
- 🔍 FAISS for vector search
- 🧰 Streamlit for frontend
- 📄 PyPDFLoader, TextLoader, CSVLoader

## 🧰 Setup Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/Niharikasingh1603/gemini-rag-assistant.git
   cd gemini-rag-assistant

2. **Install dependencies**:
    pip install -r requirements.txt

3. **Set your API key:**:
    - Local: Create a .env file with:
    GOOGLE_API_KEY=your-api-key-here

4. **Run locally:**:
    streamlit run frontend.py