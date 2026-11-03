# 📺 YouTube Transcript RAG Assistant 🤖

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Enabled-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Core-orange.svg)

An end-to-end Retrieval-Augmented Generation (RAG) application that extracts YouTube video and playlist transcripts, builds a local FAISS vector database, and answers user queries with high accuracy using Google's Gemini LLM. 

Built with a decoupled architecture featuring a **FastAPI backend** for scalable REST endpoints and a **Streamlit frontend** for an interactive chat experience.

## 🏗️ Architecture & Workflow

1. **Data Ingestion:** Automatically extracts transcripts from provided YouTube URLs using `youtube-transcript-api` and `yt-dlp`.
2. **Document Processing:** Splits massive transcripts into manageable chunks using LangChain's `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200).
3. **Semantic Embedding:** Converts text chunks into vector embeddings using Hugging Face's lightweight `all-MiniLM-L6-v2` model.
4. **Vector Storage:** Indexes and stores embeddings locally using **FAISS** for ultra-fast similarity search.
5. **Retrieval & Generation:** A FastAPI REST endpoint receives user queries, retrieves the top-4 most relevant context chunks (k=4), and generates grounded answers using `gemini-2.5-flash`.

## 📊 RAG Evaluation Metrics

The pipeline's retrieval and generation quality was rigorously tested against a custom evaluation dataset.

* **Hit Rate (Recall@4):** 100.0% (Successfully retrieved relevant context in the top 4 chunks for all queries).
* **Faithfulness:** 100.0% (Zero hallucinations; answers were strictly derived from the retrieved video context).
* **Answer Relevance:** 100.0% (Generated answers directly addressed the user's prompt).
* **Correctness:** 100.0% (Successfully captured the expected core concepts).

## 📁 Repository Structure

```text
├── app.py                   # Streamlit interactive web interface
├── fast_api.py              # FastAPI REST endpoints (/ingest, /chat)
├── rag_pipeline.py          # Core LangChain, FAISS, and Gemini logic
├── rag_evaluation.ipynb     # Pipeline testing and metric calculation
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
