import os
import re
from functools import lru_cache
from typing import Any

import yt_dlp
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv("key.env")
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key

_embedding_model: Any = None

prompt = PromptTemplate(
    template="""
    You are an expert AI assistant. Your task is to provide a clear, accurate, and concise answer to the user's question.

    Instructions:
    - Primary Answer: Base your main answer on the provided Context. Always share the content source (URL or title) provided in the context.
    - Handling Incomplete Data: If the context does not contain enough information to fully answer the question, you may provide additional relevant details from your general knowledge.
    - Outsourcing Disclaimer: If you include information that is NOT found in the context, you MUST explicitly state: "Note: This part is outsourcing from external knowledge/websites to give you the full content knowledge."
    - Format: Use bullet points for readability and keep the answer concise.

    Context:
    {context}

    Question: {query}
    """,
    input_variables=["context", "query"],
)

vector_store_cache: dict[str, Any] = {}
rag_chain_cache: dict[str, Any] = {}


def _get_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_model


embedding_model: Any = _get_embedding_model()


@lru_cache(maxsize=1)
def get_llm() -> Any:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to key.env or your environment."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_tokens=1024,
    )


def get_persist_directory(collection_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", collection_id):
        raise ValueError("Invalid collection_id.")
    return os.path.join(".", "faiss_db", collection_id)


def load_youtube_videos(playlist_url: str) -> list[str]:
    video_urls: list[str] = []
    ydl_options = {"extract_flat": True, "quiet": True}

    try:
        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
    except Exception as exc:
        match = re.search(r"(?:v=|/)([A-Za-z0-9_-]{11})", playlist_url)
        if match:
            return [f"https://www.youtube.com/watch?v={match.group(1)}"]
        print(f"Error loading playlist: {playlist_url}. Error: {exc}")
        return []

    if not info:
        return []

    entries = info.get("entries")
    if entries is None:
        video_id = info.get("id") or info.get("video_id")
        return (
            [f"https://www.youtube.com/watch?v={video_id}"]
            if video_id
            else []
        )

    for entry in entries:
        if not entry:
            continue
        video_id = entry.get("id") or entry.get("video_id")
        if video_id:
            video_urls.append(f"https://www.youtube.com/watch?v={video_id}")
    return video_urls


def process_youtube_videos(my_url: str) -> list[Document]:
    if "list=" in my_url:
        video_urls = load_youtube_videos(my_url)
    else:
        print("Single video detected!")
        video_urls = [my_url]

    documents = load_videos(video_urls)
    print(f"\nTotal Documents Loaded: {len(documents)}")
    return documents


def load_videos(video_urls_list: list[str]) -> list[Document]:
    loaded_docs: list[Document] = []
    transcript_api = YouTubeTranscriptApi()

    for video_url in video_urls_list:
        match = re.search(r"(?:v=|/)([A-Za-z0-9_-]{11})", video_url)
        if not match:
            print(f"Invalid video URL format: {video_url}")
            continue

        video_id = match.group(1)
        try:
            transcript = transcript_api.fetch(
                video_id,
                languages=["en", "hi", "en-IN"],
            )
            text = " ".join(
                item["text"] if isinstance(item, dict) else item.text
                for item in transcript
            )
        except Exception as exc:
            print(f"Skipping video {video_url} due to transcript error: {exc}")
            continue

        loaded_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": video_url,
                    "pages": 1,
                    "title": f"Transcript for {video_id}",
                    "author": "YouTube",
                },
            )
        )
    return loaded_docs


def generate_embedding(
    all_loaded_docs: list[Document], persist_directory: str = "./faiss_db"
) -> Any:
    if not all_loaded_docs:
        print("No documents to embed!")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    text_chunks = text_splitter.split_documents(all_loaded_docs)
    vectorstore = FAISS.from_documents(
        documents=text_chunks,
        embedding=_get_embedding_model(),
    )
    vectorstore.save_local(persist_directory)
    vector_store_cache[persist_directory] = vectorstore
    print(
        "Success! Your text is now embedded and stored locally in a "
        f"Vector Database named {persist_directory}."
    )
    return vectorstore


def get_vector_store(persist_directory: str) -> Any:
    if persist_directory not in vector_store_cache:
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(
                f"No vector database found at {persist_directory}"
            )
        vector_store_cache[persist_directory] = FAISS.load_local(
            folder_path=persist_directory,
            embeddings=_get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
    return vector_store_cache[persist_directory]


def retriever_func(query: str, persist_directory: str = "./faiss_db") -> list[Document]:
    vector_store = get_vector_store(persist_directory)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    return retriever.invoke(query)


def format_doc(docs: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown Website')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )


def get_rag_chain(persist_directory: str) -> Any:
    if persist_directory not in rag_chain_cache:
        setup_and_retrieval = RunnableParallel(
            context=RunnableLambda(
                lambda query: retriever_func(query, persist_directory)
            )
            | RunnableLambda(format_doc),
            query=RunnablePassthrough(),
        )
        rag_chain_cache[persist_directory] = (
            setup_and_retrieval | prompt | get_llm() | StrOutputParser()
        )
    return rag_chain_cache[persist_directory]


def generate_answers(query: str, persist_directory: str = "./faiss_db") -> str:
    rag_chain = get_rag_chain(persist_directory)
    print(f"Asking the LLM: {query}\n")
    response = rag_chain.invoke(query)
    print("$$$ FINAL ANSWER $$$")
    return response


# ==========================================
# FINAL EXECUTION
# ==========================================
if __name__ == "__main__":
    import uuid
    
    # 1. Ask for the video URL
    user_url = input("Enter the YouTube Video or Playlist URL: ").strip()
    print("\nLoading transcripts... This might take a moment.")
    
    # 2. Process transcripts
    docs = process_youtube_videos(user_url)
    if not docs:
        print("No transcripts could be loaded for this URL. Exiting.")
        raise SystemExit
        
    # 3. Build the Vector Database
    print("\nCreating the Vector Database...")
    collection_id = uuid.uuid4().hex
    persist_dir = get_persist_directory(collection_id)
    my_vector_database = generate_embedding(docs, persist_dir)
    
    print("\n" + "="*50)
    print("Database Ready! Ask questions about the video(s).")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    # 4. Interactive Q&A Loop
    while True:
        query = input("Your Question: ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            print("Ending the session. Goodbye!")
            break
        if not query:
            continue
            
        final_answer = generate_answers(query, persist_dir)
        print(final_answer)
        print('*'*50)