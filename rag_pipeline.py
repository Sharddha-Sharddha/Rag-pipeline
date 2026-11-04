import os
import uuid
import yt_dlp
import requests
import http.cookiejar
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import hashlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load API keys from key.env
load_dotenv("key.env")
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize the free local Hugging Face embedding model once
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ==========================================
# STEP 1: DATA INGESTION (YouTube Video or Playlist)
# ==========================================
def process_youtube_videos(user_url: str) -> list:
    if not user_url:
        return []

    video_urls = []

    # 1. Check if the URL is a playlist or a single video
    if "playlist" in user_url or "list=" in user_url:
        print("Playlist detected! Extracting video links...")
        ydl_opts = {
            "extract_flat": True,
            "skip_download": True,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(user_url, download=False)
            if "entries" in info:
                for entry in info["entries"]:
                    if entry and entry.get("id"):
                        video_urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
    else:
        print("Single video detected!")
        video_urls = [user_url]

    # 2. Load cookies.txt if available (prevents YouTube IP blocks)
    session = requests.Session()
    if os.path.exists("cookies.txt"):
        try:
            cookie_jar = http.cookiejar.MozillaCookieJar("cookies.txt")
            cookie_jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(cookie_jar)
        except Exception as e:
            print(f"Warning: Could not load cookies.txt: {e}")

    ytt_api = YouTubeTranscriptApi(http_client=session)
    loaded_docs = []

    # 3. Fetch transcript for each video
    for vid_url in video_urls:
        try:
            if "watch?v=" in vid_url:
                video_id = vid_url.split("watch?v=")[1].split("&")[0]
            elif "youtu.be/" in vid_url:
                video_id = vid_url.split("youtu.be/")[1].split("?")[0]
            else:
                continue

            # Fetch English or Hindi captions
            fetched_transcript = ytt_api.fetch(video_id, languages=["en", "hi","None"])
            
            # Extract text safely across both new and older versions of youtube-transcript-api
            full_text = " ".join(
                item.text if hasattr(item, "text") else item["text"]
                for item in fetched_transcript
            )

            loaded_docs.append(
                Document(
                    page_content=full_text,
                    metadata={
                        "source": vid_url,
                        "video_id": video_id,
                        "title": f"Transcript for {video_id}"
                    }
                )
            )
            print(f"Successfully loaded transcript for: {vid_url}")

        except Exception as e:
            print(f"Skipping video {vid_url} due to error: {e}")

    return loaded_docs


# ==========================================
# STEP 2: CHUNKING & VECTOR STORAGE (FAISS)
# ==========================================
def get_persist_directory(user_url: str) -> str:
    base_dir = "./faiss_db"
    cleaned_url = user_url.encode("utf-8")
    collection_id = hashlib.sha256(cleaned_url).hexdigest()[:16]  # Use first 16 chars of SHA256 hash for uniqueness
    persist_dir = os.path.join(base_dir, collection_id)
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


def generate_embedding(docs: list, persist_directory: str = "./faiss_db"):
    # Split long transcripts into 1000-character chunks with 200-character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    # Convert chunks into vectors and save locally in FAISS
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    os.makedirs(persist_directory, exist_ok=True)
    vector_store.save_local(persist_directory)
    print(f"Saved {len(chunks)} chunks to FAISS at: {persist_directory}")
    return vector_store


# ==========================================
# STEP 3: RETRIEVAL
# ==========================================
def retriever_func(query: str, persist_directory: str = "./faiss_db") -> list:
    vector_store = FAISS.load_local(
        folder_path=persist_directory,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    docs = retriever.invoke(query)
    return docs


# ==========================================
# STEP 4: AUGMENTATION & GENERATION (Groq LLM)
# ==========================================
def format_docs(docs: list) -> str:
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown Video')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )


def generate_answers(user_query: str, persist_directory: str = "./faiss_db") -> str:
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing! Please add it to your key.env file.")

    # 1. Create the Prompt Template
    prompt = PromptTemplate(
        template="""You are an expert AI assistant. Your task is to provide a clear, accurate, and concise answer to the user's question.

        Instructions:
        - Primary Answer: Base your main answer on the provided Context. Always share the content source URL provided in the context.
        - Handling Incomplete Data: If the context does not contain enough information to fully answer the question, you may provide additional relevant details from your general knowledge.
        - Outsourcing Disclaimer: If you include information that is NOT found in the context, you MUST explicitly state: "Note: This part is outsourced from external knowledge to give you full content understanding."
        - Format: Use bullet points for readability and keep the answer concise.

        Context:
        {context}

        Question: {query}
        """,
        input_variables=["context", "query"]
    )

    # 2. Initialize Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
        max_tokens=1024
    )

    # 3. Build the LCEL Chain
    setup_and_retrieval = {
        "context": RunnableLambda(lambda query: retriever_func(query, persist_directory=persist_directory)) | RunnableLambda(format_docs),
        "query": RunnablePassthrough()
    }

    rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()

    # 4. Invoke the LLM and return the answer
    response = rag_chain.invoke(user_query)
    return response


# ==========================================
# STEP 5: TERMINAL EXECUTION (Only runs if executed directly)
# ==========================================
if __name__ == "__main__":
    user_url = input("Enter the YouTube Video or Playlist URL: ").strip()
    print("\nLoading transcripts... This might take a moment.")

    docs = process_youtube_videos(user_url)
    if not docs:
        print("No transcripts could be loaded for this URL. Exiting.")
        raise SystemExit

    print("\nCreating the Vector Database...")
    persist_dir = get_persist_directory(user_url)
    generate_embedding(docs, persist_dir)

    print("\n" + "=" * 50)
    print("Database Ready! Ask questions about the video(s).")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50 + "\n")

    while True:
        query = input("Your Question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("Ending the session. Goodbye!")
            break
        if not query:
            continue

        final_answer = generate_answers(query, persist_dir)
        print("\n$$$ FINAL ANSWER $$$")
        print(final_answer)
        print("*" * 50)