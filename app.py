import os
import streamlit as st
from rag_pipeline import (
    process_youtube_videos,
    get_persist_directory,
    generate_embedding,
    generate_answers
)

# 1. Page Configuration
st.set_page_config(page_title="YouTube RAG Assistant", page_icon="🎥", layout="wide")
st.title("🎥 YouTube Video & Playlist RAG Assistant")
st.write("Paste a YouTube video or playlist link in the sidebar and ask questions about its content!")

# 2. Initialize Session State (Saves your chat history and active database path)
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Sidebar: Video / Playlist Ingestion
with st.sidebar:
    st.header("1. Load YouTube Content")
    user_url = st.text_input("Enter YouTube Video or Playlist URL:")

    if st.button("Process Video"):
        if not user_url.strip():
            st.warning("Please enter a valid YouTube URL first.")
        else:
            with st.spinner("Checking database and processing transcripts..."):
                # Get the deterministic hashlib folder path for this URL
                persist_dir = get_persist_directory(user_url.strip())
                faiss_index_file = os.path.join(persist_dir, "index.faiss")

                # Smart Cache Check: Skip downloading if this URL was already embedded!
                if os.path.exists(faiss_index_file):
                    st.session_state.persist_dir = persist_dir
                    st.success("Video already exists in FAISS! Loaded instantly from cache.")
                else:
                    docs = process_youtube_videos(user_url.strip())
                    if not docs:
                        st.error("No English or Hindi transcripts found for this URL.")
                    else:
                        generate_embedding(docs, persist_dir)
                        st.session_state.persist_dir = persist_dir
                        st.success(f"Successfully processed {len(docs)} video(s) and saved to FAISS!")

    if st.session_state.persist_dir:
        st.caption(f"Active DB Folder: `{st.session_state.persist_dir}`")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 4. Main Area: Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Chat Input Box: Ask Questions
user_query = st.chat_input("Ask a question about the video...")

if user_query:
    if not st.session_state.persist_dir:
        st.warning("Please process a YouTube URL in the left sidebar first!")
    else:
        # Show the user's question in the UI
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate and display the answer from Groq + FAISS
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answers(user_query, st.session_state.persist_dir)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating answer: {e}")