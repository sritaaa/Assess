import streamlit as st
from yt_dlp import YoutubeDL
import whisper
from groq import Groq
from dotenv import load_dotenv
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="YouTube Q&A", layout="centered")

st.title(" Q&A Assistant")
st.subheader("Ask anything about the transcribed video")

@st.cache_resource
def process_video_and_build_db(url):
    with st.status("Initializing and processing video components...", expanded=True) as status:
        st.write("Downloading video audio...")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "audio.%(ext)s",
            "quiet": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            ydl.download([url])

        video_metadata = {
            "video_id": info["id"],
            "title": info["title"],
            "channel": info["uploader"],
            "views": info.get("view_count"),
            "likes": info.get("like_count"),
            "comments": info.get("comment_count"),
        }
        likes = info.get("like_count", 0)
        comments = info.get("comment_count", 0)
        views = info.get("view_count", 1)  # avoid division by zero
        engagement = (likes + comments) / views
        st.write("Engagement Rate:", engagement)
    
        st.write("Transcribing audio with Whisper (this might take a moment)...")
        model = whisper.load_model("base")
        result = model.transcribe("audio.webm") 

        st.write("Chunking text...")
        raw_text = result["text"]
        docs = [Document(page_content=raw_text, metadata=video_metadata)]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120
        )
        chunks = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        status.update(label="System Ready! Video parsed successfully.", state="complete", expanded=False)
        return retriever
    
url = st.text_input(
    "Enter your URL",
    placeholder="https://www.youtube.com/watch?v=0CmtDk-joT4"
)

if url:
    retriever = process_video_and_build_db(url)
    
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Box 1: The User Input
user_query = st.text_input("Enter your question here:", placeholder="e.g., Can you summarize this video?")

if user_query:
    with st.spinner("🤖 Groq is thinking..."):
        try:
            
            retrieved_docs = retriever.invoke(user_query)
            context_text = format_docs(retrieved_docs)
            
            full_prompt = (
                f"Answer the user's question using ONLY the provided context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {user_query}"
            )
          
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            st.markdown("### 🤖 Answer:")
            st.info(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")