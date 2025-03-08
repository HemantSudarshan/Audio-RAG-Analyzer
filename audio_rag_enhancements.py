import os
import uuid
import tempfile
import logging
from typing import List, Dict, Optional
from rag_code import Transcribe, EmbedData, QdrantVDB_QB, Retriever, RAG
import streamlit as st
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Singleton embedding model
_embed_model_instance = None

def get_embed_model():
    global _embed_model_instance
    if _embed_model_instance is None:
        logger.info("Initializing embedding model: BAAI/bge-large-en-v1.5")
        _embed_model_instance = EmbedData(embed_model_name="BAAI/bge-large-en-v1.5", batch_size=32)
    return _embed_model_instance

class AudioRAGManager:
    def __init__(self, collection_name: str, api_key: str):
        self.collection_name = collection_name
        self.transcriber = Transcribe(api_key=api_key)  # Basic Transcribe from rag_code
        self.embeddata = get_embed_model()
        self.vector_db = QdrantVDB_QB(collection_name=collection_name, batch_size=512)
        self.retriever = None
        self.rag = None

    def process_audio(self, audio_path: str, language: str = "en") -> List[Dict]:
        transcripts = self.transcriber.transcribe_audio(audio_path, language)
        documents = [f"{t['speaker']}: {t['text']}" for t in transcripts]
        
        self.embeddata.embed(documents)
        self.vector_db.define_client()
        self.vector_db.create_collection()
        self.vector_db.ingest_data(self.embeddata)
        
        self.retriever = Retriever(vector_db=self.vector_db, embeddata=self.embeddata)
        self.rag = RAG(retriever=self.retriever, llm_name="DeepSeek-R1-Distill-Llama-70B")
        
        return transcripts

def run_enhanced_app():
    # Simple, normal UI with no blank spaces (exact CSS from your example)
    st.markdown("""
        <style>
        .app-container { padding: 10px; margin: 0; width: 100%; box-sizing: border-box; }
        .header { padding: 5px 0; margin-bottom: 10px; border-bottom: 1px solid #ccc; }
        .main-layout { display: flex; flex-direction: row; gap: 10px; }
        .control-panel { width: 30%; padding: 10px; border: 1px solid #ccc; }
        .chat-panel { width: 70%; padding: 10px; border: 1px solid #ccc; display: flex; flex-direction: column; }
        .stButton>button { background-color: #0066cc; color: white; border: none; padding: 5px 10px; }
        .stSelectbox>label, .stCheckbox>label { color: #333; font-size: 14px; }
        .chat-messages { flex-grow: 1; padding: 5px; border: 1px solid #ddd; min-height: 300px; overflow-y: auto; }
        .message { margin: 5px 0; padding: 5px; }
        .user-msg { background-color: #e6f2ff; text-align: right; }
        .assistant-msg { background-color: #f5f5f5; text-align: left; }
        .input-area { padding: 5px 0; }
        .transcript-panel { margin-top: 10px; padding: 5px; border-top: 1px solid #ccc; }
        .st-expander { margin: 0; }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='app-container'>", unsafe_allow_html=True)

        # Header
        st.markdown("<div class='header'><h3 style='margin: 0;'>Audio RAG</h3></div>", unsafe_allow_html=True)

        # Session state initialization
        if "id" not in st.session_state:
            st.session_state.id = uuid.uuid4()
            st.session_state.file_cache = {}
            st.session_state.messages = []
            st.session_state.transcripts = []

        session_id = st.session_state.id
        manager = AudioRAGManager(collection_name=f"audio_{session_id}", 
                                api_key=os.getenv("ASSEMBLYAI_API_KEY"))

        # Main layout: two columns
        st.markdown("<div class='main-layout'>", unsafe_allow_html=True)

        # Left panel: Upload and Transcript
        with st.container():
            st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
            st.write("Upload Audio")
            uploaded_file = st.file_uploader("", type=["mp3", "wav", "m4a"], label_visibility="collapsed")
            
            language_options = {"English": "en", "French": "fr", "Spanish": "es", "German": "de"}
            language = st.selectbox("Language", list(language_options.keys()), index=0)
            
            if uploaded_file:
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    with st.spinner("Processing..."):
                        transcripts = manager.process_audio(file_path, language_options[language])
                        st.session_state.transcripts = transcripts
                        st.session_state.file_cache[uploaded_file.name] = manager
                
                if st.session_state.transcripts:
                    st.markdown("<div class='transcript-panel'>", unsafe_allow_html=True)
                    st.write("Transcript")
                    for t in st.session_state.transcripts:
                        st.write(f"{t['speaker']}: {t['text']}")
                    st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Right panel: Chat
        with st.container():
            st.markdown("<div class='chat-panel'>", unsafe_allow_html=True)
            st.write("Chat")
            st.markdown("<div class='chat-messages'>", unsafe_allow_html=True)
            for message in st.session_state.messages:
                msg_class = "user-msg" if message["role"] == "user" else "assistant-msg"
                st.markdown(f"<div class='message {msg_class}'>{message['content']}</div>", 
                           unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='input-area'>", unsafe_allow_html=True)
            if prompt := st.chat_input("Ask about the audio..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.markdown(f"<div class='message user-msg'>{prompt}</div>", unsafe_allow_html=True)
                
                message_placeholder = st.empty()
                full_response = ""
                file_key = uploaded_file.name if uploaded_file else None
                manager = st.session_state.file_cache.get(file_key)
                
                if manager and manager.rag:
                    with st.spinner("Responding..."):
                        streaming_response = manager.rag.query(prompt)
                        for chunk in streaming_response:
                            try:
                                new_text = chunk.raw["choices"][0]["delta"]["content"]
                                full_response += new_text
                                message_placeholder.markdown(f"<div class='message assistant-msg'>{full_response}</div>", 
                                                           unsafe_allow_html=True)
                            except Exception as e:
                                logger.error(f"Streaming error: {e}")
                        message_placeholder.markdown(f"<div class='message assistant-msg'>{full_response}</div>", 
                                                   unsafe_allow_html=True)
                else:
                    message_placeholder.markdown("<div class='message assistant-msg'>Upload an audio file first</div>", 
                                               unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run_enhanced_app()