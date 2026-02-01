import os
import shutil
import gradio as gr
from dotenv import load_dotenv

# 1. Setup Environment
os.environ["USER_AGENT"] = "Gemma3RAGBot/2.0"
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from src.loader import load_documents
from src.transformer import chunk_data
from src.embedding import get_embedding_model
from src.database import create_vector_db

# --- BACKEND LOGIC ---
def process_documents(files, progress=gr.Progress()):
    if not files:
        return "⚠️ Please upload files first.", None
    try:
        os.makedirs("data", exist_ok=True)
        file_paths = []
        total_files = len(files)
        
        # Smart progress bar with ETA
        for f in progress.tqdm(files, desc="📂 Loading Files", unit="file"):
            dest = os.path.join("data", os.path.basename(f.name))
            shutil.copy(f.name, dest)
            file_paths.append(dest)
        
        progress(0.4, desc="📖 Reading content...")
        docs = load_documents(file_paths)
        
        progress(0.6, desc="✂️ Chunking text...")
        chunks = chunk_data(docs)
        
        progress(0.8, desc="🧠 Generating AI Embeddings...")
        embed_model = get_embedding_model()
        
        progress(0.9, desc="💾 Building Database...")
        vector_db = create_vector_db(chunks, embed_model)
        
        return f"✅ Indexed {total_files} files successfully!", vector_db
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def chat_response(message, history, vector_db):
    if vector_db is None:
        yield "Please upload documents on the left first! 📂"
        return

    # Retrieval
    search_results = vector_db.similarity_search(message, k=4)
    context = "\n\n".join([d.page_content for d in search_results])
    
    # LLM (Gemma 3)
    llm = ChatOllama(model=os.getenv("LLM_MODEL", "gemma3:1b"), temperature=0.3)
    
    messages = [
        SystemMessage(content=f"Answer using only this context:\n{context}"),
        HumanMessage(content=message)
    ]
    
    full_text = ""
    for chunk in llm.stream(messages):
        full_text += chunk.content
        yield full_text

# --- GRADIO UI (Simplified for Compatibility) ---
with gr.Blocks(title="Gemma 3 RAG") as demo:
    db_state = gr.State(None)
    
    gr.Markdown("# 🤖 Gemma 3 Local RAG")
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📂 1. Ingest Data")
            file_uploader = gr.File(label="Upload PDF/TXT", file_count="multiple")
            build_btn = gr.Button("Build Knowledge Base", variant="primary")
            status_msg = gr.Markdown("*Status: Idle*")
            
        with gr.Column(scale=3):
            gr.Markdown("### 💬 2. Knowledge Chat")
            # Simplified ChatInterface to avoid Version Errors
            gr.ChatInterface(
                fn=chat_response,
                additional_inputs=[db_state],
                examples=[
                    ["Summarize the docs", None],
                    ["Translate takeaways to Spanish", None]
                ]
            )

    build_btn.click(
        fn=process_documents, 
        inputs=[file_uploader], 
        outputs=[status_msg, db_state]
    )

# --- LAUNCH ---
if __name__ == "__main__":
    demo.launch(share=True)