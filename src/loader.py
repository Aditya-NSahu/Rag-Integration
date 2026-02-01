import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

load_dotenv()
# Fixes the User-Agent warning for WebBaseLoader
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "MyRAGChatbot/1.0")

def load_documents(file_paths):
    """Loads and combines documents from a list of file paths."""
    all_docs = []
    
    for path in file_paths:
        try:
            if path.endswith('.txt'):
                # Added encoding detection to prevent Windows crashes
                loader = TextLoader(path, encoding="utf-8")
            elif path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif path.startswith('http'):
                loader = WebBaseLoader(path)
            else:
                continue
            
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    return all_docs