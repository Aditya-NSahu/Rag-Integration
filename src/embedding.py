import os
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embedding_model():
    """Returns the specialized embedding model from .env."""
    model_name = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    return OllamaEmbeddings(model=model_name)