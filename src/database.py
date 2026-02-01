import os
from langchain_community.vectorstores import FAISS

def create_vector_db(chunks, embedding_model, db_path="vectorstore/db_faiss"):
    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.save_local(db_path)
    return vector_db

def load_vector_db(db_path, embedding_model):
    return FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)