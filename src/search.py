def similarity_search(query, vector_db, k=3):
    results = vector_db.similarity_search(query, k=k)
    return results