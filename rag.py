"""RAG engine: retrieves top-k chunks from the Chroma vector store for a given
question and calls the Claude API to generate an answer grounded only in those
chunks, with citations back to the source documents.
"""
