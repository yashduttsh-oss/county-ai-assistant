"""Ingest pipeline: loads documents from docs/, chunks them, embeds them
using sentence-transformers, and writes a local Chroma vector store to ./chroma_db/.
"""
