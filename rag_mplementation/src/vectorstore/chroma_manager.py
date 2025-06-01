"""Chroma DB vector store management."""

import os
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import settings
from loguru import logger

class ChromaManager:
    """Manages Chroma DB vector store operations."""
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        """Initialize ChromaManager."""
        self.db_path = db_path or settings.chroma_db_path
        self.collection_name = collection_name or settings.collection_name
        self.embedding_model = embedding_model or settings.embedding_model
        
        # Create db directory if it doesn't exist
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=self.embedding_model
        )
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        
        logger.info(f"ChromaManager initialized with collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            ids = self.vector_store.add_documents(chunks)
            
            logger.info(f"Added {len(chunks)} document chunks to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add raw texts to the vector store."""
        try:
            # Split texts into chunks
            chunks = self.text_splitter.split_text("\n\n".join(texts))
            
            # Prepare metadata for chunks
            if metadatas:
                chunk_metadatas = []
                for i, chunk in enumerate(chunks):
                    # Assign metadata based on original text index
                    metadata_idx = min(i, len(metadatas) - 1)
                    chunk_metadatas.append(metadatas[metadata_idx])
            else:
                chunk_metadatas = [{"source": "text"} for _ in chunks]
            
            # Add to vector store
            ids = self.vector_store.add_texts(chunks, metadatas=chunk_metadatas)
            
            logger.info(f"Added {len(chunks)} text chunks to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding texts: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform similarity search."""
        k = k or settings.retrieval_k
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """Perform similarity search with relevance scores."""
        k = k or settings.retrieval_k
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Retrieved {len(results)} documents with scores for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with score: {str(e)}")
            raise
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None):
        """Get a retriever instance."""
        search_kwargs = search_kwargs or {"k": settings.retrieval_k}
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def reset_database(self):
        """Reset the entire database."""
        try:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                logger.info(f"Reset database at: {self.db_path}")
            
            # Reinitialize
            self.__init__(self.db_path, self.collection_name, self.embedding_model)
            
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {} 