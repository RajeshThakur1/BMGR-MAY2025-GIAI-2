"""RAG pipeline implementation."""

from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from src.llm.llm_factory import LLMFactory
from src.vectorstore.chroma_manager import ChromaManager
from src.document_processor.processor import DocumentProcessor
from config.settings import settings
from loguru import logger

class RAGPipeline:
    """Complete RAG pipeline for document ingestion and querying."""
    
    def __init__(self,
                 llm_provider: Optional[str] = None,
                 chroma_manager: Optional[ChromaManager] = None,
                 document_processor: Optional[DocumentProcessor] = None):
        """Initialize RAG pipeline."""
        
        # Initialize components
        self.document_processor = document_processor or DocumentProcessor()
        self.chroma_manager = chroma_manager or ChromaManager()
        
        # Initialize LLM
        self.llm = LLMFactory.create_llm(provider=llm_provider)
        
        # Default prompt template
        self.default_prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, just say that you don't know, 
        don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        Answer:
        """
        
        # Initialize retrieval chain
        self._setup_retrieval_chain()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _setup_retrieval_chain(self):
        """Setup the retrieval QA chain."""
        prompt = PromptTemplate(
            template=self.default_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.chroma_manager.get_retriever()
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest documents from file paths."""
        all_documents = []
        ingestion_stats = {
            "total_files": len(file_paths),
            "successful_files": 0,
            "failed_files": 0,
            "total_documents": 0,
            "errors": []
        }
        
        for file_path in file_paths:
            try:
                documents = self.document_processor.load_file(file_path)
                all_documents.extend(documents)
                ingestion_stats["successful_files"] += 1
                logger.info(f"Successfully loaded: {file_path}")
                
            except Exception as e:
                ingestion_stats["failed_files"] += 1
                ingestion_stats["errors"].append(f"{file_path}: {str(e)}")
                logger.error(f"Failed to load {file_path}: {str(e)}")
        
        if all_documents:
            # Add documents to vector store
            self.chroma_manager.add_documents(all_documents)
            ingestion_stats["total_documents"] = len(all_documents)
            
            # Update retrieval chain
            self._setup_retrieval_chain()
        
        logger.info(f"Ingestion completed: {ingestion_stats}")
        return ingestion_stats
    
    def ingest_directory(self, directory_path: str, **kwargs) -> Dict[str, Any]:
        """Ingest all documents from a directory."""
        try:
            documents = self.document_processor.load_directory(directory_path, **kwargs)
            
            if documents:
                self.chroma_manager.add_documents(documents)
                self._setup_retrieval_chain()
            
            stats = {
                "directory": directory_path,
                "total_documents": len(documents),
                "status": "success"
            }
            
            logger.info(f"Directory ingestion completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error ingesting directory {directory_path}: {str(e)}")
            raise
    
    def ingest_text(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Ingest raw text data."""
        try:
            self.chroma_manager.add_texts(texts, metadatas)
            self._setup_retrieval_chain()
            
            stats = {
                "total_texts": len(texts),
                "status": "success"
            }
            
            logger.info(f"Text ingestion completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error ingesting texts: {str(e)}")
            raise
    
    def query(self, 
              question: str, 
              custom_prompt: Optional[str] = None,
              include_sources: bool = True) -> Dict[str, Any]:
        """Query the RAG system."""
        try:
            # Use custom prompt if provided
            if custom_prompt:
                prompt = PromptTemplate(
                    template=custom_prompt,
                    input_variables=["context", "question"]
                )
                
                # Create temporary chain with custom prompt
                retriever = self.chroma_manager.get_retriever()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
            else:
                qa_chain = self.qa_chain
            
            # Get response
            response = qa_chain({"query": question})
            
            result = {
                "question": question,
                "answer": response["result"],
                "source_documents": []
            }
            
            # Include source documents if requested
            if include_sources and "source_documents" in response:
                for doc in response["source_documents"]:
                    result["source_documents"].append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
            
            logger.info(f"Query processed successfully: {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform similarity search without LLM generation."""
        try:
            documents = self.chroma_manager.similarity_search(query, k=k)
            
            results = []
            for doc in documents:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.info(f"Similarity search completed for: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        try:
            collection_stats = self.chroma_manager.get_collection_stats()
            available_providers = LLMFactory.list_available_providers()
            
            return {
                "vector_store": collection_stats,
                "current_llm_provider": settings.llm_provider,
                "available_llm_providers": available_providers,
                "settings": {
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "retrieval_k": settings.retrieval_k,
                    "embedding_model": settings.embedding_model
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def reset(self):
        """Reset the RAG pipeline (clear all data)."""
        try:
            self.chroma_manager.reset_database()
            self._setup_retrieval_chain()
            logger.info("RAG pipeline reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting pipeline: {str(e)}")
            raise 