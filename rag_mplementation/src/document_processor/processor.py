"""Document processing utilities for various file formats."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)

from loguru import logger

class DocumentProcessor:
    """Handles document loading and processing from various sources."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.markdown': UnstructuredMarkdownLoader,
        '.docx': Docx2txtLoader,
        '.doc': Docx2txtLoader
    }
    
    def __init__(self):
        """Initialize DocumentProcessor."""
        logger.info("DocumentProcessor initialized")
    
    def load_file(self, file_path: str, **kwargs) -> List[Document]:
        """Load a single file and return documents."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")
        
        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        
        try:
            loader = loader_class(str(file_path), **kwargs)
            documents = loader.load()
            
            # Add file metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_extension': extension,
                    'file_size': file_path.stat().st_size
                })
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def load_directory(self, 
                      directory_path: str,
                      glob_pattern: str = "**/*",
                      recursive: bool = True,
                      exclude_extensions: Optional[List[str]] = None) -> List[Document]:
        """Load all supported files from a directory."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        exclude_extensions = exclude_extensions or []
        all_documents = []
        
        # Get all files matching the pattern
        files = list(directory_path.glob(glob_pattern))
        
        for file_path in files:
            if file_path.is_file():
                extension = file_path.suffix.lower()
                
                # Skip excluded extensions
                if extension in exclude_extensions:
                    continue
                
                # Skip unsupported extensions
                if extension not in self.SUPPORTED_EXTENSIONS:
                    logger.warning(f"Skipping unsupported file: {file_path}")
                    continue
                
                try:
                    documents = self.load_file(file_path)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Loaded {len(all_documents)} total documents from {directory_path}")
        return all_documents
    
    def load_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Create documents from raw text."""
        metadata = metadata or {}
        
        document = Document(
            page_content=text,
            metadata=metadata
        )
        
        return [document]
    
    def load_urls(self, urls: List[str]) -> List[Document]:
        """Load documents from URLs (requires additional dependencies)."""
        try:
            from langchain_community.document_loaders import WebBaseLoader
        except ImportError:
            raise ImportError("Web loading requires additional dependencies. Install with: pip install beautifulsoup4 requests")
        
        all_documents = []
        
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                documents = loader.load()
                
                # Add URL metadata
                for doc in documents:
                    doc.metadata.update({
                        'source': url,
                        'source_type': 'web'
                    })
                
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {url}")
                
            except Exception as e:
                logger.error(f"Error loading URL {url}: {str(e)}")
                continue
        
        return all_documents
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls.SUPPORTED_EXTENSIONS.keys())
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if a file can be processed."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False
        
        if not file_path.is_file():
            return False
        
        extension = file_path.suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS 