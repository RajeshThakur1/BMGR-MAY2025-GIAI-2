from pydantic_settings import BaseSettings
from typing import Optional, Literal
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "ollama", "huggingface"] = "openai"
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    
    # HuggingFace settings
    huggingface_api_key: Optional[str] = None
    huggingface_model: str = "microsoft/DialoGPT-medium"
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Chroma DB settings
    chroma_db_path: str = "./chroma_db"
    collection_name: str = "rag_documents"
    
    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # RAG settings
    retrieval_k: int = 4
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings() 