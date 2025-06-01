"""Generic LLM factory for supporting multiple LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFacePipeline
from config.settings import settings
from loguru import logger

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def create_llm(self, **kwargs) -> LLM:
        """Create and return an LLM instance."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def create_llm(self, **kwargs) -> ChatOpenAI:
        """Create OpenAI LLM instance."""
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=kwargs.get('model', settings.openai_model),
            temperature=kwargs.get('temperature', settings.temperature),
            max_tokens=kwargs.get('max_tokens', settings.max_tokens)
        )
    
    def is_available(self) -> bool:
        """Check if OpenAI is properly configured."""
        return settings.openai_api_key is not None

class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider."""
    
    def create_llm(self, **kwargs) -> ChatAnthropic:
        """Create Anthropic LLM instance."""
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=kwargs.get('model', settings.anthropic_model),
            temperature=kwargs.get('temperature', settings.temperature),
            max_tokens=kwargs.get('max_tokens', settings.max_tokens)
        )
    
    def is_available(self) -> bool:
        """Check if Anthropic is properly configured."""
        return settings.anthropic_api_key is not None

class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models."""
    
    def create_llm(self, **kwargs) -> Ollama:
        """Create Ollama LLM instance."""
        return Ollama(
            base_url=kwargs.get('base_url', settings.ollama_base_url),
            model=kwargs.get('model', settings.ollama_model),
            temperature=kwargs.get('temperature', settings.temperature)
        )
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import requests
            response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

class HuggingFaceProvider(LLMProvider):
    """HuggingFace LLM provider."""
    
    def create_llm(self, **kwargs) -> HuggingFacePipeline:
        """Create HuggingFace LLM instance."""
        from transformers import pipeline
        
        pipe = pipeline(
            "text-generation",
            model=kwargs.get('model', settings.huggingface_model),
            token=settings.huggingface_api_key,
            max_new_tokens=kwargs.get('max_tokens', settings.max_tokens),
            temperature=kwargs.get('temperature', settings.temperature)
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def is_available(self) -> bool:
        """Check if HuggingFace is properly configured."""
        try:
            import transformers
            return True
        except ImportError:
            return False

class LLMFactory:
    """Factory class for creating LLM instances."""
    
    _providers = {
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
        "ollama": OllamaProvider(),
        "huggingface": HuggingFaceProvider()
    }
    
    @classmethod
    def create_llm(cls, provider: Optional[str] = None, **kwargs) -> LLM:
        """Create an LLM instance based on the provider."""
        provider = provider or settings.llm_provider
        
        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        provider_instance = cls._providers[provider]
        
        if not provider_instance.is_available():
            logger.warning(f"Provider {provider} is not properly configured. Falling back to available providers.")
            return cls._fallback_llm(**kwargs)
        
        logger.info(f"Creating LLM instance with provider: {provider}")
        return provider_instance.create_llm(**kwargs)
    
    @classmethod
    def _fallback_llm(cls, **kwargs) -> LLM:
        """Try to create an LLM with any available provider."""
        for name, provider in cls._providers.items():
            if provider.is_available():
                logger.info(f"Using fallback provider: {name}")
                return provider.create_llm(**kwargs)
        
        raise RuntimeError("No LLM providers are available. Please configure at least one provider.")
    
    @classmethod
    def list_available_providers(cls) -> List[str]:
        """List all available (configured) providers."""
        return [name for name, provider in cls._providers.items() if provider.is_available()] 