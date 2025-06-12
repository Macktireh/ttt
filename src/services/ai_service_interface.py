from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from schemas.chat_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelsResponse,
)


class AIServiceInterface(ABC):
    """Interface abstraite pour les services d'IA."""

    available_models: list[str] = []
    provider_name: str = ""

    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Génère une réponse de chat completion."""
        pass

    @abstractmethod
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, Any]:
        """Génère un stream de réponses de chat completion."""
        pass

    @abstractmethod
    def get_model_info(self) -> list[ModelInfo]:
        """Retourne les informations des modèles supportés."""
        pass

    @staticmethod
    def get_all_models() -> ModelsResponse:
        """Retourne tous les modèles disponibles de tous les services."""
        from services.dummy_service import DummyService
        from services.gemini_service import GeminiService
        from services.ollama_service import OllamaService

        ollama_service = OllamaService()
        gemini_service = GeminiService()
        dummy_service = DummyService()

        all_models = []
        all_models.extend(ollama_service.get_model_info())
        all_models.extend(gemini_service.get_model_info())
        all_models.extend(dummy_service.get_model_info())

        return ModelsResponse(data=all_models)
