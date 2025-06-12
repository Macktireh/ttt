from typing import Annotated

from litestar import get, post
from litestar.controller import Controller
from litestar.params import Body
from litestar.response import Stream

from schemas.chat_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
)
from services.ai_service_interface import AIServiceInterface


class ChatController(Controller):
    path = "/"
    tags = ["Chat"]

    @get(
        "/models",
        summary="Lister les modèles disponibles",
        description="Retourne la liste de tous les modèles de langage disponibles",
    )
    async def get_available_models(self) -> ModelsResponse:
        """Récupère la liste des modèles disponibles depuis tous les services."""
        return AIServiceInterface.get_all_models()

    @post(
        "/chat/completions",
        summary="Chat completion",
        description="Génère une réponse de chat completion avec support du streaming",
    )
    async def chat_completion(
        self,
        data: Annotated[
            ChatCompletionRequest,
            Body(
                title="Requête de chat completion",
                description="Données de la requête pour générer une réponse de chat",
            ),
        ],
        ollama_service: AIServiceInterface,
        gemini_service: AIServiceInterface,
        dummy_service: AIServiceInterface,
    ) -> Stream | ChatCompletionResponse:
        """
        Génère une réponse de chat completion.

        Supporte le streaming si stream=True dans la requête.
        Route automatiquement vers le service approprié basé sur le modèle.
        """
        service = self._get_service_for_model(
            data.model, ollama_service, gemini_service, dummy_service
        )

        if data.stream:
            return Stream(service.chat_completion_stream(data))  # type: ignore
        else:
            return await service.chat_completion(data)

    def _get_service_for_model(
        self,
        model: str,
        ollama_service: AIServiceInterface,
        gemini_service: AIServiceInterface,
        dummy_service: AIServiceInterface,
    ) -> AIServiceInterface:
        """Détermine quel service utiliser basé sur le modèle demandé."""
        if model in ollama_service.available_models:
            return ollama_service
        elif model in gemini_service.available_models:
            return gemini_service
        else:
            return dummy_service
            from litestar.exceptions import ValidationException

            raise ValidationException(f"Modèle '{model}' non disponible")
