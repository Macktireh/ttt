Voici mon API créer avec le framwork [litestar](https://github.com/litestar-org/litestar). Ajoute moi les tests intégration des endpoints. Pour l'endpoint de chat completion, j'ai ajouté le support du streaming si le paramètre `stream=True` est présent dans la requête donc il faudra tester avec et sans streaming. Pour tester tu vas utiliser le modèle `dummy-model:1.0`, j'ai ajouté un DummyService qui retourne des textes aléatoires et qui implement bien `AIServiceInterface` comme OllamaService. Voici le payload de la requête pour tester :

```json
{
    "model": "dummy-model:1.0",
    "messages": [
        {"role": "user", "content": "Hello, what's your name?"},
        {
            "role": "assistant",
            "content": "Hello ! I'm an AI assistant. How can I help you today?",
        },
        {"role": "user", "content": "Perfect! Can you help me with some Python code?"},
    ],
    "stream": true,
}
```

Voici le lien de la documentation de testing avec litestar: [https://docs.litestar.dev/2/usage/testing.html](https://docs.litestar.dev/2/usage/testing.html). Un exemple de test en mode async avec pytest récupéré sur le site de litestar:
```py
from collections.abc import AsyncIterator

import pytest

from litestar import Litestar, MediaType, get
from litestar.status_codes import HTTP_200_OK
from litestar.testing import AsyncTestClient


@get(path="/health-check", media_type=MediaType.TEXT, sync_to_thread=False)
def health_check() -> str:
    return "healthy"


app = Litestar(route_handlers=[health_check], debug=True)


@pytest.fixture(scope="function")
async def test_client() -> AsyncIterator[AsyncTestClient[Litestar]]:
    async with AsyncTestClient(app=app) as client:
        yield client


async def test_health_check_with_fixture(test_client: AsyncTestClient[Litestar]) -> None:
    response = await test_client.get("/health-check")
    assert response.status_code == HTTP_200_OK
    assert response.text == "healthy"
```

Voici mon code de l'api :

```py
# main.py
from litestar import Litestar, get
from litestar.config.cors import CORSConfig

from config.settings import openapi_config
from controllers import chat_router


@get("/health", summary="Health Check", description="Vérifie la santé de l'API", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "Ollaix API"}


cors_config = CORSConfig(allow_origins=["*"])


app = Litestar(
    route_handlers=[health_check, chat_router],
    openapi_config=openapi_config,
    debug=True,
    cors_config=cors_config,
)


# config/settings.py
import os
from pathlib import Path

from dotenv import load_dotenv
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Contact, Tag

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Configuration des services AI
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuration OpenAPI
openapi_config = OpenAPIConfig(
    title="Ollaix API",
    version="1.0.0",
    description="API unifiée pour les modèles de chat completion Ollama et Gemini",
    contact=Contact(name="Ollaix Team", email="contact@ollaix.com"),
    path="/",
    root_schema_site="swagger",
    tags=[
        # Tag(name="Models", description="Gestion des modèles de langage disponibles"),
        Tag(name="Chat", description="Endpoints de chat completion avec support du streaming"),
        Tag(name="Health", description="Endpoints de santé et de monitoring"),
    ],
)


# schemas/chat_schemas.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from litestar.dto import DataclassDTO


@dataclass
class ChatMessage:
    """Représente un message dans une conversation."""

    role: Literal["assistant", "user", "system"]
    content: str


@dataclass
class ChatCompletionRequest:
    """Requête pour une completion de chat."""

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


@dataclass
class ChatCompletionResponse:
    """Réponse d'une completion de chat."""

    id: UUID = field(default_factory=uuid4)
    object: Literal["chat.completion"] = "chat.completion"
    created: datetime = field(default_factory=datetime.now)
    model: str = ""
    choices: list[dict] = field(default_factory=list)
    usage: dict | None = None


@dataclass
class ChatCompletionStreamChunk:
    """Chunk de données pour le streaming."""

    id: UUID = field(default_factory=uuid4)
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: datetime = field(default_factory=datetime.now)
    model: str = ""
    choices: list[dict] = field(default_factory=list)


@dataclass
class ModelInfo:
    """Information sur un modèle de langage."""

    id: str
    name: str
    description: str
    provider: Literal["dummy", "ollama", "google"]
    context_length: int | None = None


@dataclass
class ModelsResponse:
    """Réponse contenant la liste des modèles."""

    object: Literal["list"] = "list"
    data: list[ModelInfo] = field(default_factory=list)


@dataclass
class ErrorResponse:
    """Réponse d'erreur standardisée."""

    error: dict[str, str]


# DTOs pour la validation automatique
ChatCompletionRequestDTO = DataclassDTO[ChatCompletionRequest]
ChatCompletionResponseDTO = DataclassDTO[ChatCompletionResponse]
ModelsResponseDTO = DataclassDTO[ModelsResponse]


# controllers/__init__.py
from litestar import Router
from litestar.di import Provide

from controllers.chat_controller import ChatController
from services.dummy_service import DummyService
from services.gemini_service import GeminiService
from services.ollama_service import OllamaService

chat_router = Router(
    path="/v1",
    dependencies={
        "gemini_service": Provide(GeminiService, sync_to_thread=False),
        "ollama_service": Provide(OllamaService, sync_to_thread=False),
        "dummy_service": Provide(DummyService, sync_to_thread=False),
    },
    route_handlers=[ChatController],
)

# controllers/chat_controller.py
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
    ) -> Stream | ChatCompletionResponse:
        """
        Génère une réponse de chat completion.

        Supporte le streaming si stream=True dans la requête.
        Route automatiquement vers le service approprié basé sur le modèle.
        """
        service = self._get_service_for_model(data.model, ollama_service, gemini_service)

        if data.stream:
            return Stream(service.chat_completion_stream(data)) # type: ignore
        else:
            return await service.chat_completion(data)

    def _get_service_for_model(
        self,
        model: str,
        ollama_service: AIServiceInterface,
        gemini_service: AIServiceInterface,
    ) -> AIServiceInterface:
        """Détermine quel service utiliser basé sur le modèle demandé."""
        if model in ollama_service.available_models:
            return ollama_service
        elif model in gemini_service.available_models:
            return gemini_service
        else:
            from litestar.exceptions import ValidationException

            raise ValidationException(f"Modèle '{model}' non disponible")



# services/ai_service_interface.py
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


# services/ollama_service.py
import json
from collections.abc import AsyncGenerator
from typing import Any

from ollama import AsyncClient

from config.settings import OLLAMA_BASE_URL
from schemas.chat_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    ChatMessage,
    ModelInfo,
)
from services.ai_service_interface import AIServiceInterface


class OllamaService(AIServiceInterface):
    """Service pour interagir avec Ollama."""

    available_models = ["qwen3:4b", "deepseek-r1:7b"]
    provider_name = "ollama"

    def __init__(self):
        self.client = AsyncClient(host=OLLAMA_BASE_URL)

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Génère une réponse de chat completion via Ollama."""
        if request.model not in self.available_models:
            raise ValueError(f"Modèle '{request.model}' non disponible pour Ollama")

        messages = self._convert_messages(request.messages)

        response = await self.client.chat(
            model=request.model,
            messages=messages,
            stream=False,
            options={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            }
            if any([request.temperature, request.top_p, request.max_tokens])
            else None,
        )

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response["message"]["content"],
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0)
                + response.get("eval_count", 0),
            },
        )

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, Any]:
        """Génère un stream de réponses de chat completion via Ollama."""
        if request.model not in self.available_models:
            raise ValueError(f"Modèle '{request.model}' non disponible pour Ollama")

        messages = self._convert_messages(request.messages)

        async for chunk in await self.client.chat(
            model=request.model,
            messages=messages,
            stream=True,
            options={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            }
            if any([request.temperature, request.top_p, request.max_tokens])
            else None,
        ):
            if chunk.get("message", {}).get("content"):
                stream_chunk = ChatCompletionStreamChunk(
                    model=request.model,
                    choices=[
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": chunk["message"]["content"],
                            },
                            "finish_reason": None,
                        }
                    ],
                )
                yield f"data: {json.dumps(stream_chunk.__dict__, default=str)}\n\n"

            if chunk.get("done", False):
                final_chunk = ChatCompletionStreamChunk(
                    model=request.model,
                    choices=[
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                )
                yield f"data: {json.dumps(final_chunk.__dict__, default=str)}\n\n"
                yield "data: [DONE]\n\n"

    def get_model_info(self) -> list[ModelInfo]:
        """Retourne les informations des modèles Ollama."""
        return [
            ModelInfo(
                id="qwen3:4b",
                name="Qwen 3 4B",
                description="Modèle de langage Qwen 3 optimisé pour la performance",
                provider="ollama",
                context_length=32768,
            ),
            ModelInfo(
                id="deepseek-r1:7b",
                name="DeepSeek R1 7B",
                description="Modèle de raisonnement DeepSeek R1 haute performance",
                provider="ollama",
                context_length=32768,
            ),
        ]

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convertit les messages au format Ollama."""
        return [{"role": message.role, "content": message.content} for message in messages]
```

NB: le code doit être en anglais et les explications hors code en français.