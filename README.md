ollaix/
    ├── compose.yml
    ├── pyproject.toml
    ├── src/
    │   ├── __init__.py
    │   ├── main.py
    │   ├── config/
    │   │   ├── __init__.py
    │   │   └── settings.py
    │   ├── controllers/
    │   │   ├── __init__.py
    │   │   └── chat_controller.py
    │   ├── schemas/
    │   │   ├── __init__.py
    │   │   └── chat_schemas.py
    │   └── services/
    │       ├── __init__.py
    │       ├── ai_service_interface.py
    │       ├── dummy_service.py
    │       ├── gemini_service.py
    │       └── ollama_service.py
    └── tests/
        ├── __init__.py
        ├── conftest.py
        ├── test_chat_completion.py
        ├── test_chat_validation.py
        ├── test_health.py
        └── test_models.py


================================================
FILE: pyproject.toml
================================================
[project]
name = "aix"
version = "0.1.0"
description = "AI project "
authors = [
    {name = "Macktireh", email = "abdimack97@gmail.com"},
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.13"
dependencies = [
    "google-genai>=1.18.0",
    "litestar[standard]>=2.16.0",
    "ollama>=0.5.1",
    "python-dotenv>=1.1.0",
]

[dependency-groups]
lint = [
]
test = [
    "pytest>=8.4.0",
    "pytest-cov>=6.2.0",
    "pytest-asyncio>=1.0.0",
]

[tool.pdm.scripts]
dev = {cmd = "litestar --app=src.main:app run --reload --host localhost", env = {PYTHONPATH = "src"}}
test = {cmd = "pytest", env = {PYTHONPATH = "src"}}


================================================
FILE: src/main.py
================================================
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

================================================
FILE: tests/conftest.py
================================================
from collections.abc import AsyncIterator
from typing import Any

import pytest
from litestar.testing import AsyncTestClient

from src.main import app


@pytest.fixture(scope="function")
async def test_client() -> AsyncIterator[AsyncTestClient]:
    """Fixture to create an asynchronous test client."""
    async with AsyncTestClient(app=app) as client:
        yield client

j'obtiens cet erreur :
```
PS E:\develop\projects\side\ollaix> pdm run test
====================================================== test session starts ======================================================
platform win32 -- Python 3.13.2, pytest-8.4.0, pluggy-1.6.0
rootdir: E:\develop\projects\side\ollaix
configfile: pyproject.toml
plugins: anyio-4.9.0, Faker-37.3.0, asyncio-1.0.0, cov-6.2.0
asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 1 item

tests\test_health.py F                                                                                                     [100%]

=========================================================== FAILURES ============================================================
_____________________________________________ TestHealthEndpoint.test_health_check ______________________________________________
async def functions are not natively supported.
You need to install a suitable plugin for your async framework, for example:
  - anyio
  - pytest-asyncio
  - pytest-tornasync
  - pytest-trio
  - pytest-twisted
======================================================= warnings summary ========================================================
.venv\Lib\site-packages\litestar\openapi\config.py:197
  E:\develop\projects\side\ollaix\.venv\Lib\site-packages\litestar\openapi\config.py:197: DeprecationWarning: Use of deprecated attribute 'root_schema_site'. Deprecated in litestar v2.8.0. This attribute will be removed in v3.0.0. Use 'render_plugins' instead. Any 'render_plugin' with path '/' or first 'render_plugin' in list will be served at the OpenAPI root.
    warn_deprecation(

tests/test_health.py::TestHealthEndpoint::test_health_check
  E:\develop\projects\side\ollaix\.venv\Lib\site-packages\_pytest\fixtures.py:1181: PytestRemovedIn9Warning: 'test_health_check' requested an async fixture 'test_client', with no plugin or hook that handled it. This is usually an error, as pytest does not natively support it. This will turn into an error in pytest 9.
  See: https://docs.pytest.org/en/stable/deprecations.html#sync-test-depending-on-async-fixture
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
==================================================== short test summary info ====================================================
FAILED tests/test_health.py::TestHealthEndpoint::test_health_check - Failed: async def functions are not natively supported.
================================================= 1 failed, 2 warnings in 0.24s =================================================
```

================================================
FILE: tests/test_health.py
================================================
from litestar.status_codes import HTTP_200_OK
from litestar.testing import AsyncTestClient


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    async def test_health_check(self, test_client: AsyncTestClient) -> None:
        """Test the health check endpoint."""
        response = await test_client.get("/health")
        data = response.json()

        assert response.status_code == HTTP_200_OK
        assert data["status"] == "healthy"

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
from litestar.exceptions import HTTPException, ImproperlyConfiguredException, ValidationException

from config.exception_handler import app_exception_handler
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
    exception_handlers={
        HTTPException: app_exception_handler,
        ImproperlyConfiguredException: app_exception_handler,
        ValidationException: app_exception_handler,
    },
)



# config/settings.py
from pathlib import Path

from dotenv import load_dotenv
from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import ScalarRenderPlugin
from litestar.openapi.spec import Contact, Tag

from config.env import get_env_var

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# List of allowed origins for CORS (Cross-Origin Resource Sharing)
CORS_ALLOWED_ORIGINS = get_env_var("CORS_ALLOWED_ORIGINS", "*").split(",")

# Base URL for the Ollama API
OLLAMA_BASE_URL = get_env_var("OLLAMA_BASE_URL", "http://localhost:11434")

# API key for the Gemini model
GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")

# OpenAPI configuration
openapi_config = OpenAPIConfig(
    title="Ollaix API",
    version="1.0.0",
    description="Unified API for Ollama and Gemini chat completion models",
    contact=Contact(name="Ollaix Team", email="contact@ollaix.com"),
    path="/",
    tags=[
        Tag(name="Chat", description="Chat completion endpoints with streaming support"),
        Tag(name="Health", description="Health check and monitoring endpoints"),
    ],
    render_plugins=[ScalarRenderPlugin()],
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
from litestar.exceptions import ValidationException
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
        summary="List available models",
        description="Returns a list of all available language models from supported services.",
    )
    async def get_available_models(self) -> ModelsResponse:
        """Fetches all available language models from the registered AI services."""
        return AIServiceInterface.get_all_models()

    @post(
        "/chat/completions",
        summary="Chat completion",
        description="Generates a chat completion response with optional streaming support.",
    )
    async def chat_completion(
        self,
        data: Annotated[
            ChatCompletionRequest,
            Body(
                title="Chat completion request",
                description="Payload containing the chat messages and model configuration.",
            ),
        ],
        ollama_service: AIServiceInterface,
        gemini_service: AIServiceInterface,
        dummy_service: AIServiceInterface,
    ) -> Stream | ChatCompletionResponse:
        """
        Generates a response for a chat completion request.

        Supports streaming if `stream=True` is provided in the request.
        Automatically routes to the appropriate backend service based on the requested model.
        """
        if not data.messages:
            raise ValidationException("Messages list cannot be empty.")

        for message in data.messages:
            if not message.content:
                raise ValidationException("Message content cannot be empty.")
            if not message.role:
                raise ValidationException("Message role cannot be empty.")

        if not isinstance(data.stream, bool):
            raise ValidationException("Stream parameter must be a boolean.")

        service = self._get_service_for_model(
            data.model, ollama_service, gemini_service, dummy_service
        )

        if data.stream:
            return Stream(service.chat_completion_stream(data))  # type: ignore
        return await service.chat_completion(data)

    def _get_service_for_model(
        self,
        model: str,
        ollama_service: AIServiceInterface,
        gemini_service: AIServiceInterface,
        dummy_service: AIServiceInterface,
    ) -> AIServiceInterface:
        """
        Determines the appropriate AI service to handle the request based on the selected model.

        Raises:
            ValidationException: If the provided model is not supported by any service.
        """
        if model in ollama_service.available_models:
            return ollama_service
        if model in gemini_service.available_models:
            return gemini_service
        if model in dummy_service.available_models:
            return dummy_service

        raise ValidationException(f"Model '{model}' is not available.")


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