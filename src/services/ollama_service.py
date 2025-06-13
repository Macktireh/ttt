import json
from collections.abc import AsyncGenerator
from typing import Any, override

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
    """Service to interact with Ollama."""

    available_models = ["qwen3:4b", "deepseek-r1:7b"]
    provider_name = "ollama"

    def __init__(self) -> None:
        self.client = AsyncClient(host=OLLAMA_BASE_URL)

    @override
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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

    @override
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, Any]:
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

    @override
    def get_model_info(self) -> list[ModelInfo]:
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
        """Converts messages to Ollama format."""
        return [{"role": message.role, "content": message.content} for message in messages]
