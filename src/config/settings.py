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
