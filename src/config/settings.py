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
