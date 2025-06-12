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
