from fastapi import APIRouter

from app.api.routes import (
    exports,
    health,
    login,
    private,
    rallies,
    reels,
    users,
    utils,
    videos,
)
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(login.router)
api_router.include_router(users.router)
api_router.include_router(utils.router)
api_router.include_router(videos.router)
api_router.include_router(rallies.router)
api_router.include_router(reels.router)
api_router.include_router(exports.router)


if settings.FASTAPI_ENV == "development":
    api_router.include_router(private.router)
