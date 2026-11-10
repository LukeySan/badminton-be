from fastapi import APIRouter

from app.api.routes import health, login, private, reels, users, utils, videos
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(login.router)
api_router.include_router(users.router)
api_router.include_router(utils.router)
api_router.include_router(videos.router)
api_router.include_router(reels.router)


if settings.FASTAPI_ENV == "development":
    api_router.include_router(private.router)
