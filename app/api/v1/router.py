from fastapi import APIRouter
from app.api.v1.endpoints import router as endpoints_router
from app.api.v1.telegram import router as telegram_router

router = APIRouter()
router.include_router(endpoints_router)
router.include_router(telegram_router)
