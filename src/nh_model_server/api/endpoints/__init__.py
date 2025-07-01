from fastapi import APIRouter
from . import hello

router = APIRouter(prefix='/api', tags=['api'])

router.include_router(hello.router)