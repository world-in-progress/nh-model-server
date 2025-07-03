from fastapi import APIRouter
from . import hello
from . import model

router = APIRouter(prefix='/api', tags=['api'])

router.include_router(hello.router)
router.include_router(model.router)