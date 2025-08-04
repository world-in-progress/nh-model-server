from fastapi import APIRouter
from . import hello
from . import model
from . import proxy

router = APIRouter(prefix='/api', tags=['api'])

router.include_router(hello.router)
router.include_router(model.router)
router.include_router(proxy.router)