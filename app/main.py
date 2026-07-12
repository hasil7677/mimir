from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.db.duckdb_client import ensure_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_schema()
    yield


app = FastAPI(title="Mimir Engine", lifespan=lifespan)
app.include_router(router, prefix="/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
