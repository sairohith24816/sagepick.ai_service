from fastapi import FastAPI

from app.api import router
from app.core.recommender import initialize_recommender
from app.core.scheduler import start_scheduler, stop_scheduler


app = FastAPI()
app.include_router(router)


@app.on_event("startup")
async def _startup() -> None:
    initialize_recommender()
    start_scheduler()


@app.on_event("shutdown")
async def _shutdown() -> None:
    stop_scheduler()


@app.get("/")
async def read_root():
    return {
        "name": "SagePick AI Service",
        "version": "0.1.0",
        "description": "Welcome to the SagePick AI Service API!",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)